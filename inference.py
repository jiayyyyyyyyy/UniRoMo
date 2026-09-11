import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.dataset import AVDataset
from models.basic_model import AVClassifier
from utils.utils import setup_seed, weight_init


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMA-D', type=str,
                        help='Currently, we only support CREMA-D')
    parser.add_argument('--modulation', default='Normal', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE', 'QMF'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--ckpt_load_path_train', '--checkpoint',
                        dest='ckpt_load_path_train', required=True, type=str,
                        help='path to the trained checkpoint')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')
    parser.add_argument('--lorb', default='base', type=str,
                        help='model_select in [large, base]')
    parser.add_argument('--gs_flag', action='store_true')
    parser.add_argument('--av_alpha', default=0.5, type=float,
                        help='audio/visual fusion alpha in GS')
    parser.add_argument('--dynamic', action='store_true',
                        help='use dynamic fusion in GS')
    parser.add_argument('--clip', action='store_true',
                        help='run using CLIP pre-trained feature')
    parser.add_argument('--num_workers', default=8, type=int)
    return parser.parse_args()


def calculate_entropy(output):
    probabilities = F.softmax(output, dim=1)
    log_probabilities = torch.log(probabilities + 1e-8)
    entropy = -torch.sum(probabilities * log_probabilities, dim=1)
    class_count = output.shape[1]
    entropy = entropy / np.log(class_count)
    return entropy


def calculate_gating_weights(encoder_output_1, encoder_output_2):
    entropy_1 = calculate_entropy(encoder_output_1)
    entropy_2 = calculate_entropy(encoder_output_2)
    max_entropy = torch.max(entropy_1, entropy_2)

    gating_weight_1 = torch.exp(max_entropy - entropy_1)
    gating_weight_2 = torch.exp(max_entropy - entropy_2)
    sum_weights = gating_weight_1 + gating_weight_2

    gating_weight_1 = gating_weight_1 / (sum_weights + 1e-8)
    gating_weight_2 = gating_weight_2 / (sum_weights + 1e-8)
    return gating_weight_1.view(-1, 1), gating_weight_2.view(-1, 1)


def valid(args, model, device, dataloader, gs_flag=False, av_alpha=0.5):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'CREMAD' or args.dataset == 'CREMA-D':
        n_classes = 6
    else:
        n_classes = 6

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, data_packet in enumerate(dataloader):
            spec, image, label, idx = data_packet
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if len(spec.shape) == 3:
                spec = spec.unsqueeze(1).float()
            else:
                spec = spec.float()
            image = image.float()

            if not gs_flag:
                # Same forward path as the original valid() function.
                a, v, out = model(spec, image)

                out_v = (
                    torch.mm(
                        v,
                        torch.transpose(
                            model.module.fusion_module.fc_out.weight[:, 512:],
                            0,
                            1,
                        ),
                    )
                    + model.module.fusion_module.fc_out.bias / 2
                )
                out_a = (
                    torch.mm(
                        a,
                        torch.transpose(
                            model.module.fusion_module.fc_out.weight[:, :512],
                            0,
                            1,
                        ),
                    )
                    + model.module.fusion_module.fc_out.bias / 2
                )

            elif gs_flag:
                result = model(spec, image)
                a, v = result[0], result[1]

                out_a = model.module.fusion_module.fc_out(a)
                out_v = model.module.fusion_module.fc_out(v)

                if args.dynamic:
                    audio_conf, visual_conf = calculate_gating_weights(out_a, out_v)
                    out = out_a * audio_conf + out_v * visual_conf
                else:
                    out = av_alpha * out_a + (1 - av_alpha) * out_v

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):
                fused_pred = np.argmax(prediction[i].cpu().data.numpy())
                visual_pred = np.argmax(pred_v[i].cpu().data.numpy())
                audio_pred = np.argmax(pred_a[i].cpu().data.numpy())
                label_id = int(label[i].cpu().item())
                num[label_id] += 1.0

                if label_id == fused_pred:
                    acc[label_id] += 1.0
                if label_id == visual_pred:
                    acc_v[label_id] += 1.0
                if label_id == audio_pred:
                    acc_a[label_id] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    setup_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.lorb == 'base' or args.lorb == 'large':
        model = AVClassifier(args)
        model.apply(weight_init)
    else:
        raise NotImplementedError('Only base/large is supported for CREMA-D.')

    loaded_dict = torch.load(args.ckpt_load_path_train, map_location='cpu')
    state_dict = loaded_dict['model'] if 'model' in loaded_dict else loaded_dict

    # Checkpoints saved from DataParallel have an extra "module." prefix.
    if state_dict and all(key.startswith('module.') for key in state_dict):
        state_dict = {
            key[len('module.'):]: value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    print('Trained model loaded!')

    model.to(device)
    # valid() keeps the same model.module access pattern as the original code.
    model = torch.nn.DataParallel(model)

    if args.dataset == 'CREMA-D' or args.dataset == 'CREMAD':
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError(
            'Incorrect dataset name {}'.format(args.dataset)
        )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    acc, acc_a, acc_v = valid(
        args,
        model,
        device,
        test_dataloader,
        gs_flag=args.gs_flag,
        av_alpha=args.av_alpha,
    )
    print(
        'Test Acc: {:.4f} (Audio: {:.4f}, Visual: {:.4f})'.format(
            acc,
            acc_a,
            acc_v,
        )
    )


if __name__ == '__main__':
    main()
