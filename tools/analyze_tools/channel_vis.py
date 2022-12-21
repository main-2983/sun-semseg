import argparse
import sys
import os

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from mmcv.cnn.utils import revert_sync_batchnorm

from mmseg.core.hook.torch_hooks import IOHook
from mmseg.apis.inference import init_segmentor, inference_segmentor
from mmseg.models.utils import nlc_to_nchw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('data', help='path to input image')
    parser.add_argument('--device', '-d', default='cuda:0',
                        help='device')
    parser.add_argument('--print-model', action='store_true',
                        help='print all layers in model')
    parser.add_argument('--layer', default=None, type=str,
                        help='layer to visualize activation maps')
    parser.add_argument('--input', action='store_true',
                        help='visualize input features map to layer')
    parser.add_argument('--num-chans', default=None, type=int,
                        help='number of channels to visualize')
    parser.add_argument('--avg', action='store_true',
                        help='get spatial average across channels')
    parser.add_argument('--save-path', default=None, type=str,
                        help='path to save the visualization')
    parser.add_argument('--show', action='store_true',
                        help='show the visualization')
    return parser.parse_args()


def main():
    args = parse_args()
    model = init_segmentor(args.config,
                           args.checkpoint,
                           args.device)
    model = revert_sync_batchnorm(model)

    if args.print_model:
        print(model)
        sys.exit()

    layer = eval(args.layer)
    hook = IOHook(layer)
    inference_segmentor(model, args.data)

    s = "input" if args.input else "output"
    _inp = hook.input
    _out = hook.output
    if args.input:
        if len(hook.input) == 2:
            activation, hw_shape = hook.input
        else:
            activation, hw_shape = hook.input[0], None
    else:
        if len(hook.output) == 2:
            activation, hw_shape = hook.output
        else:
            activation, hw_shape = hook.output, None

    print(f"Activation map at layer {str(layer)} has shape: {activation.shape}")
    if activation.ndim == 3:
        if hw_shape is None:
            hw_shape = _inp[1] if len(_inp) == 2 else _out[1]
        b, _, c = activation.shape
        new_shape = (b, hw_shape[0], hw_shape[1], c)
        print(f"Activation map had shape {activation.shape} is now reshaped to {new_shape}")
        activation = nlc_to_nchw(activation, hw_shape)
    activation = activation[0].cpu().numpy()
    c, h, w = activation.shape
    if args.avg:
        activation = torch.mean(torch.tensor(activation), dim=0, keepdim=True)
        activation = activation[0].cpu().numpy()
        fig, ax = plt.subplots()
        ax.imshow(activation)
        if args.save_path is not None:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            plt.savefig(f"{args.save_path}/vis_{args.layer}_avg.png")
        if args.show:
            plt.show()
    c = args.num_chans or c
    nrows, ncols = int(np.sqrt(c)), int(np.sqrt(c))
    print("Creating subplot, please wait...")
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
    num_c = 0
    for i in tqdm(range(nrows)):
        for j in range(ncols):
            axes[i, j].imshow(activation[num_c, :, :])
            axes[i, j].axis('off')
            num_c += 1
    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        plt.savefig(f"{args.save_path}/vis_{args.layer}_{s}.png")
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
