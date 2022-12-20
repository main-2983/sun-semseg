import argparse
import sys
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from mmcv.cnn.utils import revert_sync_batchnorm

from mmseg.core.hook.torch_hooks import IOHook
from mmseg.apis.inference import init_segmentor, inference_segmentor


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
    if args.input:
        activation = hook.input[0]
    else:
        activation = hook.output

    print(f"Activation map at layer {str(layer)} has shape: {activation.shape}")
    activation = activation[0].cpu().numpy() # drop batch dim
    c, h, w = activation.shape
    nrows, ncols = args.num_chans or int(np.sqrt(c)), args.num_chans or int(np.sqrt(c))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
    for i in tqdm(range(nrows)):
        for j in range(ncols):
            axes[i, j].imshow(activation[i + j, :, :])
            axes[i, j].axis('off')
    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        plt.savefig(f"{args.save_path}/vis_{layer}_{s}.png")
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
