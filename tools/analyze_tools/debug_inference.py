import os
import os.path as osp
import time
import argparse
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import torch
import mmcv
from mmcv.image import tensor2imgs
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, wrap_fp16_model, load_checkpoint

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import get_device, setup_multi_processes, build_dp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # init distributed env first, since logger depends on the dist info.
    distributed = False

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))

    # build the dataloader
    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()

    cfg.device = get_device()
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

        # implement own logic for side by side comparison between gt mask and predict mask
        model.eval()
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        # The pipeline about how the data_loader retrieval samples from dataset:
        # sampler -> batch_sampler -> indices
        # The indices are passed to dataset_fetcher to get data from dataset.
        # data_fetcher -> collate_fn(dataset[index]) -> data_sample
        # we use batch_sampler to get correct data idx
        loader_indices = data_loader.batch_sampler

        for batch_indices, data in zip(loader_indices, data_loader):
            with torch.no_grad():
                result = model(return_loss=False, **data)
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                out_file = osp.join(args.work_dir, img_meta['ori_filename'])

                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]

                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                palette = dataset.PALETTE
                palette = np.array(palette)
                assert palette.shape[0] == len(model.module.CLASSES)
                assert palette.shape[1] == 3
                assert len(palette.shape) == 2
                assert 0 < args.opacity <= 1.0

                # draw prediction
                img_show = img_show.copy()
                seg = result[0]
                color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

                for label, color in enumerate(palette):
                    color_seg[seg == label, :] = color

                # convert to BGR
                #color_seg = color_seg[..., ::-1]

                pimg = img_show * (1 - args.opacity) + color_seg * args.opacity
                pimg = pimg.astype(np.uint8)

                # draw ground truth
                gt_mask = dataset.get_gt_seg_map_by_idx(batch_indices[0]) - 1
                color_gt = np.zeros(color_seg.shape, dtype=np.uint8)
                for label, color in enumerate(palette):
                    color_gt[gt_mask == label, :] = color
                # convert to BGR
                #color_gt = color_gt[..., ::-1]

                gtimg = img_show * (1 - args.opacity) + color_gt * args.opacity
                gtimg = gtimg.astype(np.uint8)

                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                legend_handles = []

                # store information about label and color of pred and gt
                existing_labels, existing_colors = [], []
                existing_labels_idxs = list(np.unique(seg))
                existing_labels_idxs.extend(list(np.unique(gt_mask)))
                existing_labels_idxs = set(existing_labels_idxs)
                for _l in existing_labels_idxs:
                    if _l != 255:
                        existing_colors.append(dataset.PALETTE[_l])
                for _l in existing_labels_idxs:
                    if _l != 255:
                        existing_labels.append(dataset.CLASSES[_l])

                for idx, (c, p) in enumerate(zip(existing_labels, existing_colors)):
                    legend_handles.append(
                        mpatches.Patch(
                            color=np.asarray(p)/255,
                            label=c
                        )
                    )

                axes[0].imshow(pimg)
                axes[1].imshow(gtimg)
                axes[0].axis('off')
                axes[1].axis('off')
                fig.legend(handles=legend_handles,
                            ncol=12)

                plt.savefig(out_file)
                plt.close()

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()


if __name__ == '__main__':
    main()