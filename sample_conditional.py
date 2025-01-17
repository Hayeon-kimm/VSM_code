import copy
import math
from functools import partial
from inspect import isfunction
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch import einsum, nn, optim
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

from dataset import create_loader
from model import select_model
from utils.util import set_seed

import argparse
import os
import os.path as osp

import numpy as np
import torch as th
import torch.distributed as dist

from model import select_model
from model.set_diffusion import dist_util, logger
from model.set_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils.path import set_folder
from utils.util import count_params, set_seed

DIR = set_folder()


def main():
    args = create_argparser().parse_args()
    print(args)
    # dct = vars(args)
    # for k in sorted(dct):
    #     print(k, dct[k])
    # print()

    # dist_util.setup_dist()
    logger.configure(
        dir=DIR,
        mode="-".join(["sampling-conditional", args.mode_conditional_sampling]),
        args=args,
        tag="",
    )

    logger.log("creating model and diffusion...")
    model = select_model(args)(args)
    print(count_params(model))

    print(args.model_path)
    _path = list(args.model_path.split("/"))
    for j in range(len(_path)):
        _p = list(_path[j].split("_"))
        _p = [i for i in _p if i not in ["", None, "None", "none"]]
        _path[j] = "_".join(_p)
    # _path = list(args.model_path.split("_"))
    # _path = [ i for i in _path if i not in ["", None, "None", "none"] ]
    model_path = "/".join(_path)
    # print(model_path)

    model.load_state_dict(
        dist_util.load_state_dict(osp.join(DIR, model_path), map_location="cpu")
    )

    model.to(args.device)
    f_save = open('./memory.pkl', 'rb')
    model.memory = pickle.load(f_save)['memory_ke']
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating data loader...")

    args.transfer = False
    if args.transfer:
        if args.image_size == 32:
            args.dataset = "minimagenet"
        else:
            args.dataset = "cub"

    if args.mode_conditional_sampling == "out-distro":
        loader = create_loader(args, split="test", shuffle=True)
    else:
        loader = create_loader(args, split="train", shuffle=True)

    logger.log("conditional sampling...")
    all_images = []
    all_labels = []
    all_conditioning_images = []

    while len(all_images) * args.batch_size < args.num_samples * args.k:
        with  torch.no_grad():
            try:
                x_set = next(loader)
            except StopIteration:
                loader = iter(loader)
                x_set = next(loader)

            x_set, label = x_set
            x_set = x_set.to(args.device)
            if args.model == "ddpm":
                c = None
            else:
                c = model.sample_conditional(x_set, args.sample_size)["c"]

                c = c.unsqueeze(1)  # attention here
                c = torch.repeat_interleave(c, args.k * args.sample_size, dim=1)
                c = c.view(-1, c.size(-2), c.size(-1))


            # c_list = []
            # for i in range(x_set.shape[1]):
            #     x_set_tmp = x_set.clone()
            #     x_set_tmp[:, i] = 0
            #     # build c
            #     out = model.sample_conditional(x_set_tmp, args.sample_size, 1)
            #     c_tmp = out["c"]
            #     c_list.append(c_tmp.unsqueeze(1))
            #     del x_set_tmp
            #     th.cuda.empty_cache()

            # c = th.cat(c_list, dim=1)
            # c = c.view(-1, c.size(-2), c.size(-1))

            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes

            sample_fn = (
                model.diffusion.p_sample_loop
                if not args.use_ddim
                else model.diffusion.ddim_sample_loop
            )
            try:
                sample = sample_fn(
                    model.generative_model,
                    (
                        args.batch_size * args.sample_size * args.k,
                        args.in_channels,
                        args.image_size,
                        args.image_size,
                    ),
                    c=c,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                # [-1, 1] ---> [0, 2] ---> [0, 255]
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                # fix this
                # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                all_images.extend([sample.cpu().numpy()])

                x_set = x_set.view(-1, args.in_channels, args.image_size, args.image_size)
                x_set = ((x_set + 1) * 127.5).clamp(0, 255).to(th.uint8)
                x_set = x_set.permute(0, 2, 3, 1)
                x_set = x_set.contiguous()
                # normalize x_set
                all_conditioning_images.extend([x_set.cpu().numpy()])

                if args.class_cond:
                    # gathered_labels = [
                    #     th.zeros_like(classes) for _ in range(dist.get_world_size())
                    # ]
                    # dist.all_gather(gathered_labels, classes)
                    # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                    all_labels.extend([classes.cpu().numpy()])

                logger.log(f"created {len(all_images) * args.batch_size} samples")
            # problem with the last batch in the loader
            except RuntimeError:
                continue

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    arr_cond = np.concatenate(all_conditioning_images, axis=0)
    arr_cond = arr_cond[: args.num_samples]

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    # if dist.get_rank() == 0:
    shape_str = "x".join([str(x) for x in arr.shape])
    if args.transfer:
        out_path = os.path.join(logger.get_dir(),
                                f"full_samples_conditional_{shape_str}_{args.mode_conditional_sampling}_{args.sample_size}_transfer_{args.dataset}.npz")
    else:
        out_path = os.path.join(logger.get_dir(),
                                f"full_samples_conditional_{shape_str}_{args.mode_conditional_sampling}_{args.sample_size}.npz")
    logger.log(f"saving to {out_path}")
    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr, arr_cond)

    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50,
        batch_size=32,
        batch_size_eval=32,
        use_ddim=False,
        model_path="/mnt/wwn-0x5000c500e040f04e-part1/lzy/rebuttal/开源code/VSM-DIFFUSION-code/fsddpm/cifar100_vfsddpm_vit_lag_mean_patch_sigma_variational_discrete/run-2023-08-09-16-00-34-650372/ema_0.995_000001.pt",
        k=1,  # multiplier for conditional samples
        model="vfsddpm",
        dataset="cifar100",
        pool='mean_patch', # mean, mean_patch
        image_size=32,
        sample_size=5,
        patch_size=4,
        hdim=256,
        in_channels=3,
        encoder_mode="vit",
        context_channels=256,
        num_classes=1,
        mode_context="variational_discrete",
        mode_conditional_sampling="out-distro",
        mode_conditioning='lag', # film, lag,
        augment=False,
        device="cuda",
        data_dir="./data",
        transfer=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    s = set_seed(0)
    main()
