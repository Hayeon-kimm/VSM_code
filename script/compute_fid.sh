#! /bin/sh

GPU=$1


CUDA_VISIBLE_DEVICES=$GPU \

python metrics/metrics.py ./fsddpm/cifar100_vfsddpm_vit_lag_mean_patch_sigma_deterministic/sampling-conditional-out-distro-2023-05-22-11-22-42-091392/full_samples_conditional_10000x32x32x3_out-distro_5.npz ./fsddpm/cifar100_vfsddpm_vit_lag_mean_patch_sigma_deterministic/sampling-conditional-out-distro-2023-06-16-15-42-35-029888/full_samples_conditional_10000x32x32x3_out-distro_5.npz

