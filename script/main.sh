#! /bin/sh

GPU=3
RUN=vfsddpm_cifar100_vit_lag_meanpatch_vid


IMAGE_SIZE=32
IN_CHANNELS=3
MODEL=vfsddpm
DATASET=cifar100
ENCODER=vit
CONDITIONING=lag
POOLING=mean_patch
CONTEXT=variational
PATCH_SIZE=8
SAMPLE_SIZE=5
BATCH_SIZE=32
BATCH_SIZE_EVAL=100


MODEL_FLAGS="--image_size ${IMAGE_SIZE} --in_channels ${IN_CHANNELS} --num_channels 64 
--context_channels 256 --dropout 0.2 --num_res_blocks 2 --model ${MODEL} --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE_EVAL} --dataset ${DATASET} --ema_rate 0.995"
ENCODER_FLAGS="--patch_size ${PATCH_SIZE} --encoder_mode ${ENCODER} --sample_size ${SAMPLE_SIZE} 
--mode_conditioning ${CONDITIONING} --pool ${POOLING} --mode_context ${CONTEXT}"

CUDA_VISIBLE_DEVICES=$GPU \
python main.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $ENCODER_FLAGS



