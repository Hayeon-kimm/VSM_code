#! /bin/sh

GPU=3
RUN=vfsddpm_cifar100_vit_lag_meanpatch_vid_outdistro
DATE=2023-05-13-22-31-50-133984
STEP=200000
NSAMPLES=10000

run()
{

IMAGE_SIZE=32
IN_CHANNELS=3
MODEL=vfsddpm
DATASET=cifar100
ENCODER=vit
CONDITIONING=lag
POOLING=mean_patch
CONTEXT=variational
SAMPLING=out-distro
PATCH_SIZE=8
SAMPLE_SIZE=5
BATCH_SIZE=100

# use ema model for sampling
MODEL_FLAGS="--image_size ${IMAGE_SIZE} --in_channels ${IN_CHANNELS} --num_channels 64 
--context_channels 256 --sample_size 5 --model ${MODEL} 
--model_path ${DATASET}_${MODEL}_${ENCODER}_${CONDITIONING}_${POOLING}_sigma_${CONTEXT}/run-${DATE}/ema_0.995_${STEP}.pt  --learn_sigma True"
SAMPLE_FLAGS="--batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} --num_samples ${NSAMPLES} --timestep_respacing 250 
--mode_conditional_sampling ${SAMPLING} --dataset ${DATASET}"
ENCODER_FLAGS="--patch_size ${PATCH_SIZE} --encoder_mode ${ENCODER} --sample_size ${SAMPLE_SIZE} 
--mode_conditioning ${CONDITIONING} --pool ${POOLING}  --mode_context ${CONTEXT}"

CUDA_VISIBLE_DEVICES=$GPU \
python sample_conditional.py $MODEL_FLAGS $SAMPLE_FLAGS $ENCODER_FLAGS

}
