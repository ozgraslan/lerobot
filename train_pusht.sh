#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=/network/scratch/o/ozgur.aslan/train_dino_diffusion_1.out

module load cuda/12.6.0/cudnn/9.3
module load anaconda/3
conda activate /network/scratch/o/ozgur.aslan/conda_envs/lerobot

export HF_HOME="/network/scratch/o/ozgur.aslan/huggingface"
export TORCH_HOME="/network/scratch/o/ozgur.aslan/torch_cache"

export WANDB_DIR=$SCRATCH
export WANDB_MODE=online

lerobot-train \
    --output_dir=outputs/train/diffusion_dino_pusht_1 \
    --policy.repo_id=ozgraslan/diffusion_dino_pusht \
    --policy.type=diffusion_dino \
    --dataset.repo_id=ozgraslan/pusht_224 \
    --env.type=pusht_224 \
    --seed=100000 \
    --batch_size=64 \
    --steps=200000 \
    --eval_freq=5000 \
    --save_freq=5000 \
    --wandb.enable=true \
    --wandb.project=lerobot-diffusion-pusht \
    --wandb.entity=jakd9 \
