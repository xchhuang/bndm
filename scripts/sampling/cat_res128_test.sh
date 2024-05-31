#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 06:00:00
#SBATCH -o slurm_outputs/slurm-%j.out
#SBATCH --gres gpu:1

mkdir -p slurm_outputs

noise_type=$1
scheduler_gamma=$2
scheduler_param=$3
out_channel=$4





# fig 12: cat (128x128), iadb (gaussian) and ours (gaussianBN)

python iadb_bn.py --dataset=cat_res128 --res=128 --batch_size=200 --train_or_test=test --nb_steps=250 --test_samples=30000 --noise_type=gaussian --scheduler_gamma=linear --scheduler_param=1 --out_channel=3

python iadb_bn.py --dataset=cat_res128 --res=128 --batch_size=200 --train_or_test=test --nb_steps=250 --test_samples=30000 --noise_type=gaussianBN --scheduler_gamma=sigmoid --scheduler_param=0.2 --out_channel=6


accelerate launch ddim_diffusers.py --dataset_name="cat_res128" --resolution=128 --train_or_test=test --eval_batch_size=200 --test_samples=30000 --random_flip --output_dir="ddim_cat_res128" --train_batch_size=32 --num_epochs=1000 --gradient_accumulation_steps=1 --learning_rate=1e-4 --lr_warmup_steps=0
