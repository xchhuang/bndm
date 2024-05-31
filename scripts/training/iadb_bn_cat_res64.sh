#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 12:00:00
#SBATCH -o slurm_outputs/slurm-%j.out
#SBATCH --gres gpu:1

mkdir -p slurm_outputs

noise_type=$1
scheduler_gamma=$2
scheduler_param=$3
out_channel=$4


# gaussian
python iadb_bn.py --dataset=cat_res64 --res=64 --batch_size=64 --epochs=1000 --train_or_test=train --lr=0.0001 --grad_clip=1.0  --noise_type=gaussian --scheduler_gamma=linear --scheduler_param=1000 --out_channel=3

# gaussianBN
python iadb_bn.py --dataset=cat_res64 --res=64 --batch_size=64 --epochs=1000 --train_or_test=train --lr=0.0001 --grad_clip=1.0  --noise_type=gaussianBN --scheduler_gamma=sigmoid --scheduler_param=1000 --out_channel=6



# sbatch scripts/cat/iadb_cat_res64.sh gaussianBN sigmoid 1000 6
# sbatch scripts/cat/iadb_cat_res64.sh gaussianRN sigmoid 1000 6
# sbatch scripts/cat/iadb_cat_res64.sh gaussian linear 1000 3





