
# gaussian
python iadb_bn.py --dataset=celeba_re128 --res=128 --batch_size=64 --epochs=1000 --train_or_test=train --lr=0.0001 --grad_clip=1.0  --noise_type=gaussian --scheduler_gamma=linear --scheduler_param=1000 --out_channel=3

# gaussianBN
python iadb_bn.py --dataset=celeba_re128 --res=128 --batch_size=64 --epochs=1000 --train_or_test=train --lr=0.0001 --grad_clip=1.0  --noise_type=gaussianBN --scheduler_gamma=sigmoid --scheduler_param=0.2 --out_channel=6




