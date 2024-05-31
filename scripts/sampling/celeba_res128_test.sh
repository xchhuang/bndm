
# fig 12: celeba (128x128), iadb (gaussian) and ours (gaussianBN)

python iadb_bn.py --dataset=celeba_res128 --res=128 --batch_size=200 --train_or_test=test --nb_steps=250 --test_samples=30000 --noise_type=gaussian --scheduler_gamma=linear --scheduler_param=1 --out_channel=3

python iadb_bn.py --dataset=celeba_res128 --res=128 --batch_size=200 --train_or_test=test --nb_steps=250 --test_samples=30000 --noise_type=gaussianBN --scheduler_gamma=sigmoid --scheduler_param=0.2 --out_channel=6


accelerate launch ddim_diffusers.py --dataset_name="celeba_res128" --train_or_test=test --eval_batch_size=200 --test_samples=30000 --resolution=128 --random_flip --output_dir="ddim_celeba_res128" --train_batch_size=2 --num_epochs=1000 --gradient_accumulation_steps=1 --learning_rate=1e-4 --lr_warmup_steps=0

