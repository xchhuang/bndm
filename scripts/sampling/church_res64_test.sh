

# fig 1, 11: lsun_church (64x64), iadb (gaussian) and ours (gaussianBN)

python iadb_bn.py --dataset=church_res64 --res=64 --batch_size=500 --train_or_test=test --nb_steps=250 --test_samples=30000 --noise_type=gaussian --scheduler_gamma=linear --scheduler_param=1 --out_channel=3

python iadb_bn.py --dataset=church_res64 --res=64 --batch_size=500 --train_or_test=test --nb_steps=250 --test_samples=30000 --noise_type=gaussianBN --scheduler_gamma=sigmoid --scheduler_param=1000 --out_channel=6

accelerate launch ddim_diffusers.py --dataset_name="church_res64" --train_or_test=test --eval_batch_size=500 --test_samples=30000 --resolution=64 --random_flip --output_dir="ddim_church_res64" --train_batch_size=2 --num_epochs=1000 --gradient_accumulation_steps=1 --learning_rate=1e-4 --lr_warmup_steps=0





