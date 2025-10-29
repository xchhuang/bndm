
# gaussianBN
accelerate launch ddim_bn_diffusers.py --dataset_name="cat_res64" --resolution=64 --random_flip --output_dir="ddim_bn_cat_res64" --train_batch_size=64 --num_epochs=1000 --gradient_accumulation_steps=1 --learning_rate=1e-4 --lr_warmup_steps=0 --noise_type=gaussianBN --scheduler_gamma=sigmoid --scheduler_param=1000
