

# gaussian
# accelerate launch latent_iadb_bn_diffusers.py --dataset_name="celeba_res256" --resolution=256 --random_flip --output_dir="latent_iadb_celeba_res256" --train_batch_size=256 --gradient_accumulation_steps=1 --learning_rate=1e-4 --lr_warmup_steps=0 --out_channels=4 --num_epochs=1000 --noise_type=gaussian

# gaussianBN
accelerate launch latent_iadb_bn_diffusers.py --dataset_name="celeba_res256" --resolution=256 --random_flip --output_dir="latent_iadb_celeba_res256" --train_batch_size=256 --gradient_accumulation_steps=1 --learning_rate=1e-4 --lr_warmup_steps=0 --out_channels=4 --num_epochs=1000 --noise_type=gaussianBN
