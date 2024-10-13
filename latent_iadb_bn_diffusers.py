import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path
import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
import diffusers
from diffusers import UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoencoderKL

import torchvision
import numpy as np
from PIL import Image
import platform

from diffusers import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin

import matplotlib.pyplot as plt
import random
import sys
sys.path.append('../')
from bluenoise.get_noise_recent import get_noise_v2
from input_args import parse_args
import lmdb

args = parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


logging_dir = os.path.join(args.output_dir, args.logging_dir)
accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.logger,
    project_config=accelerator_project_config,
    kwargs_handlers=[kwargs],
)

dimension = 3
cov_mat_L = np.load('bluenoise/cov_gaussianBN_L_res{:}_d{:}.npz'.format(64, dimension))['x'].astype(np.float32)
if args.noise_type in ['gaussianRN']:
    cov_mat_L = np.load('bluenoise/cov_gaussianRN_L_res{:}_d{:}.npz'.format(64, dimension))['x'].astype(np.float32)
cov_mat_L = torch.from_numpy(cov_mat_L).to(accelerator.device)

generator = torch.Generator(device=accelerator.device).manual_seed(seed)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae = vae.eval().to(accelerator.device).half()



class IADBScheduler(SchedulerMixin, ConfigMixin):
    """
    IADBScheduler is a scheduler for the Iterative α-(de)Blending denoising method. It is simple and minimalist.
    For more details, see the original paper: https://arxiv.org/abs/2305.03486 and the blog post: https://ggx-research.github.io/publication/2023/05/10/publication-iadb.html
    """
    def __init__(self, num_train_timesteps: int = 1000):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x_alpha: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        backward
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # print('timestep:', timestep, self.num_inference_steps)
        alpha = (timestep + 1) / self.num_inference_steps
        alpha_next = (timestep) / self.num_inference_steps

        gamma = (timestep + 1) / self.num_inference_steps
        gamma_next = (timestep) / self.num_inference_steps

        d = model_output

        
        if args.noise_type in ['gaussianBN', 'gaussianRN']:
            if args.out_channels == 4:
                x_alpha = x_alpha + (alpha - alpha_next) * d
            elif args.out_channels == 8:
                # print('t:', timestep, gamma - gamma_next)
                x_alpha = x_alpha + (alpha - alpha_next) * d[:, :4, :, :] + (gamma - gamma_next) * d[:, 4:, :, :]
            else:
                raise NotImplementedError
        elif args.noise_type in ['gaussian']:
            x_alpha = x_alpha + (alpha - alpha_next) * d
        else:
            raise NotImplementedError
        

        return x_alpha

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        alpha: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        """
        forward
        """
        # return original_samples * alpha.view(-1, 1, 1, 1) + noise * (1 - alpha.view(-1, 1, 1, 1))
        return (1 - alpha).view(-1, 1, 1, 1) * original_samples +  alpha.view(-1, 1, 1, 1) * noise
    

    def __len__(self):
        return self.num_train_timesteps



class IADBPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)





def vae_encode(image_t: torch.Tensor) -> torch.Tensor:
    image_t = image_t.to(device=accelerator.device, dtype=torch.float16).mul(2).sub(1)
    with torch.no_grad():
        latent_dist = vae.encode(image_t).latent_dist
    latents = latent_dist.sample(generator=generator)
    latents = 0.18215 * latents
    
    if False:
        plt.figure(1)
        for i in range(4):
            plt.subplot(1, 4, i+1)
            latents_plot = latents[0, i].detach().cpu().numpy()
            # latents_plot = (latents_plot - latents_plot.min()) / (latents_plot.max() - latents_plot.min())
            print('latents:', latents_plot.shape, latents_plot.min(), latents_plot.max())
            plt.imshow(latents_plot)
        plt.show()
    return latents


def vae_decode(latents: torch.Tensor) -> Image.Image:
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents.half()).sample
    # image = (image*0.5 + 0.5).clamp(0, 1)
    # return torchvision.transforms.functional.to_pil_image(image[0])
    return image


def images_to_latents(lmdb_path: str, folder: str, resolution: int=512):
    image_paths = [os.path.join(root, file) for root, _, files in os.walk(folder) for file in files
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    transform = transforms.Compose([transforms.Resize(resolution), transforms.CenterCrop(resolution)])

    # 2x for hflip, 2 bytes per float16
    max_size = int(1.5 * len(image_paths) * 2 * (4*64*64) * 2)

    env = lmdb.open(lmdb_path, map_size=max_size)
    with env.begin(write=True) as txn:
        for i, image_path in enumerate(tqdm(image_paths)):
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            for f in range(2):
                latent = vae_encode(torchvision.transforms.functional.to_tensor(image).unsqueeze(0))

                if False:
                    img = np.asarray(vae_decode(latent).convert('RGB')) / 255.0
                    print('img:', img.shape, img.min(), img.max())
                    plt.figure(1)
                    plt.subplot(1, 2, 1)
                    plt.imshow(img)
                    plt.subplot(1, 2, 2)
                    plt.imshow(image)
                    plt.show()

                txn.put(str(i*2+f).encode('utf-8'), latent.cpu().numpy().tobytes())
                image = transforms.functional.hflip(image)
    env.close()  



class LatentsDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path: str, resolution: int=512):
        self.latents = []
        env = lmdb.open(lmdb_path, readonly=True)
        stats = env.stat()
        num_entries = stats['entries']
        with env.begin() as txn:
            for index in tqdm(range(num_entries), desc="Loading latents"):
                buffer = txn.get(str(index).encode('utf-8'))
                tensor = torch.from_numpy(np.frombuffer(buffer, dtype=np.float16))
                latents = tensor.view(4, resolution//8, resolution//8)
                self.latents.append(latents)
        env.close()
        print(f"Loaded {len(self.latents)} latents")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]
    


def main():
    
    DATA_FOLDER = './data/{:}'.format(args.dataset_name)
    lmdb_path = 'data/{:}_latent_lmdb'.format(args.dataset_name)
    first_time = False   # False, True
    if first_time:
        images_to_latents(lmdb_path, DATA_FOLDER)

     # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # dataset = torchvision.datasets.ImageFolder(root=DATA_FOLDER, transform=augmentations)
    dataset = LatentsDataset(lmdb_path, args.resolution)


    args.output_dir = args.output_dir + '_{:}'.format(args.noise_type)
    if args.use_ema:
        args.output_dir = args.output_dir + '_ema'

    args.output_dir = os.path.join('results_gaussianBN', args.output_dir)
    
    # local debug
    # if platform.system() == "Windows":
    #     args.train_batch_size = 2
    #     args.eval_batch_size = 2
    
    if args.noise_type in ['gaussianBN', 'gaussianRN']:
        args.out_channels *= 2
    

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    # Initialize the model
    if args.model_config_name_or_path is None:
        
        if args.resolution == 64:
            block_out_channels=(128, 128, 256, 256, 512, 512)
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
        
        elif args.resolution in [128]:
            block_out_channels=(128, 128, 128, 256, 256, 512, 512)
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")

        # newly updated
        elif args.resolution in [256]:
            # block_out_channels=(128, 128, 128, 256, 256, 512, 512)
            # down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
            # up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
            block_out_channels=(128, 256, 256)
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D")
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D")

        elif args.resolution in [512]:
            block_out_channels=(128, 128, 256, 256, 512, 512)
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
        
        else:
            raise ValueError(f"Unsupported resolution: {args.resolution}")
        
        model = UNet2DModel(
                sample_size=args.resolution,
                in_channels=4,
                out_channels=args.out_channels,
                layers_per_block=2,
                block_out_channels=block_out_channels,
                down_block_types=down_block_types,
                up_block_types=up_block_types
            )
        
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            # if xformers_version == version.parse("0.0.16"):
            #     logger.warn(
            #         "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            #     )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    noise_scheduler = IADBScheduler(num_train_timesteps=args.ddpm_num_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True)

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)


    if args.train_or_test == 'test':
        print('===> Start testing!')
        # load model and scheduler
        if not os.path.exists(args.output_dir + '/images'):
            os.makedirs(args.output_dir + '/images', exist_ok=True)
        if not os.path.exists(args.output_dir + '/seqs'):
            os.makedirs(args.output_dir + '/seqs', exist_ok=True)
        
        # run pipeline in inference (sample random noise and denoise)
        # image = pipe(eta=0.0, num_inference_steps=1000)
        # unet = accelerator.unwrap_model(model)
        # pipeline = DDIMPipeline(unet=unet, scheduler=noise_scheduler).from_pretrained(args.output_dir)
        # pipeline = DDIMPipeline.from_pretrained(args.output_dir).to(accelerator.device)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('device:', device)
        scheduler = IADBScheduler.from_pretrained(args.output_dir+"/scheduler")
        scheduler.set_timesteps(args.ddpm_num_inference_steps)
        model = UNet2DModel.from_pretrained(args.output_dir+"/unet", use_safetensors=True).to(accelerator.device)
        # model = torch.nn.DataParallel(model)
        model.eval()

        path = '../iadb/results_gaussianBN_adamw_v4_submitted/{:}_gaussian_linear_outc3_seed0/{:}_iadb_gwn_steps250'.format(args.dataset_name, args.dataset_name)
        cnt = 0
        num_batch = int(args.test_samples // args.eval_batch_size)
        for i in tqdm(range(num_batch)):
            
            # images = pipeline(batch_size=args.eval_batch_size, num_inference_steps=args.ddpm_num_inference_steps).images
            # images = (images * 255).round().astype("uint8")
            # noise = torch.randn((args.eval_batch_size, 4, (args.resolution//8), (args.resolution//8))).to(device)
            noise = np.random.randn(args.eval_batch_size, 4, (args.resolution//8), (args.resolution//8)).astype(np.float32)

            # just to reproduce figure 9
            if True:
                if i == 0:
                    shown_image_idx = [2, 7, 31, 48]
                    noise = noise[shown_image_idx]
                elif i == 1:
                    shown_image_idx = [6]
                    noise = noise[shown_image_idx]
                else:
                    continue
            
            # noise = np.load(path + '/noise/noise_batch{:}_idx{:0>5}.npz'.format(args.eval_batch_size, i))['noise']
            noise = torch.from_numpy(noise).to(device)
            # print('noise:', noise.shape)
            
            x_alpha = noise
            # seqs = [noise[0:1]]
            seqs = []
            num_steps = scheduler.num_inference_steps

            for t in reversed(list(range(0, num_steps))):
                alpha = (t + 1) / num_steps
                with torch.no_grad():
                    # print('alpha:', t, num_steps, alpha)
                    model_output = model(x_alpha, torch.tensor(alpha, device=x_alpha.device), return_dict=False)[0]
                x_alpha = scheduler.step(model_output, t, x_alpha)

                if t == 0:
                    x_recon = vae_decode(x_alpha)
                    # print('x_recon:', x_recon.shape)
                    seqs.append(x_recon[0:1])
            
            # print('seqs:', len(seqs))
            # print('x_alpha:', x_alpha.shape, x_alpha.min(), x_alpha.max())

            images = (x_recon / 2.0 + 0.5).clamp(0, 1)
            images = (images.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()

            # print('images:', images.shape)

            if False:
                for i, image in enumerate(seqs):
                    if i == len(seqs) - 1:
                        seq = (seqs[i] / 2 + 0.5).clamp(0, 1)
                        seq = (seq.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
                        # print('seq1:', seq.shape, seq.min(), seq.max())
                    else:
                        seq = seqs[i]
                        seq = (seq - seq.min()) / (seq.max() - seq.min())
                        seq = (seq.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
                        # image = (image / 2 + 0.5).clamp(0, 1)
                        # image = (image.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
                        # print('seq2:', seq.shape, seq.min(), seq.max())
                    Image.fromarray(seq[0]).save(args.output_dir + "/seqs/iadb_img{:0>5}_step{:}.png".format(cnt, i*25))


            if args.noise_type in ['gaussian']:
                save_name = 'iadb_gwn'
            elif args.noise_type in ['gaussianBN']:
                save_name = 'iadb_gwn2gbn'
            else:
                raise ValueError(f"Unsupported noise type: {args.noise_type}")
            
            for i, image in enumerate(images):
                cnt += 1
                Image.fromarray(image).save(args.output_dir + "/images/{:}_{:0>5}.png".format(save_name, cnt))

            
        print('Done.')

        return

    print('===> Start training!')
    # print('first_epoch:', first_epoch, args.num_epochs)

    losses = []
    for epoch in tqdm(range(first_epoch, args.num_epochs)):
        model.train()
        # progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        # progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):

            # print('batch:', batch[0].shape, batch[1].shape)
            clean_images = batch.to(weight_dtype)
            bsz = clean_images.shape[0]

            # Sample a random timestep for each image
            # random sampling
            # timesteps = torch.randint(
            #     0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            # ).long()

            # antithetic sampling borrowed from ddim
            # timesteps = torch.randint(low=0, high=args.ddpm_num_steps, size=(bsz//2,)).to(clean_images.device)
            # timesteps = torch.cat([timesteps, args.ddpm_num_steps - timesteps - 1], dim=0)[:bsz].long()
            timesteps = torch.randint(low=1, high=args.ddpm_num_steps+1, size=(bsz//2,)).to(clean_images.device)
            timesteps = torch.cat([timesteps, args.ddpm_num_steps - timesteps + 1], dim=0)[:bsz].long()

            alpha = timesteps.float() / args.ddpm_num_steps

            gamma_t = timesteps.float() / args.ddpm_num_steps

            noise, noise_bn, noise_wn = get_noise_v2(accelerator.device, clean_images, cov_mat_L, gamma_t, timesteps, noise_type=args.noise_type, train_or_test='train', inplace=False)

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, alpha)    # override

            with accelerator.accumulate(model):
                
                model_output = model(noisy_images, alpha, return_dict=False)[0]
                # if args.prediction_type == "epsilon":
                # loss = F.mse_loss(model_output.float(), (clean_images - noise).float())

                if args.noise_type in ['gaussianBN', 'gaussianRN']:
                    tar1 = (clean_images - noise).float()
                    alpha_t_minus_1 = (timesteps - 1).float() / args.ddpm_num_steps
                    tar2 = alpha_t_minus_1.view(-1, 1, 1, 1) * (noise_bn - noise_wn)
                    # print('model_output:', model_output.shape)
                    split_size = int(model_output.shape[1] // 2)
                    d1 = model_output[:, :split_size, ...].float()
                    d2 = model_output[:, split_size:, ...].float()
                    gamma_t_minus_1 = (timesteps - 1).float() / args.ddpm_num_steps
                    delta_gamma_t = gamma_t - gamma_t_minus_1
                    delta_alpha_t = alpha - alpha_t_minus_1
                    loss1 = torch.sum((d1 - tar1)**2, dim=[1, 2, 3])
                    loss2 = torch.sum((d2 - tar2)**2, dim=[1, 2, 3])
                    loss1 = torch.sum(loss1 * delta_alpha_t / delta_alpha_t)    # weight is simply 1
                    loss2 = torch.sum(loss2 * delta_gamma_t / delta_alpha_t)    # weighted loss
                    loss = loss1 + loss2

                elif args.noise_type in ['gaussian']:
                    loss = torch.sum((model_output.float() - (clean_images - noise).float())**2)
                else:
                    raise ValueError(f"Unsupported noise type: {args.noise_type}")
                
                accelerator.backward(loss)

                # print('accelerator.sync_gradients:', accelerator.sync_gradients)  # True
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                losses.append(loss.item())

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                # progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            # progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        # progress_bar.close()
            # break

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the plots
                plt.figure(1)
                plt.plot(losses)
                plt.savefig(args.output_dir + "/losses.png")
                plt.clf()

                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = IADBPipeline(unet=unet, scheduler=noise_scheduler)
                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

    accelerator.end_training()


if __name__ == "__main__":
    main()


























