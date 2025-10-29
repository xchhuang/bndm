import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
# import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
# from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
# from datasets import load_dataset
# from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
# from diffusers import DDPMPipeline, DDPMScheduler
from diffusers import UNet2DModel
# from diffusers import DDIMPipeline, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from diffusers import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin
from diffusers.schedulers.scheduling_utils import SchedulerMixin

import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import platform
from input_args import parse_args
import random
import sys
sys.path.append('../')
from blue_noise_diffusion.get_noise import get_noise_v2

args = parse_args()


# seed = args.seed
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)


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
cov_mat_L = np.load('../../repo/denoising-diffusion-pytorch/simple/tests/results/crop_uniform_cov/cov_gaussianBN_L_res{:}_d{:}.npz'.format(64, dimension))['x'].astype(np.float32)
if args.noise_type in ['gaussianRN']:
    cov_mat_L = np.load('../../repo/denoising-diffusion-pytorch/simple/tests/results/crop_uniform_cov/cov_gaussianRN_L_res{:}_d{:}.npz'.format(64, dimension))['x'].astype(np.float32)
cov_mat_L = torch.from_numpy(cov_mat_L).to(accelerator.device)



beta_start = 0.0001
beta_end = 0.02
betas = torch.from_numpy(np.linspace(beta_start, beta_end, args.ddpm_num_steps)).float().to(accelerator.device)


def get_alpha(t):
    beta_cat = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    # print('beta:', beta)
    a = (1 - beta_cat).cumprod(dim=0).index_select(0, t + 1)
    a = a.view(-1, 1, 1, 1)
    return a




def get_scheduler_gamma(x, scheduler, scheduler_params):
    
    scheduler_param = scheduler_params[0]
    scheduler_param_s = scheduler_params[1]
    scheduler_param_e = scheduler_params[2]

    # input: x in [0, T], scheduler in ['linear', 'pow', 'cosinefaster', 'cosineslower', 'sigmoid', 'exp']
    # output: e.g., exp(x)
    scheduler = scheduler.lower()
    array_type = None
    if isinstance(x, np.ndarray):
        array_type = 'numpy'
    elif torch.is_tensor(x):
        array_type = 'torch'
    else:
        array_type = 'float'

    if scheduler == 'linear':
        return x / args.ddpm_num_steps
    
    elif scheduler == 'sigmoid':
        start = torch.ones_like(x) * scheduler_param_s   #scheduler_param
        end = torch.ones_like(x) * scheduler_param_e
        clip_min = 1e-9
        tau = scheduler_param   # 0.9 seems good; scheduler_param
        v_start = torch.nn.functional.sigmoid(start / tau)
        v_end = torch.nn.functional.sigmoid(end / tau)
        t = x / args.ddpm_num_steps
        output = torch.nn.functional.sigmoid((t * (end - start) + start) / tau)
        output = (v_end - output) / (v_end - v_start) 
        output = torch.clamp(output, clip_min, 1)
        x = 1 - output
    else:
        raise NotImplementedError

    return x





class DDIM_BN_Scheduler(SchedulerMixin, ConfigMixin):
    """
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
        # alpha = (timestep + 1) / self.num_inference_steps
        # alpha_next = (timestep) / self.num_inference_steps

        # gamma = (timestep + 1) / self.num_inference_steps
        # gamma_next = (timestep) / self.num_inference_steps

        # d = model_output

        
        # if args.noise_type in ['gaussianBN', 'gaussianRN']:
        #     if args.out_channels == 3:
        #         x_alpha = x_alpha + (alpha - alpha_next) * d
        #     elif args.out_channels == 6:
        #         # print('t:', timestep, gamma - gamma_next)
        #         x_alpha = x_alpha + (alpha - alpha_next) * d[:, :3, :, :] + (gamma - gamma_next) * d[:, 3:, :, :]
        #     else:
        #         raise NotImplementedError
        # elif args.noise_type in ['gaussian']:
        #     x_alpha = x_alpha + (alpha - alpha_next) * d
        # else:
        #     raise NotImplementedError
        
        # return x_alpha
        return None

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timestep: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        """
        forward
        """
        # return original_samples * alpha.view(-1, 1, 1, 1) + noise * (1 - alpha.view(-1, 1, 1, 1))
        alpha = get_alpha(timestep)

        return alpha.sqrt() * original_samples +  (1 - alpha).sqrt() * noise
    

    def __len__(self):
        return self.num_train_timesteps



class DDIM_BN_Pipeline(DiffusionPipeline):
    """
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)





def main():
    args.output_dir = args.output_dir + '_{:}'.format(args.noise_type)
    if args.use_ema:
        args.output_dir = args.output_dir + '_ema'
        # args.output_dir = os.path.join('results', args.output_dir+'_ema')  # TODO: output to folder called results
    # else:
    args.output_dir = args.output_dir + '_{:}_{:.1f}'.format(args.scheduler_gamma, args.scheduler_param)
    args.output_dir = os.path.join('results', args.output_dir)
    
    if platform.system() == "Windows":
        args.train_batch_size = 2
        args.eval_batch_size = 2

    out_channels = 3
    if args.noise_type in ['gaussian']:
        out_channels = 3
    elif args.noise_type in ['gaussianBN', 'gaussianRN']:
        out_channels = 6
    else:
        raise NotImplementedError
    

    if args.optimize_scheduler_param:
        if args.scheduler_gamma in ['sigmoid']:
            scheduler_param_min = 0.01
            scheduler_param_max = 10
            scheduler_param_s_min = -3
            scheduler_param_s_max = -0.01
            scheduler_param_e_min = 0.01
            scheduler_param_e_max = 3
        else:
            raise NotImplementedError

    else:
        scheduler_param_min = args.scheduler_param
        scheduler_param_max = args.scheduler_param
        scheduler_param_s_min = args.scheduler_param_s
        scheduler_param_s_max = args.scheduler_param_s
        scheduler_param_e_min = args.scheduler_param_e
        scheduler_param_e_max = args.scheduler_param_e

    scheduler_params = torch.rand(3).float().to(accelerator.device)
    scheduler_params[0] = scheduler_param_min + (scheduler_param_max - scheduler_param_min) * scheduler_params[0]
    scheduler_params[1] = scheduler_param_s_min + (scheduler_param_s_max - scheduler_param_s_min) * scheduler_params[1]
    scheduler_params[2] = scheduler_param_e_min + (scheduler_param_e_max - scheduler_param_e_min) * scheduler_params[2]


    # if args.logger == "tensorboard":
    #     if not is_tensorboard_available():
    #         raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    # elif args.logger == "wandb":
    #     if not is_wandb_available():
    #         raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    #     import wandb

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
    # logger.info(accelerator.state, main_process_only=False)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     diffusers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Initialize the model
    if args.model_config_name_or_path is None:
        
        if args.resolution == 64:
            model = UNet2DModel(
                sample_size=args.resolution,
                in_channels=3,
                out_channels=out_channels,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
        elif args.resolution in [128]:
            model = UNet2DModel(
                sample_size=args.resolution,
                in_channels=3,
                out_channels=out_channels,
                layers_per_block=2,
                block_out_channels=(128, 128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )

        elif args.resolution in [256]:
            model = UNet2DModel(
                sample_size=args.resolution,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(128, 128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
        else:
            raise ValueError(f"Unsupported resolution: {args.resolution}")
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
        # print('here1')
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
    # print('here2')
    # return
        
    # Initialize the scheduler
    # accepts_prediction_type = "prediction_type" in set(inspect.signature(DDIMScheduler.__init__).parameters.keys())
    # if accepts_prediction_type:
    #     noise_scheduler = DDIMScheduler(
    #         num_train_timesteps=args.ddpm_num_steps,
    #         beta_schedule=args.ddpm_beta_schedule,
    #         prediction_type=args.prediction_type,
    #     )
    # else:
    noise_scheduler = DDIM_BN_Scheduler(num_train_timesteps=args.ddpm_num_steps)

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    optimizer_scheduler_param = torch.optim.AdamW([scheduler_params.requires_grad_()], lr=0.01)   # 0.001, opt.lr


    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # if args.dataset_name is not None:
    #     dataset = load_dataset(
    #         args.dataset_name,
    #         args.dataset_config_name,
    #         cache_dir=args.cache_dir,
    #         split="train",
    #     )
    # else:
    #     dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
    #     # See more about loading custom images at
    #     # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    
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

    DATA_FOLDER = '../../repo/denoising-diffusion-pytorch/simple/data/{:}'.format(args.dataset_name)
    dataset = torchvision.datasets.ImageFolder(root=DATA_FOLDER, transform=augmentations)

    

    # def transform_images(examples):
    #     images = [augmentations(image.convert("RGB")) for image in examples["image"]]
    #     return {"input": images}

    # logger.info(f"Dataset size: {len(dataset)}")

    # dataset.set_transform(transform_images)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True)

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, optimizer_scheduler_param, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, optimizer_scheduler_param, train_dataloader, lr_scheduler
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

    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(dataset)}")
    # logger.info(f"  Num Epochs = {args.num_epochs}")
    # logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {max_train_steps}")

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
        # Test!
        # load model and scheduler
        if not os.path.exists(args.output_dir + '/images'):
            os.makedirs(args.output_dir + '/images', exist_ok=True)
        if not os.path.exists(args.output_dir + '/seqs'):
            os.makedirs(args.output_dir + '/seqs', exist_ok=True)
        
        # run pipeline in inference (sample random noise and denoise)
        # image = pipe(eta=0.0, num_inference_steps=1000)
        # unet = accelerator.unwrap_model(model)
        # pipeline = DDIMPipeline(unet=unet, scheduler=noise_scheduler).from_pretrained(args.output_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('device:', device)
        scheduler = DDIM_BN_Scheduler.from_pretrained(args.output_dir+"/scheduler")
        scheduler.set_timesteps(args.ddpm_num_steps)
        # pipeline = DDIMPipeline.from_pretrained(args.output_dir).to(accelerator.device)
        model = UNet2DModel.from_pretrained(args.output_dir+"/unet", use_safetensors=True).to(device)
        # model = torch.nn.DataParallel(model)
        model.eval()

        path = '../iadb/results_gaussianBN_adamw_v4_submitted/{:}_gaussian_linear_outc3_seed0/{:}_iadb_gwn_steps250'.format(args.dataset_name, args.dataset_name)
        # path = '../iadb/results_gaussianBN_adamw_v4_submitted/church30k_res64_gaussian_linear_outc3_seed0/church30k_res64_iadb_gwn_steps250'
        
        cnt = 0
        num_batch = int(args.test_samples // args.eval_batch_size)
        for i in tqdm(range(num_batch)):
            
            # images = pipeline(batch_size=args.eval_batch_size, num_inference_steps=args.ddpm_num_inference_steps).images
            # images = (images * 255).round().astype("uint8")
            
            # noise = torch.randn((args.eval_batch_size, 3, args.resolution, args.resolution)).to(device)
            # noise = np.random.randn(args.eval_batch_size, 3, args.resolution, args.resolution).astype(np.float32)
            noise = np.load(path + '/noise/noise_batch{:}_idx{:0>5}.npz'.format(args.eval_batch_size, i))['noise']
            noise = torch.from_numpy(noise).to(device)
            # print('noise:', noise.shape)

            input = noise
            seqs = [noise[0:1]]

            seq = range(0, args.ddpm_num_steps, 4)
            seq_next = [-1] + list(seq[:-1])

            # for t in scheduler.timesteps:
            for i, j in zip(reversed(seq), reversed(seq_next)):
                
                tt = torch.randint(low=i, high=i+1, size=(input.shape[0], )).to(device)
                tt_next = torch.randint(low=j, high=j+1, size=(input.shape[0], )).to(device)
                at = get_alpha(tt)
                at_next = get_alpha(tt_next)
                gt = get_scheduler_gamma(tt, args.scheduler_gamma, scheduler_params)
                gt_next =  get_scheduler_gamma(tt_next, args.scheduler_gamma, scheduler_params)

                with torch.no_grad():
                    d = model(input, tt).sample

                if args.noise_type in ['gaussian']:
                    input = (at_next.sqrt() / at.sqrt()) * (input - d * (1 - at).sqrt()) + (1 - at_next).sqrt() * d

                elif args.noise_type in ['gaussianBN', 'gaussianRN']:
                    d1 = d[:, :3, :, :]
                    d2 = d[:, 3:, :, :]
                    c1 = at_next.sqrt() / at.sqrt()
                    c2 = (at.sqrt() * (1 - at_next).sqrt() - at_next.sqrt() * (1 - at).sqrt()) / at_next.sqrt()
                    input = c1 * (input + c2 * d1) + (gt.view(-1, 1, 1, 1) - gt_next.view(-1, 1, 1, 1)) * d2

                if i % 100 == 0:
                    seqs.append(input[0:1])
            
            # print('seqs:', len(seqs))
            # print('input:', input.shape, input.min(), input.max())
            images = (input / 2 + 0.5).clamp(0, 1)
            images = (images.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
            # images = (input * 255).round().astype("uint8")

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
                Image.fromarray(seq[0]).save(args.output_dir + "/seqs/ddim_{:}_img{:0>5}_step{:}.png".format(args.noise_type, cnt, i*25))

            for i, image in enumerate(images):
                cnt += 1
                Image.fromarray(image).save(args.output_dir + "/images/ddim_{:}_img{:0>5}.png".format(args.noise_type, cnt))

            
        print('Done.')

        return

    # Train!
    # print('first_epoch:', first_epoch, args.num_epochs)
    losses = []
    for epoch in tqdm(range(first_epoch, args.num_epochs)):
        model.train()
        # progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        # progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):

            # if args.train_or_test == 'test':
            #     break
            # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue
            
            # print('batch:', batch[0].shape, batch[1].shape)

            # clean_images = batch["input"].to(weight_dtype)
            clean_images = batch[0].to(weight_dtype)
            
            # Sample noise that we'll add to the images
            # noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)

            
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            

            # random sampling
            timesteps = torch.randint(
                0, args.ddpm_num_steps, (bsz,), device=clean_images.device
            ).long()

            # antithetic sampling
            # timesteps = torch.randint(low=0, high=noise_scheduler.num_train_timesteps, size=(bsz//2,)).to(clean_images.device)
            # timesteps = torch.cat([timesteps, noise_scheduler.num_train_timesteps - timesteps - 1], dim=0)[:bsz].long()
            
            # gamma_t = (timesteps) / args.ddpm_num_steps
            gamma_t = get_scheduler_gamma(timesteps, args.scheduler_gamma, scheduler_params)
            noise, noise_bn, noise_wn = get_noise_v2(accelerator.device, clean_images, cov_mat_L, None, None, gamma_t, timesteps, noise_type=args.noise_type, train_or_test='train', inplace=False)
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            # noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            alpha = get_alpha(timesteps)
            noisy_images = alpha.sqrt() * clean_images +  (1 - alpha).sqrt() * noise

            # print('clean_images:', clean_images.shape, clean_images.min(), clean_images.max())

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_images, timesteps).sample

                # if args.prediction_type == "epsilon":

                if args.noise_type in ['gaussian']:
                    loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                
                elif args.noise_type in ['gaussianBN', 'gaussianRN']:
                    d1 = model_output[:, :3, :, :]
                    d2 = model_output[:, 3:, :, :]
                    loss1 = torch.sum((d1 - noise)**2, [1, 2, 3])
                    # loss1 = F.mse_loss(d1, noise)
                    
                    at = alpha
                    at_next = get_alpha(timesteps - 1)
                    c2 = at.sqrt() * (1 - at_next).sqrt() - at_next.sqrt() * (1 - at).sqrt()
                    gamma_t_minus_1 = get_scheduler_gamma(timesteps - 1, args.scheduler_gamma, scheduler_params)

                    loss2 = torch.sum((d2 - (1 - at_next).sqrt() * (noise_bn - noise_wn))**2, [1, 2, 3])
                    # loss2 = F.mse_loss(d2, (1 - at_next).sqrt() * (noise_bn - noise_wn))
                    # print('gamma_t:', gamma_t.shape, gamma_t_minus_1.shape, c2.shape, at.sqrt().shape)

                    weight = (gamma_t.view(-1, 1, 1, 1) - gamma_t_minus_1.view(-1, 1, 1, 1)) / (c2 / at.sqrt())

                    # print('weight compare:', (gamma_t.view(-1, 1, 1, 1) - gamma_t_minus_1.view(-1, 1, 1, 1))[0], (c2 / at.sqrt())[0])
                    # print('weight compare2:', torch.abs(weight)[0])

                    loss1 = torch.mean(loss1)
                    loss2 = torch.mean(torch.abs(weight) * loss2)

                    loss = loss1 + loss2   # weight
                    
                else:
                    raise NotImplementedError
                
                # elif args.prediction_type == "sample":
                #     alpha_t = _extract_into_tensor(
                #         noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                #     )
                #     snr_weights = alpha_t / (1 - alpha_t)
                #     # use SNR weighting from distillation paper
                #     loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                #     loss = loss.mean()
                # else:
                #     raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer_scheduler_param.step()

                lr_scheduler.step()

                optimizer.zero_grad()
                optimizer_scheduler_param.zero_grad()
                

                losses.append(loss.item())


                scheduler_params[0].data.clamp_(scheduler_param_min, scheduler_param_max)
                scheduler_params[1].data.clamp_(scheduler_param_s_min, scheduler_param_s_max)
                scheduler_params[2].data.clamp_(scheduler_param_e_min, scheduler_param_e_max)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                # progress_bar.update(1)
                global_step += 1

                # if accelerator.is_main_process:
                #     if global_step % args.checkpointing_steps == 0:
                #         # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                #         if args.checkpoints_total_limit is not None:
                #             checkpoints = os.listdir(args.output_dir)
                #             checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                #             checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                #             # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                #             if len(checkpoints) >= args.checkpoints_total_limit:
                #                 num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                #                 removing_checkpoints = checkpoints[0:num_to_remove]

                #                 # logger.info(
                #                 #     f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                #                 # )
                #                 # logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                #                 for removing_checkpoint in removing_checkpoints:
                #                     removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                #                     shutil.rmtree(removing_checkpoint)

                #         save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                #         accelerator.save_state(save_path)
                #         # logger.info(f"Saved state to {save_path}")

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
            # if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
            if False:#args.train_or_test == 'test':
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DDIMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="numpy",
                ).images

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                # denormalize the images and save to tensorboard
                images_processed = (images * 255).round().astype("uint8")
                print('images_processed:', images_processed.shape)
                Image.fromarray(images_processed[0]).save(args.output_dir + "/images/ddim_img{:0>5}.png".format(epoch))

                print('Done.')
                return
            #     # if args.logger == "tensorboard":
            #     #     if is_accelerate_version(">=", "0.17.0.dev0"):
            #     #         tracker = accelerator.get_tracker("tensorboard", unwrap=True)
            #     #     else:
            #     #         tracker = accelerator.get_tracker("tensorboard")
            #     #     tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
            #     # elif args.logger == "wandb":
            #     #     # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
            #     #     accelerator.get_tracker("wandb").log(
            #     #         {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
            #     #         step=global_step,
            #     #     )

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                
                # save the figure
                plt.figure(1)
                plt.plot(losses)
                plt.savefig(args.output_dir + "/losses.png")
                plt.clf()


                print('===> loss: {:.4f}, tau{:.4f},{:.4f},{:.4f}; start{:.4f},{:.4f},{:.4f}; end{:.4f},{:.4f},{:.4f}'.format(loss.item(), scheduler_params[0], scheduler_param_min, scheduler_param_max, scheduler_params[1], scheduler_param_s_min, scheduler_param_s_max, scheduler_params[2], scheduler_param_e_min, scheduler_param_e_max))


                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DDIM_BN_Pipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                # if args.push_to_hub:
                #     upload_folder(
                #         repo_id=repo_id,
                #         folder_path=args.output_dir,
                #         commit_message=f"Epoch {epoch}",
                #         ignore_patterns=["step_*", "epoch_*"],
                #     )

    accelerator.end_training()


if __name__ == "__main__":
    main()

