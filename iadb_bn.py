import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam, AdamW
import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random
# import cv2
# import imageio
import platform
from scipy.stats import qmc
from scipy.stats import norm
import sys
import time
import yaml
import glob

sys.path.append('../')
sys.path.append('../../repo')

from bluenoise.get_noise_recent import get_noise_v2


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='celeba_small', help='dataset name')
parser.add_argument('--noise_type', type=str, default='gaussian', help='type of noise')
parser.add_argument('--optimizer_type', type=str, default='adamw', help='optimizer option')

parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--res', type=int, default=64, help='resolution')
parser.add_argument('--train_or_test', type=str, default='train', help='train_or_test')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint name')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--nb_steps', type=int, default=1000, help='nb_steps')   # 128
parser.add_argument('--scheduler_alpha', type=str, default='linear', help='scheduler type')
parser.add_argument('--scheduler_gamma', type=str, default='linear', help='scheduler type')
parser.add_argument('--scheduler_param', type=float, default=0.02, help='scheduler parameter for scheduler_gamma')
parser.add_argument('--scheduler_param_s', type=float, default=0, help='scheduler parameter for scheduler_gamma')
parser.add_argument('--scheduler_param_e', type=float, default=3, help='scheduler parameter for scheduler_gamma')

parser.add_argument('--blue_noise_blur', type=float, default=None, help='blue noise blur')
parser.add_argument('--activation', type=str, default='silu', help='[silu, gelu, mish]')
parser.add_argument('--early_stopping_step', type=int, default=50, help='[200,  150, 100, 50]')
parser.add_argument('--split_step', type=int, default=900, help='experimentally chosen to be 600')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--mode_index', type=int, default=1, help='modes')
parser.add_argument('--reg_weight', type=float, default=1, help='weight of regularizer')
parser.add_argument('--alpha_min', type=float, default=0.0, help='min of alpha')
parser.add_argument('--grad_clip', type=float, default=None, help='grad norm clip')
parser.add_argument('--deterministic', type=int, default=1, help='deterministic or stochastic')
parser.add_argument("--resume_training", action="store_true", help="Whether to resume training")
parser.add_argument("--optimize_scheduler_param", action="store_true", help="Whether to optimize the scheduler_param")
parser.add_argument("--remap", action="store_true", help="remapping stratification across images")

parser.add_argument("--is_conditional", action="store_true", help="whether it is conditional image generation")
parser.add_argument('--conditional_type', type=str, default='superres', help='superres, coloring')

parser.add_argument("--fine_tune_mode_index", type=int, default=0, help="how to fine tune the model")
parser.add_argument("--skip", type=int, default=1, help="numbe of skipped steps")
parser.add_argument("--test_samples", type=int, default=10, help="numbe of generated samples")
parser.add_argument("--out_channel", type=int, default=6, help="out_channel is 3 or 6")

opt = parser.parse_args()

dimension = 3
seed = opt.seed
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha_min = opt.alpha_min
if platform.system() == 'Windows':
    # opt.batch_size = 1    # 1, 2, 4
    pass

cov_mat_L = np.load('./bluenoise/cov_gaussianBN_L_res{:}_d{:}.npz'.format(64, dimension))['x'].astype(np.float32)
if opt.noise_type in ['gaussianRN']:
    cov_mat_L = np.load('./bluenoise/cov_gaussianRN_L_res{:}_d{:}.npz'.format(64, dimension))['x'].astype(np.float32)
cov_mat_L = torch.from_numpy(cov_mat_L).to(device).detach()



def get_scheduler(x, scheduler):
    
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

    # do nothing
    if scheduler == 'linear':
        return x / opt.nb_steps
    
    elif scheduler == 'sigmoid':
        # if array_type in ['numpy', 'float']:
        #     x = np.power(x, opt.pow_index)
        # elif array_type == 'torch':
        #     x = torch.pow(x, opt.pow_index)
        # start = torch.zeros_like(x)
        start = torch.ones_like(x) * opt.scheduler_param
        end = torch.ones_like(x) * 3
        clip_min = 1e-9
        tau = 0.9   # 0.9 seems good; opt.scheduler_param
        v_start = torch.nn.functional.sigmoid(start / tau)
        v_end = torch.nn.functional.sigmoid(end / tau)
        t = x / opt.nb_steps
        output = torch.nn.functional.sigmoid((t * (end - start) + start) / tau)
        output = (v_end - output) / (v_end - v_start) 
        output = torch.clamp(output, clip_min, 1)
        x = 1 - output

    elif scheduler == 'cosine':
        start = torch.ones_like(x) * 0.2
        end = torch.ones_like(x) * 1
        clip_min = 1e-9
        tau = opt.scheduler_param
        v_start = torch.cos(start * np.pi / 2) ** (2 * tau)
        v_end = torch.cos(end * np.pi / 2) ** (2 * tau)
        t = x / opt.nb_steps
        output = torch.cos((t * (end - start) + start) * np.pi / 2) ** (2 * tau)
        output = (v_end - output) / (v_end - v_start)
        output = torch.clamp(output, clip_min, 1.0)
        x = 1 - output

    else:
        raise NotImplementedError

    return x



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
        return x / opt.nb_steps
    
    elif scheduler == 'sigmoid':
        start = torch.ones_like(x) * scheduler_param_s   #scheduler_param
        end = torch.ones_like(x) * scheduler_param_e
        clip_min = 1e-9
        tau = scheduler_param   # 0.9 seems good; scheduler_param
        v_start = torch.nn.functional.sigmoid(start / tau)
        v_end = torch.nn.functional.sigmoid(end / tau)
        t = x / opt.nb_steps
        output = torch.nn.functional.sigmoid((t * (end - start) + start) / tau)
        output = (v_end - output) / (v_end - v_start) 
        output = torch.clamp(output, clip_min, 1)
        x = 1 - output

    elif scheduler == 'cosine':
        start = torch.ones_like(x) * scheduler_param_s
        end = torch.ones_like(x) * scheduler_param_e
        clip_min = 1e-9
        tau = scheduler_param   # integer for now
        # # print('tau:', tau, torch.cos(start * np.pi / 2))
        v_start = torch.pow(torch.cos(start * np.pi / 2.0), (2.0 * tau))
        v_end = torch.pow(torch.cos(end * np.pi / 2), (2 * tau))
        # # print('output:', v_end, v_start)
        t = x / opt.nb_steps
        output = torch.pow(torch.cos((t * (end - start) + start) * np.pi / 2), (2 * tau))
        output = (v_end - output) / (v_end - v_start)
        
        output = torch.clamp(output, clip_min, 1.0)
        x = 1 - output
        # t = x / opt.nb_steps
        # output = 0.5 * torch.cos(2*np.pi*scheduler_param*t) + 0.5
        # x = output
    else:
        raise NotImplementedError

    return x



def get_model(inp_channel=3, out_channel=3):
    
    # block_out_channels=(64, 64, 128, 128, 256, 256)
    
    if opt.res in [64]:
        block_out_channels=(128, 128, 256, 256, 512, 512)
        down_block_types=( 
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D", 
            "DownBlock2D", 
            "DownBlock2D", 
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            # "DownBlock2D",
            "DownBlock2D",
        )
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            # "UpBlock2D",  # a regular ResNet upsampling block
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D"  
        )

    elif opt.res in [128]:
        block_out_channels=(128, 128, 128, 256, 256, 512, 512)
        down_block_types=( 
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D", 
            "DownBlock2D", 
            "DownBlock2D", 
            "DownBlock2D", 
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            # "DownBlock2D",
            "DownBlock2D",
        )
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            # "UpBlock2D",  # a regular ResNet upsampling block
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D"  
        )

    elif opt.res in [256]:
        block_out_channels=(128, 128, 128, 128, 256, 256, 512, 512)
        down_block_types=( 
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D", 
            "DownBlock2D", 
            "DownBlock2D", 
            "DownBlock2D", 
            "DownBlock2D", 
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            # "DownBlock2D",
            "DownBlock2D",
        )
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            # "UpBlock2D",  # a regular ResNet upsampling block
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D"  
        )

    else:
        raise NotImplementedError


    return UNet2DModel(block_out_channels=block_out_channels,out_channels=out_channel, in_channels=inp_channel, up_block_types=up_block_types, down_block_types=down_block_types, act_fn=opt.activation, add_attention=True)



@torch.no_grad()
def sample_iadb(model, x0, nb_step, scheduler_params):
    # print('sample_iadb')

    x_all = []
    x_alpha = x0

    start_step = 0#int(alpha_min * nb_step)
    seq = list(range(start_step, nb_step))
    use_reverse = True
        
    if use_reverse:
        seq = reversed(seq)
    # print('nb_step:', seq, nb_step)
    # for t in range(start_step, nb_step):
    
    inference_time = []

    for t in seq:
        
        tt = torch.randint(low=t, high=t+1, size=(x0.shape[0], )).to(device)

        # if use_reverse:
        # alpha_start = ((t+1)/nb_step)
        # alpha_end = ((t)/nb_step)
        alpha_start = get_scheduler((tt + 1).float(), opt.scheduler_alpha)
        alpha_end = get_scheduler(tt.float(), opt.scheduler_alpha)

        # if opt.optimize_scheduler_param:
        gamma_start = get_scheduler_gamma((tt + 1).float(), opt.scheduler_gamma, scheduler_params)
        gamma_end = get_scheduler_gamma(tt.float(), opt.scheduler_gamma, scheduler_params)
        
        start_time = time.time()
        d = model(x_alpha, alpha_start, return_dict=False)[0]#['sample']
        end_time = time.time()
        inference_time.append(end_time - start_time)
        
        if opt.noise_type in ['gaussianBN', 'gaussianRN']:
            # x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d[:, :3, :, :] + (gamma_start - gamma_end).view(-1, 1, 1, 1) * alpha_end.view(-1, 1, 1, 1) * d[:, 3:, :, :]
            if opt.out_channel == 3:
                x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d
            elif opt.out_channel == 6:
                # print('t:', t, gamma_start - gamma_end)
                x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d[:, :3, :, :] + (gamma_start - gamma_end).view(-1, 1, 1, 1) * d[:, 3:, :, :]
            else:
                raise NotImplementedError
            
        elif opt.noise_type in ['gaussian', 'GBN']:

            # TODO: for early stopping only
            if False:
                if opt.noise_type in ['GBN', 'gaussian'] and opt.early_stopping_step-1 == t:
                    x_alpha = d + x0
                    x_all.append(x_alpha)
                    break
                else: 
                      x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d
        
            x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d

            if False:    # motaivation figure
                if t % 25 == 0 or t == 0:
                    x1_deblended = d + x0
                    if t != 0:
                        x1_deblended = (x1_deblended - x1_deblended.min()) / (x1_deblended.max() - x1_deblended.min())
                    else:
                        x1_deblended = torch.clamp((x_alpha + 1) / 2.0, 0, 1)
                    Image.fromarray((x1_deblended[1].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)).save('results/motaivation_x1_deblended_{:}.png'.format(t))
                    # plt.figure(1)
                    # plt.subplot(121)
                    # plt.imshow(x1_deblended[1].permute(1,2,0).cpu().numpy())
                    # plt.subplot(122)
                    # plt.imshow(x0[1].permute(1,2,0).cpu().numpy())
                    # plt.show()

        else:
            raise NotImplementedError
        
        if opt.train_or_test == 'test':
            # if t == opt.early_stopping_step-1:
            #     x_all.append(x_alpha)
            #     break
            if nb_step == 1000:
                log_freq = 100
            else:
                log_freq = 25
            if t % log_freq == 0 or t == nb_step-1:
                x_all.append(x_alpha)

    # print('inferece time:', inference_time, np.mean(inference_time[1:]))

    if opt.train_or_test == 'test':
        return x_alpha, x_all, np.mean(inference_time[1:])
    return x_alpha




@torch.no_grad()
def sample_iadb_conditional(model, x0, x_c, nb_step, scheduler_params):
    x_all = []
    x_alpha = x0
    start_step = 0#int(alpha_min * nb_step)
    seq = list(range(start_step, nb_step))
    use_reverse = True
    if use_reverse:
        seq = reversed(seq)
    
    for t in seq:
        tt = torch.randint(low=t, high=t+1, size=(x0.shape[0], )).to(device)
        alpha_start = get_scheduler((tt + 1).float(), opt.scheduler_alpha)
        alpha_end = get_scheduler(tt.float(), opt.scheduler_alpha)
        # if opt.optimize_scheduler_param:
        gamma_start = get_scheduler_gamma((tt + 1).float(), opt.scheduler_gamma, scheduler_params)
        gamma_end = get_scheduler_gamma(tt.float(), opt.scheduler_gamma, scheduler_params)
        # else:
        #     gamma_start = get_scheduler((tt + 1).float(), opt.scheduler_gamma)
        #     gamma_end = get_scheduler(tt.float(), opt.scheduler_gamma)
        
        # print('x_alpha:', x_alpha.shape, x_c.shape)
        d = model(torch.cat([x_alpha, x_c], 1), alpha_start, return_dict=False)[0]#['sample']
        # d = model(x_alpha, alpha_start, return_dict=False)[0]#['sample']
        
        if opt.noise_type in ['gaussianBN', 'gaussianRN']:
            # x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d[:, :3, :, :] + (gamma_start - gamma_end).view(-1, 1, 1, 1) * alpha_end.view(-1, 1, 1, 1) * d[:, 3:, :, :]
            if opt.out_channel == 3:
                x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d
            elif opt.out_channel == 6:
                # print('t:', t, gamma_start - gamma_end)
                x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d[:, :3, :, :] + (gamma_start - gamma_end).view(-1, 1, 1, 1) * d[:, 3:, :, :]
            else:
                raise NotImplementedError
            
        elif opt.noise_type in ['gaussian', 'GBN']:
            x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d
        else:
            raise NotImplementedError
        
        
        if opt.train_or_test == 'test':
            # if t == opt.early_stopping_step-1:
            #     x_all.append(x_alpha)
            #     break
            if nb_step == 1000:
                log_freq = 100
            else:
                log_freq = 25
            if t % log_freq == 0 or t == nb_step-1:
                x_all.append(x_alpha)

    if opt.train_or_test == 'test':
        return x_alpha, x_all
    return x_alpha



DATA_FOLDER = './data/{:}'.format(opt.dataset)
transform = transforms.Compose([transforms.Resize(opt.res),transforms.CenterCrop(opt.res), transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize(opt.res),transforms.CenterCrop(opt.res), transforms.ToTensor()])
start_time = time.time()
if opt.is_conditional:
    if opt.train_or_test == 'train':
        train_dataset = torchvision.datasets.ImageFolder(root=DATA_FOLDER+'_train', transform=transform)
    else:
        train_dataset = torchvision.datasets.ImageFolder(root=DATA_FOLDER+'_test', transform=transform)     # dummy
    test_dataset = torchvision.datasets.ImageFolder(root=DATA_FOLDER+'_test', transform=test_transform)
else:
    train_dataset = torchvision.datasets.ImageFolder(root=DATA_FOLDER, transform=transform)
print('dataloader time, dataset size:', time.time() - start_time, len(train_dataset))

if platform.system() == 'Windows':
    num_workers = 0
else:
    if opt.res in [32, 64]:
        num_workers = 4
    elif opt.res in [128]:
        num_workers = 8
    elif opt.res in [256]:
        num_workers = 16
    else:
        raise NotImplementedError

is_shuffle = True
if opt.train_or_test == 'test_amin':
    is_shuffle = False
drop_last = True
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)   # True
device_count = torch.cuda.device_count()
print('device_count:', device_count)

if opt.noise_type in ['gaussianBN', 'gaussianRN']:
    pass
else:
    opt.out_channel = 3

if opt.is_conditional:
    outer_folder = 'results_gaussianBN_{:}'.format(opt.conditional_type)
else:
    outer_folder = 'results_gaussianBN'

if opt.scheduler_gamma in ['linear']:
    output_folder = outer_folder + '/{:}_{:}_{:}_outc{:}_seed{:}'.format(opt.dataset, opt.noise_type, opt.scheduler_gamma, opt.out_channel, opt.seed)
    
else:
    if opt.optimize_scheduler_param:
        output_folder = outer_folder + '/{:}_{:}_{:}_outc{:}_seed{:}'.format(opt.dataset, opt.noise_type, opt.scheduler_gamma, opt.out_channel, opt.seed)
    else:
        if opt.remap:
            output_folder = outer_folder + '/{:}_{:}_{:}_{:}_{:}_{:}_outc{:}_remap_seed{:}'.format(opt.dataset, opt.noise_type, opt.scheduler_gamma, opt.scheduler_param, opt.scheduler_param_s, opt.scheduler_param_e, opt.out_channel, opt.seed)
        else:
            output_folder = outer_folder + '/{:}_{:}_{:}_{:}_{:}_{:}_outc{:}_seed{:}'.format(opt.dataset, opt.noise_type, opt.scheduler_gamma, opt.scheduler_param, opt.scheduler_param_s, opt.scheduler_param_e, opt.out_channel, opt.seed)
    
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)



def main():

    print('output_folder:', output_folder)
    nb_iter = 0
    train_or_test = opt.train_or_test

    if opt.optimize_scheduler_param:
        # if opt.scheduler_gamma in ['cosine']:
        #     scheduler_param_min = 1
        #     scheduler_param_max = 10
        if opt.scheduler_gamma in ['sigmoid']:
            scheduler_param_min = 0.01
            scheduler_param_max = 10
            scheduler_param_s_min = -3
            scheduler_param_s_max = -0.01
            scheduler_param_e_min = 0.01
            scheduler_param_e_max = 3
        elif opt.scheduler_gamma in ['linear']:
            scheduler_param_min = 1
            scheduler_param_max = 1
            scheduler_param_s_min = 1
            scheduler_param_s_max = 1
            scheduler_param_e_min = 1
            scheduler_param_e_max = 1
        else:
            raise NotImplementedError
    else:
        scheduler_param_min = opt.scheduler_param
        scheduler_param_max = opt.scheduler_param
        scheduler_param_s_min = opt.scheduler_param_s
        scheduler_param_s_max = opt.scheduler_param_s
        scheduler_param_e_min = opt.scheduler_param_e
        scheduler_param_e_max = opt.scheduler_param_e

    scheduler_params = torch.rand(3).float().to(device)
    scheduler_params[0] = scheduler_param_min + (scheduler_param_max - scheduler_param_min) * scheduler_params[0]
    scheduler_params[1] = scheduler_param_s_min + (scheduler_param_s_max - scheduler_param_s_min) * scheduler_params[1]
    scheduler_params[2] = scheduler_param_e_min + (scheduler_param_e_max - scheduler_param_e_min) * scheduler_params[2]
    # print('scheduler_params:', scheduler_params)
    
    if opt.optimize_scheduler_param:
        # # experimentally set to better initialized values based on the image resolution
        # if opt.dataset in ['cat_res64', 'celeba_res64']:
        #     scheduler_param[:] = 99      # as it converges to around 10
        # elif opt.dataset in ['cat_res128', 'celeba_res128']:
        #     scheduler_param[:] = 0.2    # as it converges to around 0.2
        # else:
        #     raise NotImplementedError
        pass

    # conditional image generation
    is_conditional = opt.is_conditional
    inp_chanel = 3
    if is_conditional:
        if opt.conditional_type in ['superres']:
            inp_chanel = 6  # 6
        # elif opt.conditional_type in ['coloring']:
        #     inp_chanel = 4
        else:
            raise NotImplementedError
    
    model = get_model(inp_chanel, opt.out_channel)

    if train_or_test == 'test' and is_conditional:

        print('===> Start conditional sampling / superres')
        
        from piq import psnr, ssim, SSIMLoss
        model.load_state_dict(torch.load(f'{output_folder}/model.ckpt'))
        model = model.to(device)
        model = torch.nn.DataParallel(model)    # multi-gpus
        model.eval()

        total_num = 5000   # 30000, 5000
        # num_batch = int(total_num // opt.batch_size)
        cnt = 0
        
        # if total_num % opt.batch_size == 0:
        #     num_batch = int(total_num // opt.batch_size)
        #     last_batch_size = opt.batch_size
        # else:
        #     num_batch = int(total_num // opt.batch_size) + 1
        #     last_batch_size = total_num - (num_batch-1) * opt.batch_size
        
        if opt.noise_type in ['gaussianBN']:
            folder_name_noise = 'gwn2gbn'
        elif opt.noise_type in ['gaussian']:
            folder_name_noise = 'gwn'
        elif opt.noise_type in ['gaussianRN']:
            folder_name_noise = 'gwn2grn'
        elif opt.noise_type in ['GBN']:
            folder_name_noise = 'gbn'
        else:
            raise NotImplementedError
        folder_name = '{:}_iadb_{:}_{:}_steps{:}'.format(opt.dataset, folder_name_noise, opt.conditional_type, opt.nb_steps)

        for sub_folder in ['images', 'seqs', 'lowres', 'highres']:
            if not os.path.exists(output_folder + '/{:}/{:}'.format(folder_name, sub_folder)):
                os.makedirs(output_folder + '/{:}/{:}'.format(folder_name, sub_folder), exist_ok=True)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)

        if opt.optimize_scheduler_param:
            scheduler_params = np.loadtxt(f'{output_folder}/scheduler_params.txt')
        else:
            scheduler_params = np.array([opt.scheduler_param, opt.scheduler_param_s, opt.scheduler_param_e]).astype(np.float32)
        scheduler_params = torch.from_numpy(scheduler_params).float().to(device)

        avg_ssim = 0
        avg_psnr = 0
        avg_l2 = 0
        avg_l1 = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_dataloader)):     # test_dataloader
                if (i + 1) > 389:
                    return
                if (i + 1) not in [74, 104, 278, 389]:  # test the ones in the paper
                    continue
                
                x1 = (data[0].to(device)*2)-1
                downscale = 4
                x_c = torch.nn.functional.interpolate(x1, size=(opt.res//downscale, opt.res//downscale), mode='bilinear', align_corners=True)
                x_c = torch.nn.functional.interpolate(x_c, size=(opt.res, opt.res), mode='bilinear', align_corners=True)
                
                cur_batch_size = x1.shape[0]
                
                x0 = torch.from_numpy(np.random.randn(cur_batch_size, 3, opt.res, opt.res)).float().to(device)
                t = torch.randint(low=opt.nb_steps, high=opt.nb_steps+1, size=(x0.shape[0], )).to(device)
                gamma_t = get_scheduler_gamma(t.float(), opt.scheduler_gamma, scheduler_params)
                x0, _, _ = get_noise_v2(device, x0, cov_mat_L, gamma_t, t, noise_type=opt.noise_type, train_or_test='test', inplace=True)

                # print('x0:', x0.shape, x_c.shape)

                sample, sample_all = sample_iadb_conditional(model, x0, x_c, opt.nb_steps, scheduler_params)   # fair

                ssim_val = ssim(torch.clamp((sample + 1.0) / 2.0, 0.0, 1.0), (x1 + 1.0) / 2.0, data_range=1., reduction='none')
                psnr_val = psnr(torch.clamp((sample + 1.0) / 2.0, 0.0, 1.0), (x1 + 1.0) / 2.0, data_range=1., reduction='none')
                l2_val = torch.sum((sample - x1) ** 2)
                l1_val = torch.sum(torch.abs(sample - x1))
                # print('val:', ssim_val, psnr_val, l2_val)
                avg_ssim += torch.sum(ssim_val).item() / total_num
                avg_psnr += torch.sum(psnr_val).item() / total_num
                avg_l2 += l2_val.item() / total_num
                avg_l1 += l1_val.item() / total_num
                
                for j in range(0, len(sample_all), 1):
                    # plt.subplot(1, len(sample_all), j+1)
                    sample_plot = sample_all[j][0]
                    if j == len(sample_all) - 1:
                        sample_plot = torch.clamp((sample_plot + 1) / 2.0, 0.0, 1.0)
                    else:
                        sample_plot = (sample_plot - sample_plot.min()) / (sample_plot.max() - sample_plot.min()) 
                    Image.fromarray((sample_plot.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)).save(output_folder + '/{:}/seqs/{:}_img{:0>5}_step{:}.png'.format(folder_name, folder_name_noise, cnt, int((j*100)/1000*opt.nb_steps)))

                for j in range(cur_batch_size):
                    cnt += 1
                    sample_j = sample[j]
                    # print('sample_j:', sample_j.shape, sample_j.min(), sample_j.max())
                    sample_j = torch.clamp((sample_j + 1) / 2.0, 0.0, 1.0)
                    sample_j = sample_j.permute(1, 2, 0).detach().cpu().numpy()
                    Image.fromarray((sample_j*255).astype(np.uint8)).save(output_folder + '/{:}/images/image_{:}_{:0>5}.png'.format(folder_name, folder_name_noise, cnt))
                    
                    x1_plot = x1[j].permute(1, 2, 0).detach().cpu().numpy()
                    x1_plot = (x1_plot + 1) / 2
                    xc_plot = x_c[j].permute(1, 2, 0).detach().cpu().numpy()
                    xc_plot = (xc_plot + 1) / 2

                    err = np.mean(np.abs(x1_plot - sample_j))
                    # print('err:', cnt, err)

                    if opt.noise_type in ['gaussian']:      # no need to save these for all experiments
                        Image.fromarray((x1_plot*255).astype(np.uint8)).save(output_folder + '/{:}/highres/highres_{:}_{:0>5}.png'.format(folder_name, folder_name_noise, cnt))
                        Image.fromarray((xc_plot*255).astype(np.uint8)).save(output_folder + '/{:}/lowres/lowres_{:}_{:0>5}.png'.format(folder_name, folder_name_noise, cnt))

                
                # break
        
                print('conditional metrics: ssim: {:.4f}, psnr: {:.4f}, l2: {:.4f}'.format(avg_ssim, avg_psnr, avg_l2))
        return
                            

                
    if train_or_test == 'test':
        print('===> Start unconditional sampling')
        
        if opt.noise_type in ['gaussianBN']:
            folder_name_noise = 'gwn2gbn'
        elif opt.noise_type in ['gaussian']:
            folder_name_noise = 'gwn'
            # folder_name_noise = 'gwn_earlystop{:}'.format(opt.early_stopping_step)
            
        elif opt.noise_type in ['gaussianRN']:
            folder_name_noise = 'gwn2grn'
        elif opt.noise_type in ['GBN']:
            folder_name_noise = 'gbn'
            # folder_name_noise = 'gbn_earlystop{:}'.format(opt.early_stopping_step)

        else:
            raise NotImplementedError
        folder_name = '{:}_iadb_{:}_steps{:}'.format(opt.dataset, folder_name_noise, opt.nb_steps)

        for sub_folder in ['images', 'seqs', 'noise']:
            if not os.path.exists(output_folder + '/{:}/{:}/'.format(folder_name, sub_folder)):	
                os.makedirs(output_folder + '/{:}/{:}/'.format(folder_name, sub_folder), exist_ok=True)

        current_num_samples = len(glob.glob(output_folder + '/{:}/images/*.png'.format(folder_name)))
        
        current_num_samples = 0
        start_batch = int(current_num_samples // opt.batch_size)

        model.load_state_dict(torch.load(f'{output_folder}/model.ckpt'))
        model = model.to(device)
        model = torch.nn.DataParallel(model)    # multi-gpus
        model.eval()
        
        total_num = opt.test_samples   # 30000, 5000
        # num_batch = int(total_num // opt.batch_size)
        cnt = current_num_samples

        if total_num % opt.batch_size == 0:
            num_batch = int(total_num // opt.batch_size)
            last_batch_size = opt.batch_size
        else:
            num_batch = int(total_num // opt.batch_size) + 1
            last_batch_size = total_num - (num_batch-1) * opt.batch_size
        
        print('current_samples:', current_num_samples)
        print('start_batch:', start_batch)
        print('num_batch:', num_batch)
        
        if opt.optimize_scheduler_param:
            scheduler_params = np.loadtxt(f'{output_folder}/scheduler_params.txt')
        else:
            scheduler_params = np.array([opt.scheduler_param, opt.scheduler_param_s, opt.scheduler_param_e]).astype(np.float32)
        scheduler_params = torch.from_numpy(scheduler_params).float().to(device)

        inference_times = []
        noise_gen_times = []
        for i in tqdm(range(start_batch, num_batch)):
            # replicability, only one sample
            if opt.dataset in ['cat_res64'] and i not in [4]:    
                continue
            if opt.dataset in ['cat_res128'] and i not in [52]:
                continue
            if opt.dataset in ['celeba_res64'] and i not in [37]:
                continue
            if opt.dataset in ['celeba_res128'] and i not in [10]:
                continue
            if opt.dataset in ['church_res64'] and i not in [4, 23, 32, 36]:
                continue
            
            # print(opt.dataset, i, opt.batch_size)
            cur_batch_size = opt.batch_size
            if i == num_batch - 1:
                cur_batch_size = last_batch_size
            
            # x0 = torch.randn(cur_batch_size, 3, opt.res, opt.res).to(device)      # weird
            x0 = torch.from_numpy(np.random.randn(cur_batch_size, 3, opt.res, opt.res)).float().to(device)
            
            if True:
                x0 = np.load('./results_gaussianBN/{:}_gaussian_linear_outc3_seed0/{:}_iadb_gwn_steps250/noise/noise_batch{:}_idx{:05d}.npz'.format(opt.dataset, opt.dataset, opt.batch_size, i))['noise']
                x0 = torch.from_numpy(x0).float().to(device)
            x0 = x0[0:1]       # replicability, only one sample

            # print('x0:', x0.shape, x0.min(), x0.max())
            
            t = torch.randint(low=opt.nb_steps, high=opt.nb_steps+1, size=(x0.shape[0], )).to(device)
            
            gamma_t = get_scheduler_gamma(t.float(), opt.scheduler_gamma, scheduler_params)
            
            start_time = time.time()
            # x0, _, _ = get_noise_v2(device, x0, cov_mat_L, gamma_t, t, noise_type=opt.noise_type, train_or_test='test', inplace=True)
            
            end_time = time.time()
            noise_gen_times.append(end_time - start_time)


            if False:
                # print('x0:', x0.detach().cpu().numpy().shape)
                np.savez_compressed(output_folder + '/{:}/noise/noise_batch{:}_idx{:0>5}.npz'.format(folder_name, cur_batch_size, i), noise=x0.detach().cpu().numpy())
                # if i == 1:
                #     return
            
            split_alpha = (opt.split_step) / 1000
            ratio = opt.nb_steps / 1000
            split_step = int(opt.split_step * ratio)
            sample, sample_all, inference_time = sample_iadb(model, x0, opt.nb_steps, scheduler_params)   # fair
            inference_times.append(inference_time)
            
            
            if True:
                # plt.figure(1)
                for j in range(0, len(sample_all), 1):
                    # plt.subplot(1, len(sample_all), j+1)
                    sample_plot = sample_all[j][0]
                    if j == len(sample_all) - 1:
                        sample_plot = torch.clamp((sample_plot + 1) / 2.0, 0.0, 1.0)
                    else:
                        sample_plot = (sample_plot - sample_plot.min()) / (sample_plot.max() - sample_plot.min()) 
                    Image.fromarray((sample_plot.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)).save(output_folder + '/{:}/seqs/{:}_img{:0>5}_step{:}.png'.format(folder_name, folder_name_noise, cnt, int((j*100)/1000*opt.nb_steps)))
            
                #     plt.imshow(sample_plot.permute(1,2,0).cpu().numpy())
                #     plt.axis('off')
                # plt.savefig(output_folder + '/{:}/seqs/{:0>5}.png'.format(folder_name, i), bbox_inches='tight', pad_inches=0, dpi=300)
                # plt.clf()

                for j in range(cur_batch_size):
                    cnt += 1
                    if j > 0:      # replicability, only one sample
                        continue
                    sample_j = sample[j]
                    sample_j = torch.clamp((sample_j + 1) / 2.0, 0.0, 1.0)
                    Image.fromarray((sample_j.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)).save(output_folder + '/{:}/images/{:0>5}.png'.format(folder_name, cnt))


        print('np.mean(inference_times) per image with batch_size=1', np.mean(inference_times))
        print('np.mean(noise_gen_times) per image with batch_size=1', np.mean(noise_gen_times[1:]))
        return
    
    


    print('===> Start training')

    if opt.resume_training:
        model.load_state_dict(torch.load(f'{output_folder}/model.ckpt'))
        # model = model.to(device)
        # model = torch.nn.DataParallel(model)    # multi-gpus
    else:
        load_ckpt = False
        if load_ckpt:
            model.load_state_dict(torch.load(f'{output_folder}/model.ckpt'))

    model = model.to(device)
    model = torch.nn.DataParallel(model)    # multi-gpus
    
    if opt.optimizer_type in ['adam']:
        optimizer = Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer_type in ['adamw']:
        optimizer = AdamW(model.parameters(), lr=opt.lr)
    else:
        raise NotImplementedError
    
    optimizer_scheduler_param = AdamW([scheduler_params.requires_grad_()], lr=0.001)   # 0.001, opt.lr

    losses = []
    scheduler_params_0 = []
    scheduler_params_1 = []
    scheduler_params_2 = []

    iteration_count = 0

    for current_epoch in tqdm(range(opt.epochs)):
        model.train()
        
        for i, data in enumerate(tqdm(dataloader)):
            
            x1 = (data[0].to(device)*2)-1

            bs = x1.shape[0]

            # antithetic sampling following ddpm/ddim
            upper_t = int(opt.nb_steps)

            t = torch.randint(low=1, high=upper_t+1, size=(bs//2,)).to(device)
            t = torch.cat([t, upper_t - t + 1], dim=0)[:bs]
            
            alpha = t.float() / opt.nb_steps
            # print('alpha1:', alpha)
            alpha = get_scheduler(t.float(), opt.scheduler_alpha)        # other scheduler: alpha = 1 - torch.sqrt(1 - alpha)
            # print('alpha2:', alpha)
            gamma_t = get_scheduler_gamma(t.float(), opt.scheduler_gamma, scheduler_params)

            # print('gamma_t', gamma_t)
            # x0: L_t @ noise
            # noise_bn: L_b @ noise
            # noise_wn: L_w @ noise
            x0, noise_bn, noise_wn = get_noise_v2(device, x1, cov_mat_L, gamma_t, t, noise_type=opt.noise_type, train_or_test='train', inplace=False)
            

            if opt.remap:
                with torch.no_grad():
                    dist = torch.cdist(x0.view(bs, -1), x1.view(bs, -1))
                    mapping = torch.zeros(x0.shape[0], dtype=torch.long)
                    for i in range(x0.shape[0]):
                        mapping[i] = torch.argmin(dist[i])
                        # dist[:,mapping[i]] *= 100
                        dist[:,mapping[i]] = 10000
                    x1 = x1[mapping]
                    # print('mapping:', mapping)


            # debug backward
            if False:
                gamma_t_1 = get_scheduler((t-1).float(), opt.scheduler_gamma)
                alpha_t_1 = get_scheduler((t-1).float(), opt.scheduler_alpha)
                noise = torch.randn_like(x1)
                x0, noise_bn, noise_wn = get_noise_v2(device, noise, cov_mat_L, gamma_t, t, noise_type=opt.noise_type, train_or_test='train', inplace=True)
                x0_1, _, _ = get_noise_v2(device, noise, cov_mat_L, gamma_t_1, t, noise_type=opt.noise_type, train_or_test='train', inplace=True)

                # x_alpha = alpha.view(-1,1,1,1) * x0 + (1-alpha).view(-1,1,1,1) * x1
                # x_alpha_t_1 = alpha_t_1.view(-1,1,1,1) * x0_1 + (1-alpha_t_1).view(-1,1,1,1) * x1
                
                # x_alpha_t_1_recon = x_alpha + (alpha - alpha_t_1).view(-1, 1, 1, 1) * (x1 - x0) + (gamma_t - gamma_t_1).view(-1, 1, 1, 1) * (alpha_t_1).view(-1, 1, 1, 1) * (noise_bn - noise_wn)
                # print('err:', torch.mean((x_alpha_t_1 - x_alpha_t_1_recon)**2))
                # continue
                return
            
            iteration_count += 1

            # x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0
            x_alpha = alpha.view(-1,1,1,1) * x0 + (1-alpha).view(-1,1,1,1) * x1     # be careful: x1 is data, x0 is noise
            
            
            if False:
                print('alpha:', alpha, t)
                plt.figure(1)
                plt.subplot(121)
                plt.imshow(x1[0].permute(1,2,0).detach().cpu().numpy())
                plt.subplot(122)
                plt.imshow(x0[0].permute(1,2,0).detach().cpu().numpy())
                plt.show()

            if is_conditional:
                if opt.conditional_type in ['superres']:
                    # downsample and upsample to get bad initialized low-res image as input
                    downscale = 4
                    x_c = torch.nn.functional.interpolate(x1, size=(opt.res//downscale, opt.res//downscale), mode='bilinear', align_corners=True)
                    x_c = torch.nn.functional.interpolate(x_c, size=(opt.res, opt.res), mode='bilinear', align_corners=True)
                    
                # elif opt.conditional_type in ['coloring']:
                #     x_c = torch.nn.functional.interpolate(x1, size=(opt.res//2, opt.res//2), mode='bilinear', align_corners=True)
                #     x_c = torch.nn.functional.interpolate(x_c, size=(opt.res, opt.res), mode='bilinear', align_corners=True)
                
                d = model(torch.cat([x_alpha, x_c], 1), alpha, return_dict=False)[0]
                
            else:
                d = model(x_alpha, alpha, return_dict=False)[0]
                # d = model(x_alpha, t.float())
            
            if opt.noise_type in ['gaussianBN', 'gaussianRN']:

                alpha_t_minus_1 = get_scheduler((t - 1).float(), opt.scheduler_alpha)

                if opt.out_channel == 3:
                    tar = x1 - x0 + alpha_t_minus_1.view(-1, 1, 1, 1) * (noise_bn - noise_wn)
                    loss = torch.sum((d - tar)**2)

                elif opt.out_channel == 6:
                    tar1 = x1 - x0
                    tar2 = alpha_t_minus_1.view(-1, 1, 1, 1) * (noise_bn - noise_wn)
                    d1 = d[:, :3, ...]
                    d2 = d[:, 3:, ...]

                    # print(d2.shape, tar2.shape, alpha_t_minus_1.shape, gamma_t.shape, t, gamma_t)

                    gamma_t_minus_1 = get_scheduler_gamma((t-1).float(), opt.scheduler_gamma, scheduler_params)
                    delta_gamma_t = gamma_t - gamma_t_minus_1
                    delta_alpha_t = alpha - alpha_t_minus_1

                    loss1 = torch.sum((d1 - tar1)**2, dim=[1, 2, 3])
                    loss2 = torch.sum((d2 - tar2)**2, dim=[1, 2, 3])
                    loss1 = torch.sum(loss1 * delta_alpha_t / delta_alpha_t)    # weight is simply 1
                    loss2 = torch.sum(loss2 * delta_gamma_t / delta_alpha_t)    # weighted loss
                    loss = loss1 + loss2
                    
                else:
                    raise NotImplementedError
                
            elif opt.noise_type in ['gaussian', 'GBN']:
                loss = torch.sum((d - (x1-x0))**2)
            else:
                raise NotImplementedError
            
            optimizer.zero_grad()
            optimizer_scheduler_param.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            except Exception:
                pass

            optimizer.step()
            optimizer_scheduler_param.step()
            nb_iter += 1

            losses.append(loss.item())
            
            scheduler_params[0].data.clamp_(scheduler_param_min, scheduler_param_max)
            scheduler_params[1].data.clamp_(scheduler_param_s_min, scheduler_param_s_max)
            scheduler_params[2].data.clamp_(scheduler_param_e_min, scheduler_param_e_max)
            
            scheduler_params_0.append(scheduler_params[0].item())
            scheduler_params_1.append(scheduler_params[1].item())
            scheduler_params_2.append(scheduler_params[2].item())

            # print('loss:', loss.item(), opt.noise_type)
            # break
            
        # continue
        print('np.array(losses):', np.mean(np.array(losses)))

        # if opt.optimize_scheduler_param:
        print('scheduler_params: tau{:.4f},{:.4f},{:.4f}; start{:.4f},{:.4f},{:.4f}; end{:.4f},{:.4f},{:.4f}'.format(scheduler_params_0[-1], scheduler_param_min, scheduler_param_max, scheduler_params_1[-1], scheduler_param_s_min, scheduler_param_s_max, scheduler_params_2[-1], scheduler_param_e_min, scheduler_param_e_max))

        # moving saving things outside training loop
        plt.figure(1)
        plt.plot(losses)
        # plt.plot(np.mean(np.array(losses)))     # track mean of losses
        plt.savefig(output_folder + '/losses.png')
        plt.clf()
        np.savetxt(output_folder + '/losses.txt', np.array(losses))

        plt.figure(1)
        plt.plot(scheduler_params_0)
        plt.plot(scheduler_params_1)
        plt.plot(scheduler_params_2)
        plt.savefig(output_folder + '/scheduler_params.png')
        plt.clf()

        np.savetxt(f'{output_folder}/scheduler_params.txt', scheduler_params.detach().cpu().numpy())
        
        save_model_name = 'model'
        torch.save(model.module.state_dict(), f'{output_folder}/{save_model_name}.ckpt')


if __name__ == '__main__':
    main()
    

