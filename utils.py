import torch
import numpy as np
from diffusers import UNet2DModel
import time


def get_model(inp_channel=3, out_channel=3, res=64):
    
    # block_out_channels=(64, 64, 128, 128, 256, 256)
    
    if res in [64]:
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

    elif res in [128]:
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

    elif res in [256]:
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


    return UNet2DModel(block_out_channels=block_out_channels,out_channels=out_channel, in_channels=inp_channel, up_block_types=up_block_types, down_block_types=down_block_types, act_fn='silu', add_attention=True)









def get_scheduler(x, scheduler, nb_steps):
    
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
        return x / nb_steps

    else:
        raise NotImplementedError

    return x



def get_scheduler_gamma(x, scheduler, scheduler_params, nb_steps):
    
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
        return x / nb_steps
    
    elif scheduler == 'sigmoid':
        start = torch.ones_like(x) * scheduler_param_s   #scheduler_param
        end = torch.ones_like(x) * scheduler_param_e
        clip_min = 1e-9
        tau = scheduler_param   # 0.9 seems good; scheduler_param
        v_start = torch.nn.functional.sigmoid(start / tau)
        v_end = torch.nn.functional.sigmoid(end / tau)
        t = x / nb_steps
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
        t = x / nb_steps
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




@torch.no_grad()
def sample_iadb(model, x0, nb_step, scheduler_gamma, scheduler_params, out_channel, noise_type, train_or_test, scheduler_alpha='linear'):

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
        
        tt = torch.randint(low=t, high=t+1, size=(x0.shape[0], )).to(x0.device)

        # if use_reverse:
        # alpha_start = ((t+1)/nb_step)
        # alpha_end = ((t)/nb_step)
        alpha_start = get_scheduler((tt + 1).float(), scheduler_alpha, nb_step)
        alpha_end = get_scheduler(tt.float(), scheduler_alpha, nb_step)

        # if opt.optimize_scheduler_param:
        gamma_start = get_scheduler_gamma((tt + 1).float(), scheduler_gamma, scheduler_params, nb_step)
        gamma_end = get_scheduler_gamma(tt.float(), scheduler_gamma, scheduler_params, nb_step)
        
        start_time = time.time()
        d = model(x_alpha, alpha_start, return_dict=False)[0]#['sample']
        end_time = time.time()
        inference_time.append(end_time - start_time)
        
        if noise_type in ['gaussianBN', 'gaussianRN']:
            # x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d[:, :3, :, :] + (gamma_start - gamma_end).view(-1, 1, 1, 1) * alpha_end.view(-1, 1, 1, 1) * d[:, 3:, :, :]
            if out_channel == 3:
                x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d
            elif out_channel == 6:
                # print('t:', t, gamma_start - gamma_end)
                x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d[:, :3, :, :] + (gamma_start - gamma_end).view(-1, 1, 1, 1) * d[:, 3:, :, :]
            else:
                raise NotImplementedError
            
        elif noise_type in ['gaussian', 'GBN']:
            x_alpha = x_alpha + (alpha_start - alpha_end).view(-1, 1, 1, 1) * d
        else:
            raise NotImplementedError
        
        if train_or_test == 'test':
            if nb_step == 1000:
                log_freq = 100
            else:
                log_freq = 1
            if t % log_freq == 0 or t == nb_step-1:
                x_all.append(x_alpha)

    if train_or_test == 'test':
        return x_alpha, x_all, np.mean(inference_time[1:])
    return x_alpha


