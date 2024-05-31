import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def noise_padding(noise_small, res=128):
    if res == 128:
        
        t1, t2, t3, t4 = noise_small[:, 0, ...], noise_small[:, 1, ...], noise_small[:, 2, ...], noise_small[:, 3, ...]
        if True:
            top_row = torch.cat((t1, t2), dim=-2)     # Concatenate along width
            bottom_row = torch.cat((t3, t4), dim=-2)  # Concatenate along width
            noise = torch.cat((top_row, bottom_row), dim=-1)  # Concatenate along height
            # print('here1', noise.shape)

    else:
        raise NotImplementedError
    return noise



def get_noise_v2(device, x, cov_mat_L, alpha_t, time_step, noise_type='gaussian', train_or_test='train', inplace=False):
    
    # eps = 1e-10
    # print('x:', x.shape)
    dimension = x.shape[1]
    res = x.shape[2]
    # assert x.shape[2] == x.shape[3]

    if noise_type == 'gaussian':
        
        if x.shape[-1] == 64:
            if inplace:
                noise = x
            else:
                noise = torch.randn_like(x)

        elif x.shape[-1] == 128:

            up_scale = int(x.shape[-1] // 64)
            up_scale_sqr = up_scale * up_scale
            bs = x.shape[0]

            if inplace:
                noise = x
            else:
                noise = torch.randn_like(x)
            
            # only for consistent with gaussianBN
            if train_or_test == 'test':
                t1, t2, t3, t4 = x[:, :, 0:64, 0:64], x[:, :, 0:64, 64:128], x[:, :, 64:128, 0:64], x[:, :, 64:128, 64:128]
                noise_small = torch.cat((t1, t2, t3, t4), dim=0)
                noise_small = noise_small.view(bs * up_scale_sqr, dimension, 64 * 64).permute(0, 2, 1)
                noise = noise_small.contiguous().view(bs, up_scale_sqr, dimension, 64, 64)
                noise = noise_padding(noise, res=128)

        else:
            raise NotImplementedError
        
        # plt.figure(1)
        # plt.imshow(noise[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32))
        # plt.show()
            
        # dummy noises, just align with gaussianBN
        noise_bn = noise
        noise_wn = noise

    elif noise_type == 'uniform':
        noise = torch.rand_like(x)
        noise = (noise * 2 - 1) * np.sqrt(3)


    elif noise_type in ['gaussianBN', 'gaussianRN', 'GBN']:
        
        if x.shape[-1] == 64:
        # if False:
            if inplace:
                noise = x
            else:
                noise = torch.randn_like(x)
            B, C, H, W = noise.shape
            noise_wn = noise.clone()
            noise = noise.view(B, C, -1).permute(0, 2, 1)
            
            noise_bn = torch.matmul(cov_mat_L, noise).permute(0, 2, 1).contiguous().view(B, C, H, W)
            
            if noise_type in ['gaussianBN', 'gaussianRN']:
                noise = noise_bn * (1 - alpha_t.view(-1, 1, 1, 1)) + noise_wn * alpha_t.view(-1, 1, 1, 1)
            elif noise_type in ['GBN']:
                noise = noise_bn
            else:
                raise NotImplementedError
            # print('time_step:', time_step, delta_t)

            # noise = torch.matmul(cov_mat_L, noise).permute(0, 2, 1).contiguous().view(B, C, H, W)    # cov_mat_L.to(device) is super slow
        

        elif x.shape[-1] == 128:
            up_scale = int(x.shape[-1] // 64)
            up_scale_sqr = up_scale * up_scale
            bs = x.shape[0]
            
            if inplace:
                t1, t2, t3, t4 = x[:, :, 0:64, 0:64], x[:, :, 0:64, 64:128], x[:, :, 64:128, 0:64], x[:, :, 64:128, 64:128]
                noise_small = torch.cat((t1, t2, t3, t4), dim=0)
                # top_row = torch.cat((t1, t2), dim=-2)     # Concatenate along width
                # bottom_row = torch.cat((t3, t4), dim=-2)  # Concatenate along width
                # noise = torch.cat((top_row, bottom_row), dim=-1)  # Concatenate along height
            else:
                noise_small = torch.randn(bs * up_scale_sqr, dimension, 64, 64).float().to(device)
                # noise = torch.randn_like(x)
                # t1, t2, t3, t4 = x[:, :, 0:64, 0:64], x[:, :, 0:64, 64:128], x[:, :, 64:128, 0:64], x[:, :, 64:128, 64:128]
                # noise_small = torch.cat((t1, t2, t3, t4), dim=0)

            noise_small = noise_small.view(bs * up_scale_sqr, dimension, 64 * 64).permute(0, 2, 1)
            noise_wn = noise_small.contiguous().view(bs, up_scale_sqr, dimension, 64, 64)
            
            noise_bn = torch.matmul(cov_mat_L, noise_small).permute(0, 2, 1).contiguous().view(bs, up_scale_sqr, dimension, 64, 64)
            
            # t1, t2, t3, t4 = noise_bn[:, 0, ...], noise_bn[:, 1, ...], noise_bn[:, 2, ...], noise_bn[:, 3, ...]
            # top_row = torch.cat((t1, t2), dim=-2)     # Concatenate along width
            # bottom_row = torch.cat((t3, t4), dim=-2)  # Concatenate along width
            # noise_bn = torch.cat((top_row, bottom_row), dim=-1)  # Concatenate along height

            # print(alpha_t.shape, noise_bn.shape, noise_wn.shape)
            noise_bn = noise_padding(noise_bn, res=128)
            noise_wn = noise_padding(noise_wn, res=128)
            # print(alpha_t.shape, noise_bn.shape, noise_wn.shape)

            if noise_type in ['gaussianBN', 'gaussianRN']:
                # print('alpha_t:', alpha_t)
                noise = noise_bn * (1 - alpha_t.view(-1, 1, 1, 1)) + noise_wn * alpha_t.view(-1, 1, 1, 1)
            elif noise_type in ['GBN']:
                noise = noise_bn
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
        
        # print('cov_mat_L:', cov_mat_L.shape, noise.shape)
        # print('noise:', noise.shape, noise.min(), noise.max())
        
        if False:
            print('t:', time_step)
            noise_plot = noise[0]
            noise_plot_wn = torch.randn_like(noise_plot)
            noise_plot = (noise_plot - noise_plot.min()) / (noise_plot.max() - noise_plot.min())
            noise_plot_wn = (noise_plot_wn - noise_plot_wn.min()) / (noise_plot_wn.max() - noise_plot_wn.min())
            # print('noise_plot:', noise_plot.shape, noise_plot.min(), noise_plot.max())
            # Image.fromarray((noise_plot.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save('bn_res{:}.png'.format(x.shape[-1]))
            plt.figure(1)
            plt.subplot(121)
            plt.imshow(noise_plot.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32))
            plt.subplot(122)
            plt.imshow(noise_plot_wn.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32))
            plt.show()

    else:
        raise NotImplementedError
    
    # print('x0:', x0.shape, x0.min(), x0.max())
    # print('noise_type:', opt.noise_type)
    # plt.figure(1)
    # plt.imshow(x0[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32), cmap='gray')
    # plt.show()

    return noise, noise_bn, noise_wn
