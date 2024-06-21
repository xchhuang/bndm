import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
from bluenoise.get_noise_recent import get_noise_v2
import os
from tqdm import tqdm
from PIL import Image
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nb_steps = 1000
cov_mat_L = np.load('./bluenoise/cov_gaussianBN_L_res{:}_d{:}.npz'.format(64, 3))['x'].astype(np.float32)
cov_mat_L = torch.from_numpy(cov_mat_L).to(device)

cov_mat_L_rn = np.load('./bluenoise/cov_gaussianRN_L_res{:}_d{:}.npz'.format(64, 3))['x'].astype(np.float32)
cov_mat_L_rn = torch.from_numpy(cov_mat_L_rn).to(device)

scheduler_gamma = 'sigmoid'
scheduler_params = [1000, 0, 3]
batch_size = 1
noise_type = 'gaussianBN'
output_dir = 'scripts/results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def compute_fft(x):
    x_0 = torch.fft.fftshift(torch.fft.fft2(x[:, 0, :, :]))
    x_1 = torch.fft.fftshift(torch.fft.fft2(x[:, 1, :, :]))
    x_2 = torch.fft.fftshift(torch.fft.fft2(x[:, 2, :, :]))
    x = torch.stack([x_0, x_1, x_2], dim=1)
    return x


def get_scheduler_gamma(x, scheduler, scheduler_params):
    scheduler_param = scheduler_params[0]
    scheduler_param_s = scheduler_params[1]
    scheduler_param_e = scheduler_params[2]
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
    else:
        raise NotImplementedError

    return x



def fig_main_inset():
    x = torch.linspace(0, nb_steps, nb_steps)
    scheduler_param_list = [0.1, 0.2, 0.5, 1.0, 1000.0]
    for tau in scheduler_param_list:
        y = get_scheduler_gamma(x, 'sigmoid', [tau, 0, 3])
        plt.plot(x / nb_steps, y)
       
    plt.legend([r'$\tau={:}$'.format(str(x)) for x in scheduler_param_list], prop={'size': 15})
    plt.gca().set_ylabel(r'$\gamma_t$', fontsize=15) 
    plt.gca().set_xlabel(r'$t/T$', fontsize=15)
    plt.savefig(output_dir+'/inset.png')
    plt.clf()

    

def fig_main_10():
    
    for cur_step in [0]:
        t = torch.randint(low=cur_step, high=cur_step+1, size=(batch_size, )).to(device)
        gamma_t = get_scheduler_gamma(t.float(), scheduler_gamma, scheduler_params)
        gaussian_white_noise = torch.randn(batch_size, 3, 64, 64).float().to(device)
        gaussian_red_noise, _, _ = get_noise_v2(device, gaussian_white_noise, cov_mat_L_rn, gamma_t, t, noise_type=noise_type, train_or_test='test', inplace=True)
        
        gaussian_red_noise_plot = (gaussian_red_noise - gaussian_red_noise.min()) / (gaussian_red_noise.max() - gaussian_red_noise.min())

        gaussian_red_noise_fft = compute_fft(gaussian_red_noise)
        gaussian_red_noise_fft = torch.abs(gaussian_red_noise_fft)
        
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(gaussian_red_noise_plot[0].permute(1, 2, 0).cpu().numpy())
        plt.subplot(122)
        plt.imshow(gaussian_red_noise_fft[0, 0].real.detach().cpu().numpy(), cmap='gray')
        plt.savefig(output_dir+'/gaussianRN_res64_and_spectrum_{:}.png'.format(cur_step))
        plt.clf()


def fig_main_3_4():
    
    for cur_step in [0, 500, 999]:
        t = torch.randint(low=cur_step, high=cur_step+1, size=(batch_size, )).to(device)
        gamma_t = get_scheduler_gamma(t.float(), scheduler_gamma, scheduler_params)
        gaussian_white_noise = torch.randn(batch_size, 3, 64, 64).float().to(device)
        gaussian_blue_noise, _, _ = get_noise_v2(device, gaussian_white_noise, cov_mat_L, gamma_t, t, noise_type=noise_type, train_or_test='test', inplace=True)
        
        gaussian_blue_noise_plot = (gaussian_blue_noise - gaussian_blue_noise.min()) / (gaussian_blue_noise.max() - gaussian_blue_noise.min())

        gaussian_blue_noise_fft = compute_fft(gaussian_blue_noise)
        gaussian_blue_noise_fft = torch.abs(gaussian_blue_noise_fft)
        
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(gaussian_blue_noise_plot[0].permute(1, 2, 0).cpu().numpy())
        plt.subplot(122)
        plt.imshow(gaussian_blue_noise_fft[0, 0].real.detach().cpu().numpy(), cmap='gray')
        plt.savefig(output_dir+'/gaussianBN_res64_and_spectrum_{:}.png'.format(cur_step))
        plt.clf()


def fig_supp_1_2():
    # we use non-repetitive padding as repetitive will lead to artifacts in the spectrum
    for repetitive in [True, False]:
        realizations = 100
        avg_gaussian_blue_noise_fft = 0
        for realization in tqdm(range(realizations)):
            for res in [128]:
                cur_step = 0
                t = torch.randint(low=cur_step, high=cur_step+1, size=(batch_size, )).to(device)
                gamma_t = get_scheduler_gamma(t.float(), scheduler_gamma, scheduler_params)
                gaussian_white_noise = torch.randn(batch_size, 3, res, res).float().to(device)

                if repetitive:
                    patch = gaussian_white_noise[:, :, 0:64, 0:64].clone()
                    num_per_axis = res // 64
                    for i in range(num_per_axis):
                        for j in range(num_per_axis):
                            gaussian_white_noise[:, :, i*64:(i+1)*64, j*64:(j+1)*64] = patch
                gaussian_blue_noise, _, _ = get_noise_v2(device, gaussian_white_noise, cov_mat_L, gamma_t, t, noise_type=noise_type, train_or_test='test', inplace=True)
                
                gaussian_blue_noise_plot = (gaussian_blue_noise - gaussian_blue_noise.min()) / (gaussian_blue_noise.max() - gaussian_blue_noise.min())

                gaussian_blue_noise_fft = compute_fft(gaussian_blue_noise)
                gaussian_blue_noise_fft = torch.abs(gaussian_blue_noise_fft)
                
                avg_gaussian_blue_noise_fft += gaussian_blue_noise_fft

        avg_gaussian_blue_noise_fft = avg_gaussian_blue_noise_fft / realizations
        # print(avg_gaussian_blue_noise_fft.shape, avg_gaussian_blue_noise_fft.min(), avg_gaussian_blue_noise_fft.max())
        # plt.figure(1)
        # plt.subplot(121)
        # plt.imshow(gaussian_blue_noise_plot[0].permute(1, 2, 0).cpu().numpy())
        # plt.subplot(122)
        # plt.imshow(avg_gaussian_blue_noise_fft[0, 0].real.detach().cpu().numpy(), cmap='gray')
        # plt.savefig(output_dir+'/gaussianBN_res{:}_and_spectrum_{:}_repetitive_{:}.png'.format(res, cur_step, repetitive))
        # plt.clf()

        Image.fromarray((gaussian_blue_noise_plot[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(output_dir+'/gaussianBN_res{:}_and_spectrum_{:}_repetitive_{:}_noise.png'.format(res, cur_step, repetitive))

        avg_gaussian_blue_noise_fft_exr = avg_gaussian_blue_noise_fft[0].permute(1,2,0).detach().cpu().numpy()[..., 0].astype(np.float32)
        avg_gaussian_blue_noise_fft_exr = avg_gaussian_blue_noise_fft_exr / avg_gaussian_blue_noise_fft_exr.max()
        # print(avg_gaussian_blue_noise_fft_exr.shape, avg_gaussian_blue_noise_fft_exr.min(), avg_gaussian_blue_noise_fft_exr.max())
        
        # imageio.imsave('float_img.exr', avg_gaussian_blue_noise_fft_exr)
        cv2.imwrite(output_dir+'/gaussianBN_res{:}_and_spectrum_{:}_repetitive_{:}_spectrum.exr'.format(res, cur_step, repetitive), avg_gaussian_blue_noise_fft_exr)


def main():
    fig_main_3_4()
    fig_main_inset()
    fig_main_10()
    fig_supp_1_2()
    


if __name__ == '__main__':
    main()
