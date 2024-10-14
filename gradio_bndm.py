import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse
from utils import get_model, sample_iadb
import numpy as np
from bluenoise.get_noise_recent import get_noise_v2
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMPipeline, DDIMScheduler

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dimension = 3
inp_chanel = 3

cov_mat_L = np.load('./bluenoise/cov_gaussianBN_L_res{:}_d{:}.npz'.format(64, dimension))['x'].astype(np.float32)
if opt.noise_type in ['gaussianRN']:
    cov_mat_L = np.load('./bluenoise/cov_gaussianRN_L_res{:}_d{:}.npz'.format(64, dimension))['x'].astype(np.float32)
cov_mat_L = torch.from_numpy(cov_mat_L).to(device).detach()



# Load the diffusion model

model_iadb = get_model(inp_chanel, 3)
model_bndm = get_model(inp_chanel, 6)

outer_folder = 'results_gaussianBN'
output_folder_iadb = outer_folder + '/{:}_{:}_{:}_outc{:}_seed{:}'.format(opt.dataset, 'gaussian', 'linear', 3, 0)
output_folder_bndm = outer_folder + '/{:}_{:}_{:}_{:}_{:}_{:}_outc{:}_seed{:}'.format(opt.dataset, 'gaussianBN', 'sigmoid', opt.scheduler_param, opt.scheduler_param_s, opt.scheduler_param_e, 6, 0)

model_iadb.load_state_dict(torch.load(f'{output_folder_iadb}/model.ckpt'))
model_iadb = model_iadb.to(device).eval()

model_bndm.load_state_dict(torch.load(f'{output_folder_bndm}/model.ckpt'))
model_bndm = model_bndm.to(device).eval()


scheduler = DDIMScheduler.from_pretrained(outer_folder + "/ddim_church_res64/scheduler")
scheduler.set_timesteps(opt.nb_steps)
model_ddim = UNet2DModel.from_pretrained(outer_folder + "/ddim_church_res64/unet", use_safetensors=True).to(device).eval()


resize_res = 128



def generate_images(seed, num_steps):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scheduler_params = np.array([opt.scheduler_param, opt.scheduler_param_s, opt.scheduler_param_e]).astype(np.float32)
    scheduler_params = torch.from_numpy(scheduler_params).float().to(device)
    x0 = torch.from_numpy(np.random.randn(1, 3, opt.res, opt.res)).float().to(device)

    sample_last_iadb, sample_all_iadb, inference_time = sample_iadb(model_iadb, x0, opt.nb_steps, 'linear', scheduler_params, out_channel=opt.out_channel, noise_type='gaussian', train_or_test='test', scheduler_alpha='linear')
    # print('sample_last_iadb:', sample_last_iadb.shape, len(sample_all_iadb))
    
    sample_last_bndm, sample_all_bndm, inference_time = sample_iadb(model_bndm, x0, opt.nb_steps, opt.scheduler_gamma, scheduler_params, out_channel=opt.out_channel, noise_type='gaussianBN', train_or_test='test', scheduler_alpha='linear')
    # print('sample_last_bndm:', sample_last_bndm.shape, len(sample_all_bndm))
    
    sample_all_ddim = []
    input = x0
    for t in scheduler.timesteps:
        with torch.no_grad():
            noisy_residual = model_ddim(input, t).sample
        previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = previous_noisy_sample
        sample_all_ddim.append(input[0:1])
    # print('sample_all_ddim:', len(sample_all_ddim))

    show_iadb = sample_all_iadb[num_steps]
    show_bndm = sample_all_bndm[num_steps]
    show_ddim = sample_all_ddim[num_steps]#torch.rand(*show_bndm.shape)
    
    # Generate images from different methods
    images = []
    # methods = ["DDIM", "IADB", "Ours"]  # Assuming these are methods that can be switched in your model settings
    # for method in methods:
    # For demonstration, assuming there is a way to specify the method in your model call
    # This part needs to be adjusted based on how your actual model can switch between methods
    # image = model("A sample prompt", num_inference_steps=num_steps, guidance_scale=7.5).images[0]

    image_iadb = np.clip(show_iadb[0].detach().cpu().numpy().transpose(1, 2, 0)*0.5+0.5, 0, 1)
    image_iadb = Image.fromarray((image_iadb * 255).astype(np.uint8)).convert("RGB").resize((resize_res, resize_res))
    
    image_bndm = np.clip(show_bndm[0].detach().cpu().numpy().transpose(1, 2, 0)*0.5+0.5, 0, 1)
    image_bndm = Image.fromarray((image_bndm * 255).astype(np.uint8)).convert("RGB").resize((resize_res, resize_res))
    
    image_ddim = np.clip(show_ddim[0].detach().cpu().numpy().transpose(1, 2, 0)*0.5+0.5, 0, 1)
    image_ddim = Image.fromarray((image_ddim * 255).astype(np.uint8)).convert("RGB").resize((resize_res, resize_res))
    
    images.append(image_ddim)
    images.append(image_iadb)
    images.append(image_bndm)

    return images

# Define Gradio interface
interface = gr.Interface(
    fn=generate_images,
    inputs=[gr.Number(label="Seed Number", value=0), gr.Slider(minimum=1, maximum=249, step=1, value=180, label="Intermediate steps")],
    outputs=[gr.Image(label="DDIM"), gr.Image(label="IADB/RectifiedFlow"), gr.Image(label="Our BNDM")],
    title="Visualizing intermediate denoising diffusion steps: DDIM vs. IADB/Rectifiedflow vs. Our BNDM",
    description="""
    Enter a seed and number of steps to generate images using different methods (DDIM, IADB, Our BNDM). Learn more about <a href='https://xchhuang.github.io/bndm' target='_blank'>our method: Blue noise for diffusion models</a>.
    """
)

# Launch the interface
interface.launch()
