from argparse import ArgumentParser
from omegaconf import OmegaConf
import os
import torch
import numpy as np
from PIL import Image
import imageio

from pytorch_lightning import seed_everything

from scripts.evaluation.funcs_mp import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_gif
from scripts.evaluation.funcs_mp import base_ddim_sampling, fifo_ddim_sampling
from utils.utils import instantiate_from_config
from lvdm.models.samplers.ddim import DDIMSampler
import torch.multiprocessing as mp

def set_directory(args, prompt):
    if args.output_dir is None:
        output_dir = f"results/videocraft_v2_fifo/random_noise/{prompt[:100]}"
        if args.eta != 1.0:
            output_dir += f"/eta{args.eta}"

        if args.new_video_length != 100:
            output_dir += f"/{args.new_video_length}frames"
        if not args.lookahead_denoising:
            output_dir = output_dir.replace(f"{prompt[:100]}", f"{prompt[:100]}/no_lookahead_denoising")
        if args.num_partitions != 4:
            output_dir = output_dir.replace(f"{prompt[:100]}", f"{prompt[:100]}/n={args.num_partitions}")
        if args.video_length != 16:
            output_dir = output_dir.replace(f"{prompt[:100]}", f"{prompt[:100]}/f={args.video_length}")

    else:
        output_dir = args.output_dir

    latents_dir = f"results/videocraft_v2_fifo/latents/{args.num_inference_steps}steps/{prompt[:100]}/eta{args.eta}"

    print("The results will be saved in", output_dir)
    print("The latents will be saved in", latents_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)
    
    return output_dir, latents_dir


def main(args):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    #data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    models = [instantiate_from_config(model_config) for _ in range(args.num_gpus)]
    models = [model.to(f"cuda:{i}")for i, model in enumerate(models)]
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    models = [load_model_checkpoint(model, args.ckpt_path) for model in models]
    models = [model.eval() for model in models]


    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = args.video_length
    channels = models[0].channels

    ## step 2: load data
    ## -----------------------------------------------------------------
    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    prompt_list = load_prompts(args.prompt_file)
    num_samples = len(prompt_list)

    indices = list(range(num_samples))
    indices = indices[args.rank::args.num_processes]

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    for idx in indices:
        prompt = prompt_list[idx]
        output_dir, latents_dir = set_directory(args, prompt)

        batch_size = 1
        noise_shape = [batch_size, channels, frames, h, w]
        fpss = [torch.tensor([args.fps]*batch_size).to(model.device).long() for model in models]

        prompts = [prompt]
        text_embs = [model.get_learned_conditioning(prompts) for model in models]

        conds = [{"c_crossattn": [text_emb], "fps": fps} for text_emb, fps in zip(text_embs, fpss)]

        ## inference
        is_run_base = not (os.path.exists(latents_dir+f"/{args.num_inference_steps}.pt") and os.path.exists(latents_dir+f"/0.pt"))

        if is_run_base:
            base_tensor, ddim_sampler, _ = base_ddim_sampling(models[0], conds[0], noise_shape, \
                                                args.num_inference_steps, args.eta, args.unconditional_guidance_scale, \
                                                latents_dir=latents_dir)
            save_gif(base_tensor, output_dir, "origin")

            del base_tensor, ddim_sampler

        ddim_samplers = [DDIMSampler(model) for model in models]
        for ddim_sampler in ddim_samplers:
            ddim_sampler.make_schedule(ddim_num_steps=args.num_inference_steps, ddim_eta=args.eta, verbose=False)

        video_frames = fifo_ddim_sampling(
            args, models, conds, noise_shape, ddim_samplers, args.unconditional_guidance_scale, output_dir=output_dir, latents_dir=latents_dir, save_frames=args.save_frames
        )
        if args.output_dir is None:
            output_path = output_dir+"/fifo"
        else:
            output_path = output_dir+f"/{prompt[:100]}"

        if args.use_mp4:
            imageio.mimsave(output_path+".mp4", video_frames[-args.new_video_length:], fps=args.output_fps)
        else:
            imageio.mimsave(output_path+".gif", video_frames[-args.new_video_length:], duration=int(1000/args.output_fps)) 
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='videocrafter_models/base_512_v2/model.ckpt', help="checkpoint path")
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml", help="config (yaml) path")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--video_length", type=int, default=16, help="f in paper")
    parser.add_argument("--num_partitions", "-n", type=int, default=4, help="n in paper")
    parser.add_argument("--num_inference_steps", type=int, default=16, help="number of inference steps, it will be f * n forcedly")
    parser.add_argument("--prompt_file", "-p", type=str, default="prompts/test_prompts.txt", help="path to the prompt file")
    parser.add_argument("--new_video_length", "-l", type=int, default=100, help="N in paper; desired length of the output video")
    parser.add_argument("--num_processes", type=int, default=1, help="number of processes if you want to run only the subset of the prompts")
    parser.add_argument("--rank", type=int, default=0, help="rank of the process(0~num_processes-1)")
    parser.add_argument("--height", type=int, default=320, help="height of the output video")
    parser.add_argument("--width", type=int, default=512, help="width of the output video")
    parser.add_argument("--save_frames", action="store_true", default=False, help="save generated frames for each step")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="prompt classifier-free guidance")
    parser.add_argument("--lookahead_denoising", "-ld", action="store_false", default=True, help="use lookahead denoising")
    parser.add_argument("--eta", "-e", type=float, default=1.0)
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--output_dir", type=str, default=None, help="custom output directory")
    parser.add_argument("--use_mp4", action="store_true", default=False, help="use mp4 format for the output video")
    parser.add_argument("--output_fps", type=int, default=10, help="fps of the output video")

    args = parser.parse_args()

    args.num_inference_steps = args.video_length * args.num_partitions

    mp.set_start_method("spawn")

    seed_everything(args.seed)

    main(args)
