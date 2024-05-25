import os, sys, glob, math
import numpy as np
from collections import OrderedDict
from decord import VideoReader, cpu
import cv2
import torch
import torchvision
import imageio
from tqdm import trange
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler


def prepare_latents(args, latents_dir, sampler):
    latents_list = []

    video = torch.load(latents_dir+f"/{args.num_inference_steps}.pt")
    if args.lookahead_denoising:
        for i in range(args.video_length // 2):
            alpha = sampler.ddim_alphas[0]
            beta = 1 - alpha
            latents = alpha**(0.5) * video[:,:,[0]] + beta**(0.5) * torch.randn_like(video[:,:,[0]])
            latents_list.append(latents)

    for i in range(args.num_inference_steps):
        alpha = sampler.ddim_alphas[i] # image -> noise
        beta = 1 - alpha
        frame_idx = max(0, i-(args.num_inference_steps - args.video_length))
        latents = (alpha)**(0.5) * video[:,:,[frame_idx]] + (1-alpha)**(0.5) * torch.randn_like(video[:,:,[frame_idx]])
        latents_list.append(latents)

    latents = torch.cat(latents_list, dim=2)

    return latents


def shift_latents(latents):
    # shift latents
    latents[:,:,:-1] = latents[:,:,1:].clone()

    # add new noise to the last frame
    latents[:,:,-1] = torch.randn_like(latents[:,:,-1])

    return latents


def batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, **kwargs):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    x_T = None
    batch_variants = []
    #batch_variants1, batch_variants2 = [], []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=True,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            **kwargs
                                            )
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage_2DAE(samples) # b,c,f,h,w
        batch_variants.append(batch_images)
    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1) # b,n,c,f,h,w
    return batch_variants

def base_ddim_sampling(model, cond, noise_shape, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, latents_dir=None, **kwargs):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]
    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq": # True
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'): # False
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict): # True
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else: # False
            uc = uc_emb
    else:
        uc = None
    
    x_T = None

    if ddim_sampler is not None:
        kwargs.update({"clean_cond": True})
        samples, _ = ddim_sampler.sample(S=ddim_steps,
                                        conditioning=cond,
                                        batch_size=noise_shape[0],
                                        shape=noise_shape[1:],
                                        verbose=True,
                                        unconditional_guidance_scale=cfg_scale,
                                        unconditional_conditioning=uc,
                                        eta=ddim_eta,
                                        temporal_length=noise_shape[2],
                                        conditional_guidance_scale_temporal=temporal_cfg_scale,
                                        x_T=x_T,
                                        latents_dir=latents_dir,
                                        **kwargs
                                        )
    ## reconstruct from latent to pixel space
    # samples: b,c,f,h,w
    batch_images = model.decode_first_stage_2DAE(samples) # b,c,f,H,W

    return batch_images, ddim_sampler, samples

def fifo_ddim_sampling(args, model, conditioning, noise_shape, ddim_sampler,\
                        cfg_scale=1.0, output_dir=None, latents_dir=None, save_frames=False, **kwargs):
    batch_size = noise_shape[0]
    kwargs.update({"clean_cond": True})

    # check condition bs
    if conditioning is not None:
        if isinstance(conditioning, dict):
            try:
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
            except:
                cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

            if cbs != batch_size:
                print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
        else:
            if conditioning.shape[0] != batch_size:
                print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
    
    cond = conditioning

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        prompts = batch_size * [""]
        #prompts = N * T * [""]  ## if is_imgbatch=True
        uc_emb = model.get_learned_conditioning(prompts)
        
        uc = {key:cond[key] for key in cond.keys()}
        uc.update({'c_crossattn': [uc_emb]})
        
    else:
        uc = None
    
    latents = prepare_latents(args, latents_dir, ddim_sampler)

    num_frames_per_gpu = args.video_length
    if args.save_frames:
        fifo_dir = os.path.join(output_dir, "fifo")
        os.makedirs(fifo_dir, exist_ok=True)

    fifo_video_frames = []

    timesteps = ddim_sampler.ddim_timesteps
    indices = np.arange(args.num_inference_steps)

    if args.lookahead_denoising:
        timesteps = np.concatenate([np.full((args.video_length//2,), timesteps[0]), timesteps])
        indices = np.concatenate([np.full((args.video_length//2,), 0), indices])
    for i in trange(args.new_video_length + args.num_inference_steps - args.video_length, desc="fifo sampling"):
        for rank in reversed(range(2 * args.num_partitions if args.lookahead_denoising else args.num_partitions)):
            start_idx = rank*(num_frames_per_gpu // 2) if args.lookahead_denoising else rank*num_frames_per_gpu
            midpoint_idx = start_idx + num_frames_per_gpu // 2
            end_idx = start_idx + num_frames_per_gpu

            t = timesteps[start_idx:end_idx]
            idx = indices[start_idx:end_idx]

            input_latents = latents[:,:,start_idx:end_idx].clone()
            output_latents, _ = ddim_sampler.fifo_onestep(
                                            cond=cond,
                                            shape=noise_shape,
                                            latents=input_latents,
                                            timesteps=t,
                                            indices=idx,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            **kwargs
                                            )
            if args.lookahead_denoising:
                latents[:,:,midpoint_idx:end_idx] = output_latents[:,:,-(num_frames_per_gpu//2):]
            else:
                latents[:,:,start_idx:end_idx] = output_latents
            del output_latents
        

        # reconstruct from latent to pixel space
        first_frame_idx = args.video_length // 2 if args.lookahead_denoising else 0
        frame_tensor = model.decode_first_stage_2DAE(latents[:,:,[first_frame_idx]]) # b,c,1,H,W
        image = tensor2image(frame_tensor)
        if save_frames:
            fifo_path = os.path.join(fifo_dir, f"{i}.png")
            image.save(fifo_path)
        fifo_video_frames.append(image)
            
        latents = shift_latents(latents)

    return fifo_video_frames

def fifo_ddim_sampling_multiprompts(args, model, conditioning, noise_shape, ddim_sampler, multiprompts,
                                    cfg_scale=1.0, output_dir=None, latents_dir=None, save_frames=False, **kwargs):
    batch_size = noise_shape[0]
    kwargs.update({"clean_cond": True})

    prompt_lengths = np.array([int(i) for i in multiprompts[-1].split(',')]).cumsum()
    multiprompts_embed = [model.get_learned_conditioning(prompt) for prompt in multiprompts[:-1]]
    assert len(prompt_lengths) == len(multiprompts_embed)

    # check condition bs
    if conditioning is not None:
        if isinstance(conditioning, dict):
            try:
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
            except:
                cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

            if cbs != batch_size:
                print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
        else:
            if conditioning.shape[0] != batch_size:
                print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
    
    cond = conditioning
    ## construct unconditional guidance
    if cfg_scale != 1.0:
        prompts = batch_size * [""]
        #prompts = N * T * [""]  ## if is_imgbatch=True
        uc_emb = model.get_learned_conditioning(prompts)
        
        uc = {key:cond[key] for key in cond.keys()}
        uc.update({'c_crossattn': [uc_emb]})    
    else:
        uc = None

    latents = prepare_latents(args, latents_dir, ddim_sampler)

    num_frames_per_gpu = args.video_length
    fifo_dir = os.path.join(output_dir, "fifo")
    # os.makedirs(fifo_dir, exist_ok=True)

    fifo_video_frames = []

    timesteps = ddim_sampler.ddim_timesteps
    indices = np.arange(args.num_inference_steps)

    if args.lookahead_denoising:
        timesteps = np.concatenate([np.full((args.video_length//2,), timesteps[0]), timesteps])
        indices = np.concatenate([np.full((args.video_length//2,), 0), indices])
    
    j = 0
    for i in trange(prompt_lengths[-1] + args.num_inference_steps - args.video_length, desc="fifo sampling"):

        if i - (args.num_inference_steps - args.video_length) >= prompt_lengths[j]:
            j = j +1
        embed = multiprompts_embed[j]

        cond.update({'c_crossattn':[embed]})
        for rank in reversed(range(2 * args.num_partitions if args.lookahead_denoising else args.num_partitions)):
            start_idx = rank*(num_frames_per_gpu // 2) if args.lookahead_denoising else rank*num_frames_per_gpu
            midpoint_idx = start_idx + num_frames_per_gpu // 2
            end_idx = start_idx + num_frames_per_gpu

            t = timesteps[start_idx:end_idx]
            idx = indices[start_idx:end_idx]

            input_latents = latents[:,:,start_idx:end_idx].clone()
            output_latents, _ = ddim_sampler.fifo_onestep(
                                            cond=cond,
                                            shape=noise_shape,
                                            latents=input_latents,
                                            timesteps=t,
                                            indices=idx,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            **kwargs
                                            )
            if args.lookahead_denoising:
                latents[:,:,midpoint_idx:end_idx] = output_latents[:,:,-(num_frames_per_gpu//2):]
            else:
                latents[:,:,start_idx:end_idx] = output_latents
            del output_latents
        

        # reconstruct from latent to pixel space
        first_frame_idx = args.video_length // 2 if args.lookahead_denoising else 0
        frame_tensor = model.decode_first_stage_2DAE(latents[:,:,[first_frame_idx]]) # b,c,1,H,W
        image = tensor2image(frame_tensor)
        if save_frames:
            fifo_path = os.path.join(fifo_dir, f"{i}.png")
            image.save(fifo_path)
        fifo_video_frames.append(image)
            
        latents = shift_latents(latents)
    return fifo_video_frames

def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def get_dirlist(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        try:
            ## deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model
    load_checkpoint(model, ckpt, full_strict=True)
    print('>>> model checkpoint loaded.')
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_video_batch(filepath_list, frame_stride, video_size=(256,256), video_frames=16):
    '''
    Notice about some special cases:
    1. video_frames=-1 means to take all the frames (with fs=1)
    2. when the total video frames is less than required, padding strategy will be used (repreated last frame)
    '''
    fps_list = []
    batch_tensor = []
    assert frame_stride > 0, "valid frame stride should be a positive interge!"
    for filepath in filepath_list:
        padding_num = 0
        vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames-1) // frame_stride + 1
        if video_frames < 0:
            ## all frames are collected: fs=1 is a must
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [frame_stride*i for i in range(query_frames)]

        ## [t,h,w,c] -> [c,t,h,w]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frame_tensor = (frame_tensor / 255. - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = torch.cat([frame_tensor, *([frame_tensor[:,-1:,:,:]]*padding_num)], dim=1)
            print(f'{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded.')
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps/frame_stride)
        fps_list.append(sample_fps)
    
    return torch.stack(batch_tensor, dim=0)

from PIL import Image
def load_image_batch(filepath_list, image_size=(256,256)):
    batch_tensor = []
    for filepath in filepath_list:
        _, filename = os.path.split(filepath)
        _, ext = os.path.splitext(filename)
        if ext == '.mp4':
            vidreader = VideoReader(filepath, ctx=cpu(0), width=image_size[1], height=image_size[0])
            frame = vidreader.get_batch([0])
            img_tensor = torch.tensor(frame.asnumpy()).squeeze(0).permute(2, 0, 1).float()
        elif ext == '.png' or ext == '.jpg':
            img = Image.open(filepath).convert("RGB")
            rgb_img = np.array(img, np.float32)
            #bgr_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            #bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (image_size[1],image_size[0]), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        else:
            print(f'ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]')
            raise NotImplementedError
        img_tensor = (img_tensor / 255. - 0.5) * 2
        batch_tensor.append(img_tensor)
    return torch.stack(batch_tensor, dim=0)


def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [t, n*h, w, 3]
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})

def save_gif(batch_tensors, savedir, name):
    vid_tensor = torch.squeeze(batch_tensors) # c,f,h,w

    video = vid_tensor.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    video = video.permute(1, 0, 2, 3) # f,c,h,w

    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1) # f,h,w,c

    frames = video.chunk(video.shape[0], dim=0)
    frames = [frame.squeeze(0) for frame in frames]
    savepath = os.path.join(savedir, f"{name}.gif")

    imageio.mimsave(savepath, frames, duration=100)

def tensor2image(batch_tensors):
    img_tensor = torch.squeeze(batch_tensors) # c,h,w

    image = img_tensor.detach().cpu()
    image = torch.clamp(image.float(), -1., 1.)

    image = (image + 1.0) / 2.0
    image = (image * 255).to(torch.uint8).permute(1, 2, 0) # h,w,c
    image = image.numpy()
    image = Image.fromarray(image)
    
    return image
