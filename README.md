## FIFO-Diffusion: Generating Infinite Videos from Text without Training
<div align="center">

<p>
💾 <b> VRAM < 10GB </b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
🚀 <b> Infinitely Long Videos</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
⭐️ <b> Tuning-free</b>
</p>

<a href="https://arxiv.org/abs/2405.11473"><img src='https://img.shields.io/badge/arXiv-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://jjihwan.github.io/projects/FIFO-Diffusion"><img src='https://img.shields.io/badge/Project-Page-Green'></a>

</div>

## 📽️ See more video samples in our <a href="https://jjihwan.github.io/projects/FIFO-Diffusion"> project page</a>!
<div align="center">

<img src="https://github.com/jjihwan/FIFO-Diffusion_public/assets/63445348/aafafa52-5ddf-4093-9d29-681fe469e447">

"An astronaut floating in space, high quality, 4K resolution.", 

VideoCrafter2, 100 frames, 320 X 512 resolution

<img src="assets/opensora_fifo.gif">

"A corgi vlogging itself in tropical Maui."

Open-Sora Plan, 512 X 512 resolution


</div>


## News 📰
**[2024.06.06]** 🔥🔥🔥 We are excited to release the code for **Open-Sora Plan v1.1.0**. Thanks to the authors for open-sourcing the awesome baseline!

**[2024.05.25]** 🥳🥳🥳 We are thrilled to present our official PyTorch implementation for FIFO-Diffusion. We are releasing the code based on **VideoCrafter2**.

**[2024.05.19]** 🚀🚀🚀 Our paper, *FIFO-Diffusion: Generating Infinite Videos from Text without Training*, has been archived.

## Clone our repository
```
git clone git@github.com:jjihwan/FIFO-Diffusion_public.git
cd FIFO-Diffusion_public
```

## ☀️ Start with <a href="https://github.com/AILab-CVC/VideoCrafter">VideoCrafter</a>

### 1. Environment Setup ⚙️ (python==3.10.14 recommended)
```
python3 -m venv .fifo
source .fifo/bin/activate

pip install -r requirements.txt
```

### 2.1 Download the models from Hugging Face🤗
|Model|Resolution|Checkpoint
|:----|:---------|:---------
|VideoCrafter2 (Text2Video)|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)

### 2.2 Set file structure
Store them as following structure:
```
cd FIFO-Diffusion_public
    .
    └── videocrafter_models
        └── base_512_v2
            └── model.ckpt      # VideoCrafter2 checkpoint
```

### 3.1. Run with VideoCrafter2 (Single GPU)
Requires less than **9GB VRAM** with Titan XP.
```
python3 videocrafter_main.py --save_frames
```

### 3.2. Distributed Parallel inference with VideoCrafter2 (Multiple GPUs)
May consume slightly more memory than the single GPU inference (**11GB** with Titan XP).
Please note that our implementation for parallel inference might not be optimal.
Pull requests are welcome! 🤓

```
python3 videocrafter_main_mp.py --num_gpus 8 --save_frames
```

### 3.3. Multi-prompt generation
Comming soon.

## ☀️ Start with <a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan">Open-Sora Plan v1.1.0</a>
For simple implementation, we use the DDPM scheduler for Open-Sora Plan v1.1.0.
Since Open-Sora Plan recommends using the PNDM scheduler, the results might not show the optimal performance.
Multi-processing (parallelizable inference) and adapting PNDM scheduler are our next plan.

### 1. Environment Setup ⚙️ (python==3.10.14 recommended)
```
cd FIFO-Diffusion_public
git clone git@github.com:PKU-YuanGroup/Open-Sora-Plan.git

python -m venv .sora
source .sora/bin/activate

cd Open-Sora-Plan
pip install -e .

pip install deepspeed
```

### 2. Run with Open-Sora Plan v1.1.0, 65x512x512 model
Requires about 40GB VRAM with A6000.
It uses *n=8* by default.
```
sh scripts/opensora_fifo_65.sh
```

### 3. Run with Open-Sora Plan v1.1.0, 221x512x512 model
Requires about 40GB VRAM with A6000.
It uses *n=4* by default.
```
sh scripts/opensora_fifo_221.sh
```

### 4. Distributed Parallel inference with Open-Sora Plan (WIP)
Comming soon.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jjihwan/FIFO-Diffusion_public&type=Date)](https://star-history.com/#jjihwan/FIFO-Diffusion_public&Date)

## 😆 Citation
```
@article{kim2024fifo,
	title = {FIFO-Diffusion: Generating Infinite Videos from Text without Training},
	author = {Jihwan Kim and Junoh Kang and Jinyoung Choi and Bohyung Han},
	journal = {arXiv preprint arXiv:2405.11473},
	year = {2024},
}
```


## 🤓 Acknowledgements
Our codebase builds on [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter), [Open-Sora Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w). 
Thanks to the authors for sharing their awesome codebases!
