## FIFO-Diffusion: Generating Infinite Videos from Text without Training
<div align="center">

<p>
💾 <b> VRAM < 10GB</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
🚀 <b> Infinitely Long Videos</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
⭐️ <b> Tuning-free</b>
</p>

<a href="https://jjihwan.github.io/projects/FIFO-Diffusion"><img src='https://img.shields.io/badge/arXiv-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://jjihwan.github.io/projects/FIFO-Diffusion"><img src='https://img.shields.io/badge/Project-Page-Green'></a>

</div>

## 📽️ See video samples in our <a href="https://jjihwan.github.io/projects/FIFO-Diffusion"> project page</a>!

</div>

## Clone our repository
```
git clone git@github.com:jjihwan/FIFO-Diffusion.git
cd FIFO-Diffusion
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
|:---------|:---------|:--------
|VideoCrafter2 (Text2Video)|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|VideoCrafter1 (Text2Video)|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-512/blob/main/model.ckpt)

### 2.2 Set file structure
Store them as following structure:
```
cd FIFO-Diffusion
    .
    └── videocrafter_models
        ├── base_512_v2
        │   └── model.ckpt      # VideoCrafter2 checkpoint
        └── base_512_v1
            └── model.ckpt      # VideoCrafter1 checkpoint
```

### 3.1. Run with VideoCrafter2
```
python3 videocrafter_main.py
```

### 3.2. Distributed Parallel inference with VideoCrafter2 (Multiple GPUs required)

```
python3 videocrafter_main_mp.py --num_gpus 8
```

### 3.3. Run with VideoCrafter1
```
python3 videocrafter_main.py -ver=1
```

## ☀️ Start with <a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan">Open-Sora Plan</a>

### 1. Environment Setup ⚙️ (python==3.10.14 recommended)
```
cd FIFO-Diffusion
git clone git@github.com:PKU-YuanGroup/Open-Sora-Plan.git

python -m venv .sora
source .sora/bin/activate

cd Open-Sora-Plan
pip install -e .
```

### 2. Run with Open-Sora Plan
```
sh scripts/opensora_fifo_ddpm.sh
```

## ☀️ Start with <a href="https://huggingface.co/cerspense/zeroscope_v2_576w">zeroscope</a>

### 1. Environment Setup ⚙️ (python==3.10.14 recommended)
```
python3 -m venv .fifo
source .fifo/bin/activate

pip install -r requirements.txt
```

### 2. Run with zeroscope(Recommended)
```
mkdir zeroscope_models         # directory where the model will be stored
python3 zeroscope_main.py
```
