# LongCat-AudioDiT: High-Fidelity Diffusion Text-to-Speech in the Waveform Latent Space

<div align="center">
  <img src="assets/LongCat-AudioDiT.svg" width="45%" alt="LongCat-AudioDiT" />
</div>
<hr>

<div align="center" style="line-height: 1;">
    <a href="https://arxiv.org/abs/2603.29339">
    <img alt="Paper" src="https://img.shields.io/badge/arXiv-2603.29339-b31b1b.svg" style="display: inline-block; vertical-align: middle;"/>  
    </a>
    <a href="https://github.com/meituan-longcat/LongCat-AudioDiT" target="_blank" style="margin: 2px;">
        <img alt="GitHub" src="https://img.shields.io/badge/GitHub-LongCatAudioDiT-white?logo=github&logoColor=white&color=a4b5d5" style="display: inline-block; vertical-align: middle;"/>
    </a>
        <a href="https://aria-k-alethia.github.io/LongCat-AudioDiT-demo" target="_blank" style="margin: 2px;">
        <img alt="Demo" src="https://img.shields.io/badge/Demo-LongCatAudioDiT-white?logo=googleplay&logoColor=white&color=eabcdd" style="display: inline-block; vertical-align: middle;"/>
    </a>
</div>
<div align="center" style="line-height: 1;">
    <a href="https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B" target="_blank" style="margin: 2px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCatAudioDiT3.5B-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B" target="_blank" style="margin: 2px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCatAudioDiT1B-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/assets/wechat_official_accounts.png" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-LongCat-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/Meituan_LongCat" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-LongCat-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
    <a href="https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Introduction

LongCat-AudioDiT is a state-of-the-art (SOTA) diffusion-based text-to-speech (TTS) model that directly operates in the waveform latent space.
> **Abstract**: We present LongCat-TTS, a novel, non-autoregressive diffusion-based text-to-speech (TTS) model that achieves state-of-the-art (SOTA) performance.
Unlike previous methods that rely on intermediate acoustic representations such as mel-spectrograms, the core innovation of LongCat-TTS lies in operating directly within the waveform latent space. This approach effectively mitigates compounding errors and drastically simplifies the TTS pipeline, requiring only a waveform variational autoencoder (Wav-VAE) and a diffusion backbone.
Furthermore, we introduce two critical improvements to the inference process: first, we identify and rectify a long-standing training-inference mismatch; second, we replace traditional classifier-free guidance with adaptive projection guidance to elevate generation quality.
Experimental results demonstrate that, despite the absence of complex multi-stage training pipelines or high-quality human-annotated datasets, LongCat-TTS achieves SOTA zero-shot voice cloning performance on the Seed benchmark while maintaining competitive intelligibility.
Specifically, our largest variant, LongCat-TTS-3.5B, outperforms the previous SOTA model (Seed-TTS), improving the speaker similarity (SIM) scores from 0.809 to 0.818 on Seed-ZH, and from 0.776 to 0.797 on Seed-Hard.
Finally, through comprehensive ablation studies and systematic analysis, we validate the effectiveness of our proposed modules.
Notably, we investigate the interplay between the Wav-VAE and the TTS backbone, revealing the counterintuitive finding that superior reconstruction fidelity in the Wav-VAE does not necessarily lead to better overall TTS performance.
Code and model weights are released to foster further research within the speech community.

![image](assets/architecture.png)

This repository provides the HuggingFace-compatible implementation, including model definition, weight conversion, and inference scripts.

## Experimental Results on Seed Benchmark
LongCat-AudioDiT obtains state-of-the-art (SOTA) voice cloning performance on the Seed-benchmark, surpassing both close-source and open-source modles.

| **Model** | **ZH CER (%)** ↓ | **ZH SIM** ↑ | **EN WER (%)** ↓ | **EN SIM** ↑ | **ZH-Hard CER (%)** ↓ | **ZH-Hard SIM** ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| GT | 1.26 | 0.755 | 2.14 | 0.734 | - | - |
| Seed-DiT | 1.18 | 0.809 | 1.73 | **0.790** | - | - |
| MaskGCT | 2.27 | 0.774 | 2.62 | 0.714 | 10.27 | 0.748 |
| E2 TTS | 1.97 | 0.730 | 2.19 | 0.710 | - | - |
| F5 TTS | 1.56 | 0.741 | 1.83 | 0.647 | 8.67 | 0.713 |
| F5R-TTS | 1.37 | 0.754 | - | - | 8.79 | 0.718 |
| ZipVoice | 1.40 | 0.751 | 1.64 | 0.668 | - | - |
| Seed-ICL | 1.12 | 0.796 | 2.25 | 0.762 | 7.59 | 0.776 |
| SparkTTS | 1.20 | 0.672 | 1.98 | 0.584 | - | - |
| FireRedTTS | 1.51 | 0.635 | 3.82 | 0.460 | 17.45 | 0.621 |
| Qwen2.5-Omni | 1.70 | 0.752 | 2.72 | 0.632 | 7.97 | 0.747 |
| Qwen2.5-Omni_RL | 1.42 | 0.754 | 2.33 | 0.641 | 6.54 | 0.752 |
| CosyVoice | 3.63 | 0.723 | 4.29 | 0.609 | 11.75 | 0.709 |
| CosyVoice2 | 1.45 | 0.748 | 2.57 | 0.652 | 6.83 | 0.724 |
| FireRedTTS-1S | 1.05 | 0.750 | 2.17 | 0.660 | 7.63 | 0.748 |
| CosyVoice3-1.5B | 1.12 | 0.781 | 2.21 | 0.720 | *5.83* | 0.758 |
| IndexTTS2 | 1.03 | 0.765 | 2.23 | 0.706 | 7.12 | 0.755 |
| DiTAR | 1.02 | 0.753 | 1.69 | 0.735 | - | - |
| MiniMax-Speech | 0.99 | 0.799 | 1.90 | 0.738 | - | - |
| VoxCPM | *0.93* | 0.772 | 1.85 | 0.729 | 8.87 | 0.730 |
| MOSS-TTS | 1.20 | 0.788 | 1.85 | 0.734 | - | - |
| Qwen3-TTS | 1.22 | 0.770 | **1.23** | 0.717 | 6.76 | 0.748 |
| CosyVoice3.5 | **0.87** | 0.797 | 1.57 | 0.738 | **5.71** | 0.786 |
| LongCat-AudioDiT-1B | 1.18 | *0.812* | 1.78 | 0.762 | 6.33 | *0.787* |
| LongCat-AudioDiT-3.5B | 1.09 | **0.818** | *1.50* | *0.786* | 6.04 | **0.797** |

*Notes*:

1. Results of MOSS-TTS are from [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
2. Results of CosyVoice3.5 are from [CosyVoice3.5](https://mp.weixin.qq.com/s/sTNC7bVphs9zofly3lBoUQ)

## Installation

### As a Python Package
```bash
pip install git+https://github.com/meituan-longcat/LongCat-AudioDiT
```

### Development

```bash
git clone https://github.com/meituan-longcat/LongCat-AudioDiT
pip install -r requirements.txt
```

## CLI Inference

```bash
# TTS
python inference.py --text "今天晴暖转阴雨，空气质量优至良，空气相对湿度较低。" --output_audio output.wav --model_dir meituan-longcat/LongCat-AudioDiT-1B

# Voice cloning
python inference.py \
    --text "今天晴暖转阴雨，空气质量优至良，空气相对湿度较低。" \
    --prompt_text "小偷却一点也不气馁，继续在抽屉里翻找。" \
    --prompt_audio assets/prompt.wav \
    --output_audio output.wav \
    --model_dir meituan-longcat/LongCat-AudioDiT-1B \
    --guidance_method apg

# Batch inference (SeedTTS eval format, one item per line: uid|prompt_text|prompt_wav_path|gen_text)
python batch_inference.py \
    --lst /path/to/meta.lst \
    --output_dir /path/to/output \
    --model_dir meituan-longcat/LongCat-AudioDiT-1B \
    --guidance_method apg
```

## Inference (Python API)

### 1. TTS
```python
import audiodit  # auto-registers with transformers
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
import torch, soundfile as sf

# Load model
model = AudioDiTModel.from_pretrained("meituan-longcat/LongCat-AudioDiT-1B").to("cuda")
model.vae.to_half()  # VAE runs in fp16 (matching original)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

# Zero-shot synthesis
inputs = tokenizer(["今天晴暖转阴雨，空气质量优至良，空气相对湿度较低。"], padding="longest", return_tensors="pt")
output = model(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    duration=62,  # latent frames
    steps=16,
    cfg_strength=4.0,
    guidance_method="cfg",  # or "apg"
)
sf.write("output.wav", output.waveform.squeeze().cpu().numpy(), 24000)
```

### 2. Voice Cloning (with prompt audio)

```python
import librosa, torch

# Load prompt audio
audio, _ = librosa.load("assets/prompt.wav", sr=24000, mono=True)
prompt_wav = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

# Concatenate prompt_text + gen_text for the text encoder
prompt_text = "小偷却一点也不气馁，继续在抽屉里翻找。"
gen_text = "今天晴暖转阴雨，空气质量优至良，空气相对湿度较低。"
inputs = tokenizer([f"{prompt_text} {gen_text}"], padding="longest", return_tensors="pt")

output = model(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    prompt_audio=prompt_wav,
    duration=138,  # prompt_frames + gen_frames
    steps=16,
    cfg_strength=4.0,
    guidance_method="apg",
)
```

## License Agreement
This repository, including both the model weights and the source code, is released under the **MIT License**.

Any contributions to this repository are licensed under the MIT License, unless otherwise stated. This license does not grant any rights to use Meituan trademarks or patents.

For details, see the [LICENSE](./LICENSE) file.

## Contact
Please contact us at <a href="mailto:longcat-team@meituan.com">longcat-team@meituan.com</a> or open an issue if you have any questions.

#### WeChat Group
<img src=./assets/longcat_wechat_group.jpeg width="200px">