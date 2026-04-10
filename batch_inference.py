"""Batch AudioDiT inference for SeedTTS-like evaluation lists.

Format per line:  uid|prompt_text|prompt_wav_relative_path|gen_text

Usage:
    python batch_inference.py \
        --lst /path/to/meta.lst \
        --output_dir /path/to/output \
        --model_dir meituan-longcat/LongCat-AudioDiT-1B \
        --guidance_method apg
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch

import audiodit
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from utils import normalize_text, load_audio, approx_duration_from_text

torch.backends.cudnn.benchmark = False

@torch.no_grad()
def infer_one(gen_text, prompt_text, prompt_wav_path, model, tokenizer, device,
              nfe=16, cfg_strength=4.0, guidance_method="cfg"):
    sr = model.config.sampling_rate
    full_hop = model.config.latent_hop
    max_duration = model.config.max_wav_duration

    prompt_text = normalize_text(prompt_text)
    gen_text = normalize_text(gen_text)

    sep = ' ' if prompt_text[-1] == '.' else ''
    full_text = f"{prompt_text}{sep}{gen_text}"
    inputs = tokenizer([full_text], padding="longest", return_tensors="pt")
    prompt_wav = load_audio(prompt_wav_path, sr).unsqueeze(0)

    # Encode prompt audio once (reused for duration estimation and generation)
    prompt_latent, prompt_dur = model.encode_prompt_audio(prompt_wav.to(device))

    prompt_time = prompt_dur * full_hop / sr
    dur_sec = approx_duration_from_text(gen_text, max_duration - prompt_time)
    approx_pd = approx_duration_from_text(prompt_text, max_duration)
    ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
    dur_sec *= ratio
    duration = int(dur_sec * sr // full_hop)
    duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

    output = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_latent=prompt_latent,
        prompt_duration_frames=prompt_dur,
        duration=duration,
        steps=nfe,
        cfg_strength=cfg_strength,
        guidance_method=guidance_method,
    )
    return output.waveform.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Batch AudioDiT inference")
    parser.add_argument("--lst", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="hf_audiodit_1b")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--guidance_strength", type=float, default=4.0)
    parser.add_argument("--guidance_method", type=str, default="cfg", choices=["cfg", "apg"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    lst_dir = os.path.dirname(os.path.abspath(args.lst))
    os.makedirs(args.output_dir, exist_ok=True)

    items = []
    with open(args.lst) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            items.append((parts[0], parts[1], os.path.join(lst_dir, parts[2]), parts[3]))
    print(f"Loaded {len(items)} items from {args.lst}")

    model = AudioDiTModel.from_pretrained(args.model_dir).to(device)
    model.vae.to_half()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    print(f"Model loaded on {device}, method={args.guidance_method}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    total = len(items)
    t0 = time.time()
    for i, (uid, pt, pwa, gt) in enumerate(items):
        out_path = os.path.join(args.output_dir, f"{uid}.wav")
        if os.path.exists(out_path):
            continue
        try:
            wav = infer_one(gt, pt, pwa, model, tokenizer, device,
                            args.nfe, args.guidance_strength, args.guidance_method)
            sf.write(out_path, wav, model.config.sampling_rate)
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (total - i - 1) / speed if speed > 0 else 0
            print(f"[{i+1}/{total}] {uid}  {len(wav)/model.config.sampling_rate:.1f}s  ({speed:.1f} it/s, ETA {eta/60:.0f}min)")
        except Exception as e:
            print(f"[{i+1}/{total}] ERROR {uid}: {e}")

    print(f"\nDone. {total} items in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
