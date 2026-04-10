"""AudioDiT inference using AudioDiTModel.from_pretrained.

Usage:
    # TTS:
    python inference.py --text "hello world" --output_audio output.wav --model_dir meituan-longcat/LongCat-AudioDiT-1B

    # Voice cloning (with prompt audio):
    python inference.py \
        --text "要合成的文本" \
        --prompt_text "参考音频对应的文本" \
        --prompt_audio prompt.wav \
        --output_audio output.wav \
        --model_dir meituan-longcat/LongCat-AudioDiT-1B

    # APG guidance:
    python inference.py --text "hello" --output_audio out.wav --guidance_method apg
"""

import argparse
import numpy as np
import soundfile as sf
import torch

import audiodit  # auto-registers AudioDiTConfig/AudioDiTModel
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from utils import normalize_text, load_audio, approx_duration_from_text

torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="AudioDiT TTS inference")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--prompt_text", type=str, default=None, help="Text of the prompt audio")
    parser.add_argument("--prompt_audio", type=str, default=None, help="Path to prompt audio for voice cloning")
    parser.add_argument("--output_audio", type=str, required=True, help="Output wav path")
    parser.add_argument("--model_dir", type=str, default="meituan-longcat/LongCat-AudioDiT-1B", help="Path to HF model directory")
    parser.add_argument("--nfe", type=int, default=16, help="Number of ODE steps")
    parser.add_argument("--guidance_strength", type=float, default=4.0, help="CFG/APG strength")
    parser.add_argument("--guidance_method", type=str, default="cfg", choices=["cfg", "apg"])
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load model
    model = AudioDiTModel.from_pretrained(args.model_dir).to(device)
    model.vae.to_half()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

    sr = model.config.sampling_rate
    full_hop = model.config.latent_hop
    max_duration = model.config.max_wav_duration

    # Text
    text = normalize_text(args.text)
    no_prompt = args.prompt_audio is None

    if not no_prompt:
        prompt_text = normalize_text(args.prompt_text)
        full_text = f"{prompt_text} {text}"
    else:
        full_text = text

    print(f"Text: {full_text}")
    inputs = tokenizer([full_text], padding="longest", return_tensors="pt")

    # Prompt audio
    if not no_prompt:
        prompt_wav = load_audio(args.prompt_audio, sr).unsqueeze(0)

        # Encode prompt audio once (reused for duration estimation and generation)
        with torch.no_grad():
            prompt_latent, prompt_dur = model.encode_prompt_audio(prompt_wav.to(device))
    else:
        prompt_wav = None
        prompt_latent = None
        prompt_dur = 0

    # Duration estimation
    prompt_time = prompt_dur * full_hop / sr
    dur_sec = approx_duration_from_text(text, max_duration=max_duration - prompt_time)
    if not no_prompt:
        approx_pd = approx_duration_from_text(prompt_text, max_duration=max_duration)
        ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
        dur_sec = dur_sec * ratio
    print(f"Approx duration: {dur_sec:.3f}s")
    duration = int(dur_sec * sr // full_hop)
    duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

    # Generate
    output = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_latent=prompt_latent,
        prompt_duration_frames=prompt_dur if prompt_latent is not None else None,
        duration=duration,
        steps=args.nfe,
        cfg_strength=args.guidance_strength,
        guidance_method=args.guidance_method,
    )

    wav = output.waveform.squeeze().detach().cpu().numpy()
    sf.write(args.output_audio, wav, sr)
    print(f"Saved: {args.output_audio} ({len(wav)/sr:.2f}s)")


if __name__ == "__main__":
    main()
