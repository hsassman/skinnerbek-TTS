"""Clone a voice from a short reference clip (Base model) and speak new text.

This is the most reliable way to get an authentic Cape Flats / Capetonian
accent out of Qwen3-TTS: supply a real reference recording of someone
speaking in that accent, plus an exact transcript of what they said, and the
model will carry the accent/timbre over to new text.

Reference audio requirements (per the Qwen3-TTS docs): a few seconds of
clean, single-speaker audio works; --ref-text must match what's actually
said in the clip.

Examples:
    # Local reference file
    python scripts/generate_voice_clone.py \
        --ref-audio samples/cape_flats_ref.wav \
        --ref-text "Ja no, it's lekker hot today, hey." \
        --text "Howzit my bru, you good?" \
        --out output/clone_1.wav

    # Multiple lines from a text file (one per line), reusing the same
    # cloned voice prompt so the reference audio is only encoded once
    python scripts/generate_voice_clone.py \
        --ref-audio samples/cape_flats_ref.wav \
        --ref-text "Ja no, it's lekker hot today, hey." \
        --text-file samples/lines.txt \
        --out-dir output/clone_batch
"""
import argparse
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from common import add_common_args, model_id, resolve_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-audio", required=True, help="Path or URL to reference audio")
    parser.add_argument("--ref-text", required=True, help="Exact transcript of the reference audio")
    parser.add_argument("--language", default="English")

    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Single line of text to synthesize")
    text_group.add_argument("--text-file", help="File with one line of text per generation")

    parser.add_argument("--out", default="output/clone.wav", help="Output path (single --text mode)")
    parser.add_argument("--out-dir", default="output", help="Output directory (--text-file mode)")
    add_common_args(parser)
    args = parser.parse_args()

    model = Qwen3TTSModel.from_pretrained(
        model_id("Base", args.model_size),
        device_map=args.device,
        dtype=resolve_dtype(args.device),
    )

    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
    )

    if args.text:
        wavs, sr = model.generate_voice_clone(
            text=args.text,
            language=args.language,
            voice_clone_prompt=voice_clone_prompt,
        )
        sf.write(args.out, wavs[0], sr)
        print(f"Wrote {args.out} ({sr} Hz)")
        return

    lines = [line.strip() for line in Path(args.text_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs, sr = model.generate_voice_clone(
        text=lines,
        language=[args.language] * len(lines),
        voice_clone_prompt=voice_clone_prompt,
    )
    for i, wav in enumerate(wavs, start=1):
        out_path = out_dir / f"clone_{i:03d}.wav"
        sf.write(out_path, wav, sr)
        print(f"Wrote {out_path} ({sr} Hz)")


if __name__ == "__main__":
    main()
