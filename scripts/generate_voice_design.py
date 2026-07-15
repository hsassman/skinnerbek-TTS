"""Generate speech from a natural-language voice description (VoiceDesign model).

This is the first thing worth trying for a Cape Flats / Capetonian accent
without reference audio: describe the accent directly in --instruct. Qwen3-TTS
was not demonstrated on South African English in its own docs, so treat the
output as experimental -- it may drift toward a generic English or another
accent entirely. If it doesn't sound right, voice cloning (generate_voice_clone.py)
with a real reference clip is the more reliable path.

Example:
    python scripts/generate_voice_design.py \
        --text "Ja no, it's lekker hot today, hey." \
        --language English \
        --instruct "South African English, Cape Flats/Cape Coloured accent, \
Cape Town, warm conversational male voice, mid-20s" \
        --out output/voice_design.wav
"""
import argparse

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from common import add_common_args, model_id, resolve_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", default="English")
    parser.add_argument(
        "--instruct",
        required=True,
        help="Natural-language description of the voice/accent to design",
    )
    parser.add_argument("--out", default="output/voice_design.wav")
    add_common_args(parser)
    args = parser.parse_args()
    # VoiceDesign only ships as 1.7B on Hugging Face; no 0.6B checkpoint exists.
    args.model_size = "1.7B"

    model = Qwen3TTSModel.from_pretrained(
        model_id("VoiceDesign", args.model_size),
        device_map=args.device,
        dtype=resolve_dtype(args.device),
    )

    wavs, sr = model.generate_voice_design(
        text=args.text,
        language=args.language,
        instruct=args.instruct,
    )
    sf.write(args.out, wavs[0], sr)
    print(f"Wrote {args.out} ({sr} Hz)")


if __name__ == "__main__":
    main()
