"""Generate speech using a Qwen3-TTS preset voice (CustomVoice model).

Note: none of the built-in preset voices (Ryan, Aiden for English) carry a
South African / Cape Flats accent -- this script is useful for testing the
model pipeline and general preset voices, not for accent work. See
generate_voice_design.py and generate_voice_clone.py for accent attempts.

Example:
    python scripts/generate_custom_voice.py \
        --text "Howzit, long time no see." \
        --language English --speaker Ryan \
        --instruct "Speak casually and warmly." \
        --out output/custom_voice.wav
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
        "--speaker",
        default="Ryan",
        help="Preset speaker name, e.g. Ryan, Aiden, Vivian, Serena, Uncle_Fu, Dylan, Eric, Ono_Anna, Sohee",
    )
    parser.add_argument("--instruct", default="", help="Optional style/emotion instruction")
    parser.add_argument("--out", default="output/custom_voice.wav")
    add_common_args(parser)
    args = parser.parse_args()

    model = Qwen3TTSModel.from_pretrained(
        model_id("CustomVoice", args.model_size),
        device_map=args.device,
        dtype=resolve_dtype(args.device),
    )

    wavs, sr = model.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
    )
    sf.write(args.out, wavs[0], sr)
    print(f"Wrote {args.out} ({sr} Hz)")


if __name__ == "__main__":
    main()
