"""Launch the bundled qwen_tts Gradio UI with settings that work on this machine.

The UI that ships with qwen_tts (`qwen-tts-demo`) defaults to --device cuda:0
and FlashAttention-2, neither of which exists here (AMD GPU, CPU-only torch, no
flash-attn). Running it unmodified fails. This wrapper flips those defaults to
cpu/no-flash-attn and picks a dtype that fits in RAM (see common.resolve_dtype:
the 1.7B models need bfloat16 on this 8GB box).

One checkpoint per process -- the UI renders whichever capability the loaded model
supports, so pick the one you want:

    python scripts/launch_ui.py preset   # built-in speakers + style instructions
    python scripts/launch_ui.py design   # design a voice from a description (1.7B)
    python scripts/launch_ui.py clone    # clone from reference audio, save/load voices

Generation is slow on CPU -- expect tens of seconds to minutes per line. The UI's
"Save / Load Voice" tab (clone mode) writes the same .pt format that
`tts.py voices` uses, so voices are shareable between the two; save from the UI
into voices/ to have `tts.py clone --voice <name>` pick it up.
"""
import argparse
import sys

from common import (
    MODEL_SIZES,
    VOICES_DIR,
    add_model_args,
    configure_hf,
    check_memory,
    model_id,
    resolve_dtype,
)

KINDS = {"preset": "CustomVoice", "design": "VoiceDesign", "clone": "Base"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=sorted(KINDS), help="Which capability to load the UI for")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--ip", default="127.0.0.1", help="Bind address (default: 127.0.0.1, local only)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    add_model_args(parser)
    # `mode` is positional, so --model-size can't be restricted per-kind at parse time the way
    # tts.py does it. Default to None and resolve below, so we only complain when the size was
    # actually asked for rather than on every design run.
    parser.set_defaults(model_size=None)
    args = parser.parse_args()

    kind = KINDS[args.mode]
    available = MODEL_SIZES[kind]
    if args.model_size is None:
        args.model_size = available[0]
    elif args.model_size not in available:
        raise SystemExit(
            f"{kind} is only published as {'/'.join(available)}; --model-size {args.model_size} "
            f"does not exist."
        )

    if args.mode == "preset" and args.model_size == "0.6B":
        print(
            "Note: the UI's Instruction box does nothing on 0.6B (the library drops it silently). "
            "Re-run with --model-size 1.7B for style control.",
            file=sys.stderr,
        )

    repo = model_id(kind, args.model_size)
    # Same reasoning as common.resolve_dtype: float32 for the 1.7B models would need ~7.7GB of
    # weights and crash this box, so let the shared policy decide.
    dtype = resolve_dtype(args.device, args.model_size, args.dtype)
    check_memory(args.model_size, dtype)
    dtype_name = str(dtype).split(".")[-1]

    demo_argv = [
        repo,
        "--device", args.device,
        "--dtype", dtype_name,
        "--ip", args.ip,
        "--port", str(args.port),
    ]
    if not args.device.startswith("cuda"):
        demo_argv.append("--no-flash-attn")
    if args.share:
        demo_argv.append("--share")

    if args.mode == "clone":
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Tip: save voices into {VOICES_DIR} so `tts.py clone --voice <name>` finds them.", file=sys.stderr)

    print(f"Loading {repo} on {args.device} -- first run downloads weights, and CPU load is slow.", file=sys.stderr)
    print(f"UI will be at http://{args.ip}:{args.port}", file=sys.stderr)

    # Must precede the qwen_tts import: huggingface_hub reads HF_HUB_OFFLINE at import time.
    configure_hf(args)
    from qwen_tts.cli.demo import main as demo_main

    return demo_main(demo_argv)


if __name__ == "__main__":
    raise SystemExit(main())
