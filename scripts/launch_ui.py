"""Launch a Qwen3-TTS web UI with settings that work on this machine.

Two UIs are available:

    python scripts/launch_ui.py            # the simple all-in-one app (scripts/app.py)
    python scripts/launch_ui.py preset     # the bundled qwen_tts demo, preset voices
    python scripts/launch_ui.py design     # the bundled qwen_tts demo, voice design (1.7B)
    python scripts/launch_ui.py clone      # the bundled qwen_tts demo, cloning + save/load

With no mode this hands off to `scripts/app.py`: one page with a script box, a
voice list (built-in speakers plus anything you've saved), a "describe a voice"
tab and a "voice library" tab. That's the one to use unless you specifically
want the original demo.

The bundled demo (`qwen-tts-demo`) defaults to --device cuda:0 and
FlashAttention-2, neither of which exists here (AMD GPU, CPU-only torch, no
flash-attn). Running it unmodified fails. The mode wrappers below flip those
defaults to cpu/no-flash-attn and pick a dtype that fits in RAM (see
common.resolve_dtype: the 1.7B models need bfloat16 on this 8GB box). One
checkpoint per process.

Generation is slow on CPU -- expect tens of seconds to minutes per line. Voices
use the same .pt format in both UIs and `tts.py voices`, so a
voice made in one place works everywhere; save into voices/ to share it.
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
    parser.add_argument(
        "mode",
        nargs="?",
        choices=sorted(KINDS),
        help="Bundled-demo capability to load. Omit for the simple all-in-one app (scripts/app.py).",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--ip", default="127.0.0.1", help="Bind address (default: 127.0.0.1, local only)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="(all-in-one app only) lines generated per model call; "
                             "1 = lowest RAM, 0 = one batch. See scripts/app.py.")
    add_model_args(parser)
    # `mode` is positional, so --model-size can't be restricted per-kind at parse time the way
    # tts.py does it. Default to None and resolve below, so we only complain when the size was
    # actually asked for rather than on every design run.
    parser.set_defaults(model_size=None)
    args = parser.parse_args()

    if args.mode is None:
        # No capability picked -> the simple all-in-one app. Forward the flags it shares.
        app_argv = ["app.py", "--host", args.ip, "--port", str(args.port), "--device", args.device,
                    "--dtype", args.dtype]
        if args.share:
            app_argv.append("--share")
        if args.offline:
            app_argv.append("--offline")
        if getattr(args, "low_memory", False):
            app_argv.append("--low-memory")
        app_argv += ["--chunk-size", str(args.chunk_size)]
        if getattr(args, "threads", None):
            app_argv += ["--threads", str(args.threads)]
        sys.argv = app_argv
        import app  # local module; scripts/ is on sys.path (same as `from common import ...` above)

        return app.main()

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
