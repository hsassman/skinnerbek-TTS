"""Launch a web UI.

    python scripts/launch_ui.py            # Skinnerbek Studio (scripts/app.py)
    python scripts/launch_ui.py preset     # bundled qwen_tts demo: preset voices
    python scripts/launch_ui.py design     # bundled qwen_tts demo: voice design (1.7B)
    python scripts/launch_ui.py clone      # bundled qwen_tts demo: cloning

The bundled demo defaults to CUDA and FlashAttention-2; the modes switch it to
CPU and a dtype that fits in RAM. It has no login, so it only runs locally.
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
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (app only, needs --auth)")
    parser.add_argument("--auth", metavar="USER:PASSWORD", help="(all-in-one app only) require this login")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="(all-in-one app only) lines generated per model call; "
                             "1 = lowest RAM, 0 = one batch. See scripts/app.py.")
    add_model_args(parser)
    # Resolved per mode below (`mode` is positional, so argparse can't restrict it).
    parser.set_defaults(model_size=None)
    args = parser.parse_args()

    if args.mode is None:
        # No capability picked -> the simple all-in-one app. Forward the flags it shares.
        app_argv = ["app.py", "--host", args.ip, "--port", str(args.port), "--device", args.device,
                    "--dtype", args.dtype]
        if args.share:
            app_argv.append("--share")
        if args.auth:
            app_argv += ["--auth", args.auth]
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

    if args.share or args.ip not in ("127.0.0.1", "localhost", "::1"):
        parser.error("the bundled demo has no login, so it only runs on this machine "
                     "(no --share, --ip 127.0.0.1)")

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
    # Shared dtype policy (1.7B as float32 doesn't fit in 8 GB).
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
