"""One-click start: fetch the models once, then open Skinnerbek Studio in the browser.

Run by start.bat (Windows) and start.command (macOS) inside the app's Python
environment. Safe to run again: anything already downloaded is skipped.
"""
import os
import socket
import sys
import urllib.request
import webbrowser

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

QWEN_MODELS = ["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"]
WHISPER_MODELS = ["large-v3-turbo", "small"]
PORTS = range(7860, 7880)


def say(msg: str = "") -> None:
    print(msg, flush=True)


def online(host: str = "huggingface.co") -> bool:
    try:
        socket.create_connection((host, 443), timeout=5).close()
        return True
    except OSError:
        return False


def qwen_cached(repo: str) -> bool:
    from pathlib import Path

    from huggingface_hub import constants, try_to_load_from_cache

    needed = ["config.json", "model.safetensors", "speech_tokenizer/model.safetensors"]
    if not all(isinstance(try_to_load_from_cache(repo, f), str) for f in needed):
        return False
    blobs = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "blobs"
    return not any(blobs.glob("*.incomplete"))


def fetch_models() -> bool:
    """Download whatever is missing. True when every model is on disk."""
    import transcribe
    from huggingface_hub import snapshot_download

    missing_qwen = [r for r in QWEN_MODELS if not qwen_cached(r)]
    missing_asr = [m for m in WHISPER_MODELS if not (transcribe.MODEL_DIR / m / "model.bin").exists()]
    if not missing_qwen and not missing_asr:
        return True
    if not online():
        say("No internet connection, and some voice models are not downloaded yet.")
        say("Connect to the internet and run start again. Starting with what is available.")
        return False

    say("First run: downloading the voice models (about 7 GB, one time only).")
    say("This can take a while on a slow connection. Progress is shown below.")
    ok = True
    for repo in missing_qwen:
        say(f"\n- {repo}")
        try:
            snapshot_download(repo)
        except Exception as e:
            say(f"  Download failed: {e}")
            ok = False
    for name in missing_asr:
        say(f"\n- speech recognition ({name})")
        try:
            transcribe.model_path(name)
        except Exception as e:
            say(f"  Download failed: {e}")
            ok = False
    return ok


def running_studio(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/config", timeout=2) as r:
            return b"Skinnerbek" in r.read(200_000)
    except Exception:
        return False


def free_port() -> int:
    for port in PORTS:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"Ports {PORTS.start}-{PORTS.stop - 1} are all in use. Close some programs and try again.")


def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for port in PORTS:
        if running_studio(port):
            say(f"Skinnerbek Studio is already running. Opening http://127.0.0.1:{port}")
            webbrowser.open(f"http://127.0.0.1:{port}")
            return 0

    all_models = fetch_models()
    port = free_port()
    say()
    say("Starting Skinnerbek Studio. Your browser will open by itself.")
    say(f"If it does not, go to http://127.0.0.1:{port}")
    say("Keep this window open while you use the app. Close it to stop the app.")
    say()

    sys.argv = ["app.py", "--port", str(port)] + (["--offline"] if all_models else [])
    import app

    return app.main()


if __name__ == "__main__":
    raise SystemExit(main())
