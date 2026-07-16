"""Shared helpers for the Qwen3-TTS scripts in this project.

This machine has an AMD GPU (no CUDA, no Windows ROCm support), so every
script defaults to CPU inference. If you later run this on a CUDA-capable
box, pass --device cuda:0 to any script to use the GPU path instead.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "samples"
OUTPUT_DIR = REPO_ROOT / "output"
VOICES_DIR = REPO_ROOT / "voices"

# Published checkpoints on Hugging Face. VoiceDesign only ships as 1.7B.
MODEL_SIZES = {
    "Base": ["0.6B", "1.7B"],
    "CustomVoice": ["0.6B", "1.7B"],
    "VoiceDesign": ["1.7B"],
}


def add_model_args(
    parser: argparse.ArgumentParser,
    kind: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Model/device flags. `kind` restricts --model-size to the sizes actually published,
    so `design --model-size 0.6B` is rejected by argparse instead of being silently
    overridden to 1.7B."""
    sizes = MODEL_SIZES[kind] if kind else ["0.6B", "1.7B"]
    default = sizes[0]
    size_help = (
        f"Model size (only {sizes[0]} is published for {kind})"
        if len(sizes) == 1
        else f"Model size to load (default: {default}, faster on CPU)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device to run on, e.g. 'cpu' or 'cuda:0' (default: cpu)",
    )
    parser.add_argument(
        "--model-size",
        choices=sizes,
        default=default,
        help=size_help,
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
        help="Weight dtype. auto = float32 for 0.6B on CPU, bfloat16 for 1.7B (float32 would "
             "need ~7.7GB of RAM) and on CUDA.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load from the local HF cache only, skipping network checks. Use when the model is "
             "already downloaded and huggingface.co is slow or unreachable.",
    )
    return parser


def validate_ref_audio(ref: str) -> None:
    """Reference audio may be a local path, an http(s) URL, or a base64 blob.

    Only local paths can be checked up front -- do that, so a typo fails immediately with a
    clear message instead of surfacing from inside the tokenizer minutes into a model load.
    """
    if ref.startswith(("http://", "https://", "data:audio")):
        return
    if len(ref) > 256 and "/" not in ref and "\\" not in ref:
        return  # base64 blob, per qwen_tts's own heuristic
    if not Path(ref).exists():
        raise SystemExit(f"Reference audio not found: {ref}")


def configure_hf(args: argparse.Namespace) -> None:
    """Set up huggingface_hub before it gets imported. Call before importing qwen_tts.

    Two Windows/network problems are worth heading off here:

    1. Symlinks. The hub cache symlinks blobs into snapshot dirs, but Windows only permits
       that under Developer Mode or as admin -- otherwise `os.symlink` raises WinError 1314
       and kills the download mid-way. Copying instead costs some disk and always works.
    2. Offline. Every load revalidates each file against the Hub; on a slow network each
       check burns 5 retries with backoff, turning a cached load into many minutes.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if not getattr(args, "offline", False):
        return

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _stub_mistral_check()
    print("Offline mode: loading from the local HF cache only.", file=sys.stderr)


def _stub_mistral_check() -> None:
    """Stop transformers from phoning home about Mistral regexes while loading our tokenizer.

    qwen_tts calls AutoProcessor.from_pretrained(..., fix_mistral_regex=True), which sends
    transformers into `is_base_mistral()` -> `huggingface_hub.model_info()`. That call is
    unguarded: no cache, no offline branch, no try/except, so it hard-fails offline and
    stalls on a flaky network. It checks for a `base_model:.*mistralai` tag, which a Qwen
    repo can never have, so the answer is always False -- we just return that without asking
    the Hub. Scoped to Qwen3-TTS repos so any other model_info call behaves normally.

    transformers does `from huggingface_hub import model_info` inside the function body, so
    it resolves the attribute at call time and picks this up.
    """
    from types import SimpleNamespace

    import huggingface_hub

    real_model_info = huggingface_hub.model_info

    def model_info(repo_id, *args, **kwargs):
        if str(repo_id).startswith("Qwen/Qwen3-TTS"):
            return SimpleNamespace(tags=None)
        return real_model_info(repo_id, *args, **kwargs)

    huggingface_hub.model_info = model_info


def add_sampling_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Generation knobs. All default to None so the model's own generate_config wins."""
    group = parser.add_argument_group("sampling")
    group.add_argument("--temperature", type=float, help="Higher = more varied delivery (model default ~0.9)")
    group.add_argument("--top-k", type=int, help="Top-k sampling (model default ~50)")
    group.add_argument("--top-p", type=float, help="Nucleus sampling (model default ~1.0)")
    group.add_argument("--repetition-penalty", type=float, help="Penalty on repeated codes (model default ~1.05)")
    group.add_argument("--max-new-tokens", type=int, help="Cap on generated codec tokens (model default ~2048)")
    group.add_argument(
        "--seed",
        type=int,
        help="Seed torch RNG so a given generation is reproducible",
    )
    return parser


def sampling_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect only the sampling flags the user actually set."""
    mapping = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def apply_seed(args: argparse.Namespace) -> None:
    if getattr(args, "seed", None) is not None:
        torch.manual_seed(args.seed)


# Approximate parameter counts, for the memory preflight below.
PARAM_COUNTS = {"0.6B": 0.75e9, "1.7B": 1.92e9}

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_dtype(device: str, model_size: str = "0.6B", requested: str = "auto") -> torch.dtype:
    """Pick a dtype that actually fits.

    The checkpoints are stored in bfloat16. Loading as float32 upcasts them and doubles the
    resident size: the 1.7B models become ~7.7GB of weights, which this 8GB box cannot hold --
    it segfaults partway through loading. So float32 (faster, better-supported CPU kernels) is
    only the default for 0.6B; 1.7B falls back to bfloat16, matching how it's stored.
    """
    if requested != "auto":
        return DTYPES[requested]
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32 if model_size == "0.6B" else torch.bfloat16


def check_memory(model_size: str, dtype: torch.dtype) -> None:
    """Warn before a load that physically can't fit, instead of segfaulting mid-way."""
    import psutil

    per_param = {torch.float32: 4, torch.bfloat16: 2, torch.float16: 2}[dtype]
    need_gb = PARAM_COUNTS[model_size] * per_param / 1e9
    avail_gb = psutil.virtual_memory().available / 1e9
    total_gb = psutil.virtual_memory().total / 1e9

    # Weights alone, before activations and the Python/torch runtime; leave headroom.
    if need_gb > total_gb * 0.9:
        raise SystemExit(
            f"{model_size} in {str(dtype).split('.')[-1]} needs ~{need_gb:.1f}GB of weights but this "
            f"machine has only {total_gb:.1f}GB of RAM total. It will crash partway through loading. "
            f"Try --dtype bfloat16, or use --model-size 0.6B."
        )
    if need_gb > avail_gb:
        print(
            f"WARNING: {model_size} in {str(dtype).split('.')[-1]} needs ~{need_gb:.1f}GB but only "
            f"{avail_gb:.1f}GB is free right now (of {total_gb:.1f}GB). Expect heavy swapping or a "
            f"crash -- close other apps first.",
            file=sys.stderr,
        )


def model_id(kind: str, size: str) -> str:
    """kind: 'CustomVoice' | 'VoiceDesign' | 'Base'"""
    available = MODEL_SIZES[kind]
    if size not in available:
        raise ValueError(
            f"{kind} is only published as {'/'.join(available)} on Hugging Face (no {size} checkpoint)"
        )
    return f"Qwen/Qwen3-TTS-12Hz-{size}-{kind}"


def load_model(kind: str, args: argparse.Namespace):
    """Load a Qwen3TTSModel, printing what's happening since CPU loads are slow."""
    dtype = resolve_dtype(args.device, args.model_size, getattr(args, "dtype", "auto"))
    check_memory(args.model_size, dtype)

    # Must precede the qwen_tts import: huggingface_hub reads HF_HUB_OFFLINE at import time.
    configure_hf(args)
    from qwen_tts import Qwen3TTSModel

    repo = model_id(kind, args.model_size)
    print(
        f"Loading {repo} on {args.device} as {str(dtype).split('.')[-1]} "
        f"(first run downloads weights)...",
        file=sys.stderr,
    )
    return Qwen3TTSModel.from_pretrained(
        repo,
        device_map=args.device,
        dtype=dtype,
    )


def warn_if_instruct_ignored(kind: str, args: argparse.Namespace, instruct: Optional[str]) -> None:
    """The 0.6B CustomVoice model silently drops `instruct`.

    qwen_tts's generate_custom_voice does `if self.model.tts_model_size in "0b6": instruct = None`,
    so style instructions on the 0.6B checkpoint are discarded with no error. Say so out loud
    rather than letting the user think the instruction had an effect.
    """
    if kind == "CustomVoice" and args.model_size == "0.6B" and instruct:
        print(
            "WARNING: --instruct is ignored by the 0.6B CustomVoice model (the library drops it "
            "silently). Re-run with --model-size 1.7B for style control.",
            file=sys.stderr,
        )


def read_lines(text: Optional[str], text_file: Optional[str]) -> List[str]:
    """One generation per line. Blank lines separate nothing -- they're just skipped."""
    if text is not None:
        if not text.strip():
            raise SystemExit("--text is empty; give it something to say.")
        return [text]
    path = Path(text_file)
    if not path.exists():
        raise SystemExit(f"Script file not found: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise SystemExit(f"No usable lines in {path} (blank lines and #comments are skipped)")
    return lines


def peak_limit(wavs, peak: float):
    """Scale the whole batch by one gain so nothing clips.

    The model occasionally generates samples at or past full scale -- long reads have been
    observed flat-topping for a few samples. Attenuate (never boost), and use a single gain
    across every line rather than per-line: normalising each line separately would give each
    sentence a different level and make a joined paragraph pump.
    """
    import numpy as np

    if not peak:
        return wavs
    loudest = max(float(np.abs(np.asarray(w)).max()) for w in wavs)
    if loudest <= peak or loudest == 0:
        return wavs
    gain = peak / loudest
    return [np.asarray(w, dtype=np.float32) * gain for w in wavs]


def write_wavs(
    wavs,
    sample_rate: int,
    out: str,
    join: bool,
    gap_ms: int = 300,
    peak: float = 0.97,
) -> List[Path]:
    """Write generated audio.

    Single wav -> `out` as given. Multiple wavs -> `out` gets a _001, _002 ... suffix,
    unless --join, in which case they're concatenated into `out` with a short gap.
    """
    import numpy as np
    import soundfile as sf

    wavs = peak_limit(wavs, peak)

    out_path = Path(out)
    if not out_path.suffix:
        # soundfile infers the container from the extension; without one it raises deep in
        # libsndfile, and the numbered path would produce files like `read_001` with no type.
        out_path = out_path.with_suffix(".wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(wavs) == 1:
        sf.write(out_path, wavs[0], sample_rate)
        return [out_path]

    if join:
        gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
        pieces = []
        for i, wav in enumerate(wavs):
            if i:
                pieces.append(gap)
            pieces.append(np.asarray(wav, dtype=np.float32))
        sf.write(out_path, np.concatenate(pieces), sample_rate)
        return [out_path]

    written = []
    for i, wav in enumerate(wavs, start=1):
        numbered = out_path.with_name(f"{out_path.stem}_{i:03d}{out_path.suffix}")
        sf.write(numbered, wav, sample_rate)
        written.append(numbered)
    return written
