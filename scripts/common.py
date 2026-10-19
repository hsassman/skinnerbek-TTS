"""Shared helpers for the Qwen3-TTS scripts in this project.

This machine has an AMD GPU (no CUDA, no Windows ROCm support), so every
script defaults to CPU inference. If you later run this on a CUDA-capable
box, pass --device cuda:0 to any script to use the GPU path instead.
"""
import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from textprep import prepare_script

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
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch intra-op threads (default: torch's own choice, the physical core count)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Load 0.6B as bfloat16 too (~1.5GB instead of ~3GB). On a CPU without hardware "
             "bf16 this is ~25x SLOWER, not faster -- only use it when float32 will not fit. "
             "Auto-enabled when free RAM is too low for float32.",
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


# Named delivery presets. "natural" = the model's own generate_config (change
# nothing). "steady" pulls temperature down and adds a little repetition penalty:
# on the 0.6B model that noticeably reduces the rushed, occasionally stammering
# delivery. "expressive" opens it back up. Explicit --temperature etc. always win.
DELIVERY_PRESETS: Dict[str, Dict[str, float]] = {
    "steady": {"temperature": 0.7, "repetition_penalty": 1.1},
    "natural": {},
    "expressive": {"temperature": 1.0, "repetition_penalty": 1.0},
}


def add_sampling_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Generation knobs. All default to None so the model's own generate_config wins."""
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--delivery",
        choices=sorted(DELIVERY_PRESETS),
        default="natural",
        help="Delivery preset: steady (calmer, more consistent -- best for the 0.6B model), "
             "natural (model default), expressive (more varied). Overridden by any explicit knob below.",
    )
    group.add_argument("--temperature", type=float, help="Higher = more varied delivery (model default ~0.9)")
    group.add_argument("--top-k", type=int, help="Top-k sampling (model default ~50)")
    group.add_argument("--top-p", type=float, help="Nucleus sampling (model default ~1.0)")
    group.add_argument("--repetition-penalty", type=float, help="Penalty on repeated codes (model default ~1.05)")
    group.add_argument("--max-new-tokens", type=int,
                   help="Cap on generated codec frames for every line. Default: sized from "
                        "the text by common.token_budget (the library's own default is 4096, "
                        "~5.7 min of audio per line, which is a runaway risk)")
    group.add_argument(
        "--seed",
        type=int,
        help="Seed torch RNG so a given generation is reproducible",
    )
    return parser


def delivery_sampling(name: Optional[str]) -> Dict[str, Any]:
    """Sampling kwargs for a named delivery preset (see DELIVERY_PRESETS). Case-insensitive."""
    return dict(DELIVERY_PRESETS.get((name or "natural").strip().lower(), {}))


def sampling_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Delivery preset first, then any sampling flag the user set explicitly on top."""
    out = delivery_sampling(getattr(args, "delivery", "natural"))
    explicit = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }
    out.update({k: v for k, v in explicit.items() if v is not None})
    return out


def apply_seed(args: argparse.Namespace) -> None:
    if getattr(args, "seed", None) is not None:
        torch.manual_seed(args.seed)


# Approximate parameter counts, for the memory preflight below.
PARAM_COUNTS = {"0.6B": 0.75e9, "1.7B": 1.92e9}

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

# Below this much free RAM, 0.6B float32 (~3GB of weights) thrashes swap on this box
# -- a generation still finishes but takes 15-20 min. Drop to bfloat16 automatically.
LOW_MEMORY_GB = 4.0

# ...but only where bfloat16 is actually cheap. See cpu_bf16_is_native().
#
# Where bf16 is emulated the trade is so lopsided that float32 wins even when it
# does NOT fit. Measured on the i5-7200U with 1.7GB free -- float32 paging hard
# against a 3GB weight set -- one 33-character line took 72s in float32 versus
# 510s in bfloat16 (7.1x), because emulated bf16 costs more than the page faults
# do. So only fall back when float32 would leave essentially nothing to run in.
LOW_MEMORY_GB_EMULATED_BF16 = 1.5


def cpu_bf16_is_native() -> bool:
    """True when this CPU does bfloat16 in hardware (AVX512-BF16 or AMX).

    This matters enormously and is not obvious. torch has no bf16 kernels for a
    pre-AVX512 x86 CPU, so it emulates: measured on the i5-7200U (Kaby Lake,
    AVX2) in this project, bfloat16 is **25x slower than float32 for a decode
    matvec and 265x slower for a prefill matmul**. Halving the weights to save
    RAM is a straight win on a Xeon with AMX and a catastrophe here, so the dtype
    policy has to ask the CPU first rather than assume smaller is faster.
    """
    if not hasattr(torch, "cpu"):
        return False
    for probe in ("_is_avx512_bf16_supported", "_is_amx_tile_supported"):
        try:
            if getattr(torch.cpu, probe, lambda: False)():
                return True
        except Exception:
            continue
    return False


def low_memory_threshold_gb() -> float:
    """Free-RAM level below which 0.6B drops to bfloat16."""
    return LOW_MEMORY_GB if cpu_bf16_is_native() else LOW_MEMORY_GB_EMULATED_BF16


def free_gb() -> Optional[float]:
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        return None


def auto_low_memory() -> bool:
    """True when free RAM is tight enough that 0.6B should load as bfloat16.

    On a CPU without native bf16 this is a last resort that trades ~25x speed for
    half the RAM, not an optimisation -- load_model says so out loud when it fires.
    """
    avail = free_gb()
    return False if avail is None else avail < low_memory_threshold_gb()


def resolve_dtype(
    device: str,
    model_size: str = "0.6B",
    requested: str = "auto",
    low_memory: bool = False,
) -> torch.dtype:
    """Pick a dtype that actually fits.

    The checkpoints are stored in bfloat16. Loading as float32 upcasts them and doubles the
    resident size: the 1.7B models become ~7.7GB of weights, which this 8GB box cannot hold --
    it segfaults partway through loading. So float32 (faster, better-supported CPU kernels) is
    only the default for 0.6B; 1.7B falls back to bfloat16, matching how it's stored.
    `low_memory` also drops 0.6B to bfloat16 -- half the RAM. Note that "a little slower per
    step" only holds where the CPU has hardware bf16; where it doesn't, torch emulates and the
    same switch costs ~25x. See cpu_bf16_is_native() and auto_low_memory().
    """
    if requested != "auto":
        return DTYPES[requested]
    if device.startswith("cuda"):
        return torch.bfloat16
    if model_size == "0.6B":
        return torch.bfloat16 if low_memory else torch.float32
    return torch.bfloat16


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
    requested = getattr(args, "dtype", "auto")
    low_mem = bool(getattr(args, "low_memory", False)) or auto_low_memory()
    if low_mem and requested == "auto" and args.model_size == "0.6B" and not args.device.startswith("cuda"):
        why = ("requested" if getattr(args, "low_memory", False)
               else f"under {low_memory_threshold_gb():.1f}GB free")
        print(f"Low-memory mode ({why}): loading 0.6B as bfloat16 (~1.5GB instead of ~3GB).",
              file=sys.stderr)
        if not cpu_bf16_is_native():
            avail = free_gb()
            short = max(0.0, 3.4 - avail) if avail is not None else 0.0
            print(
                "WARNING: this CPU has no hardware bfloat16, so torch emulates it -- roughly "
                "25x slower per decode step and far worse on the first pass of each line. "
                "float32 is the fast path here despite being twice the size."
                + (f" Free about {short:.1f}GB more RAM (close the browser) and it will be used "
                   f"automatically." if short else ""),
                file=sys.stderr,
            )
    dtype = resolve_dtype(args.device, args.model_size, requested, low_memory=low_mem)
    check_memory(args.model_size, dtype)

    # Must precede the qwen_tts import: huggingface_hub reads HF_HUB_OFFLINE at import time.
    configure_hf(args)
    from qwen_tts import Qwen3TTSModel

    repo = model_id(kind, args.model_size)
    threads = configure_threads(getattr(args, "threads", None))
    print(
        f"Loading {repo} on {args.device} as {str(dtype).split('.')[-1]} "
        f"({threads} torch thread(s); first run downloads weights)...",
        file=sys.stderr,
    )
    # transformers >= ~4.50 always streams weights straight into place (the old
    # low_cpu_mem_usage knob is now a no-op), so there's nothing to tune at load
    # time here. The RAM lever on this box is generating one line at a time --
    # see common.synth / --chunk-size.
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


def read_script(
    text: Optional[str],
    text_file: Optional[str],
    *,
    normalize: bool = True,
    lexicon: bool = True,
    max_words: int = 24,
    sentence_gap_ms: int = 300,
    paragraph_gap_ms: int = 650,
) -> "tuple[List[str], List[int]]":
    """Read + prepare a script: (lines_to_speak, gap_before_each_line_ms).

    Wraps textprep.prepare_script and turns its ValueError into a CLI-friendly
    SystemExit. `normalize=False` falls back to plain line splitting.
    """
    if text is not None:
        src = text
    else:
        path = Path(text_file)
        if not path.exists():
            raise SystemExit(f"Script file not found: {path}")
        src = path.read_text(encoding="utf-8")
    try:
        return prepare_script(
            src,
            normalize=normalize,
            lexicon=lexicon,
            max_words=max_words,
            sentence_gap_ms=sentence_gap_ms,
            paragraph_gap_ms=paragraph_gap_ms,
        )
    except ValueError as e:
        raise SystemExit(str(e))


# --- The one place each capability is actually invoked -------------------------
# The CLI subcommands and the GUI callbacks both call these, so "the GUI does
# exactly what the CLI does" is true by construction rather than by review.

def run_preset(model, lines: List[str], *, language: str, speaker: str,
               instruct: Optional[str], sampling: Dict[str, Any]):
    return model.generate_custom_voice(
        text=lines,
        language=language,
        speaker=speaker,
        instruct=(instruct or None),
        **sampling,
    )


def run_design(model, lines: List[str], *, language: str, instruct: str,
               sampling: Dict[str, Any]):
    return model.generate_voice_design(
        text=lines,
        language=language,
        instruct=instruct,
        **sampling,
    )


def run_clone(model, lines: List[str], *, language: str, voice_clone_prompt=None,
              ref_audio: Optional[str] = None, ref_text: Optional[str] = None,
              x_vector_only: bool = False, sampling: Dict[str, Any]):
    if voice_clone_prompt is not None:
        return model.generate_voice_clone(
            text=lines,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            **sampling,
        )
    return model.generate_voice_clone(
        text=lines,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only,
        **sampling,
    )


# Bound generation by text length: the library's 4096-frame default permits
# about 5.7 minutes of audio per line. These conservative limits leave headroom
# for slow speech; --max-new-tokens overrides them.
CODEC_HZ = 12.0
CHARS_PER_SECOND = 8.0
TOKEN_HEADROOM = 3.0
MIN_NEW_TOKENS = 128
MAX_NEW_TOKENS_CAP = 4096


def token_budget(text: str) -> int:
    """A generous ceiling on codec frames for `text`, as a runaway guard."""
    seconds = max(1.0, len(text) / CHARS_PER_SECOND)
    return max(MIN_NEW_TOKENS, min(MAX_NEW_TOKENS_CAP, int(seconds * CODEC_HZ * TOKEN_HEADROOM)))


def configure_threads(threads: Optional[int] = None) -> int:
    """Set torch's intra-op thread count, returning what's in effect.

    torch defaults to the *physical* core count, which is usually right; this
    exists so a machine that reports the wrong topology (or a run sharing the box
    with the Gradio server) can be pinned explicitly with --threads.
    """
    if threads and threads > 0:
        torch.set_num_threads(int(threads))
    return torch.get_num_threads()


# --- machine facts and a speed estimate ---------------------------------------

def system_report() -> Dict[str, Any]:
    """Everything that decides how fast a generation will be, in one dict.

    Exists because the single worst performance trap on this project was
    invisible: a CPU without hardware bfloat16 silently taking a ~25x slower
    path. Surfacing `bf16_native` and the resolved dtype turns that into
    something you can see before you wait an hour for it.
    """
    import platform

    cores = threads = None
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
        threads = psutil.cpu_count(logical=True)
    except Exception:
        pass

    avail = free_gb()
    low = auto_low_memory()
    return {
        "cpu": (platform.processor() or platform.machine() or "unknown"),
        "cores": cores,
        "threads": threads,
        "torch_threads": torch.get_num_threads(),
        "simd": _cpu_capability(),
        "bf16_native": cpu_bf16_is_native(),
        "free_gb": avail,
        "total_gb": _total_gb(),
        "low_memory": low,
        "dtype": str(resolve_dtype("cpu", "0.6B", "auto", low_memory=low)).split(".")[-1],
        "sec_per_frame": seconds_per_frame(),
    }


def _cpu_capability() -> str:
    try:
        return str(torch.backends.cpu.get_cpu_capability())
    except Exception:
        return "unknown"


def _total_gb() -> Optional[float]:
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except Exception:
        return None


# Measured generation speed, so the UI can estimate before you commit to a run.
# Keyed by dtype because that is the factor that moves it by an order of
# magnitude on this hardware. Kept in .cache (derived data, gitignored).
#
# The estimate is the MEDIAN of the last SPEED_WINDOW samples, not a running
# average. An average is permanently skewed by a handful of outliers -- a line
# generated against a stubbed model, a run that thrashed swap, a machine change
# -- and there is no way for it to recover. A median of a bounded window shrugs
# those off and re-converges once real samples outnumber them.
SPEED_PATH = REPO_ROOT / ".cache" / "speed.json"
SPEED_WINDOW = 15


def _speed_table() -> Dict[str, Any]:
    import json

    try:
        data = json.loads(SPEED_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _samples(table: Dict[str, Any], key: str) -> List[float]:
    """Sample list for `key`, tolerating the older {sec_per_frame, samples:int} shape."""
    entry = table.get(key)
    if not isinstance(entry, dict):
        return []
    raw = entry.get("samples")
    if isinstance(raw, list):
        return [float(v) for v in raw if isinstance(v, (int, float)) and v > 0]
    return []


def _dtype_key() -> str:
    try:
        return str(resolve_dtype("cpu", "0.6B", "auto",
                                 low_memory=auto_low_memory())).split(".")[-1]
    except Exception:
        return "float32"


def record_speed(seconds: float, frames: float, dtype_name: Optional[str] = None) -> None:
    """Fold one finished line into the rolling seconds-per-frame samples.

    Silently does nothing on any error -- a stats file is never worth failing a
    generation the user just waited minutes for.
    """
    import json

    try:
        if not seconds or not frames or seconds <= 0 or frames <= 0:
            return
    except TypeError:
        return
    key = dtype_name or _dtype_key()
    try:
        table = _speed_table()
        samples = _samples(table, key)
        samples.append(float(seconds) / float(frames))
        samples = samples[-SPEED_WINDOW:]
        table[key] = {"sec_per_frame": round(_median(samples), 4),
                      "samples": [round(v, 5) for v in samples]}
        SPEED_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPEED_PATH.write_text(json.dumps(table, indent=2), encoding="utf-8")
    except Exception:
        pass


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def seconds_per_frame(dtype_name: Optional[str] = None) -> Optional[float]:
    """Median seconds of compute per codec frame, or None if never measured."""
    key = dtype_name or _dtype_key()
    samples = _samples(_speed_table(), key)
    return _median(samples) if samples else None


def reset_speed() -> bool:
    """Forget the speed calibration. True if a file was actually removed."""
    try:
        SPEED_PATH.unlink()
        return True
    except OSError:
        return False


def speed_sample_count(dtype_name: Optional[str] = None) -> int:
    return len(_samples(_speed_table(), dtype_name or _dtype_key()))


def estimate_seconds(text_lines: List[str]) -> Optional[float]:
    """Rough wall-clock estimate for generating `text_lines`, or None if unknown.

    Uses the same characters-per-second model as token_budget, but without its
    safety headroom -- that headroom exists to catch runaways, and applying it
    here would triple every estimate.
    """
    spf = seconds_per_frame()
    if not spf:
        return None
    frames = sum(max(1.0, len(t) / CHARS_PER_SECOND) * CODEC_HZ for t in text_lines)
    return frames * spf


class Cancelled(Exception):
    """Raised by synth() when its should_cancel callback asks it to stop."""


def synth(
    capability: str,
    model,
    lines: List[str],
    *,
    language: str,
    sampling: Dict[str, Any],
    chunk_size: int = 1,
    on_line=None,
    on_progress=None,
    should_cancel=None,
    speaker: Optional[str] = None,
    instruct: Optional[str] = None,
    voice_clone_prompt=None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    x_vector_only: bool = False,
):
    """Generate lines in chunks; return (waveforms, sample_rate).

    qwen_tts batches every text in one call, which on an 8 GB box builds a KV
    cache and holds every waveform at once -- multiple GB that pages to disk. So
    chunk_size=1 (the default) generates one line per call with gc.collect()
    between chunks, keeping peak RAM at a single line. chunk_size=0, or any value
    >= len(lines), batches everything (faster on a big machine). Changing the
    chunk size changes seeded output. Each call gets a text-based token budget
    unless sampling explicitly sets max_new_tokens.

    on_line(index, waveform, sample_rate) runs for each completed line, then
    on_progress(done, total) runs per chunk. should_cancel is polled between
    chunks and raises Cancelled; on_line can preserve completed audio on disk.
    """
    total = len(lines)
    # max(1, ...) only matters for an empty `lines`: step would be 0 and
    # range() rejects a zero stride. Non-empty inputs are unaffected.
    step = max(1, total if not chunk_size or chunk_size < 1 else int(chunk_size))
    wavs: List = []
    sr = None
    if capability not in ("preset", "design", "clone"):
        raise ValueError(f"unknown capability {capability!r}")

    for start in range(0, total, step):
        # Checked between chunks only, never mid-generation: stopping a line
        # half-way would leave a truncated waveform. At the default chunk_size
        # of 1 that is a cancellation point before every line.
        if should_cancel is not None and should_cancel():
            raise Cancelled(f"stopped after {len(wavs)} of {total} line(s)")
        batch = lines[start:start + step]
        # Bound each call by the longest line in it (see token_budget), unless the
        # caller set an explicit ceiling.
        call = dict(sampling)
        call.setdefault("max_new_tokens", token_budget(max(batch, key=len)))

        if capability == "preset":
            part, sr = run_preset(model, batch, language=language, speaker=speaker,
                                  instruct=instruct, sampling=call)
        elif capability == "design":
            part, sr = run_design(model, batch, language=language, instruct=instruct,
                                  sampling=call)
        else:
            part, sr = run_clone(model, batch, language=language,
                                 voice_clone_prompt=voice_clone_prompt, ref_audio=ref_audio,
                                 ref_text=ref_text, x_vector_only=x_vector_only,
                                 sampling=call)

        for j, one in enumerate(part):
            if on_line is not None:
                on_line(start + j, one, sr)
        wavs.extend(part)
        if on_progress is not None:
            on_progress(min(start + step, total), total)
        gc.collect()
    return wavs, sr


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


DEFAULT_LUFS = -16.0


def loudness_normalize(wavs, sample_rate: int, target_lufs: float = DEFAULT_LUFS,
                       peak: float = 0.97):
    """Bring the whole batch to `target_lufs` with one gain, then peak-limit as a safety net.

    One gain across all lines (measured on the concatenation) so relative levels between
    sentences are preserved -- same reasoning as peak_limit. Falls back to peak_limit alone
    when `target_lufs` is 0/None, when pyloudnorm isn't installed, or when the signal is too
    short/quiet to measure (ITU-R BS.1770 needs ~400ms).
    """
    import numpy as np

    if not target_lufs:
        return peak_limit(wavs, peak)
    try:
        import pyloudnorm as pyln
    except ImportError:
        print("pyloudnorm not installed (`pip install pyloudnorm`); using peak limiting only.",
              file=sys.stderr)
        return peak_limit(wavs, peak)

    arrs = [np.asarray(w, dtype=np.float32) for w in wavs]
    concat = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    try:
        current = pyln.Meter(sample_rate).integrated_loudness(concat)
    except Exception:
        return peak_limit(wavs, peak)
    if not np.isfinite(current):  # silence / below the gate
        return peak_limit(wavs, peak)

    gain = float(10.0 ** ((target_lufs - current) / 20.0))
    return peak_limit([a * gain for a in arrs], peak)


def write_wavs(
    wavs,
    sample_rate: int,
    out: str,
    join: bool,
    gap_ms: int = 300,
    peak: float = 0.97,
    gaps: Optional[List[int]] = None,
    lufs: Optional[float] = DEFAULT_LUFS,
) -> List[Path]:
    """Write generated audio.

    Single wav -> `out` as given. Multiple wavs -> `out` gets a _001, _002 ... suffix,
    unless --join, in which case they're concatenated into `out` with a short gap.

    `gaps`, when given, is a per-line silence in ms placed *before* each line
    (gaps[0] is ignored); it lets a joined read use a longer pause between
    paragraphs than between sentences. Falls back to a uniform `gap_ms`.

    `lufs` is the integrated-loudness target every take is brought to (one gain
    across all lines) before the peak-limit safety net; pass 0/None for the old
    peak-only behaviour.
    """
    import numpy as np
    import soundfile as sf

    wavs = loudness_normalize(wavs, sample_rate, target_lufs=lufs, peak=peak)

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
        use_gaps = gaps if (gaps and len(gaps) == len(wavs)) else [gap_ms] * len(wavs)
        pieces = []
        for i, wav in enumerate(wavs):
            if i:
                n = max(0, int(sample_rate * use_gaps[i] / 1000))
                pieces.append(np.zeros(n, dtype=np.float32))
            pieces.append(np.asarray(wav, dtype=np.float32))
        sf.write(out_path, np.concatenate(pieces), sample_rate)
        return [out_path]

    written = []
    for i, wav in enumerate(wavs, start=1):
        numbered = out_path.with_name(f"{out_path.stem}_{i:03d}{out_path.suffix}")
        sf.write(numbered, wav, sample_rate)
        written.append(numbered)
    return written


def write_raw_parts(wavs, sample_rate: int, out_dir: str, prefix: str) -> List[Path]:
    """Write each line as its own wav, with NO loudness/peak processing.

    These are the raw per-line takes kept so a single line can be regenerated and
    the read re-stitched without re-running the rest. Levelling happens later, in
    `stitch`, across the whole set.
    """
    import numpy as np
    import soundfile as sf

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, wav in enumerate(wavs, start=1):
        p = d / f"{prefix}_{i:03d}.wav"
        sf.write(p, np.asarray(wav, dtype=np.float32), sample_rate)
        paths.append(p)
    return paths


def stitch(
    part_paths: List[str],
    out: str,
    gaps: Optional[List[int]] = None,
    gap_ms: int = 300,
    peak: float = 0.97,
    lufs: Optional[float] = DEFAULT_LUFS,
) -> Path:
    """Join raw per-line wavs (see write_raw_parts) into one levelled wav."""
    import soundfile as sf

    arrs = []
    sample_rate = None
    for p in part_paths:
        a, rate = sf.read(p, dtype="float32")
        if sample_rate is not None and rate != sample_rate:
            raise ValueError("Cannot stitch audio parts with different sample rates")
        # Parts are mono model output, and the join pads with 1-D silence, so a
        # multichannel part could never be joined: refuse it up front.
        if a.ndim != 1:
            raise ValueError("Cannot stitch multichannel audio parts; parts must be mono")
        sample_rate = rate
        arrs.append(a)
    if sample_rate is None:
        raise ValueError("stitch() got no parts")
    return write_wavs(arrs, sample_rate, out, join=True, gap_ms=gap_ms, peak=peak,
                      gaps=gaps, lufs=lufs)[0]
