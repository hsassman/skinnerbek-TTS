"""Shared helpers for the Qwen3-TTS scripts in this project.

This machine has an AMD GPU (no CUDA, no Windows ROCm support), so every
script defaults to CPU inference. If you later run this on a CUDA-capable
box, pass --device cuda:0 to any script to use the GPU path instead.
"""
import argparse

import torch


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device to run on, e.g. 'cpu' or 'cuda:0' (default: cpu)",
    )
    parser.add_argument(
        "--model-size",
        choices=["0.6B", "1.7B"],
        default="0.6B",
        help="Model size to load (default: 0.6B, faster on CPU)",
    )
    return parser


def resolve_dtype(device: str) -> torch.dtype:
    # bfloat16 has no real speed benefit on CPU and some CPU kernels don't
    # support it well; use float32 off-GPU, bfloat16 on CUDA.
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def model_id(kind: str, size: str) -> str:
    """kind: 'CustomVoice' | 'VoiceDesign' | 'Base'"""
    if kind == "VoiceDesign" and size != "1.7B":
        raise ValueError("VoiceDesign is only published as 1.7B on Hugging Face (no 0.6B checkpoint)")
    return f"Qwen/Qwen3-TTS-12Hz-{size}-{kind}"
