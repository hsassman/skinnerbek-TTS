# Qwen3-TTS — local setup

Local setup of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for experimenting with
South African English / Cape Town / Cape Flats accented speech.

## Hardware reality check

This machine has an **AMD Radeon RX 5700 XT**, not an NVIDIA GPU. Qwen3-TTS's PyTorch
backend needs CUDA for GPU acceleration; AMD ROCm has no Windows build (and even on Linux,
RX 5700 XT / gfx1010 support is unofficial). **All scripts here default to `--device cpu`.**
Expect generation to take tens of seconds to a few minutes per sentence, especially with the
1.7B models. If you ever run this on a CUDA machine, pass `--device cuda:0`.

## Accent reality check

Qwen3-TTS does **not** ship a South African English / Cape Flats preset. Its English presets
are just `Ryan` and `Aiden`, and its only built-in dialect variation is for Chinese (Beijing,
Sichuan). Three ways to get at the accent, in order of how likely they are to actually sound
right:

1. **Voice cloning (`generate_voice_clone.py`, Base model)** — give it a real reference clip
   of someone speaking with a Cape Flats / Capetonian accent plus its transcript. This is the
   only approach likely to sound authentic, because the model isn't inventing the accent, it's
   copying it from the reference audio. Put reference clips in `samples/`.
2. **Voice design (`generate_voice_design.py`, VoiceDesign model)** — describe the accent in
   plain English in `--instruct` (e.g. "South African English, Cape Flats/Cape Coloured accent,
   Cape Town"). Experimental — the model wasn't demonstrated on South African English in its
   own docs, so it may not render convincingly.
3. **Preset voices (`generate_custom_voice.py`, CustomVoice model)** — useful for testing the
   pipeline, but no accent control at all.

## Setup

Already done in this checkout: a `.venv` with `qwen-tts` installed. To recreate from scratch:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -U qwen-tts
```

Model weights are **not** downloaded up front — each script downloads the relevant model
from Hugging Face the first time it runs (cached under `%USERPROFILE%\.cache\huggingface`
after that).

## Usage

Activate the venv first: `.\.venv\Scripts\Activate.ps1`

```powershell
# Preset voice, just to sanity-check the pipeline works
python scripts\generate_custom_voice.py --text "Hello, this is a test." --out output\test.wav

# Voice design attempt at a Cape Flats accent (experimental)
python scripts\generate_voice_design.py `
  --text "Ja no, it's lekker hot today, hey." `
  --instruct "South African English, Cape Flats/Cape Coloured accent, Cape Town, warm conversational male voice" `
  --out output\voice_design.wav

# Voice cloning from a real reference clip (best shot at authenticity)
python scripts\generate_voice_clone.py `
  --ref-audio samples\cape_flats_ref.wav `
  --ref-text "exact transcript of what's said in the reference clip" `
  --text "Howzit my bru, you good?" `
  --out output\clone.wav
```

All three scripts accept `--model-size 0.6B|1.7B` (0.6B is the default — faster on CPU) and
`--device cpu|cuda:0`.

## Layout

- `scripts/` — inference scripts (`common.py` has shared device/model-id helpers)
- `samples/` — put your reference audio clips here for voice cloning
- `output/` — generated `.wav` files land here
