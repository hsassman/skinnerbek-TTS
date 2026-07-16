# Qwen3-TTS — local setup

Local setup of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for experimenting with
South African English / Cape Town / Cape Flats accented speech.

## The three capabilities

Qwen3-TTS ships **three separate checkpoints**, one per capability. You can't get all three
from one model — each subcommand below loads its own:

| Capability | Subcommand | Model | Sizes |
|---|---|---|---|
| Preset voices | `preset` | CustomVoice | 0.6B, 1.7B |
| Voice design (describe a voice in words) | `design` | VoiceDesign | **1.7B only** |
| Voice cloning (copy a real recording) | `clone` | Base | 0.6B, 1.7B |

## Hardware reality check

This machine has an **AMD Radeon RX 5700 XT**, not an NVIDIA GPU. Qwen3-TTS's PyTorch
backend needs CUDA for GPU acceleration; AMD ROCm has no Windows build (and even on Linux,
RX 5700 XT / gfx1010 support is unofficial). **Everything defaults to `--device cpu`.**
Expect tens of seconds to a few minutes per line, especially on the 1.7B models. If you ever
run this on a CUDA machine, pass `--device cuda:0`.

**RAM is the binding constraint: this box has 7.9 GB.** The checkpoints ship in bfloat16, and
loading them as float32 doubles their resident size:

| | float32 | bfloat16 |
|---|---|---|
| 0.6B (0.75B params) | ~3.0 GB | ~1.5 GB |
| 1.7B (1.92B params) | **~7.7 GB — will not fit** | ~3.8 GB |

So `--dtype auto` (the default) uses float32 for 0.6B (better-supported CPU kernels) and
bfloat16 for 1.7B, which is the only way it fits. A preflight check refuses a load that can't
fit rather than letting it segfault halfway through. **Close other apps before running a 1.7B
model** — it needs ~3.8 GB free, and a browser can easily deny it that.

## Gotchas worth knowing

- **`--instruct` does nothing on 0.6B.** `qwen_tts` silently drops style instructions on the
  0.6B CustomVoice checkpoint (`if self.model.tts_model_size in "0b6": instruct = None`).
  Pass `--model-size 1.7B` for style control — the CLI warns you if you don't.
- **VoiceDesign is 1.7B-only.** There is no 0.6B checkpoint, so `design` only accepts
  `--model-size 1.7B` (it's the default there). It's a ~3.5 GB download and the slowest to run.
- **Cloning wants a transcript.** `--ref-text` must match the clip word-for-word. Without one,
  `--x-vector-only` clones timbre alone, at noticeably lower fidelity.
- **"SoX could not be found!" is harmless — ignore it.** It comes from `qwen_tts`'s *25Hz*
  tokenizer (`core/tokenizer_25hz/vq/speech_vq.py`), which is imported at package load but
  never used here: every model in this project is 12Hz. Same for "flash-attn is not installed"
  (no CUDA on this box anyway). Neither affects output.
- **`--offline` when Hugging Face is slow.** Every load revalidates each file against the Hub,
  and `qwen_tts` also triggers a `model_info()` API call in `transformers` (it passes
  `fix_mistral_regex=True`, and that check is unguarded — no cache, no offline branch). If the
  network is flaky, each check burns 5 retries with backoff and a cached load takes many
  minutes. `--offline` skips all of it and reads the cache directly. Needs the model downloaded
  once first.

## Accent reality check

Qwen3-TTS has **no** South African English preset or language. Its English presets are just
`Ryan` and `Aiden`, and its only built-in dialect variation is Chinese (Beijing via speaker
`Dylan`, Sichuan via `Eric`). In order of how likely they are to sound right:

1. **`clone`** — give it a real Cape Flats / Capetonian reference clip + transcript. The only
   approach likely to sound authentic: the model copies the accent rather than inventing it.
2. **`design`** — describe the accent in words. Experimental; Qwen never demonstrated South
   African English, so it may drift to a generic or wrong accent.
3. **`preset`** — no accent control at all. Useful for testing the pipeline.

## TODO: test the 1.7B models on the home machine (32 GB RAM)

Everything verified so far ran on the 8 GB desktop, which forces the compromises above. The
home machine has **32 GB RAM and more disk**, so it can run the 1.7B models comfortably —
that's the next test.

Why it's worth doing: cloning on **Base 0.6B** works and the accent survives (confirmed by ear
on a real Cape Flats reference), **but every clone comes out ~20 Hz above the reference** —
94.5 Hz reference vs 110–117 Hz cloned, consistently across takes. That's a systematic 0.6B
limitation, not a bad seed. Base 1.7B is the obvious lever.

On the home machine:

```powershell
# 32 GB fits 1.7B in float32 -- no need for the bfloat16 compromise
python tts.py voices add <name> --model-size 1.7B --ref-audio ..\samples\<clip>.wav --ref-text "<transcript>"
python tts.py clone --voice <name> --model-size 1.7B --text-file ..\samples\script.txt --join
```

Also worth trying there, all previously RAM-blocked:
- `preset --model-size 1.7B --instruct "..."` — style control (does nothing on 0.6B, see Gotchas)
- `design --model-size 1.7B` — 1.7B is its only size; slow but usable on 32 GB
- Running the Gradio UI and a CLI generation at the same time (8 GB can only do one at once)

Note: `output/`, `samples/` and `voices/` are gitignored, so **the reference clips, saved voices
and generated wavs do not travel with a clone.** Copy them across manually, or re-register the
voice at home with `voices add`.

## Setup

Already done in this checkout: a `.venv` with `qwen-tts` installed. To recreate:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -U qwen-tts
```

Weights are **not** downloaded up front — each subcommand pulls its model from Hugging Face on
first use, then caches it under `%USERPROFILE%\.cache\huggingface`.

## Usage

Run from the `scripts/` directory (activate the venv first: `.\.venv\Scripts\Activate.ps1`):

```powershell
# What speakers/languages does the model actually support?
python tts.py info

# Preset voice
python tts.py preset --text "Hello, this is a test." --speaker Ryan --out ..\output\test.wav

# Preset voice with style control (1.7B required)
python tts.py preset --text "Hello, this is a test." --speaker Vivian `
  --instruct "Speak slowly and warmly." --model-size 1.7B --out ..\output\warm.wav

# Design a voice from a description (experimental for this accent)
python tts.py design --text "Ja no, it's lekker hot today, hey." `
  --instruct "South African English, Cape Flats accent, warm conversational male, mid-20s" `
  --out ..\output\designed.wav

# Clone straight from a reference clip
python tts.py clone --ref-audio ..\samples\cape_flats.wav `
  --ref-text "exact transcript of the clip" `
  --text "Howzit my bru, you good?" --out ..\output\clone.wav
```

### The voice library

Encoding a reference clip is the slow part of cloning, and it's identical every time you reuse
the same clip. Register it **once**, then refer to it by name:

```powershell
# Drop your clips in samples\ first, then:
python tts.py voices add cape_flats --ref-audio ..\samples\cape_flats.wav `
  --ref-text "exact transcript of the clip"

python tts.py voices list
python tts.py voices show cape_flats
python tts.py voices remove cape_flats

# Reuse it -- no --ref-audio or --ref-text needed
python tts.py clone --voice cape_flats --text "Howzit my bru, you good?" --out ..\output\clone.wav
```

Voices are saved to `voices\<name>.pt` in the same format the Gradio UI uses, so they're
interchangeable between the two. A voice records which Base model size encoded it (the speaker
embedding comes from that model), and you get a warning if you later load it at a different
size — re-register with `voices add <name> --force --model-size <size>` to match.

### Reading a script or paragraph

Any subcommand takes `--text-file` instead of `--text`: one generation per line, with blank
lines and `#` comments skipped. You get numbered files (`read_001.wav`, `read_002.wav`, ...),
or one continuous wav with `--join`:

```powershell
python tts.py clone --voice cape_flats --text-file ..\samples\script.txt --join --out ..\output\read.wav
```

**For anything longer than one line, split it into one sentence per line — don't pass a whole
paragraph as a single generation.** Measured on the same 82-word paragraph, same voice, same
seed, only the splitting differs:

| | Pace | Median pitch (reference: 94.5 Hz) |
|---|---|---|
| One sentence per line + `--join` | **155 wpm** (natural) | **102.5 Hz** |
| Whole paragraph, one generation | 186 wpm (rushed) | 112.0 Hz |

Splitting wins twice. It stops the model rushing, and it halves the pitch error — each sentence
re-conditions on the reference clip independently, so it keeps getting pulled back toward the
real voice, while one long generation drifts away and never recovers.

`--join` puts 300ms of silence between lines; tune it with `--gap-ms`. Output is peak-limited to
0.97 by default (`--peak`) because long reads have been observed clipping; the gain is applied
once across all lines, so relative levels between sentences are preserved. `--peak 0` disables.

### Tuning the delivery

All generation subcommands accept `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`,
`--max-new-tokens`, and `--seed`. Anything you don't set falls back to the model's own
`generate_config.json`. `--seed` makes a take reproducible; re-roll a bad take by changing it or
nudging `--temperature`.

`--dtype` and `--device` are also on every subcommand — see the RAM table above before forcing
`--dtype float32` on a 1.7B model.

### Web UI

The UI bundled with `qwen_tts` defaults to CUDA + FlashAttention, which don't exist here.
`launch_ui.py` fixes those defaults. One capability per process:

```powershell
python launch_ui.py preset    # add --model-size 1.7B for the Instruction box to work
python launch_ui.py design
python launch_ui.py clone     # has a Save/Load Voice tab; save into voices\ to share with the CLI
```

Serves on `http://127.0.0.1:8000` (local only; `--share` for a public link). Add `--offline` if
startup stalls on Hugging Face — see Gotchas above.

## Layout

- `scripts/tts.py` — the CLI (`info`, `preset`, `design`, `clone`, `voices`)
- `scripts/voice_library.py` — save/load named voices
- `scripts/launch_ui.py` — Gradio UI launcher with CPU-correct defaults
- `scripts/common.py` — shared device/model/output helpers
- `samples/` — put your reference clips here (gitignored)
- `voices/` — saved voice prompts (gitignored; they embed a speaker fingerprint)
- `output/` — generated wavs land here (gitignored)
