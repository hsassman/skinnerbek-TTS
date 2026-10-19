<img src="assets/icon-256.png" alt="" width="88" align="right">

# Skinnerbek TTS

Skinnerbek TTS is a local text-to-speech toolkit for **South African English with a Cape Flats /
Coloured Capetonian accent**, built on Alibaba's open-weight
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) models. No mainstream TTS model ships this
accent, and Qwen3-TTS has no South African voice either. So the project gets the accent from
**voice cloning**: a short reference recording of a real speaker conditions the model. Around
that sits the engineering that makes a small model read well on ordinary hardware: text
preparation, a reusable voice library, per-line generation with repair, and loudness mastering.
It all runs fully offline on a CPU-only Windows PC, with a command-line tool and a browser app.
Fine-tuning a dedicated accent model is the next stage (see [Status and roadmap](#status-and-roadmap)).

## Features

- **Three ways to make a voice.** Clone one from a reference clip (the accent path), design one
  from a written description, or use the model's built-in preset speakers.
- **Voice library.** Encode a reference clip once and reuse it by name. Ad-hoc clips are cached
  automatically.
- **Text preparation for natural reads.** Numbers, money, dates, times, units and
  abbreviations are spelt out. Markdown is stripped. Long scripts are split into sentences and
  paragraphs, with tuned pauses between them.
- **Pronunciation lexicon.** A plain-text `lexicon.txt` of overrides for names, slang and
  acronyms the model gets wrong. The web app reloads it live.
- **Delivery presets** (steady / natural / expressive) and reproducible seeds.
- **Loudness mastering** to −16 LUFS, so separate takes match in volume.
- **Web app (Skinnerbek Studio).** Opens on **Clone a voice**: drop in a clip or record
  yourself, and the app writes out what was said, reads how the person speaks, saves the voice
  and lets you hear it say anything, all on one screen. Speak, My voices, Design a voice,
  Recordings and Settings tabs follow, in light and dark themes, with a skippable guided tour
  for first-time visitors. Renders show live per-line progress with an ETA, stop safely between
  lines, and have "Fix one line" repair.
- **Automatic transcripts.** A clip is transcribed locally with Whisper as soon as it arrives,
  word for word including "um" and "uh" (the cloning model hears them), tuned with South
  African vocabulary and your lexicon. Words it was unsure of are highlighted for checking.
  Accuracy was measured against hand-typed transcripts (see *Transcription* below).
- **A read of the speaker's delivery.** Pitch and range, pace, pauses, fillers and clarity,
  with a pitch contour. Each saved voice keeps it, and Speak uses it to set that voice's
  delivery and sentence pauses automatically.
- **Record a voice in the browser.** A guided recorder with a read-aloud passage, live level
  meter and target-length timer. Every clip gets a quality report (length, silence, clipping,
  loudness) before it becomes a voice.
- **Runs on modest hardware.** It works CPU-only and picks a memory-aware precision
  automatically. A preflight check refuses to load a model that won't fit in RAM, instead of
  letting it crash mid-load.
- **Private by design.** Inference is local, there's an offline mode, and reference
  recordings and voice fingerprints never leave the machine.

## Architecture

```
            text / script                       reference clip + transcript
                 │                                         │
                 ▼                                         ▼
   textprep.py  ─ normalise, lexicon,          voice_library.py / audio_prompt_cache.py
                  sentence + paragraph split     ─ encode once → voices/<name>.pt
                 │                                         │
                 └──────────────┬──────────────────────────┘
                                ▼
   common.py  ─ model loading (device, dtype, RAM preflight), delivery presets,
                per-line generation loop (synth), loudness, stitching
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                  ▼
     Qwen3-TTS Base     Qwen3-TTS CustomVoice   Qwen3-TTS VoiceDesign
       (clone)              (preset)               (design)
                                │
                                ▼
                    output/*.wav  (+ per-line parts, session file)

   Front ends:  tts.py (CLI)   ·   app.py (Gradio web app)   ·   launch_ui.py
```

Both front ends call the same `common.synth` / `run_clone` / `run_preset` / `run_design`
functions. A take from the web app therefore matches the equivalent CLI command. Model
weights come from Hugging Face on first use and are cached locally. They aren't stored in
this repository.

## Requirements

| | |
|---|---|
| OS | Windows 10/11 (developed and tested on Windows 10) |
| Python | 3.10+, tested on **3.12.2** |
| GPU | **Not required.** Runs CPU-only by default. A CUDA GPU can be used with `--device cuda:0` (not yet tested; CUDA version follows your PyTorch build) |
| GPU VRAM | n/a on CPU. Roughly 2 GB for the 0.6B models and 4 GB for 1.7B in bfloat16 |
| RAM | 8 GB minimum (0.6B models; 1.7B only in bfloat16 with other apps closed). **16–32 GB recommended** |
| Disk | A few GB for the Hugging Face model cache (VoiceDesign 1.7B alone is ~3.5 GB), plus 0.5–1.6 GB for the speech-recognition models |

Pinned dependency versions are in [requirements.txt](requirements.txt) (mirrored in
[pyproject.toml](pyproject.toml)).

## Install

```powershell
git clone https://github.com/hsassman/skinnerbek-TTS.git
cd skinnerbek-TTS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt     # or requirements.txt for runtime only
```

No API keys or environment variables are needed. The base models download from Hugging Face
the first time each capability runs, then load from `%USERPROFILE%\.cache\huggingface`. The
speech-recognition models download on the first transcription into
`%USERPROFILE%\.cache\skinnerbek\whisper` (plain files: the Hugging Face cache's symlinks fail
on Windows without Developer Mode). Pass `--offline` after that.

## Run

```powershell
cd scripts
python app.py                                   # web app at http://127.0.0.1:7860

python tts.py info                              # supported speakers and languages
python tts.py preset --text "Hello, this is a test." --speaker Ryan --out ..\output\test.wav

# Register a consented reference recording once, then reuse it by name
python tts.py voices add cape_flats --ref-audio ..\samples\cape_flats.wav --ref-text "exact transcript"
python tts.py clone --voice cape_flats --text-file ..\samples\quick_test.txt --join --out ..\output\read.wav
```

The [technical reference](#the-three-capabilities) below covers every option.

## Training and fine-tuning

**Not implemented yet.** The current system adapts the accent at inference time through voice
cloning, and the model weights are unchanged. The planned fine-tuning stage is:

1. Build a consented Cape Flats speech dataset: recordings plus verified transcripts, with
   written speaker consent for training use.
2. Fine-tune the Qwen3-TTS Base checkpoint on it.
3. Publish the resulting weights to a **private Hugging Face repository** and add a download
   script. Weights and voice data will never be committed to this repository.

## Project structure

```
scripts/
  tts.py                 CLI: info, preset, design, clone, voices
  app.py                 all-in-one Gradio web app
  common.py              model loading, delivery presets, generation loop, loudness, stitching
  textprep.py            text normalisation, lexicon, segmentation (--selftest)
  voice_library.py       save/load named voices (voices/<name>.pt)
  audio_prompt_cache.py  disk cache for encoded ad-hoc reference clips
  session.py             persists finished takes so they can be repaired after a restart
  launch_ui.py           launches app.py, or the bundled qwen_tts demo
  feature_tour.py        narrated smoke test of the read-quality features
  refclip.py             reference-clip quality report and clean-up
  transcribe.py          local speech-to-text (faster-whisper) and the speaker's delivery read
  mock_model.py          fast stand-in model for `app.py --mock` (UI development)
  ui/                    studio.css, studio.js, recorder.js: the web app's styles and browser code
  ui/vendor/             driver.js 1.8.0 (MIT) for the guided tour, bundled so it works offline
assets/                  logo, favicon, masthead banners (night and dawn)
tests/                   pytest suite (99 tests, no model download needed)
lexicon.txt              pronunciation overrides
samples/quick_test.txt   short script for smoke-testing an install
samples/ voices/ output/ local-only: reference clips, voice fingerprints, generated audio (gitignored)
```

## Licence

- **This repository's code:** proprietary, all rights reserved. See [LICENSE](LICENSE).
- **Qwen3-TTS models and the `qwen-tts` package:** Apache License 2.0, © the Qwen team /
  Alibaba Cloud. They're downloaded at runtime and not redistributed here.
- **Voice data:** reference recordings and derived voice prompts belong to their speakers.
  They're excluded from the repository and must only be used with the speaker's consent.

## Status and roadmap

**Status: working prototype.** Built and tested on an 8 GB, CPU-only Windows desktop.

- Done: cloning with the 0.6B Base model keeps the Cape Flats accent, judged by ear against a
  real reference speaker.
- Done: text prep, the voice library, the web app with line-level repair, and loudness
  mastering. All 91 unit tests pass.
- Known limitation: 0.6B clones come out consistently about 20 Hz higher in pitch than the
  reference.
- Next: evaluate the 1.7B models on a 32 GB machine. This is the expected fix for the pitch
  offset and adds style control.
- Next: run the fine-tuning stage above, using consented training data.

---

# Technical reference

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

**`--low-memory` drops 0.6B to bfloat16 too** (~1.5 GB instead of ~3 GB). It's on every
subcommand and `app.py`, and it turns itself on automatically when RAM is too tight for float32.

**Do not reach for it on a CPU without hardware bfloat16.** torch has no bf16 kernels below
AVX512-BF16/AMX, so it emulates them. Measured on this box (Intel i5-7200U, 2 cores, AVX2):

| operation | float32 | bfloat16 | penalty |
|---|---|---|---|
| decode matvec (`1×1024 @ 1024×3072`) | 1.11 ms | 28.39 ms | **25.6×** |
| prefill matmul (`64×1024 @ 1024×3072`) | 5.34 ms | 1417.77 ms | **265.6×** |

End to end, one 33-character line took **72 s in float32 vs 510 s in bfloat16 — 7.1× — and the
float32 run was paging** against only 1.7 GB free. Emulated bf16 costs more than the page faults
do, so `common.auto_low_memory()` asks `cpu_bf16_is_native()` first and only falls back to
bfloat16 below ~1.5 GB free, where float32 would leave nothing to run in. On a CPU that *does*
have bf16 kernels the old ~4 GB threshold still applies, because there the trade is real.

If you see the "this CPU has no hardware bfloat16" warning, close a browser and re-run —
`--dtype float32` forces it explicitly.

**`--chunk-size` controls how many lines are generated per model call.** qwen_tts pads and
batches every line you hand it into one tensor, so a 12-line script builds a 12-wide KV cache
and holds every waveform before decoding — several GB that pages to disk on this box and never
finishes. The default **`--chunk-size 1`** generates one line at a time: the peak footprint
stays at a single line, and a run prints `...N/M lines` as it goes so you can see it moving.
**`--chunk-size 0`** restores the single batched call — faster where RAM is plentiful, so use
it on the 32 GB machine. A given seed produces different audio at different chunk sizes (the
sampler draws in a different order); each take is still a valid sample.

**Every line gets a generation ceiling.** `qwen_tts`'s `generate()` defaults to
`max_new_tokens=4096` codec frames regardless of how short the line is — at 12 Hz that is ~5.7
minutes of audio for one sentence, and nothing relates it to the text you passed in. A line that
never emits its EOS code (long digit runs and other odd text make that much more likely) runs
all the way to that ceiling; on a CPU box at a few frames per second that is *hours* on one
sentence, with no output to show for it. `common.token_budget()` sizes a ceiling from the
character count instead, with ~3× headroom over the slowest read measured here, and `synth`
applies it per call. `--max-new-tokens` still overrides it explicitly.

**`--threads` pins torch's intra-op thread count.** The default is torch's own choice (the
physical core count) and is usually right; set it when a machine reports its topology badly or
when a CLI run is sharing the box with the Gradio server.

**Practical advice for the 8 GB box:** close the browser and other apps before a run (aim for
3–4 GB free), and feed short scripts — two or three sentences — not a dense page. Watch Task
Manager: if "In use" is pegged and the disk is at 100%, it's paging, not hung. A useful sanity
check is CPU time versus wall time: much under 100% of `--threads` cores means it is stalling on
page faults rather than computing.

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

## Next test: the 1.7B models on a 32 GB machine

Everything verified so far ran on an 8 GB desktop, which forces the compromises above. The
home machine has **32 GB RAM and more disk**, so it can run the 1.7B models comfortably —
that's the next test.

Why it's worth doing: cloning on **Base 0.6B** works and the accent survives (confirmed by ear
on a real Cape Flats reference), **but every clone comes out ~20 Hz above the reference** —
94.5 Hz reference vs 110–117 Hz cloned, consistently across takes. That's a systematic 0.6B
limitation, not a bad seed. Base 1.7B is the obvious lever.

On the home machine:

```powershell
# 32 GB fits 1.7B in float32 -- no need for the bfloat16 compromise.
# --chunk-size 0 batches all lines in one call (faster; the 8 GB box can't).
python tts.py voices add <name> --model-size 1.7B --ref-audio ..\samples\<clip>.wav --ref-text "<transcript>"
python tts.py clone --voice <name> --model-size 1.7B --chunk-size 0 --text-file ..\samples\script.txt --join
```

Also worth trying there, all previously RAM-blocked:
- `preset --model-size 1.7B --instruct "..."` — style control (does nothing on 0.6B, see Gotchas)
- `design --model-size 1.7B` — 1.7B is its only size; slow but usable on 32 GB
- `--chunk-size 0` everywhere — one batched call instead of line-by-line
- Running the Gradio UI and a CLI generation at the same time (8 GB can only do one at once)

Note: `output/`, `samples/` and `voices/` are gitignored, so **the reference clips, saved voices
and generated wavs do not travel with a clone.** Copy them across manually, or re-register the
voice at home with `voices add`.

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

**Ad-hoc `clone --ref-audio` runs also cache the encoding** (under `.cache/voice_prompts/`,
keyed by the clip's contents + transcript + mode + size), so repeated takes of the same
one-off clip skip the slow encode step without needing a named voice. `--no-cache` forces a
re-encode; delete `.cache/` to clear it.

### Reading a script or paragraph

Any subcommand takes `--text-file` instead of `--text`. You get numbered files
(`read_001.wav`, `read_002.wav`, ...), or one continuous wav with `--join`:

```powershell
python tts.py clone --voice cape_flats --text-file ..\samples\script.txt --join --out ..\output\read.wav
```

**Text is cleaned up before it reaches the model** (turn off with `--no-normalize`). This is
where most of the 0.6B model's read quality comes from — it stumbles badly on anything that
isn't plain prose:

- **Numbers, money, dates, times, percentages spelt out**: `$1,250.50` → "one thousand two
  hundred and fifty dollars and fifty cents", `1985` → "nineteen eighty-five", `10:30` → "ten
  thirty", `25%` → "twenty-five percent", `3rd` → "third", long digit runs read digit by digit.
- **Abbreviations expanded**: `Dr.` → "Doctor", `e.g.` → "for example", `&` → "and", `etc.`,
  `vs.`, `No. 5` → "number five".
- **Markdown stripped**: `**bold**`, `_italic_`, `` `code` ``, `# headings`, `- bullets`,
  `[links](url)` all reduced to plain words.
- **Segmented for pace**: each sentence becomes its own generation, and any sentence over
  `--max-words` (default 24) is broken at a clause boundary near its middle. A blank line is a
  paragraph break and gets a longer silence (`--paragraph-gap-ms`, default 650) than the
  between-sentence gap (`--gap-ms`, default 300). A missing full stop is added so the model
  doesn't trail off or clip the last word.
- **Your pronunciations applied** from `lexicon.txt` (see below) — this happens even with
  `--no-normalize`, disable it with `--no-lexicon`.

You can still hand-format — explicit line breaks are always honoured. Measured on the same
82-word paragraph, same voice, same seed, only the splitting differs:

| | Pace | Median pitch (reference: 94.5 Hz) |
|---|---|---|
| One sentence per line + `--join` | **155 wpm** (natural) | **102.5 Hz** |
| Whole paragraph, one generation | 186 wpm (rushed) | 112.0 Hz |

Splitting wins twice. It stops the model rushing, and it halves the pitch error — each sentence
re-conditions on the reference clip independently, so it keeps getting pulled back toward the
real voice, while one long generation drifts away and never recovers.

`--join` puts 300ms of silence between sentences and 650ms between paragraphs; tune with
`--gap-ms` / `--paragraph-gap-ms`.

Normalisation also handles the things that most often make the model stumble: currency in any of
`$ £ € ¥` (including the correct minor unit — "pence", not "pences"), units after a number
(`12.5 cm` → "twelve point five centimetres"), `~` → "about", `+` → "plus", `#` → "number",
email addresses (`support@example.com` → "support at example dot com"), roman numerals after a
label (`Mk III` → "Mark three"), and abbreviations that end in a full stop (`excl.`, `incl.`,
`est.`) which would otherwise also be mistaken for a sentence end. An initial no longer ends a
sentence either, so "Dr. A. Nguyen" stays on one line. A bare four-digit number in 1100–2099 is
read as a year (`1985` → "nineteen eighty-five"); write it with a comma (`1,150`) when it is a
count, not a year.

### Pronunciation lexicon

`lexicon.txt` at the repo root holds pronunciation overrides, one rule per line:

```
SQL      -> sequel
Nguyen   -> win
/\bkm\/h\b/ -> kilometres per hour
```

A bare pattern matches whole words case-insensitively; wrap it in `/.../` for a raw regex.
Rules run **before** number and abbreviation expansion, and the file is re-read every run (and
live-reloaded by the web app), so edits take effect immediately. It's the reliable fix for
proper nouns and acronyms the model gets wrong. `--no-lexicon` skips it for one run.

### Loudness

Every take is normalised to **−16 LUFS** (`--lufs`), one gain across all lines so relative
levels between sentences are preserved, then peak-limited to 0.97 (`--peak`) as a safety net.
This keeps separate takes and joined paragraphs at a consistent perceived loudness rather than
just "not clipping". `--lufs 0` falls back to peak-limiting only. Needs `pyloudnorm`
(in `pyproject.toml`); without it, it degrades to peak-limiting with a warning.

### Tuning the delivery

`--delivery` picks a sampling preset:

| Preset | Effect | Use for |
|---|---|---|
| `steady` | `temperature 0.7`, `repetition_penalty 1.1` | narration on 0.6B — calmer, fewer stammers and speed-ups |
| `natural` (default) | the model's own `generate_config.json` | general use |
| `expressive` | `temperature 1.0` | more variation, at the cost of the odd wander |

`--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--max-new-tokens` override the
preset when set explicitly. `--seed` makes a take reproducible; re-roll a bad take by changing
it or the delivery preset.

`--dtype` and `--device` are also on every subcommand — see the RAM table above before forcing
`--dtype float32` on a 1.7B model.

### Web UI

`scripts/app.py` is **Skinnerbek Studio**, the browser app. Start here:

```powershell
python app.py                 # http://127.0.0.1:7860, opens your browser
python app.py --offline       # skip Hugging Face network checks (see Gotchas)
python app.py --port 8000 --share
python app.py --mock          # development: a fast stand-in model, no checkpoint loaded
```

`python launch_ui.py` with no argument launches the same app. It has a light and a dark theme
(toggle at the top right, remembered per browser; the header banner changes with it, night or
dawn over Table Mountain), and **Ctrl+Enter** renders from the Speak or Design tab.

- **Clone a voice** (the landing tab): one screen from clip to hearing the new voice.
  - A large drop zone takes an MP3, WAV, M4A or FLAC; **Or record yourself** opens the
    browser recorder with a passage to read.
  - The clip gets a **quality report** at once, then an automatic **transcript** in an
    editable box, with any words the recogniser was unsure of highlighted above it. If you read
    the on-screen passage, its exact wording is used instead.
  - **Your voice at a glance** shows pitch, range, pace, pauses, fillers and clarity, a pitch
    contour, and the delivery and pause it will use for this voice.
  - Name it, press **Create my voice**, and **Hear your new voice** unlocks: type anything and
    press Speak. The name is suggested from the file name.

- **Speak**: write a script or start from a sample, pick a voice, press **Speak**.
  - A live counter shows words, sentences and roughly how long the read will be.
  - A card explains the selected voice: built-in or cloned, model size, what it was cloned from.
  - **Delivery** (Steady / Natural / Expressive) and **Tidy the text** sit above the buttons.
    Tidy-up does the number, abbreviation and markdown cleanup and the one-sentence-per-line
    split described under *Reading a script*, the same as `tts.py --text-file --join`.
  - **Preview** shows the exact lines the model will get, with a time estimate, before you
    commit to a slow run.
  - **Advanced** holds the model size, language, the 1.7B style instruction, the pause lengths
    and a seed.
  - A seed of **0 means a fresh take**. Any other number is repeatable. Gradio turns an empty
    number box into 0, so 0 can't double as a real seed.
- **Design a voice**: invent a voice from a written description (VoiceDesign, 1.7B), with
  one-click description ideas and the same render column.
- **My voices**: every cloned voice as a card (with its pitch and pace), plus a panel to
  audition the clip it was made from, send it to Speak, or delete it with a two-step confirm.
  Saves the same `voices\<name>.pt` the CLI reads.

  The **quality report** on the Clone tab shows length (against the ideal 5–30 s), speech
  time, loudness and peak, and explains problems in words: silence, clipping, too short or too
  long, mostly pauses. On save, the clip is trimmed of leading and trailing silence, levelled,
  and kept as `voices/<name>.ref.wav` beside the voice, so it can be auditioned later.
- **Recordings**: every finished read, newest first, with playback, **Download**, and a
  two-step **Delete**.
- **Settings**: the machine report, restoring an earlier take, the pronunciation lexicon
  editor, and housekeeping.

**Transcription.** `scripts/transcribe.py` runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
locally (int8 on CPU). Two modes, under *Transcription options*: **Accurate** (Whisper
large-v3-turbo, the default) and **Fast** (Whisper small). Choices made for cloning rather
than dictation:

- *Verbatim by default.* A disfluent prompt keeps "um" and "uh", because the cloning model
  aligns the transcript against the audio it hears.
- *Spoken numbers.* "24" is written "twenty-four", the way it was said; codes such as HTD-189
  or MP3 are left alone.
- *Tailored vocabulary.* South African words and place names, plus every plain pattern in
  `lexicon.txt`, are passed as hotwords.
- *Memory.* The recogniser is loaded on first use and freed before a voice model loads, so the
  two never share the 8 GB.

Accuracy was checked against the hand-typed transcripts of the four voices in this repo
(advert and voice-over reads, 9–47 s), as word error rate after spelling numbers out on both
sides so formatting doesn't count as an error. On this 8 GB, 2-core machine:

| Mode | Mean WER | Time per clip | Extra RAM |
|---|---|---|---|
| Accurate (large-v3-turbo), plain Whisper | 6.5 % | 50–105 s | ~1 GB |
| Fast (small), with the app's tuning | 8.9 % | 10–30 s | ~0.3 GB |
| Fast (small), plain Whisper | 9.3 % | 10–30 s | ~0.3 GB |
| Fast (small), tuning but no "um/uh" priming | 10.6 % | 10–30 s | ~0.3 GB |

The app's tuning (verbatim priming, hotwords, spoken numbers) also helped turbo on the two clips
re-run with it (11.9 % → 8.5 %, and 0 %); the other two weren't re-run. Turbo can still be
confidently wrong ("light train" for "late train" at 98 % confidence), which is why the app
asks for a read-through even when nothing is highlighted. Clip language defaults to English:
Whisper's auto-detect labelled a Cape Town English clip as Afrikaans.

Most remaining "errors" are spelling choices (Bancorp vs Bank Corp, HTD-189 vs HTD189), and
at least one disagreement may be the hand-typed reference rather than the recogniser. Four
clips is a small sample: treat these as a sanity check, not a benchmark.

**Delivery read.** Pitch comes from YIN (librosa), gated on energy, with octave slips
dropped; medians matched Praat to within 3 Hz on every clip here. Expressiveness is the
10th–90th percentile pitch span in semitones (under 8 steady, 8–13 natural, over 13
animated), which picks the suggested Delivery. Pace is words per minute over the speaking
time, and the suggested sentence pause is the median pause after a full stop.

**Guided tour.** First-time visitors get a short tour (driver.js) of the Clone tab and the
tabs. It can be skipped at any step, has a "Don't show this again" box, and stops offering
itself after three visits either way. **Take the tour** in the header replays it.

**Recording in the browser.** The recorder is custom (`scripts/ui/recorder.js`) rather than
Gradio's built-in microphone input. Gradio's recorder swallows permission and device errors, so
its Record button just did nothing when the prompt was missed, the mic was blocked, or Windows'
microphone privacy setting was off. It also showed "No microphone found" until permission was
granted, and never refreshed. This recorder:
- asks explicitly, and explains each failure with a fix (address-bar permission, Windows
  Settings › Privacy › Microphone, mic in use by another app);
- fills in the microphone picker once permission is granted;
- gives a 3-2-1 count-in, a live waveform and level meter ("Speak up", "Good level", "Too
  loud"), and a timer against the 5–30 s target;
- encodes a 24 kHz mono 16-bit WAV in the browser, so no ffmpeg is needed.

Recording works on `http://127.0.0.1` / `localhost` and over `--share` (https). Browsers block
the microphone on a plain-http LAN address.

**Live render progress.** A generation streams its own progress: a bar, an elapsed clock, an ETA
computed from the lines that have already finished in *this* run, and a per-line list showing
which line is rendering, which are done (with their time and audio length) and which are waiting.
It is a pure view over the `on_line` callback `common.synth` already fires, so it adds nothing to
the model path.

**Stop.** Cancellation is checked between lines, never mid-line, because a half-generated
waveform is not usable audio. Whatever finished is still joined and saved, so a stopped run
leaves a real file.

**Machine report.** The header shows cores, SIMD level, torch threads, free RAM, the resolved
dtype and the measured speed. If the CPU lacks hardware bfloat16 *and* RAM has forced the slow
path, it says so in as many words, along with how much RAM to free. That specific trap cost
hours before it was visible.

**Time estimate.** "Preview" shows the exact lines and roughly how long they will take. It uses
a seconds-per-frame figure that `common.record_speed` calibrates from your own completed lines
(never from `--mock` runs).

**Restore a take.** Each finished read writes a small session file next to its per-line audio,
so a take stays repairable after a restart. Restore it from Settings, then use "Fix one line".

**Housekeeping.** Settings can delete per-line audio older than 24 hours that no restorable take
refers to, with a preview first.

**Lexicon editor.** `lexicon.txt` is editable in Settings. `textprep` re-reads it whenever the
file changes, so a saved rule applies to the next generation with no restart.

**Design assets.** The logo is inline SVG (it animates, and stays crisp at any size), and
`assets/favicon.svg` is its static copy. The masthead banners (`assets/banner.webp` at night
for the dark theme, `assets/banner-light.webp` at dawn for the light one) were generated for
the project; the app falls back to drawn SVG art without them. Styles and browser
behaviour live in `scripts/ui/` (`studio.css`, `studio.js`, `recorder.js`) as ordinary files.

`python app.py --low-memory` forces 0.6B to bfloat16 (it also auto-enables when RAM is tight);
the RAM line at the top of the page says when low-memory mode is active. The app generates
**one line at a time** (`--chunk-size 1`, the default): the progress bar counts `Generated N
of M lines`, each finished line is written to `output/parts/` as it lands (so a killed run
still leaves audio), and the peak memory stays at a single line. Pass `--chunk-size 0` on a
big machine to batch the whole script in one call.

Every generation calls the exact helpers the CLI uses (`common.synth` over `run_preset` /
`run_design` / `run_clone`, then `common.write_wavs` / `stitch`), so a result from the app
matches the equivalent `tts.py` command at the same `--chunk-size`. One model fits in RAM at a
time here, so switching between Speak / Describe / a different model size swaps the loaded
checkpoint; the first use of each is a slow CPU load.

**The bundled `qwen_tts` demo** is still available for the original per-capability UI. It
defaults to CUDA + FlashAttention, which don't exist here; `launch_ui.py` fixes those
defaults. One capability per process:

```powershell
python launch_ui.py preset    # add --model-size 1.7B for the Instruction box to work
python launch_ui.py design
python launch_ui.py clone     # has a Save/Load Voice tab; save into voices\ to share with the CLI
```

Serves on `http://127.0.0.1:8000` (local only; `--share` for a public link). Add `--offline` if
startup stalls on Hugging Face — see Gotchas above.

## Testing

```powershell
python -m pytest                    # unit tests (tests/), no model needed
python scripts/feature_tour.py      # a narrated walk-through of every read-quality feature
python scripts/feature_tour.py --generate   # + one real 0.6B synthesis end-to-end (slow)
python scripts/textprep.py --selftest       # just the text-prep cases
```

`feature_tour.py` is the quickest way to see numbers/dates expansion, the lexicon, delivery
presets, low-memory dtype selection, loudness normalisation, the reference-clip cache and the
per-line re-stitch all working on one worked example.
