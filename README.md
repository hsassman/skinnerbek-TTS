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
- **Web app.** Speak, Describe a voice, Voice library, Recordings and Settings tabs. It shows
  live per-line progress with an ETA, can stop safely between lines, and has "Fix one line"
  repair. Finished takes can be restored after a restart.
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
| Disk | A few GB for the Hugging Face model cache (VoiceDesign 1.7B alone is ~3.5 GB) |

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
the first time each capability runs, then load from `%USERPROFILE%\.cache\huggingface`. Pass
`--offline` after that.

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
tests/                   pytest suite (85 tests, no model download needed)
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
  mastering. All 85 unit tests pass.
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

The app has five tabs: **Speak**, **Describe a voice**, **Voices**, **Recordings** and
**Settings**.

**Live render progress.** A generation streams its own progress: a bar, an elapsed clock, an ETA
computed from the lines that have already finished in *this* run, and a per-line list showing
which line is rendering, which are done (with their time and audio length) and which are waiting.
It is a pure view over the `on_line` callback `common.synth` already fires, so it adds nothing to
the model path.

**Stop.** Cancellation is checked between lines, never mid-line — a half-generated waveform is not
usable audio. Whatever finished is still joined and saved, so a stopped run leaves a real file.

**Machine report.** The header shows cores, SIMD level, torch threads, free RAM, the resolved
dtype and the measured speed. If the CPU lacks hardware bfloat16 *and* RAM has forced the slow
path, it says so in as many words along with how much RAM to free. That specific trap cost hours
before it was visible.

**Time estimate.** "Preview text" shows the exact lines and roughly how long they will take, using
a seconds-per-frame figure `common.record_speed` calibrates from your own completed lines.

**Restore a take.** Each finished read writes a small session file next to its per-line audio, so a
take stays repairable after a restart — restore it from Settings and use "Fix one line".

**Housekeeping.** Settings can delete per-line audio older than 24 hours that no restorable take
refers to, with a preview first.

**Lexicon editor.** `lexicon.txt` is editable in Settings; `textprep` re-reads it on mtime change,
so a saved rule applies to the next generation with no restart.


**The simple app — start here.** `scripts/app.py` is one page for everything:

```powershell
python app.py                 # http://127.0.0.1:7860, opens your browser
python app.py --offline       # skip Hugging Face network checks (see Gotchas)
python app.py --port 8000 --share
```

`python launch_ui.py` with no argument launches the same app.

- **Speak** — type or paste a script, pick a voice, press one button. The voice list holds
  the nine built-in speakers *and* every voice you've saved under `voices\`. A **Delivery**
  choice (Steady / Natural / Expressive) and a **Tidy the text** toggle sit above the button;
  tidy-up does the number/abbreviation/markdown cleanup and one-sentence-per-line segmentation
  described under *Reading a script* — the same behaviour as `tts.py --text-file --join`.
  **Preview text** shows the exact lines that will be spoken before you commit to a slow run.
  After a multi-line take, **Fix one line** re-generates a single sentence (optionally with a
  new seed) and re-stitches the read, keeping the other lines. An **Advanced** panel exposes
  model size, language, the 1.7B style instruction, the between-sentence and between-paragraph
  gaps, and a seed.
- **Describe a voice** — invent a voice from a written description (VoiceDesign, 1.7B). Same
  Preview text and Fix one line controls.
- **Voice library** — turn a recording into a reusable named voice (upload or record in the
  browser, add the transcript, name it), and list or delete saved voices. Saves the same
  `voices\<name>.pt` the CLI reads.

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
