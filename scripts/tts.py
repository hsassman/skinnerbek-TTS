"""Unified Qwen3-TTS CLI: preset voices, voice design, and voice cloning.

Qwen3-TTS ships three separate checkpoints, one per capability, and this CLI
exposes all three plus a reusable voice library:

    preset   CustomVoice model  -- 9 built-in speakers (+ style instructions on 1.7B)
    design   VoiceDesign model  -- invent a voice from a written description
    clone    Base model         -- copy a voice from a reference recording
    voices   register/list/inspect/remove saved cloned voices
    info     list the speakers and languages the model actually supports

Every subcommand takes either --text "one line" or --text-file script.txt
(one generation per line; blank lines and #comments skipped, --join to
concatenate the results into a single wav).

Examples:
    # What can this thing actually say?
    python scripts/tts.py info

    # Preset speaker
    python scripts/tts.py preset --text "Hello there." --speaker Ryan --out output/hi.wav

    # Preset speaker with style control (needs 1.7B -- 0.6B ignores --instruct)
    python scripts/tts.py preset --text "Hello there." --speaker Vivian \
        --instruct "Speak slowly and warmly." --model-size 1.7B --out output/warm.wav

    # Design a voice from a description
    python scripts/tts.py design --text "Ja no, it's lekker hot today, hey." \
        --instruct "South African English, Cape Flats accent, warm male voice, mid-20s" \
        --out output/designed.wav

    # Register a reference clip once, then reuse it by name
    python scripts/tts.py voices add cape_flats --ref-audio samples/ref.wav \
        --ref-text "exact transcript of the clip"
    python scripts/tts.py clone --voice cape_flats --text-file samples/script.txt \
        --join --out output/read.wav
"""
import argparse
import sys

from audio_prompt_cache import encode_prompt
from common import (
    OUTPUT_DIR,
    add_model_args,
    add_sampling_args,
    apply_seed,
    load_model,
    read_script,
    sampling_kwargs,
    synth,
    validate_ref_audio,
    warn_if_instruct_ignored,
    write_wavs,
)
from voice_library import delete_voice, list_voices, load_voice, save_voice, voice_meta, voice_path


def add_text_args(parser: argparse.ArgumentParser, default_out: str) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="A single line of text to speak")
    group.add_argument("--text-file", help="Script file: one generation per line")
    parser.add_argument("--out", default=default_out, help=f"Output wav (default: {default_out})")
    parser.add_argument(
        "--join",
        action="store_true",
        help="With --text-file, concatenate all lines into one wav instead of numbered files",
    )
    parser.add_argument(
        "--gap-ms",
        type=int,
        default=300,
        help="Silence between sentences when using --join (default: 300)",
    )
    parser.add_argument(
        "--paragraph-gap-ms",
        type=int,
        default=650,
        help="Silence between paragraphs (a blank line in the script) when using --join (default: 650)",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Don't clean up the text before speaking (skip number/abbreviation expansion, "
             "markdown stripping and long-sentence splitting). Use if the cleanup mangles something.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=24,
        help="Split any sentence longer than this at a clause boundary so the model doesn't rush "
             "(default: 24; 0 disables clause splitting)",
    )
    parser.add_argument(
        "--no-lexicon",
        dest="lexicon",
        action="store_false",
        help="Ignore lexicon.txt pronunciation overrides for this run.",
    )
    parser.set_defaults(normalize=True, lexicon=True)
    parser.add_argument(
        "--lufs",
        type=float,
        default=-16.0,
        help="Loudness target in LUFS every take is normalised to, one gain across all lines "
             "(default: -16.0). 0 disables it and falls back to peak limiting only.",
    )
    parser.add_argument(
        "--peak",
        type=float,
        default=0.97,
        help="Scale output down to this peak so it can't clip; one gain across all lines, applied "
             "after --lufs as a safety net. 0 disables (default: 0.97)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Lines generated per model call. 1 (default) = one at a time, the lowest peak RAM "
             "-- required on the 8GB box, where a whole batch pages to disk. 0 (or a big number) "
             "= all lines in one batched call, faster where RAM is plentiful. Output for a given "
             "seed differs between chunk sizes.",
    )
    parser.add_argument("--language", default="Auto", help="Language name, or Auto to detect (default: Auto)")


def report(paths, sample_rate: int) -> None:
    for p in paths:
        print(f"Wrote {p} ({sample_rate} Hz)")


def line_progress(total: int):
    """A synth on_progress callback that reports advancing lines to stderr."""
    if total <= 1:
        return None

    def _cb(done: int, _total: int) -> None:
        print(f"  ...{done}/{total} lines generated", file=sys.stderr)

    return _cb


def prepared_script(args: argparse.Namespace):
    """(lines, gaps) after text cleanup + sentence/paragraph segmentation."""
    return read_script(
        args.text,
        args.text_file,
        normalize=args.normalize,
        lexicon=args.lexicon,
        max_words=args.max_words,
        sentence_gap_ms=args.gap_ms,
        paragraph_gap_ms=args.paragraph_gap_ms,
    )


def cmd_info(args: argparse.Namespace) -> int:
    model = load_model("CustomVoice", args)
    print("\nPreset speakers (for `preset --speaker`):")
    for spk in sorted(model.get_supported_speakers() or []):
        note = ""
        dialect = model.model.config.talker_config.spk_is_dialect.get(spk)
        if dialect:
            note = f"  -- speaks {dialect.replace('_', ' ')} when language is Chinese/Auto"
        print(f"  {spk}{note}")
    print("\nLanguages (for --language):")
    for lang in sorted(model.get_supported_languages() or []):
        print(f"  {lang}")
    print(
        "\nNote: there is no South African / Cape Flats preset or language. "
        "Use `clone` with a reference clip for that accent."
    )
    return 0


def cmd_preset(args: argparse.Namespace) -> int:
    # Read the script before loading: a model load is minutes on CPU, and a typo'd path or an
    # empty --text should fail now, not after the wait.
    lines, gaps = prepared_script(args)
    warn_if_instruct_ignored("CustomVoice", args, args.instruct)
    model = load_model("CustomVoice", args)
    apply_seed(args)

    wavs, sr = synth(
        "preset", model, lines,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
        sampling=sampling_kwargs(args),
        chunk_size=args.chunk_size,
        on_progress=line_progress(len(lines)),
    )
    report(write_wavs(wavs, sr, args.out, args.join, args.gap_ms, args.peak, gaps=gaps, lufs=args.lufs), sr)
    return 0


def cmd_design(args: argparse.Namespace) -> int:
    # --model-size is restricted to 1.7B at the parser level (see build_parser), so there's
    # nothing to override here -- argparse rejects 0.6B with a real message.
    lines, gaps = prepared_script(args)
    model = load_model("VoiceDesign", args)
    apply_seed(args)

    wavs, sr = synth(
        "design", model, lines,
        language=args.language,
        instruct=args.instruct,
        sampling=sampling_kwargs(args),
        chunk_size=args.chunk_size,
        on_progress=line_progress(len(lines)),
    )
    report(write_wavs(wavs, sr, args.out, args.join, args.gap_ms, args.peak, gaps=gaps, lufs=args.lufs), sr)
    return 0


def cmd_clone(args: argparse.Namespace) -> int:
    if not args.voice and not args.ref_audio:
        raise SystemExit("Pass either --voice <saved name> or --ref-audio <clip> (see `tts.py voices --help`).")
    if args.voice and args.ref_audio:
        raise SystemExit("Pass --voice or --ref-audio, not both.")
    if args.voice and (args.ref_text or args.x_vector_only):
        # A saved voice already baked in its transcript and mode; accepting these would
        # look like they applied when they don't.
        raise SystemExit(
            "--ref-text/--x-vector-only don't apply to --voice: a saved voice already has them "
            "baked in. Re-register it with `voices add --force` to change them."
        )
    if args.ref_audio and not args.ref_text and not args.x_vector_only:
        raise SystemExit(
            "--ref-text is required unless you pass --x-vector-only (which clones timbre only, "
            "without a transcript, at noticeably lower fidelity)."
        )
    if args.ref_audio and args.ref_text and args.x_vector_only:
        raise SystemExit(
            "--x-vector-only ignores --ref-text. Drop one: --ref-text alone gives the better clone."
        )
    if args.ref_audio:
        validate_ref_audio(args.ref_audio)
    if args.voice and not voice_path(args.voice).exists():
        known = list_voices()
        hint = f" Known voices: {', '.join(known)}" if known else " No voices saved yet."
        raise SystemExit(f"No saved voice named '{args.voice}'.{hint}")

    # Everything above is checked before the load: it's minutes on CPU, so bad input should
    # fail immediately rather than after the wait.
    lines, gaps = prepared_script(args)

    model = load_model("Base", args)
    apply_seed(args)

    if args.voice:
        prompt = load_voice(args.voice, expect_model_size=args.model_size)
    else:
        # Ad-hoc reference clip: encode it once and cache the result, so repeat
        # takes of the same clip skip the (slow, identical) encoding step.
        prompt, _ = encode_prompt(
            model,
            args.ref_audio,
            args.ref_text,
            args.x_vector_only,
            args.model_size,
            use_cache=not args.no_cache,
        )
    wavs, sr = synth(
        "clone", model, lines,
        language=args.language,
        voice_clone_prompt=prompt,
        sampling=sampling_kwargs(args),
        chunk_size=args.chunk_size,
        on_progress=line_progress(len(lines)),
    )
    report(write_wavs(wavs, sr, args.out, args.join, args.gap_ms, args.peak, gaps=gaps, lufs=args.lufs), sr)
    return 0


def cmd_voices_add(args: argparse.Namespace) -> int:
    if not args.ref_text and not args.x_vector_only:
        raise SystemExit(
            "--ref-text is required unless you pass --x-vector-only (timbre only, lower fidelity)."
        )
    if args.ref_text and args.x_vector_only:
        raise SystemExit(
            "--x-vector-only ignores --ref-text. Drop one: --ref-text alone gives the better clone."
        )
    validate_ref_audio(args.ref_audio)
    if voice_path(args.name).exists() and not args.force:
        raise SystemExit(f"Voice '{args.name}' already exists. Pass --force to overwrite.")

    model = load_model("Base", args)
    print(f"Encoding {args.ref_audio}...", file=sys.stderr)
    items = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only,
    )
    path = save_voice(args.name, items, ref_audio=args.ref_audio, model_size=args.model_size)
    print(f"Saved voice '{args.name}' -> {path}")
    print(f"Use it with: python scripts/tts.py clone --voice {args.name} --text \"...\"")
    return 0


def cmd_voices_list(args: argparse.Namespace) -> int:
    names = list_voices()
    if not names:
        print("No saved voices yet. Add one with:")
        print('  python scripts/tts.py voices add <name> --ref-audio samples/clip.wav --ref-text "transcript"')
        return 0
    print(f"Saved voices ({len(names)}):")
    for name in names:
        meta = voice_meta(name) or {}
        mode = "x-vector only" if meta.get("x_vector_only_mode") else "ICL"
        src = meta.get("ref_audio", "?")
        print(f"  {name}  [{mode}]  from {src}")
    return 0


def cmd_voices_show(args: argparse.Namespace) -> int:
    meta = voice_meta(args.name)
    if meta is None:
        raise SystemExit(f"No metadata for '{args.name}' (does it exist? try `voices list`).")
    for key, value in meta.items():
        print(f"{key}: {value}")
    return 0


def cmd_voices_remove(args: argparse.Namespace) -> int:
    path = voice_path(args.name)
    if not delete_voice(args.name):
        raise SystemExit(f"No saved voice named '{args.name}'.")
    print(f"Removed voice '{args.name}' ({path})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # info
    p_info = sub.add_parser("info", help="List supported speakers and languages")
    add_model_args(p_info, "CustomVoice")
    p_info.set_defaults(func=cmd_info)

    # preset
    p_preset = sub.add_parser("preset", help="Speak with a built-in preset voice (CustomVoice)")
    add_text_args(p_preset, str(OUTPUT_DIR / "preset.wav"))
    p_preset.add_argument("--speaker", default="Ryan", help="Preset speaker name (see `info`; default: Ryan)")
    p_preset.add_argument(
        "--instruct",
        default="",
        help="Style/emotion instruction, e.g. 'Speak angrily'. Requires --model-size 1.7B; ignored on 0.6B.",
    )
    add_model_args(p_preset, "CustomVoice")
    add_sampling_args(p_preset)
    p_preset.set_defaults(func=cmd_preset)

    # design
    p_design = sub.add_parser("design", help="Invent a voice from a written description (VoiceDesign, 1.7B only)")
    add_text_args(p_design, str(OUTPUT_DIR / "design.wav"))
    p_design.add_argument(
        "--instruct",
        required=True,
        help="Description of the voice to design, e.g. 'gravelly older male, slow, wry'",
    )
    add_model_args(p_design, "VoiceDesign")
    add_sampling_args(p_design)
    p_design.set_defaults(func=cmd_design)

    # clone
    p_clone = sub.add_parser("clone", help="Copy a voice from a reference recording (Base)")
    add_text_args(p_clone, str(OUTPUT_DIR / "clone.wav"))
    p_clone.add_argument("--voice", help="Name of a saved voice (see `voices list`)")
    p_clone.add_argument("--ref-audio", help="Reference clip: path, URL, or base64")
    p_clone.add_argument("--ref-text", help="Exact transcript of --ref-audio")
    p_clone.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Clone timbre from the speaker embedding alone, no transcript needed. Lower fidelity.",
    )
    p_clone.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't read or write the cached encoding of --ref-audio (re-encode from scratch).",
    )
    add_model_args(p_clone, "Base")
    add_sampling_args(p_clone)
    p_clone.set_defaults(func=cmd_clone)

    # voices
    p_voices = sub.add_parser("voices", help="Manage the saved voice library")
    vsub = p_voices.add_subparsers(dest="voices_command", required=True)

    v_add = vsub.add_parser("add", help="Encode a reference clip and save it under a name")
    v_add.add_argument("name", help="Name to save the voice as")
    v_add.add_argument("--ref-audio", required=True, help="Reference clip: path, URL, or base64")
    v_add.add_argument("--ref-text", help="Exact transcript of --ref-audio")
    v_add.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Skip the transcript and clone timbre only. Lower fidelity.",
    )
    v_add.add_argument("--force", action="store_true", help="Overwrite an existing voice of the same name")
    add_model_args(v_add, "Base")
    v_add.set_defaults(func=cmd_voices_add)

    v_list = vsub.add_parser("list", help="List saved voices")
    v_list.set_defaults(func=cmd_voices_list)

    v_show = vsub.add_parser("show", help="Show details of a saved voice")
    v_show.add_argument("name")
    v_show.set_defaults(func=cmd_voices_show)

    v_rm = vsub.add_parser("remove", help="Delete a saved voice")
    v_rm.add_argument("name")
    v_rm.set_defaults(func=cmd_voices_remove)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
