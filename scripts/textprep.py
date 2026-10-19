"""Prepare written text so Qwen3-TTS *reads* it well.

The 0.6B CustomVoice checkpoint has no style control (`instruct` is dropped) and
tends to rush and mangle anything that isn't plain prose. We can't change the
weights, but we can change what the model sees:

    * normalise the text        -- ASCII punctuation, expanded abbreviations,
                                   numbers/money/years/times spelt out, markdown
                                   noise stripped, a full stop added so the model
                                   doesn't trail off or clip the last word.
    * segment it sensibly       -- one sentence per generation, and any sentence
                                   longer than `max_words` broken at a clause
                                   boundary near its middle. Each chunk
                                   re-conditions on the voice, which is what stops
                                   the model speeding up and drifting off pitch.
    * pace it                   -- a longer silence between paragraphs (blank
                                   line) than between sentences.
    * fix names it gets wrong   -- an optional user lexicon (`lexicon.txt` at the
                                   repo root): `Nguyen -> win`, `SQL -> sequel`.

`prepare_script(text)` is the single entry point. It returns
`(lines, gaps_ms)`: the text to speak line by line, and the silence to place
*before* each line when the lines are joined (`gaps_ms[0]` is always 0).

Everything here is pure string work -- no model, no torch -- so it is cheap and
unit-testable on its own (`python scripts/textprep.py --selftest`).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

# --- user pronunciation lexicon ----------------------------------------------

LEXICON_PATH = Path(__file__).resolve().parent.parent / "lexicon.txt"

_LEXICON_CACHE: Optional[List[Tuple["re.Pattern[str]", str]]] = None
_LEXICON_MTIME: Optional[float] = None


def load_lexicon(path=None) -> List[Tuple["re.Pattern[str]", str]]:
    """Parse `lexicon.txt` into (compiled pattern, replacement) rules.

    Format: `pattern -> replacement`, one per line, `#` starts a comment.
    A bare pattern matches a whole word case-insensitively; wrap it in `/.../`
    for a raw regex. Missing file -> no rules.
    """
    p = Path(path) if path is not None else LEXICON_PATH
    rules: List[Tuple["re.Pattern[str]", str]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return rules
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "->" not in line:
            continue
        pat, rep = (s.strip() for s in line.split("->", 1))
        rep = rep.strip('"')
        if not pat:
            continue
        try:
            if len(pat) > 2 and pat[0] == "/" and pat[-1] == "/":
                rx = re.compile(pat[1:-1])
            else:
                rx = re.compile(r"\b" + re.escape(pat) + r"\b", re.IGNORECASE)
        except re.error:
            continue
        rules.append((rx, rep))
    return rules


def _lexicon() -> List[Tuple["re.Pattern[str]", str]]:
    """Cached lexicon, reloaded automatically when the file changes on disk."""
    global _LEXICON_CACHE, _LEXICON_MTIME
    try:
        mtime = LEXICON_PATH.stat().st_mtime
    except OSError:
        mtime = None
    if _LEXICON_CACHE is None or mtime != _LEXICON_MTIME:
        _LEXICON_CACHE = load_lexicon()
        _LEXICON_MTIME = mtime
    return _LEXICON_CACHE


def apply_lexicon(text: str, rules=None) -> str:
    for rx, rep in (_lexicon() if rules is None else rules):
        text = rx.sub(rep, text)
    return text

# --- unicode -> ASCII -------------------------------------------------------

_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": " - ", "—": " - ", "―": " - ",
    "…": "...",
    " ": " ", " ": " ", " ": " ", "​": "",
    "·": ".", "•": " ",
    "½": " one half", "¼": " one quarter", "¾": " three quarters",
    "×": " by ", "−": "-",
}


# Non-ASCII characters that still carry meaning and must survive to the number
# expander. Dropping the currency symbols here used to strip the "£" off
# "£999.99" before _expand_numbers ever saw it, so the money rule never fired.
_KEEP_NON_ASCII = "£€¥°"


def _to_ascii_punct(text: str) -> str:
    for bad, good in _PUNCT_MAP.items():
        text = text.replace(bad, good)
    # Anything else non-ASCII: keep letters (accented names), drop stray symbols.
    return "".join(
        ch if ord(ch) < 128 or ch.isalpha() or ch in _KEEP_NON_ASCII else " "
        for ch in text
    )


# --- markdown / layout noise ---------------------------------------------------

_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_MD_EMPH = re.compile(r"(\*\*|__|\*|_|`)")
_MD_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s+")
_MD_QUOTE = re.compile(r"^\s*>\s?")


def _strip_markdown(line: str) -> str:
    line = _MD_HEADING.sub("", line)
    line = _MD_QUOTE.sub("", line)
    line = _MD_BULLET.sub("", line)
    line = _MD_LINK.sub(r"\1", line)
    line = _MD_EMPH.sub("", line)
    return line


# --- numbers -----------------------------------------------------------------

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"),
           (10 ** 3, "thousand"), (100, "hundred")]
_ORDINAL = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


def _int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        head = _TENS[n // 10]
        return head if n % 10 == 0 else f"{head}-{_ONES[n % 10]}"
    for value, name in _SCALES:
        if n >= value:
            head = _int_to_words(n // value)
            rest = n % value
            if not rest:
                return f"{head} {name}"
            joiner = " and " if rest < 100 else " "
            return f"{head} {name}{joiner}{_int_to_words(rest)}"
    return _ONES[n]


def _ordinal_words(n: int) -> str:
    words = _int_to_words(n)
    last = words.split()[-1].split("-")[-1]
    if last in _ORDINAL:
        return words[: len(words) - len(last)] + _ORDINAL[last]
    if last.endswith("y"):
        return words[:-1] + "ieth"
    return words + "th"


def _year_to_words(n: int) -> str:
    """1985 -> nineteen eighty-five; 2007 -> two thousand seven; 1900 -> nineteen hundred."""
    if n < 1100 or n > 2099:
        return _int_to_words(n)
    hi, lo = divmod(n, 100)
    # 2000-2009 first: "two thousand", "two thousand and seven". Checking lo == 0
    # ahead of this used to catch 2000 and read it as "twenty hundred".
    if 2000 <= n <= 2009:
        return _int_to_words(n)
    if lo == 0:
        return f"{_int_to_words(hi)} hundred"
    lo_words = _int_to_words(lo) if lo >= 10 else f"oh {_ONES[lo]}"
    return f"{_int_to_words(hi)} {lo_words}"


def _digits_to_words(s: str) -> str:
    return " ".join("oh" if d == "0" else _ONES[int(d)] for d in s)


# (major singular, major plural, minor singular, minor plural). Naive "+s"
# pluralisation gets "pences" and "rands" wrong, so spell both forms out.
_CURRENCY = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
    "¥": ("yen", "yen", "sen", "sen"),
    "R": ("rand", "rand", "cent", "cents"),
}


def _expand_numbers(text: str) -> str:
    # $1,250.50 / £3.99  -> words
    def money(m: re.Match) -> str:
        sym, whole, frac = m.group(1), m.group(2).replace(",", ""), m.group(3)
        one, many, minor_one, minor_many = _CURRENCY.get(sym, _CURRENCY["$"])
        n = int(whole)
        out = f"{_int_to_words(n)} {one if n == 1 else many}"
        if frac and int(frac.ljust(2, "0")[:2]):
            c = int(frac.ljust(2, "0")[:2])
            out += f" and {_int_to_words(c)} {minor_one if c == 1 else minor_many}"
        return out

    text = re.sub(r"([$£€¥])\s?(\d[\d,]*)(?:\.(\d{1,2}))?", money, text)
    text = re.sub(r"\bR\s?(\d[\d,]*)(?:\.(\d{1,2}))?",
                  lambda m: money(re.match(r"(R)\s?(\d[\d,]*)(?:\.(\d{1,2}))?", m.group(0))), text)

    # 25% -> twenty-five percent
    text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%",
                  lambda m: _decimalish(m.group(1)) + " percent", text)

    # times: 10:30 -> ten thirty ; 9:05 -> nine oh five ; 12:00 -> twelve o'clock
    def clock(m: re.Match) -> str:
        h, mm = int(m.group(1)), int(m.group(2))
        if mm == 0:
            return f"{_int_to_words(h)} o'clock"
        if mm < 10:
            return f"{_int_to_words(h)} oh {_ONES[mm]}"
        return f"{_int_to_words(h)} {_int_to_words(mm)}"

    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", clock, text)

    # decades: 1980s / '90s / 90s -> nineteen eighties / nineties
    def decade(m: re.Match) -> str:
        core = m.group(1)
        if len(core) == 4:
            base = _year_to_words(int(core))
        else:
            base = _TENS[int(core[0])] if core[0] in "23456789" else _int_to_words(int(core))
        return re.sub(r"y$", "ie", base.split()[-1]).replace("-", " ") \
            .join([base[: -len(base.split()[-1])], "s"]) if base.split()[-1].endswith("y") \
            else base + "s"

    text = re.sub(r"\b'?(\d{2}|\d{4})s\b", decade, text)

    # ordinals: 3rd -> third, 21st -> twenty-first
    text = re.sub(r"\b(\d{1,3})(?:st|nd|rd|th)\b",
                  lambda m: _ordinal_words(int(m.group(1))), text)

    # bare 4-digit years in a plausible range
    text = re.sub(r"(?<!\d)\b(1[1-9]\d{2}|20\d{2})\b(?!\d)",
                  lambda m: _year_to_words(int(m.group(1))), text)

    # long digit runs (phone numbers, PINs): read digit by digit
    text = re.sub(r"\b\d{5,}\b", lambda m: _digits_to_words(m.group(0)), text)

    # decimals and plain integers (with thousands separators).
    #
    # The integer lookahead deliberately rejects only a *digit* after an optional
    # dot, not any dot. `(?![\w.])` used to reject a sentence-final full stop too,
    # so "roughly 1,150." backtracked until it matched the bare leading "1" and
    # produced "roughly one,150." -- any number ending a sentence was mangled.
    text = re.sub(r"(?<![\w.])\d[\d,]*\.\d+(?![\w])", lambda m: _decimalish(m.group(0)), text)
    text = re.sub(r"(?<![\w.])(\d[\d,]*)(?!\.?\d)(?!\w)",
                  lambda m: _int_to_words(int(m.group(1).replace(",", ""))), text)
    return text


def spell_out_numbers(text: str) -> str:
    """Numbers, money, times and percentages as words, nothing else touched.
    Used on automatic transcripts: a cloning transcript should read the way
    the clip *sounds* ("twenty-four", not "24"). Codes that mix letters and
    digits (HTD-189, MP3, A4) are left as written: nobody says "HTD one
    hundred and eighty-nine", and the person can't know how it was read."""
    codes: List[str] = []

    def keep(m: re.Match) -> str:
        codes.append(m.group(0))
        # One private-use character per code: no digits for the expander to see.
        return chr(0xE000 + len(codes) - 1)

    masked = re.sub(r"\b(?:[A-Za-z]+\d[A-Za-z0-9]*|[A-Z]{2,}-\d[A-Za-z0-9]*)\b", keep, text)
    out = _expand_numbers(masked)
    return re.sub("[\ue000-\uf8ff]", lambda m: codes[ord(m.group(0)) - 0xE000], out)


def _decimalish(s: str) -> str:
    s = s.replace(",", "")
    if "." not in s:
        return _int_to_words(int(s))
    whole, frac = s.split(".", 1)
    return f"{_int_to_words(int(whole or 0))} point {_digits_to_words(frac)}"


# --- abbreviations ----------------------------------------------------------

_ABBREV = [
    (r"\bMr\.\s", "Mister "), (r"\bMrs\.\s", "Missus "), (r"\bMs\.\s", "Miz "),
    (r"\bDr\.\s", "Doctor "), (r"\bProf\.\s", "Professor "), (r"\bSt\.\s", "Saint "),
    (r"\bMt\.\s", "Mount "), (r"\bvs\.?\s", "versus "), (r"\betc\.", "et cetera"),
    (r"\be\.g\.\s", "for example, "), (r"\bi\.e\.\s", "that is, "),
    (r"\bapprox\.\s", "approximately "), (r"\bDept\.\s", "Department "),
    (r"\bNo\.\s?(?=\d)", "number "), (r"\ba\.m\.\b", "AM"), (r"\bp\.m\.\b", "PM"),
    (r"\bU\.S\.\b", "US"), (r"\bU\.K\.\b", "UK"),
    # These end in a full stop and are routinely followed by a capitalised word
    # ("excl. VAT"), so leaving them unexpanded also creates a false sentence
    # boundary in _SENT_SPLIT -- expanding them fixes the read and the split.
    (r"\bexcl\.\s", "excluding "), (r"\bincl\.\s", "including "),
    (r"\best\.\s", "estimated "), (r"\bcf\.\s", "compare "),
    (r"\bFig\.\s", "Figure "), (r"\bJr\.", "Junior"), (r"\bSr\.", "Senior"),
]

# Only expanded directly after a number, so "5 cm" reads but the word "cm" alone
# (or a name like "Mg") is left untouched. Case-sensitive on purpose: "MB" is a
# megabyte, "Mb" usually is not what the writer meant to say aloud.
_UNITS = [
    ("mm", "millimetres"), ("cm", "centimetres"), ("km", "kilometres"),
    ("kg", "kilograms"), ("mg", "milligrams"), ("ml", "millilitres"),
    ("GB", "gigabytes"), ("MB", "megabytes"), ("TB", "terabytes"), ("KB", "kilobytes"),
    ("kHz", "kilohertz"), ("MHz", "megahertz"), ("GHz", "gigahertz"),
    ("lb", "pounds"), ("oz", "ounces"), ("ft", "feet"), ("hrs", "hours"), ("hr", "hours"),
]

_ROMAN = {
    "I": "one", "II": "two", "III": "three", "IV": "four", "V": "five",
    "VI": "six", "VII": "seven", "VIII": "eight", "IX": "nine", "X": "ten",
}
# Longest-first so "VIII" isn't matched as "V". Only after a label word, so a
# stray "I" or "X" in ordinary prose is never touched.
_ROMAN_RX = re.compile(
    r"\b(Mk|MK|Mark|Type|Phase|Part|Chapter|Volume|Book|Class)\s+"
    r"(VIII|VII|VI|IV|IX|III|II|I|X|V)\b"
)


def _expand_units(text: str) -> str:
    for abbr, full in _UNITS:
        text = re.sub(rf"(?<=\d)\s*{abbr}\b", " " + full, text)
    return text


def _expand_roman(text: str) -> str:
    def rep(m: re.Match) -> str:
        label = "Mark" if m.group(1) in ("Mk", "MK") else m.group(1)
        return f"{label} {_ROMAN[m.group(2)]}"

    return _ROMAN_RX.sub(rep, text)


def _expand_abbrev(text: str) -> str:
    for pat, rep in _ABBREV:
        text = re.sub(pat, rep, text)
    text = re.sub(r"\s&\s", " and ", text)
    # "support@example.com" -> "support at example dot com". Without the second
    # rule the dot survives into normalize_line's "space after punctuation" pass
    # and comes out as "example. com", which the model reads as a sentence end.
    text = re.sub(r"(?<=\w)@(?=\w)", " at ", text)
    text = re.sub(r"(?<=\w)\.(?=(?:com|net|org|io|co|gov|edu|ai)\b)", " dot ", text, flags=re.I)
    text = re.sub(r"\bw/\b", "with ", text)
    # Symbols that survive the ASCII pass but have no spoken form of their own.
    text = re.sub(r"~\s*(?=[\d.])", "about ", text)     # ~40% -> about 40%
    text = re.sub(r"(?<!\w)\+\s?(?=\d)", "plus ", text)  # +1 555... -> plus 1 555...
    text = re.sub(r"#\s?(?=\d)", "number ", text)
    text = _expand_units(text)
    text = _expand_roman(text)
    return text


# --- one line at a time ---------------------------------------------------------

_TERMINAL = tuple('.!?:;,"\')]')


def normalize_line(line: str, lexicon: bool = True) -> str:
    """Clean a single line of text for the model. Safe to call twice.

    `lexicon` applies the user's `lexicon.txt` pronunciation overrides first,
    before abbreviations and numbers are expanded.
    """
    line = _to_ascii_punct(line)
    line = _strip_markdown(line)
    if lexicon:
        line = apply_lexicon(line)
    line = _expand_abbrev(line)
    line = _expand_numbers(line)
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"\s+([.,!?;:])", r"\1", line)      # no space before punctuation
    line = re.sub(r"([.,!?;:])(?=[A-Za-z])", r"\1 ", line)  # space after it
    if line[:1].isalpha():
        line = line[:1].upper() + line[1:]
    if line and not line.endswith(_TERMINAL):
        line += "."
    return line


# --- segmentation ------------------------------------------------------------

# `(?<!\s[A-Z]\.)` keeps an initial from ending a sentence: "Doctor A. Nguyen"
# used to split into "...Doctor A." | "Nguyen joined...", which stranded the
# surname on its own line and made the model read the initial as a full stop.
_SENT_SPLIT = re.compile(r"(?<=[.!?])(?<!\s[A-Z]\.)(?:\"|')?\s+(?=[A-Z0-9\"'(])")
# Strong boundaries first -- a semicolon/colon/dash is nearly always a safe place
# to breathe. Only fall back to commas and conjunctions when there's no strong one.
_STRONG_MARKS = ["; ", ": ", " - "]
_WEAK_MARKS = [", ", " and ", " but ", " so ", " because ", " which ",
               " while ", " although ", " though ", " whereas "]

_NUM_WORDS = set(_ONES) | {t for t in _TENS if t} | {
    "hundred", "thousand", "million", "billion", "and", "point", "oh",
    "dollars", "dollar", "cents", "cent", "percent",
}


def _word_count(s: str) -> int:
    return len(s.split())


def _is_num_edge(text: str, at_end: bool) -> bool:
    parts = text.split()
    if not parts:
        return False
    word = (parts[-1] if at_end else parts[0]).strip(",.-;:").lower()
    word = word.split("-")[-1] if at_end else word.split("-")[0]
    return word in _NUM_WORDS


def _split_long(sentence: str, max_words: int) -> List[str]:
    """Break one over-long sentence at a clause boundary near its middle.

    Strong marks (; : -) are tried first; commas and conjunctions only if there
    is no strong one. A candidate is rejected if either side of it is the middle
    of a spelt-out number ("two hundred | and fifty"), so number expansion never
    gets chopped in half.
    """
    if max_words <= 0 or _word_count(sentence) <= int(max_words * 1.15):
        return [sentence]

    mid = len(sentence) / 2
    for markers in (_STRONG_MARKS, _WEAK_MARKS):
        best = None  # (distance_from_mid, index, marker_len)
        for marker in markers:
            start = 0
            while True:
                i = sentence.find(marker, start)
                if i == -1:
                    break
                left, right = sentence[:i], sentence[i + len(marker):]
                start = i + len(marker)
                if _word_count(left) < 4 or _word_count(right) < 4:
                    continue
                if marker != ", " and (_is_num_edge(left, True) or _is_num_edge(right, False)):
                    continue
                d = abs((i + len(marker) / 2) - mid)
                if best is None or d < best[0]:
                    best = (d, i, len(marker))
        if best is not None:
            _, i, mlen = best
            left = sentence[:i].strip()
            right = sentence[i + mlen:].strip()
            if left and left[-1] not in ".!?,;:-":
                left += ","
            return _split_long(left, max_words) + _split_long(right, max_words)

    return [sentence]


def segment_text(text: str, max_words: int = 24, lexicon: bool = True) -> List[Tuple[str, bool]]:
    """(chunk, starts_new_paragraph) in reading order.

    Hard newlines in the input are respected as breaks; a blank line marks a
    paragraph. Within a block, text is split into sentences and any sentence
    longer than `max_words` is split further at a clause boundary.
    """
    out: List[Tuple[str, bool]] = []
    blocks = re.split(r"\n[ \t]*\n+", (text or "").replace("\r\n", "\n").replace("\r", "\n"))
    for b_idx, block in enumerate(blocks):
        raw_lines = [ln.strip() for ln in block.split("\n")]
        raw_lines = [ln for ln in raw_lines if ln and not ln.startswith("#")]
        if not raw_lines:
            continue
        first_chunk_of_block = True
        for raw in raw_lines:
            norm = normalize_line(raw, lexicon=lexicon)
            if not norm:
                continue
            for sentence in _SENT_SPLIT.split(norm):
                sentence = sentence.strip()
                if not sentence:
                    continue
                for chunk in _split_long(sentence, max_words):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    new_para = first_chunk_of_block and b_idx > 0
                    out.append((chunk, new_para))
                    first_chunk_of_block = False
    return out


# --- public entry point ----------------------------------------------------------

def prepare_script(
    text: str,
    *,
    normalize: bool = True,
    lexicon: bool = True,
    max_words: int = 24,
    sentence_gap_ms: int = 300,
    paragraph_gap_ms: int = 650,
) -> Tuple[List[str], List[int]]:
    """Turn a block of writing into (lines_to_speak, gap_before_each_line_ms).

    `gaps[0]` is always 0. When `normalize` is False the text is still split into
    explicit lines but not rewritten -- use it if the cleanup mangles something;
    `lexicon.txt` pronunciation overrides are still applied unless `lexicon=False`.
    Raises ValueError if there is nothing to say.
    """
    if normalize:
        segments = segment_text(text, max_words=max_words, lexicon=lexicon)
    else:
        # honour explicit line breaks only, like the old behaviour -- but still
        # run the pronunciation lexicon, which is the point of maintaining it.
        raw = [ln.strip() for ln in (text or "").splitlines()]
        raw = [ln for ln in raw if ln and not ln.startswith("#")]
        segments = [((apply_lexicon(ln) if lexicon else ln), False) for ln in raw]

    if not segments:
        raise ValueError(
            "Nothing to say: the script is empty (blank lines and #comments are skipped)."
        )

    lines = [s for s, _ in segments]
    gaps = [0]
    for _, new_para in segments[1:]:
        gaps.append(paragraph_gap_ms if new_para else sentence_gap_ms)
    return lines, gaps


# --- self test ---------------------------------------------------------------

def _selftest() -> int:
    cases = [
        ("Dr. Smith paid $1,250.50 on 3rd March 1985.",
         "Doctor Smith paid one thousand two hundred and fifty dollars and fifty cents "
         "on third March nineteen eighty-five."),
        ("It's 25% off at 10:30 today — no joke.",
         "It's twenty-five percent off at ten thirty today - no joke."),
        ("**Bold** and _italic_ and `code`.", "Bold and italic and code."),
        ("- a bullet point", "A bullet point."),
        ("Call 5551234567 now", "Call five five five one two three four five six seven now."),
        ("The 1990s were fine", "The nineteen nineties were fine."),
        ("no terminal punctuation here", "No terminal punctuation here."),
    ]
    ok = True
    for src, want in cases:
        got = normalize_line(src)
        flag = "ok " if got == want else "BAD"
        if got != want:
            ok = False
        print(f"[{flag}] {src!r}\n       -> {got!r}")
        if got != want:
            print(f"       want {want!r}")

    lines, gaps = prepare_script(
        "First paragraph, one sentence. And a second sentence that keeps going and "
        "going and going and going and going and going until it really must break.\n\n"
        "Second paragraph here.",
        max_words=12,
    )
    print("\nsegmentation:")
    for ln, g in zip(lines, gaps):
        print(f"  ({g:4d} ms) {ln}")
    assert 650 in gaps, "expected a paragraph-sized gap"
    assert len(lines) >= 4, "expected the long sentence to be split"

    lex = [(re.compile(r"\bSQL\b", re.IGNORECASE), "sequel")]
    got = apply_lexicon("The SQL query", rules=lex)
    lex_ok = got == "The sequel query"
    print(f"\nlexicon: {'ok ' if lex_ok else 'BAD'} {got!r}")
    ok = ok and lex_ok

    print("\nself-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
