
import pytest

import textprep
from textprep import (
    apply_lexicon,
    load_lexicon,
    normalize_line,
    prepare_script,
    segment_text,
)


@pytest.mark.parametrize("src, want", [
    ("Dr. Smith paid $1,250.50 on 3rd March 1985.",
     "Doctor Smith paid one thousand two hundred and fifty dollars and fifty cents "
     "on third March nineteen eighty-five."),
    ("It's 25% off at 10:30 today — no joke.",
     "It's twenty-five percent off at ten thirty today - no joke."),
    ("**Bold** and _italic_ and `code`.", "Bold and italic and code."),
    ("- a bullet point", "A bullet point."),
    ("# Heading text", "Heading text."),
    ("Call 5551234567 now", "Call five five five one two three four five six seven now."),
    ("The 1990s were fine", "The nineteen nineties were fine."),
    ("no terminal punctuation here", "No terminal punctuation here."),
    ("Meet at 9:05.", "Meet at nine oh five."),
    ("See [the docs](http://x.y) please.", "See the docs please."),
])
def test_normalize_line(src, want):
    assert normalize_line(src) == want


def test_normalize_is_idempotent():
    for src in ["Dr. Smith paid $4.50.", "The 1980s, 25% off!", "plain sentence"]:
        once = normalize_line(src)
        assert normalize_line(once) == once


def test_segment_splits_long_sentence_without_chopping_numbers():
    text = ("On the third of March nineteen eighty-five, Doctor Adams paid one thousand "
            "two hundred and fifty dollars and fifty cents for a device.")
    chunks = [c for c, _ in segment_text(text, max_words=12)]
    assert len(chunks) >= 2
    # the spelt-out number must survive intact in exactly one chunk
    joined = " ".join(chunks)
    assert "one thousand two hundred and fifty dollars and fifty cents" in joined
    assert not any(c.rstrip(",.").endswith("two hundred") for c in chunks)


def test_prepare_script_paragraph_gaps():
    lines, gaps = prepare_script(
        "First para. Still first para.\n\nSecond para starts here.",
        sentence_gap_ms=300, paragraph_gap_ms=650,
    )
    assert gaps[0] == 0
    assert 650 in gaps            # paragraph boundary
    assert gaps.count(300) >= 1   # sentence boundary
    assert len(lines) == len(gaps)


def test_prepare_script_no_normalize_keeps_raw_lines_but_runs_lexicon(tmp_path, monkeypatch):
    lex = tmp_path / "lexicon.txt"
    lex.write_text("SQL -> sequel\n", encoding="utf-8")
    monkeypatch.setattr(textprep, "LEXICON_PATH", lex)
    textprep._LEXICON_CACHE = None
    textprep._LEXICON_MTIME = None

    lines, _ = prepare_script("run the SQL now 1985", normalize=False)
    assert lines == ["run the sequel now 1985"]   # lexicon applied, numbers NOT expanded


def test_prepare_script_empty_raises():
    with pytest.raises(ValueError):
        prepare_script("   \n # comment only\n")


def test_lexicon_parsing(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text(
        "# a comment\n"
        "SQL -> sequel\n"
        'Qwen -> "chwen"\n'
        "/\\bkm\\b/ -> kilometres\n"
        "bad line without arrow\n",
        encoding="utf-8",
    )
    rules = load_lexicon(p)
    assert len(rules) == 3
    text = "The SQL and Qwen ran 5 km."
    assert apply_lexicon(text, rules) == "The sequel and chwen ran 5 kilometres."


def test_lexicon_missing_file_is_no_op(tmp_path):
    assert load_lexicon(tmp_path / "nope.txt") == []
    assert apply_lexicon("unchanged", []) == "unchanged"


def test_max_words_zero_disables_clause_split():
    long_sentence = "one two three four five six seven eight nine ten " * 4
    chunks = [c for c, _ in segment_text(long_sentence.strip() + ".", max_words=0)]
    assert len(chunks) == 1


# --- regressions found by reading a real script aloud -------------------------
# Each of these produced audibly wrong speech (and in two cases a bad sentence
# split, which costs a whole extra generation) before being fixed.

def test_number_ending_a_sentence_is_not_mangled():
    # "(?![\w.])" rejected the final full stop, so the match backtracked to the
    # bare leading digit and emitted "one,150."
    assert normalize_line("It is roughly 1,150.") == "It is roughly one thousand one hundred and fifty."
    assert normalize_line("We counted 42.") == "We counted forty-two."
    assert normalize_line("Stock is 8,400.") == "Stock is eight thousand four hundred."


def test_bare_four_digit_number_still_reads_as_a_year():
    # Deliberate: a bare 1100-2099 integer is far more often a year than a count,
    # and the comma form ("1,150") is there for when it isn't.
    assert normalize_line("There were 1150 people.") == "There were eleven fifty people."


def test_non_dollar_currency_survives_the_ascii_pass():
    assert normalize_line("It costs £999.99.") == \
        "It costs nine hundred and ninety-nine pounds and ninety-nine pence."
    assert normalize_line("About €1,150.") == "About one thousand one hundred and fifty euros."


def test_currency_minor_units_pluralise_correctly():
    assert "pence" in normalize_line("£5.99") and "pences" not in normalize_line("£5.99")
    assert "one penny" in normalize_line("£5.01")


def test_an_initial_does_not_end_a_sentence():
    chunks = [c for c, _ in segment_text("Dr. A. Nguyen joined the lab.")]
    assert chunks == ["Doctor A. Nguyen joined the lab."]


def test_excl_is_expanded_and_does_not_split_the_sentence():
    chunks = [c for c, _ in segment_text("It costs ten pounds excl. VAT today.")]
    assert len(chunks) == 1
    assert "excluding VAT" in chunks[0]


def test_units_after_a_number_are_spelt_out():
    got = normalize_line("It measures 12.5 cm and weighs 3.4 kg.")
    assert got == "It measures twelve point five centimetres and weighs three point four kilograms."


def test_bare_unit_word_is_left_alone():
    assert normalize_line("The cm scale.") == "The cm scale."


def test_plus_tilde_and_hash_have_spoken_forms():
    assert normalize_line("Call +1 5551234567 now") == \
        "Call plus one five five five one two three four five six seven now."
    assert "about forty percent" in normalize_line("Throughput is up ~40%.")
    assert "number seven" in normalize_line("See #7.")


def test_domain_dot_does_not_become_a_sentence_break():
    assert normalize_line("Email support@example.com today.") == \
        "Email support at example dot com today."


def test_roman_numerals_after_a_label():
    assert normalize_line("The Mk II versus the Mk III.") == "The Mark two versus the Mark three."
    # A stray capital I in prose must not be touched.
    assert normalize_line("I went home.") == "I went home."


def test_year_2000_reads_as_two_thousand():
    # The lo == 0 branch used to fire before the 2000-2009 special case, so the
    # round year came out as "twenty hundred".
    assert normalize_line("It happened in 2000.") == "It happened in two thousand."
    assert normalize_line("The 2000s changed things.") == "The two thousands changed things."
    # The neighbouring cases must not regress.
    assert normalize_line("Back in 1900 it differed.") == "Back in nineteen hundred it differed."
    assert normalize_line("By 2007 it was done.") == "By two thousand and seven it was done."
    assert normalize_line("We shipped in 2020.") == "We shipped in twenty twenty."


def test_scales_reach_trillion():
    from textprep import _int_to_words

    assert _int_to_words(10 ** 12) == "one trillion"
    assert _int_to_words(10 ** 9) == "one billion"
