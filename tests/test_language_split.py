"""Clone tab: a bilingual clip's transcript is shown per language, and picking
one language builds the voice from only that language's parts."""
import app
from transcribe import Part, Transcript


def _bilingual():
    parts = [Part("en", 0.0, 4.0, "Good morning, Cape Town."),
             Part("af", 5.0, 9.0, "Goeie more, Kaapstad."),
             Part("en", 10.0, 12.0, "Have a lekker day.")]
    text = " ".join(p.text for p in parts)
    return Transcript(text=text, words=[], language="en", language_prob=1.0, duration=12.0,
                      model="test", parts=parts), text


def test_parts_group_by_language_in_spoken_order():
    t, text = _bilingual()
    parts = app._parts_state(t, text)
    assert list(parts["langs"]) == ["English", "Afrikaans"]
    assert parts["langs"]["English"]["text"] == "Good morning, Cape Town. Have a lekker day."
    assert app.do_show_language("Afrikaans", parts) == "Goeie more, Kaapstad."
    assert app.do_show_language(app.ALL_LANGUAGES, parts) == text


def test_one_language_keeps_only_its_spans():
    t, text = _bilingual()
    parts = app._parts_state(t, text)
    assert app._kept_spans("English", parts) == [(0.0, 4.0), (10.0, 12.0)]
    assert app._kept_spans(app.ALL_LANGUAGES, parts) is None
    assert app._kept_spans("English", None) is None


def test_single_language_clip_hides_the_picker():
    t = Transcript(text="Hello.", words=[], language="en", language_prob=1.0, duration=3.0,
                   model="test", parts=[Part("en", 0.0, 3.0, "Hello.")])
    parts = app._parts_state(t, t.text)
    assert app._show_update(parts)["visible"] is False
    assert app._kept_spans("English", parts) is None
