"""Clone tab: a bilingual clip gets one transcript per language, and picking one
language builds the voice from only that language's parts and words."""
import app
from transcribe import Part, Transcript


def _bilingual():
    parts = [Part("en", 0.0, 4.0, "Good morning, Cape Town."),
             Part("af", 5.0, 9.0, "Goeie more, Kaapstad."),
             Part("en", 10.0, 12.0, "Have a lekker day.")]
    text = " ".join(p.text for p in parts)
    return Transcript(text=text, words=[], language="en", language_prob=1.0, duration=12.0,
                      model="test", parts=parts), text


def test_each_language_gets_its_own_transcript():
    t, text = _bilingual()
    parts = app._parts_state(t, text)
    panel, en, af, use = app._split_updates(parts)
    assert 'data-on="true"' in panel and use["value"] == app.USE_BOTH
    assert en["value"] == "Good morning, Cape Town. Have a lekker day."
    assert af["value"] == "Goeie more, Kaapstad."


def test_the_chosen_language_supplies_words_and_audio():
    t, text = _bilingual()
    parts = app._parts_state(t, text)
    assert app._voice_text("English only", parts, text, "EN edited", "AF edited") == "EN edited"
    assert app._voice_text("Afrikaans only", parts, text, "EN", "AF edited") == "AF edited"
    assert app._voice_text(app.USE_BOTH, parts, text, "EN", "AF") == text
    assert app._kept_spans("English only", parts) == [(0.0, 4.0), (10.0, 12.0)]
    assert app._kept_spans("Afrikaans only", parts) == [(5.0, 9.0)]
    assert app._kept_spans(app.USE_BOTH, parts) is None


def test_single_language_clip_hides_the_panel_and_uses_the_main_box():
    t = Transcript(text="Hello.", words=[], language="en", language_prob=1.0, duration=3.0,
                   model="test", parts=[Part("en", 0.0, 3.0, "Hello.")])
    parts = app._parts_state(t, t.text)
    assert 'data-on="false"' in app._split_updates(parts)[0]
    assert app._voice_text("English only", parts, "Hello.", "", "") == "Hello."
    assert app._kept_spans("English only", parts) is None


def test_a_new_clip_never_keeps_a_saved_voices_name(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "voice_path", lambda n: tmp_path / f"{n}.pt")
    (tmp_path / "glen.pt").write_text("saved")
    (tmp_path / "mandri.pt").write_text("saved")
    # The box still says the voice just made: replaced by the new file's name.
    assert app._name_for_clip("C:/clips/mandri_take.wav", "glen") == "mandri_take"
    # A suggestion that is already taken gets a number.
    assert app._name_for_clip("C:/clips/mandri.wav", "glen") == "mandri_2"
    # A recording has no useful file name: the box is cleared.
    assert app._name_for_clip("C:/tmp/recording.wav", "glen") == ""
    # A new name typed for this clip is kept.
    assert app._name_for_clip("C:/clips/mandri_take.wav", "Auntie Faiza") == "Auntie Faiza"


def test_a_new_clip_resets_the_previous_voice():
    coach, create, status, saved, note, try_btn, *_ = app._clip_changed("C:/clips/new.wav")
    assert saved is None and status == "" and try_btn["interactive"] is False
    assert create["interactive"] is False and 'data-wait="#tts-save-card' in coach


def test_missing_models_are_labelled_and_refused(monkeypatch):
    monkeypatch.setattr(app, "_INSTALLED", {("Base", "0.6B"): True, ("Base", "1.7B"): False})
    assert app.size_choices("Base") == [("0.6B · installed", "0.6B"), ("1.7B · not installed", "1.7B")]
    assert "tts-mchip--off" in app.model_chip("Base", "1.7B")
    assert "isn't on this computer" in app.missing_model_text("Base", "1.7B")
