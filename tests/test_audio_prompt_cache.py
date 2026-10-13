import audio_prompt_cache as apc
from audio_prompt_cache import cache_key, encode_prompt


def test_cache_key_is_stable_and_sensitive(tmp_path):
    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"RIFF....fake audio....")

    k1 = cache_key(str(clip), "hello world", False, "0.6B")
    k2 = cache_key(str(clip), "hello world", False, "0.6B")
    assert k1 == k2 and len(k1) == 32

    # every input that changes the encoding changes the key
    assert k1 != cache_key(str(clip), "different text", False, "0.6B")
    assert k1 != cache_key(str(clip), "hello world", True, "0.6B")
    assert k1 != cache_key(str(clip), "hello world", False, "1.7B")

    clip.write_bytes(b"RIFF....DIFFERENT audio....")
    assert k1 != cache_key(str(clip), "hello world", False, "0.6B")


class FakeModel:
    def __init__(self):
        self.encode_calls = 0

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        self.encode_calls += 1
        return [{"ref_text": ref_text, "tag": "ENCODED", "x_vector_only_mode": x_vector_only_mode}]


def test_encode_prompt_misses_then_hits(tmp_path, monkeypatch):
    # keep the roundtrip trivial -- we're testing the cache, not the serialiser
    monkeypatch.setattr(apc, "payload_from_items", lambda items: {"items": items})
    monkeypatch.setattr(apc, "rebuild_items", lambda payload: payload["items"])

    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"fake audio bytes")
    cache_dir = tmp_path / "cache"
    model = FakeModel()

    items1, cached1 = encode_prompt(model, str(clip), "a transcript", False, "0.6B",
                                    cache_dir=cache_dir, verbose=False)
    assert cached1 is False
    assert model.encode_calls == 1
    assert items1[0]["tag"] == "ENCODED"

    items2, cached2 = encode_prompt(model, str(clip), "a transcript", False, "0.6B",
                                    cache_dir=cache_dir, verbose=False)
    assert cached2 is True
    assert model.encode_calls == 1          # not re-encoded
    assert items2[0]["ref_text"] == "a transcript"


def test_encode_prompt_use_cache_false_always_encodes(tmp_path, monkeypatch):
    monkeypatch.setattr(apc, "payload_from_items", lambda items: {"items": items})
    monkeypatch.setattr(apc, "rebuild_items", lambda payload: payload["items"])
    clip = tmp_path / "r.wav"
    clip.write_bytes(b"x")
    model = FakeModel()

    encode_prompt(model, str(clip), "t", False, "0.6B",
                  use_cache=False, cache_dir=tmp_path / "c", verbose=False)
    encode_prompt(model, str(clip), "t", False, "0.6B",
                  use_cache=False, cache_dir=tmp_path / "c", verbose=False)
    assert model.encode_calls == 2
