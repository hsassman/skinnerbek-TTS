"""A loaded model keeps its precision: free RAM drops *because* it is loaded,
and re-deciding per take used to reload it as slow emulated bfloat16."""
import types

import torch

import app


class _Fake:
    def __init__(self, dtype):
        self.model = torch.nn.Linear(1, 1).to(dtype)


def _setup(monkeypatch, low):
    loads = []

    def load_model(kind, args):
        loads.append((kind, args.model_size, low["now"]))
        return _Fake(torch.bfloat16 if low["now"] else torch.float32)

    monkeypatch.setattr(app, "load_model", load_model)
    monkeypatch.setattr(app, "auto_low_memory", lambda: low["now"])
    monkeypatch.setattr(app, "_CACHE", {})
    monkeypatch.setattr(app, "_LOADED", {"dtype": None})
    monkeypatch.setattr(app, "CFG", types.SimpleNamespace(
        device="cpu", offline=True, low_memory=False, chunk_size=1, threads=None,
        dtype="auto", mock=False))
    monkeypatch.setattr(app.asr, "unload", lambda: None)
    return loads


def test_loaded_model_survives_ram_getting_tight(monkeypatch):
    low = {"now": False}
    loads = _setup(monkeypatch, low)
    first = app.get_model("Base", "0.6B")
    low["now"] = True                      # the model itself now fills the RAM
    assert app.get_model("Base", "0.6B") is first
    assert len(loads) == 1 and app._LOADED["dtype"] == "float32"


def test_switching_model_reloads_and_records_its_precision(monkeypatch):
    low = {"now": False}
    loads = _setup(monkeypatch, low)
    app.get_model("Base", "0.6B")
    low["now"] = True
    app.get_model("CustomVoice", "0.6B")
    assert [k for k, _s, _l in loads] == ["Base", "CustomVoice"]
    assert list(app._CACHE) == [("CustomVoice", "0.6B")] and app._LOADED["dtype"] == "bfloat16"
