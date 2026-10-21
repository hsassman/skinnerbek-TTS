"""Voice library paths must stay within their configured directory."""
import pytest

from voice_library import ref_clip_path, voice_path


@pytest.mark.parametrize('name', ['../outside', r'..\outside', '/absolute', r'C:\outside', '', '..', 'bad\0name'])
def test_invalid_names_cannot_escape_voice_library(tmp_path, name):
    for path_for in (voice_path, ref_clip_path):
        with pytest.raises(ValueError):
            path_for(name, tmp_path)


def test_voice_paths_allow_readable_names(tmp_path):
    assert voice_path('Auntie Faiza', tmp_path) == tmp_path / 'Auntie Faiza.pt'
    assert ref_clip_path('cape_flats', tmp_path) == tmp_path / 'cape_flats.ref.wav'


@pytest.mark.parametrize('name', ['nul', 'CON', 'com1', 'Lpt9', 'aux.backup', 'com\u00b9',
                                  'nul .x', 'CONIN$', 'com0'])
def test_windows_device_names_are_rejected(tmp_path, name, monkeypatch):
    import voice_library

    monkeypatch.setattr(voice_library, "_WINDOWS", True)
    with pytest.raises(voice_library.InvalidVoiceName):
        voice_path(name, tmp_path)


def test_device_names_are_fine_off_windows(tmp_path, monkeypatch):
    import voice_library

    monkeypatch.setattr(voice_library, "_WINDOWS", False)
    assert voice_path('aux', tmp_path) == tmp_path / 'aux.pt'


def test_listing_skips_files_the_library_cannot_address(tmp_path, monkeypatch):
    import voice_library
    from voice_library import list_voices

    monkeypatch.setattr(voice_library, "_WINDOWS", True)
    for stem in ('good', 'nul', 'old.'):
        (tmp_path / f'{stem}.pt').write_bytes(b'')
    assert list_voices(tmp_path) == ['good']
