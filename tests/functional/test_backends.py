import pytest

from spartorch.functional import backends


def test_resolves_a_preset_name_and_passes_a_record_through(backend):
    preset = backends.resolve(backend)
    assert preset.name == backend
    assert backends.resolve(preset) is preset


def test_unknown_preset():
    with pytest.raises(ValueError, match=r"unknown backend 'nope'; presets: \['csr', 'dense'\]"):
        backends.resolve("nope")
