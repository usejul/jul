"""`jul models add`: the choice logic on synthetic vectors, and presets saved as JSON."""

import numpy as np
import pytest

from jul.calibrate import choose, smooth
from jul.presets import (ONE_WORD, QUESTION_OPTIONS, Formulation, Preset, fitted_presets, load_preset,
                         one_word_preset, resolve, save_preset)


def test_smoothing_prefers_a_plateau_to_a_lucky_peak():
    grid = np.array([0.5, 0.5, 0.9, 0.5, 0.5, 0.7, 0.72, 0.7, 0.5])
    assert np.argmax(grid) == 2
    assert np.argmax(smooth(grid)) == 6
    assert smooth(np.ones((4, 4))).shape == (4, 4)


def _pack(rng, layers, good, n=40, k=4, d=16):
    """Texts point to their label only at the layers in `good`; elsewhere they are noise."""
    y = np.arange(n) % k
    labels = {l: rng.normal(size=(k, d)) for l in layers}
    def texts(l):
        signal = labels[l][y] if l in good else 0
        return signal + rng.normal(scale=0.3 if l in good else 3, size=(n, d))
    part = lambda: {"X": {l: texts(l) for l in layers}, "L": labels, "U": {l: texts(l) for l in layers}}
    return {"name": "synthetic", "y": y, "ow": part(), "qo": part()}


def test_choose_finds_the_informative_layers():
    rng = np.random.default_rng(0)
    layers = list(range(10, 20))
    packs = [_pack(rng, layers, good={15, 16, 17}) for _ in range(2)]
    report = choose(packs, {l: np.zeros(16) for l in layers}, layers)
    assert report["layers"]["one_word"] in (15, 16, 17)
    assert report["layers"]["question_options"] in (15, 16, 17)
    assert report["dev_accuracy"] > 0.9
    assert report["tau"] < 0.05          # separable data: a sharp temperature


def _preset(name="my-model", backend="torch", home=None):
    return Preset(name=name, repo="", torch_repo="org/My-Model",
                  formulations=(Formulation("one_word", ONE_WORD, 20),
                                Formulation("question_options", QUESTION_OPTIONS, 21)),
                  tau=0.05, latency_ms="~40", quality="dev 0.6", center="options",
                  one_word=(19, 0.04), backend=backend, calibration={"n_dev": 200})


def test_a_fitted_preset_round_trips_through_json(tmp_path):
    preset = _preset()
    loaded = load_preset(save_preset(preset, tmp_path))
    assert loaded == preset
    assert loaded.repos == {"torch": "org/My-Model"}
    assert loaded.calibration == {"n_dev": 200}


def test_a_fitted_preset_is_resolved_on_its_backend_only(tmp_path):
    save_preset(_preset(), tmp_path)
    assert resolve("my-model", "torch", tmp_path).tau == 0.05
    assert resolve("my-model", None, tmp_path).backend == "torch"
    with pytest.raises(ValueError, match="fitted for torch only"):
        resolve("my-model", "mlx", tmp_path)
    assert one_word_preset("my-model", "torch", tmp_path).formulations[0].layer == 19
    assert [p.name for p in fitted_presets(tmp_path)][:1] == ["my-model"]


def test_a_fitted_preset_overrides_the_built_in_one_on_its_backend(tmp_path):
    save_preset(_preset(name="minicpm5-2b"), tmp_path)
    assert resolve("minicpm5-2b", "torch", tmp_path).tau == 0.05
    assert resolve("minicpm5-2b", "mlx", tmp_path).tau == 0.04133


def test_minicpm_ships_a_preset_fitted_on_torch():
    """Fitted by `jul models add minicpm5-2b --backend torch`."""
    preset = resolve("minicpm5-2b", "torch")
    assert preset.backend == "torch" and preset.center == "generic"
    assert preset.generic_center(preset.formulations[0], "torch") is not None
    assert resolve("minicpm5-2b", "mlx") is resolve("minicpm5-2b")


def test_minicpm5_1b_is_built_in_on_both_backends():
    """Fitted by `jul models add minicpm5-1b` on MLX (built in) and on torch (shipped asset)."""
    mlx, torch = resolve("minicpm5-1b"), resolve("minicpm5-1b", "torch")
    assert mlx.repos == {"mlx": "openbmb/MiniCPM5-1B-MLX", "torch": "openbmb/MiniCPM5-1B"}
    assert torch.backend == "torch" and torch.repos["torch"] == "openbmb/MiniCPM5-1B"
    for preset, backend in ((mlx, "mlx"), (torch, "torch")):
        assert preset.center == "generic"
        assert preset.generic_center(preset.formulations[0], backend) is not None
    assert one_word_preset("minicpm5-1b").formulations[0].layer == 21
