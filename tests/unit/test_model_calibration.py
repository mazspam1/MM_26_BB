"""Unit tests for model calibration persistence."""

from pathlib import Path

from packages.models.calibration import ModelCalibrationParams, load_model_calibration, save_model_calibration


def test_save_and_load_calibration(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    params = ModelCalibrationParams(
        spread_bias=1.2,
        spread_scale=0.95,
        total_bias=-0.8,
        total_scale=1.02,
        base_spread_std=10.5,
        base_total_std=12.3,
        market_anchor_weight_spread=0.15,
        market_anchor_weight_total=0.10,
        n_samples=1234,
        fitted_at="2025-01-01T00:00:00Z",
        source_run_id="bt_20250101_000000",
    )

    save_model_calibration(params, path)
    loaded = load_model_calibration(path)

    assert loaded is not None
    assert loaded.spread_bias == params.spread_bias
    assert loaded.spread_scale == params.spread_scale
    assert loaded.total_bias == params.total_bias
    assert loaded.total_scale == params.total_scale
    assert loaded.base_spread_std == params.base_spread_std
    assert loaded.base_total_std == params.base_total_std
    assert loaded.market_anchor_weight_spread == params.market_anchor_weight_spread
    assert loaded.market_anchor_weight_total == params.market_anchor_weight_total
    assert loaded.n_samples == params.n_samples
    assert loaded.source_run_id == params.source_run_id
