from __future__ import annotations

from tools.pip_boundary_auc_experiment import (
    boundary_sample_index,
    choose_theta_5,
    normalize_aa,
    pip_real_boundary_sample_index,
)


def test_2m_boundary_conversion_uses_phy_rate() -> None:
    # 2M: 2 preamble + 4 AA + 2 header + 20 payload + 3 CRC = 31 bytes.
    assert boundary_sample_index(1000, 20, 100_000_000, "2m") == 13_400


def test_pip_real_boundary_has_no_second_preamble() -> None:
    # 2 + 4 + 2 + 4 + 2 + 9 + 3 = 26 bytes at 400 samples/byte.
    assert pip_real_boundary_sample_index(1000, 9, 100_000_000, "2m") == 11_400


def test_access_address_normalization_preserves_air_order() -> None:
    assert normalize_aa("0xA058FC36") == "A058FC36"
    assert normalize_aa("A0:58:FC:36") == "A058FC36"


def test_theta_5_does_not_invert_or_flip_scores() -> None:
    # With 20 benign calibration observations, the top observed score gives a
    # 5% operating point; the direction is preserved rather than inverted.
    assert choose_theta_5([1.0] * 19 + [2.0]) == 2.0
