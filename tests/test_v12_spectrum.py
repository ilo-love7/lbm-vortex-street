"""v1.2 单测：探针频谱 / 对流时滞 / 谐波比。"""
import numpy as np
import pytest

from lbm.spectrum import convection_delay, dual_st, harmonic_ratio, probe_st


def _sine(n, st, d, u_in, dt=1.0, noise=0.0, seed=0):
    t = np.arange(n) * dt
    f = st * u_in / d
    x = np.sin(2 * np.pi * f * t)
    if noise:
        rng = np.random.default_rng(seed)
        x = x + noise * rng.standard_normal(n)
    return t, x


def test_probe_st_recovers_shedding():
    _, x = _sine(4000, st=0.2, d=28.0, u_in=0.1, noise=0.01)
    out = probe_st(x, d=28.0, u_in=0.1)
    assert abs(out["st"] - 0.2) < 0.005


def test_dual_st_agreement():
    _, a = _sine(4000, st=0.18, d=28.0, u_in=0.1, noise=0.01, seed=1)
    _, b = _sine(4000, st=0.18, d=28.0, u_in=0.1, noise=0.01, seed=2)
    out = dual_st(a, b, d=28.0, u_in=0.1)
    assert out["rel_diff"] < 0.03


def test_convection_delay_known_shift():
    # b 是 a 平移 40 步（延迟） → U_c = dx / 40
    n, dx, dt = 8000, 112.0, 1.0
    _, a = _sine(n, st=0.2, d=28.0, u_in=0.1, noise=0.005, seed=3)
    delay_true = 40
    b = np.roll(a, delay_true)
    out = convection_delay(a, b, d=28.0, u_in=0.1, dx=dx, dt=dt)
    assert abs(out["delay_steps"] - delay_true) < 2.0
    assert abs(out["u_conv"] - dx / delay_true) < 0.1 * dx / delay_true


def test_harmonic_ratio():
    n = 4000
    t = np.arange(n, dtype=float)
    f0 = 0.2 * 0.1 / 28.0
    x = np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
    out = harmonic_ratio(x, d=28.0, u_in=0.1)
    assert abs(out["ratio"] - 0.25) < 0.02


def test_harmonic_ratio_offbin_neighborhood():
    """谐波频率偏离整数 bin 时，单 bin 取样会漏检，邻域峰值口径应找回。"""
    n = 40000
    t = np.arange(n, dtype=float)
    f0 = 28.57 / n                    # 基波峰 ≈ bin 29
    f2 = 60.5 / n                     # 二次谐波峰 ≈ bin 60/61（非 2×idx）
    x = np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * f2 * t)
    out = harmonic_ratio(x, d=28.0, u_in=0.1)
    assert 0.18 < out["ratio"] < 0.32
    assert out["harm_idx"] != 2 * out["fund_idx"]
    # 旧口径（精确 2×idx 单 bin）会明显低估
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n)))
    assert spec[2 * out["fund_idx"]] / spec[out["fund_idx"]] < 0.15


def test_too_short_raises():
    with pytest.raises(ValueError):
        probe_st(np.ones(5), d=28.0, u_in=0.1)
