"""v1.9 强迫/边界消融单测：锁纯归约判据与随机种子可复现性。

100T 消融档约 120 min/档，不进 pytest；生产脚本用 ``--smoke`` 单独验。
这里守的是四类易错口径：None→nan（短窗缺谱量）、指数增长率与幅值无关、
"幅值不敏感 + 峰值移动 + 零扰动二维"三读数的联合判据、以及默认种子必须
逐位复现历史初场。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbm.solver3d import Case3D, Solver3D  # noqa: E402


@pytest.fixture(scope="module")
def v19():
    spec = importlib.util.spec_from_file_location(
        "v19_forcing", ROOT / "scripts" / "18_v19_forcing.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _row(uz, *, cd=1.6, cl=0.6, st=0.225, peak_t=48.0, p3d=0.08):
    w = [{"index": j, "t_start": 10.0 * j, "t_end": 10.0 * (j + 1),
          "cd_mean": cd, "cl_rms": cl, "st": st,
          "uz_rel_mean": u, "uz_rel_peak": 1.1 * u,
          "uz_peak_t": 10.0 * j + 8.0, "m_centroid": 3.2, "p3d_rel": p3d}
         for j, u in enumerate(uz)]
    return {"windows": w, "full": {"uz_peak_t": peak_t,
                                   "uz_rel_peak": max(uz),
                                   "cd_mean": cd, "cl_rms": cl, "st": st}}


# ---------------------------------------------------------------- 基础读数
def test_f_maps_none_and_bad_values_to_nan(v19):
    assert np.isnan(v19._f(None))
    assert np.isnan(v19._f(float("inf")))
    assert v19._f("0.25") == pytest.approx(0.25)


def test_late_stats_uses_only_last_n_and_handles_missing_spectrum(v19):
    r = _row([0.1, 0.2, 0.3, 0.09, 0.10, 0.11])
    r["windows"][-1]["m_centroid"] = None
    s = v19.late_stats(r, n=3)
    assert s["n"] == 3
    assert s["uz_rel_mean_mean"] == pytest.approx(0.10)
    assert s["cd_mean_mean"] == pytest.approx(1.6)
    assert s["t_start"] == pytest.approx(30.0)
    assert s["m_centroid_mean"] == pytest.approx(3.2)  # None 不计入
    assert np.isnan(v19.late_stats(r, n=1)["uz_rel_mean_slope"])  # 单点不拟合


def test_growth_rate_is_exponential_not_amplitude_dependent(v19):
    """两条相差 100 倍的指数曲线必须给出同一增长率。"""
    t = np.arange(0.0, 26.0, 0.5)
    for amp in (1e-6, 1e-4):
        y = amp * np.exp(0.06 * t)
        s = v19.growth_rate({"uz_series": {"t_over_T": list(t),
                                            "uz_rel": list(y)}})
        assert s["rate_per_T"] == pytest.approx(0.06, rel=1e-9)
        assert s["doubling_T"] == pytest.approx(np.log(2) / 0.06, rel=1e-9)


def test_growth_rate_of_zero_field_is_nan_not_negative(v19):
    t = list(np.arange(0.0, 26.0, 0.5))
    s = v19.growth_rate({"uz_series": {"t_over_T": t,
                                        "uz_rel": [0.0] * len(t)}})
    assert np.isnan(s["rate_per_T"]) and np.isnan(s["doubling_T"])
    assert s["n_points"] == 0


# ---------------------------------------------------------------- 消融判据
def test_e_fold_delay_is_log_amplitude_over_growth(v19):
    t = list(np.arange(0.0, 26.0, 0.5))
    y = [1e-4 * np.exp(0.06 * x) for x in t]
    row = {"uz_series": {"t_over_T": t, "uz_rel": y}}
    d = v19.e_fold_delay(row, 10.0)
    assert d["delay_t"] == pytest.approx(np.log(10) / 0.06, rel=1e-9)
    assert np.isnan(v19.e_fold_delay(row, -1.0)["delay_t"])


def test_delayed_ignition_is_not_mislabelled_as_forced(v19):
    """小扰动档在窗内未追平，但峰值平移与 ln(10)/g 一致 ⇒ 不是幅度依赖。"""
    g = 0.06
    pred = np.log(10) / g
    cmp = {"trip_lo": {"uz_rel_mean_ratio": 0.5, "peak_shift_t": pred}}
    r = v19.attribution_verdict(cmp, {"is_2d": True},
                                {"trip_lo": {"delay_t": pred}})
    assert r["e_fold_delay_consistent"] is True
    assert r["label"] == "delayed_ignition_not_forced"
    bad = v19.attribution_verdict(
        {"trip_lo": {"uz_rel_mean_ratio": 0.5, "peak_shift_t": 5.0}},
        {"is_2d": True}, {"trip_lo": {"delay_t": pred}})
    assert bad["label"] == "unresolved"  # 单点比值 1.0 不足以判"幅度依赖"


def test_zero_trip_control_requires_two_readouts(v19):
    r = _row([1e-12, 2e-12, 1e-12], p3d=1e-12)
    assert v19.is_2d_control(r)["is_2d"] is True
    r2 = _row([1e-12, 2e-12, 1e-12], p3d=0.05)  # uz 机器零但展向有结构
    assert v19.is_2d_control(r2)["is_2d"] is False


def test_compare_to_base_ratios_and_peak_shift(v19):
    rows = {"base": _row([0.08] * 10, peak_t=50.0),
            "trip_lo": _row([0.08] * 10, peak_t=70.0, cd=1.55)}
    c = v19.compare_to_base(rows)
    assert c["trip_lo"]["uz_rel_mean_ratio"] == pytest.approx(1.0)
    assert c["trip_lo"]["cd_mean_ratio"] == pytest.approx(1.55 / 1.6)
    assert c["trip_lo"]["peak_shift_t"] == pytest.approx(20.0)
    with pytest.raises(KeyError):
        v19.compare_to_base({"trip_lo": rows["trip_lo"]})


def test_verdict_self_sustained_needs_all_three_readouts(v19):
    cmp = {"trip_lo": {"uz_rel_mean_ratio": 0.95, "peak_shift_t": 20.0},
           "trip_hi": {"uz_rel_mean_ratio": 1.04, "peak_shift_t": -10.0}}
    good = v19.attribution_verdict(cmp, {"is_2d": True})
    assert good["label"] == "self_sustained_attractor"
    assert good["amp_insensitive"] and good["phase_shifted"]
    assert v19.attribution_verdict(cmp, None)["label"] == "unresolved"
    small_shift = v19.attribution_verdict(
        {"trip_lo": {"uz_rel_mean_ratio": 0.95, "peak_shift_t": 1.0}},
        {"is_2d": True})
    assert small_shift["label"] == "same_state_no_phase_resolution"


def test_zero_trip_control_does_not_poison_amplitude_verdict(v19):
    """trip=0 的机器零 uz 不是"1e-16 倍幅度"，不得进幅值比判据。"""
    cmp = v19.compare_to_base({
        "base": _row([0.08] * 10),
        "trip_lo": _row([0.076] * 10),
        "notrip": _row([1e-16] * 10, p3d=0.0)})
    assert cmp["notrip"]["in_amplitude_verdict"] is False
    assert cmp["trip_lo"]["in_amplitude_verdict"] is True
    r = v19.attribution_verdict(cmp, {"is_2d": True})
    assert r["amp_insensitive"] is True
    assert r["label"] in ("self_sustained_attractor",
                          "same_state_no_phase_resolution")


def test_verdict_amplitude_dependent_when_state_scales_with_trip(v19):
    cmp = {"trip_lo": {"uz_rel_mean_ratio": 0.4, "peak_shift_t": 20.0},
           "trip_hi": {"uz_rel_mean_ratio": 1.6, "peak_shift_t": -5.0}}
    r = v19.attribution_verdict(cmp, {"is_2d": True})
    assert r["label"] == "amplitude_dependent"
    assert r["amp_ratio_range"] == pytest.approx([0.4, 1.6])


def test_baseline_crosscheck_flags_any_window_mismatch(v19):
    new = _row([0.1] * 10)
    old = _row([0.1] * 10)
    ck = v19.baseline_crosscheck(new, old)
    assert all(v["identical"] for k, v in ck.items() if "identical" in v)
    old["windows"][3]["cd_mean"] = 1.7
    ck = v19.baseline_crosscheck(new, old)
    assert ck["cd_mean"]["identical"] is False
    assert ck["cd_mean"]["max_abs_diff"] == pytest.approx(0.1)
    assert ck["full_cd_mean"]["abs_diff"] == pytest.approx(0.0)  # full 未改


# ---------------------------------------------------------------- 求解器开关
def test_default_seed_reproduces_legacy_initial_field():
    a = Solver3D(Case3D(nx=32, ny=16, nz=8, d=0.0, tau_direct=0.6, trip=1e-3,
                        trip_seed=42, inlet="periodic", outlet="periodic"))
    b = Solver3D(Case3D(nx=32, ny=16, nz=8, d=0.0, tau_direct=0.6, trip=1e-3,
                        inlet="periodic", outlet="periodic"))
    assert np.array_equal(a.f, b.f)


def test_custom_seed_changes_initial_field_but_not_magnitude():
    a = Solver3D(Case3D(nx=32, ny=16, nz=8, d=0.0, tau_direct=0.6, trip=1e-3,
                        trip_seed=42, inlet="periodic", outlet="periodic"))
    b = Solver3D(Case3D(nx=32, ny=16, nz=8, d=0.0, tau_direct=0.6, trip=1e-3,
                        trip_seed=43, inlet="periodic", outlet="periodic"))
    c = Solver3D(Case3D(nx=32, ny=16, nz=8, d=0.0, tau_direct=0.6, trip=0.0,
                        inlet="periodic", outlet="periodic"))
    assert not np.array_equal(a.f, b.f)
    rho, ux, uy, uz = c.macros()
    # 平衡态反解宏观量有 ~3e-18 舍入，不能要求逐位 == 0
    assert np.max(np.abs(uy)) < 1e-14 and np.max(np.abs(uz)) < 1e-14
    assert np.isfinite(a.f).all() and a.f.shape == b.f.shape
