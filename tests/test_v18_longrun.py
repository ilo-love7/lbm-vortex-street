"""v1.8 超长窗单测：只锁纯归约函数与交叉核对口径。

100T 生产算例约 160 min/档，不进 pytest；生产脚本用 ``--smoke`` 单独验。
这里守的是最容易写错的三件事：尾窗斜率/漂移的解析值、regime 双读数判据、
以及"前 40T 必须逐位相同 / 40–50T 因尾窗并容许 1–2 步差"的核对口径。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def v18():
    spec = importlib.util.spec_from_file_location(
        "v18_longrun", ROOT / "scripts" / "16_v18_longrun.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _windows(uz, *, cd=1.6, cl=0.6, st=0.225, mc=3.2):
    return [{"index": j, "t_start": 10.0 * j, "t_end": 10.0 * (j + 1),
             "cd_mean": cd, "cl_rms": cl, "st": st,
             "uz_rel_mean": u, "uz_rel_peak": 1.1 * u,
             "uz_peak_t": 10.0 * j + 8.0, "m_centroid": mc}
            for j, u in enumerate(uz)]


def _row(uz, peak_t=48.0, peak_val=None):
    w = _windows(uz)
    pv = peak_val if peak_val is not None else max(uz)
    return {"re": 350.0, "windows": w,
            "full": {"uz_peak_t": peak_t, "uz_rel_peak": pv,
                     "cd_mean": 1.6, "cl_rms": 0.6, "st": 0.225}}


def test_tail_stats_linear_sequence_is_exact(v18):
    """末 3 窗 0.10/0.08/0.06：斜率 -0.02/窗、漂移 -40%、均值 0.08。"""
    s = v18.tail_stats(_windows([0.2, 0.15, 0.10, 0.08, 0.06]),
                       "uz_rel_mean", 3)
    assert s["n"] == 3
    assert s["slope_per_window"] == pytest.approx(-0.02, rel=1e-9)
    assert s["drift_pct"] == pytest.approx(-40.0)
    assert s["mean"] == pytest.approx(0.08)
    assert s["first"] == pytest.approx(0.10)
    assert s["last"] == pytest.approx(0.06)


def test_tail_stats_single_finite_point_has_nan_slope(v18):
    """点数不足时给 nan，不能把 1 点拟合成 0 斜率（假"稳定"）。"""
    w = _windows([0.1])
    s = v18.tail_stats(w, "uz_rel_mean", 3)
    assert s["n"] == 1
    assert np.isnan(s["slope_per_window"]) and np.isnan(s["drift_pct"])
    assert s["first"] == pytest.approx(0.1)


def test_tail_stats_ignores_nan_windows(v18):
    w = _windows([0.1, 0.2, 0.3, 0.4])
    w[-1]["uz_rel_mean"] = float("nan")
    s = v18.tail_stats(w, "uz_rel_mean", 3)
    assert s["n"] == 2
    assert s["slope_per_window"] == pytest.approx(0.1)
    assert np.isnan(s["drift_pct"]) or s["drift_pct"] > 0  # 0.2→0.3


def test_regime_settled_low_needs_both_low_ratio_and_flat_tail(v18):
    r = v18.regime_label(_row([0.09, 0.07, 0.05, 0.04, 0.04, 0.04],
                              peak_val=0.10))
    assert r["label"] == "settled_low"
    assert r["last_over_peak"] == pytest.approx(0.4)


def test_regime_persistent_high_is_not_mistaken_for_settled(v18):
    r = v18.regime_label(_row([0.08, 0.09, 0.088, 0.09, 0.087, 0.089],
                              peak_val=0.10))
    assert r["label"] == "persistent_high"
    assert r["last_over_peak"] > 0.8


def test_regime_still_growing_detected_by_tail_slope(v18):
    r = v18.regime_label(_row([0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
                              peak_val=0.10))
    assert r["label"] == "still_growing"
    assert r["tail_slope_per_window"] > 0


def test_regime_unresolved_when_peak_is_zero(v18):
    r = v18.regime_label(_row([0.04] * 6, peak_val=0.0))
    assert r["label"] == "unresolved"
    assert np.isnan(r["last_over_peak"])


def test_split_crosscheck_first4_bitwise_but_window4_tolerant(v18):
    new = _row([0.09, 0.08, 0.07, 0.06, 0.05] + [0.04] * 5)
    old = json_like = {
        "windows": [dict(w) for w in new["windows"][:5]],
        "full": {"cd_mean": 1.6, "cl_rms": 0.6, "st": 0.225}}
    # 模拟 v1.7 窗4 多 1 步：把它的 cl_rms 轻微改动
    old["windows"][4]["cl_rms"] = 0.601
    old["full"]["cl_rms"] = 0.601
    ck = v18.split_crosscheck(new, old)
    assert ck["cd_mean"]["bitwise_identical"] is True
    assert ck["cd_mean"]["first4_max_abs_diff"] == 0.0
    assert ck["cd_mean"]["window4_abs_diff"] == 0.0  # cd 未改动
    assert ck["cl_rms"]["bitwise_identical"] is True   # 只看前 4 窗
    assert ck["cl_rms"]["window4_abs_diff"] == pytest.approx(0.001)
    assert ck["full50_cl_rms"]["rel_pct"] == pytest.approx(100 / 0.601 * 0.001)
    assert json_like is old  # 纯函数不修改入参


def test_split_crosscheck_rejects_drift_in_first4(v18):
    new = _row([0.09] * 10)
    old = {"windows": [dict(w) for w in new["windows"][:5]],
           "full": {"cd_mean": 1.6, "cl_rms": 0.6, "st": 0.225}}
    old["windows"][1]["st"] = 0.226
    ck = v18.split_crosscheck(new, old)
    assert ck["st"]["bitwise_identical"] is False
    assert ck["st"]["first4_max_abs_diff"] == pytest.approx(0.001)


def test_block_shift_sign_and_requires_ten_windows(v18):
    row = _row([0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
                0.045, 0.043, 0.020, 0.019])
    s = v18.block_shift(row)
    assert s["uz_rel_mean"] < 0        # 80–100T 比 60–80T 低
    assert len(s) == 5
    assert v18.block_shift({"windows": row["windows"][:6]}) == {}


def test_deviation_sign_and_missing_reference(v18):
    row = _row([0.09] * 10)
    dev = v18.deviation_by_window([row], [{"re": 350.0, "cd_mean": 1.7,
                                           "cl_rms": 0.8, "st": 0.23}])
    assert len(dev) == 10
    assert all(d["dcl_rms_pct"] < 0 and d["dcd_pct"] < 0 and d["dst_pct"] < 0
               for d in dev)
    assert v18.deviation_by_window([row], []) == []


def test_make_summary_has_all_readouts(v18):
    rows = [_row([0.09, 0.08, 0.07, 0.06, 0.05] + [0.04] * 5)]
    old = [{"re": 350.0,
            "windows": [dict(w) for w in rows[0]["windows"][:5]],
            "full": {"cd_mean": 1.6, "cl_rms": 0.6, "st": 0.225}}]
    ref = [{"re": 350.0, "cd_mean": 1.7, "cl_rms": 0.8, "st": 0.23}]
    s = v18.make_summary(rows, old, ref)
    assert set(("regime", "tail_stats", "crosscheck_v17",
                "block_shift_60_80_vs_80_100", "peak_timing",
                "deviation_by_window")) <= set(s)
    assert s["regime"]["350"]["label"] in (
        "settled_low", "settling", "persistent_high", "still_growing",
        "decaying", "growing", "flat_mid", "unresolved")
    assert len(s["peak_timing"]["350"]["window_mean"]) == 10
    assert s["crosscheck_v17"]["350"]["cd_mean"]["bitwise_identical"] is True
