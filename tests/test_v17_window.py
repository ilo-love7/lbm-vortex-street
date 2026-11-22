"""v1.7 多窗长归约单测：只锁纯函数与可解析核对的口径。

不做 50T 生产算例的端到端测试（那要 1 h/档）；生产代码由
``scripts/15_v17_window.py --smoke`` 单独验。这里守的是最容易出错的
**归约口径**：窗切分（含余数）、峰值时刻、零分母、漂移方向、随 Re 趋势、
谱重心的解析值，以及"3D 相对 2D 偏差"的符号。
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
def v17():
    spec = importlib.util.spec_from_file_location(
        "v17", ROOT / "scripts" / "15_v17_window.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_window_bounds_even_split(v17):
    """整分：500 步、每 T 10 步、窗长 10T → 5 个等长半开区间。"""
    b = v17.window_bounds(500, 10.0, 10.0)
    assert b == [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]
    assert sum(e - s for s, e in b) == 500


def test_window_bounds_never_discards_steps(v17):
    """有余数时末窗补齐，既不丢步也不造出超出模拟的步号。"""
    b = v17.window_bounds(523, 10.0, 10.0)
    assert b[-1] == (500, 523)
    assert sum(e - s for s, e in b) == 523
    assert all(e <= 523 for _, e in b)
    assert all(s < e for s, e in b)


def test_window_bounds_single_tiny_run(v17):
    """总步数小于窗长时只给一个窗（smoke 路径）。"""
    assert v17.window_bounds(33, 10.0, 10.0) == [(0, 33)]


def test_window_bounds_merges_one_step_tail(v17):
    """Re=400 实测：39816 步 / 7963 步每窗 ⇒ 不能留下 1 步的尾窗。

    尾窗太短会让 strouhal() 抛 ValueError（v1.7 首跑第 4 档在此失败）；
    并进前一窗后，所有窗都 ≥16 步且总步数不丢。
    """
    b = v17.window_bounds(39816, 796.3, 10.0)
    assert all(e - s >= 16 for s, e in b)
    assert b[-1][1] == 39816
    assert sum(e - s for s, e in b) == 39816
    assert len(b) == 5


def test_peak_timing_returns_time_not_index(v17):
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 9.0, 4.0, 0.0])
    assert v17.peak_timing(t, y) == (9.0, 1.0)


def test_peak_timing_empty_is_nan(v17):
    p, t = v17.peak_timing(np.array([]), np.array([]))
    assert np.isnan(p) and np.isnan(t)


def test_drift_pct_sign_and_zero_denominator(v17):
    assert v17.drift_pct(100.0, 110.0) == pytest.approx(10.0)
    assert v17.drift_pct(100.0, 90.0) == pytest.approx(-10.0)
    assert np.isnan(v17.drift_pct(0.0, 1.0))   # 机器零上不报假涨幅
    assert np.isnan(v17.drift_pct(np.nan, 1.0))


def test_stability_flag_is_a_threshold_helper_not_proof(v17):
    assert v17.stability_flag(100.0, 102.0, 2.0) is True
    assert v17.stability_flag(100.0, 102.01, 2.0) is False
    # nan 不能"碰巧稳定"
    assert v17.stability_flag(np.nan, 100.0) is False


def test_re_trend_detects_monotonicity_and_slope(v17):
    re = [250.0, 300.0, 350.0, 400.0]
    down = v17.re_trend(re, [10.0, 8.0, 6.0, 4.0])
    assert down["monotone_down"] and not down["monotone_up"]
    assert down["slope"] < 0
    assert down["ratio_last_first"] == pytest.approx(0.4)
    mixed = v17.re_trend(re, [10.0, 8.0, 9.0, 4.0])
    assert not mixed["monotone_down"] and not mixed["monotone_up"]
    assert np.isnan(v17.re_trend(re, [np.nan] * 4)["slope"])


def test_spectral_centroid_of_pure_mode_is_exact(v17):
    """单一 m=3：重心必须逐位为 3，λ = Lz/m = 2D。"""
    power = [0.0] * 12
    power[3] = 1.0
    c = v17.spectral_centroid({"power": power}, 96, 16.0)
    assert c["m_centroid"] == pytest.approx(3.0)
    assert c["lambda_centroid_over_d"] == pytest.approx(2.0)
    assert c["m_std"] == pytest.approx(0.0)


def test_spectral_centroid_mixed_moves_between_modes(v17):
    p = [0.0] * 8
    p[2], p[4] = 0.5, 0.5
    c = v17.spectral_centroid({"power": p}, 96, 16.0)
    assert c["m_centroid"] == pytest.approx(3.0)
    assert c["m_std"] == pytest.approx(1.0)


def _fake_rows():
    rows = []
    for i, re in enumerate((250.0, 300.0, 350.0, 400.0)):
        windows = []
        for j in range(5):
            # 只在窗 4（末窗）把 uz 峰值压低：检验 peak_window 读数
            uz = 0.05 + 0.01 * j if j < 4 else 0.02
            windows.append({
                "index": j, "t_start": 10.0 * j, "t_end": 10.0 * (j + 1),
                "cd_mean": 1.6 - 0.01 * i - 0.002 * j,
                "cl_rms": 0.7 - 0.02 * i - 0.005 * j,
                "st": 0.23 - 0.001 * i - 0.002 * j,
                "uz_rel_mean": 0.04 + 0.002 * j,
                "uz_rel_peak": uz, "uz_peak_t": 10.0 * j + 8.0,
                "m_centroid": 2.4 + 0.2 * j,
            })
        rows.append({"re": re, "windows": windows,
                     "full": {"uz_peak_t": 38.0, "uz_rel_peak": 0.05,
                              "cd_mean": 1.5, "cl_rms": 0.6, "st": 0.22}})
    return rows


def test_summarize_anchors_at_20_30t_and_uses_last_window(v17):
    s = v17.summarize(_fake_rows())
    d = s["drift_anchor_to_last"]["250"]
    # 窗2→窗4：cd 1.596→1.592，即 -0.2506%
    assert d["cd_mean"] == pytest.approx(-100 * 0.004 / 1.596, rel=1e-6)
    assert d["cl_rms"] < 0            # 力幅在两窗之间继续下降
    assert d["st"] < 0
    assert d["m_centroid"] > 0        # 谱重心反向：正说明两量不是同一件事


def test_summarize_peak_window_is_argmax_over_windows(v17):
    s = v17.summarize(_fake_rows())
    for re in ("250", "300", "350", "400"):
        assert s["peak_timing"][re]["peak_window"] == 3
    assert s["peak_timing"]["250"]["global_peak_t"] == 38.0


def test_summarize_re_trend_has_five_windows(v17):
    s = v17.summarize(_fake_rows())
    assert len(s["re_trend_by_window"]) == 5
    assert s["re_trend_by_window"][0]["cl_rms"]["monotone_down"] is True
    assert s["re_trend_by_window"][0]["cl_rms"]["slope"] < 0


def test_deviation_sign_3d_lower_than_2d(v17):
    ref = [{"re": re, "cd_mean": 1.7, "cl_rms": 0.8, "st": 0.23}
           for re in (250.0, 300.0, 350.0, 400.0)]
    dev = v17.summarize(_fake_rows(), ref)["deviation_by_window"]
    assert len(dev) == 5
    assert all(x["dcl_rms_pct"] < 0 and x["dcd_pct"] < 0 for x in dev[0])


def test_summarize_without_reference_has_empty_deviation(v17):
    """没有 2D 参考时偏差表必须为空（不是伪造的零偏差）——调用方需显式检查。"""
    assert v17.summarize(_fake_rows())["deviation_by_window"] == []
