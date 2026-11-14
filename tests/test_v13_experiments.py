"""v1.3 后处理纯函数 + 汇总写盘单测（不跑 CFD）。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbm.v13 import (blockage_ratio, contour_consistency, fit_extrapolate_zero,
                     method_gap, nu_ratio_summary, onset_time, period_average,
                     st_residual)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "deep", ROOT / "scripts" / "06_v13_deepening.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------- 基础口径 ----------
def test_blockage_ratio():
    assert blockage_ratio(20, 400) == 0.05
    assert blockage_ratio(28, 280) == 0.10
    assert blockage_ratio(20, 100) == 0.20


def test_period_average_removes_phase_residual():
    T = 100
    n = np.arange(550)
    x = np.sin(2 * np.pi * n / T)
    r = period_average(x, T)
    assert r["periods"] == 5
    assert abs(r["mean"]) < 1e-12           # 整 5 周期严格为 0
    assert abs(r["full_mean"]) > 1e-3       # 全段（5.5 周期）残留相位
    assert abs(r["phase_residual"]) > 1e-3


def test_period_average_falls_back_on_short_series():
    x = np.array([1.0, 2.0, 3.0])
    r = period_average(x, 100)
    assert r["periods"] == 0
    assert r["mean"] == r["full_mean"] == 2.0
    assert r["phase_residual"] == 0.0


def test_fit_extrapolate_zero_linear_exact():
    beta = [0.20, 0.10, 0.05]
    st = [0.19 + 0.05 * b for b in beta]
    f = fit_extrapolate_zero(beta, st, 1)
    assert f["intercept"] == pytest.approx(0.19, abs=1e-12)
    assert f["slope"] == pytest.approx(0.05, abs=1e-12)
    assert f["r2"] == pytest.approx(1.0, abs=1e-12)
    assert f["rmse"] < 1e-12
    assert f["beta_range"] == [0.05, 0.20]


def test_fit_extrapolate_zero_quadratic_and_validation():
    beta = [0.20, 0.14, 0.10, 0.067, 0.05]
    st = [0.177 + 0.04 * b + 0.3 * b * b for b in beta]
    q = fit_extrapolate_zero(beta, st, 2)
    assert q["intercept"] == pytest.approx(0.177, abs=1e-10)
    lin = fit_extrapolate_zero(beta, st, 1)
    assert lin["r2"] < q["r2"] <= 1.0 + 1e-12
    with pytest.raises(ValueError):
        fit_extrapolate_zero([0.1], [0.18])
    with pytest.raises(ValueError):
        fit_extrapolate_zero([0.1, 0.2], [0.18])


def test_contour_consistency_and_method_gap():
    assert contour_consistency({"18": 1.5, "22": 1.5, "28": 1.5})["spread"] == 0.0
    r = contour_consistency({"18": 1.50, "22": 1.53, "28": 1.47})
    assert r["n"] == 3 and r["rel_spread"] == pytest.approx(0.06 / 1.5)
    assert np.isnan(contour_consistency({})["spread"])
    g = method_gap(1.53, 1.50)
    assert g["delta"] == pytest.approx(0.03)
    assert g["rel_delta"] == pytest.approx(0.02)


def test_onset_time_detects_switch():
    T = 100
    x = np.r_[np.zeros(300), np.sin(2 * np.pi * np.arange(700) / T)]
    r = onset_time(x, T)
    assert 270 < r["onset_steps"] < 315
    assert r["onset_periods"] == pytest.approx(r["onset_steps"] / T)
    assert r["sat_rms"] > 0.6
    short = onset_time(np.ones(5), T)
    assert short["onset_steps"] == 0


def test_st_residual_and_nu_ratio_summary():
    s = st_residual(0.19, 0.18)
    assert s["abs_err"] == pytest.approx(0.01)
    assert s["rel_err"] == pytest.approx(0.01 / 0.18)
    r = nu_ratio_summary([0.1, 0.2, 0.9, np.inf, np.nan])
    assert r["median"] == pytest.approx(0.2)
    assert r["max"] == pytest.approx(0.9)
    assert np.isnan(nu_ratio_summary([])["median"])


# ---------- 汇总写盘（用 tmp_path，不污染仓库） ----------
@pytest.fixture
def synth():
    blk = []
    for re, nys in ((100.0, (100, 140, 200, 300, 400)),
                    (250.0, (100, 200, 400))):
        for ny in nys:
            b = blockage_ratio(20, ny)
            blk.append({"re": re, "ny": ny, "beta": b,
                        "st": 0.18 + 0.05 * b, "st_lit": 0.1665,
                        "cd_mean": 1.4 + 0.8 * b, "diverged": False})
    cv = []
    for re in (100.0, 300.0):
        for coll in ("bgk", "mrt"):
            mem = 1.50 + (0.02 if coll == "mrt" else 0)
            link = {"18": mem, "22": mem, "28": mem}  # 恒等式：逐位复现 MEM
            wake = {"3.0": mem - 0.01, "6.0": mem, "10.0": mem + 0.01}
            cv.append({"re": re, "collision": coll,
                       "cd_mem_period": mem,
                       "cd_cv": {"18": 1.49, "22": 1.50, "28": 1.51},
                       "cd_link": link,
                       "cd_link_raw": {"18": 1.48, "22": 1.47, "28": 1.46},
                       "cd_wake": wake,
                       "link_vs_mem_maxdiff": 0.0,
                       "link_contour": contour_consistency(link),
                       "wake": contour_consistency(wake),
                       "gap_wake": method_gap(mem, mem),
                       "st": 0.18, "gap": method_gap(mem, 1.50),
                       "contour": contour_consistency({"18": 1.49,
                                                       "22": 1.50,
                                                       "28": 1.51}),
                       "diverged": False})
    les = [{"re": re, "cs": cs, "st": 0.19, "cd_mean": 1.4,
            "cl_peak_amp": 0.6, "nu_ratio_median": 0.1 * cs,
            "diverged": False}
           for re in (200.0, 400.0, 800.0) for cs in (0.0, 0.16, 0.30)]
    inlet = [{"re": 200.0, "eps": e, "st": 0.19, "st_lit": 0.1894,
              "cd_mean": 1.4, "onset_periods": 3.0 - 5 * e,
              "diverged": False}
             for e in (0.0, 0.05, 0.15, 0.30)]
    return blk, cv, les, inlet


def test_summary_writers_produce_valid_json(tmp_path, synth, monkeypatch):
    m = _load_script()
    monkeypatch.setattr(m, "RESULTS", tmp_path)
    blk_rows, cv_rows, les_rows, inlet_rows = synth
    blk = m.write_blockage(blk_rows)
    cv = m.write_cv(cv_rows)
    les = m.write_les_inlet(les_rows, inlet_rows)
    rows = m.write_rows_csv(blk, cv, les)

    assert (tmp_path / "v13_blockage.json").exists()
    assert (tmp_path / "v13_cv_force.json").exists()
    assert (tmp_path / "v13_les_inlet.json").exists()
    assert (tmp_path / "v13_case_rows.csv").exists()
    for name in ("v13_blockage.json", "v13_cv_force.json", "v13_les_inlet.json"):
        json.loads((tmp_path / name).read_text(encoding="utf-8"))
    assert len(rows) == len(blk_rows) + len(cv_rows) + len(les_rows) + len(inlet_rows)
    assert set(blk["extrapolation"]) == {"100", "250"}
    assert "st_fit_quad" in blk["extrapolation"]["100"]   # 5 点可二次拟合
    assert "st_fit_quad" not in blk["extrapolation"]["250"]  # 3 点不给二次
    assert blk["extrapolation"]["100"]["st_fit_lin"]["intercept"] == \
        pytest.approx(0.18, abs=1e-10)
    assert set(cv["groups"]) == {"100", "300"}
    assert cv["groups"]["100"]["delta_mem_rel"] > 0
    assert cv["groups"]["100"]["link_vs_mem_maxdiff"] == 0.0
    assert "delta_wake_rel" in cv["groups"]["100"]
    assert cv["groups"]["100"]["contour_spread_link"] == 0.0
    assert set(les["grid"]) == {"200", "400", "800"}


def test_build_tasks_are_unique_and_structured():
    from lbm.config import default_config_path, load_config
    m = _load_script()
    tasks = m.build_tasks(load_config(default_config_path()))
    keys = [t["key"] for t in tasks]
    assert len(keys) == len(set(keys))
    assert len(tasks) == 8 + 4 + 9 + 4
    assert {t["kind"] for t in tasks} == {"e13", "e14", "e15_les", "e15_inlet"}
    for t in tasks:
        assert "key" in t and "kind" in t
