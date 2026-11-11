"""v1.2 并行实验调度器的轻量单测（不跑真实仿真）。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "05_mrt_re_vorticity.py"
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def exp():
    spec = importlib.util.spec_from_file_location("exp05_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cfg(exp):
    from lbm.config import default_config_path, load_config
    return load_config(default_config_path())


def test_package_exports_v12_api():
    import lbm
    for name in ("mrt_collide", "mrt_matrices", "mrt_meq", "mrt_relax_rates",
                 "probe_st", "dual_st", "convection_delay", "harmonic_ratio"):
        assert name in lbm.__all__
        assert callable(getattr(lbm, name))


def test_build_tasks_count_and_uniqueness(exp, cfg):
    tasks = exp.build_tasks(cfg)
    assert len(tasks) == 12
    keys = [t["key"] for t in tasks]
    assert len(set(keys)) == 12
    kinds = {k: sum(t["kind"] == k for t in tasks) for k in ("e10", "e11", "e12")}
    assert kinds == {"e10": 6, "e11": 4, "e12": 2}


def test_build_tasks_e10_covers_low_tau_and_both_collisions(exp, cfg):
    e10 = [t for t in exp.build_tasks(cfg) if t["kind"] == "e10"]
    assert {(t["re"], t["coll"]) for t in e10} == {
        (100.0, "bgk"), (100.0, "mrt"),
        (300.0, "bgk"), (300.0, "mrt"),
        (350.0, "bgk"), (350.0, "mrt"),
    }
    # τ = 3UD/Re + 0.5 = 8.4/Re + 0.5；Re=350 已低于 v1.1 红线 0.528
    assert 8.4 / 350.0 + 0.5 < 0.528


def test_make_case_collision_switch(exp, cfg):
    bgk = exp.make_case(cfg, 100.0, collision="bgk")
    mrt = exp.make_case(cfg, 100.0, collision="mrt")
    assert bgk.tau == mrt.tau == pytest.approx(0.584, abs=1e-6)
    assert str(bgk.collision).lower() == "bgk"
    assert str(mrt.collision).lower() == "mrt"


def test_cache_roundtrip(exp, tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "CACHE", tmp_path / "cache")
    key = "unit_smoke"
    payload = {"key": key, "ok": True, "data": {"cd_mean": 1.25}}
    exp._save_cache(key, payload)
    p = exp._cache_path(key)
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8"))["data"]["cd_mean"] == 1.25
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_collect_skips_failed_workers(exp):
    results = {
        "e10_re100_bgk": {"ok": True, "data": {"re": 100.0, "collision": "bgk"}},
        "e10_re100_mrt": {"ok": False, "data": None, "error": "RuntimeE divider"},
        "e11_re75": {"ok": True, "data": {"re": 75.0}},
        "e12_re100": {"ok": True, "data": {"re": 100.0}},
    }
    assert len(exp.collect(results, "e10", "e10_")) == 1
    assert len(exp.collect(results, "e11", "e11_")) == 1
    assert exp.collect(results, "e12", "e12_")[0]["re"] == 100.0


def test_repost_harmonics_refreshes_cache(exp, cfg, tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setattr(exp, "RESULTS", tmp_path)
    monkeypatch.setattr(exp, "CACHE", tmp_path / "cache")
    n = 4000
    t = np.arange(n, dtype=float)
    f0 = 0.2 * 0.1 / 28.0
    x = np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
    np.savez(tmp_path / "probes_v12_re100.npz",
             cl=x, uy_0=x, uy_1=x, uy_2=x)
    stations = {f"x{s}D": {"harm_ratio_uy": 0.0, "harm_ratio_force": 0.0}
                for s in exp.STATIONS}
    exp._save_cache("e12_re100", {
        "key": "e12_re100", "kind": "e12", "ok": True,
        "data": {"re": 100.0, "st_force": 0.2, "wall_min": 1.0,
                 "stations": stations}})
    changed = exp.repost_harmonics(cfg)
    assert "e12_re100" in changed
    q = json.loads(exp._cache_path("e12_re100").read_text(encoding="utf-8"))
    assert q["data"]["stations"]["x1.5D"]["harm_ratio_uy"] > 0.2
    assert q["data"]["stations"]["x1.5D"]["harm_idx_uy"] > 0
    assert (tmp_path / "v12_probes.json").exists()


def test_write_sweep_merges_v12_with_v11(exp, tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "RESULTS", tmp_path)
    # 造一个 v1.1 基线 JSON，验证复用分支
    (tmp_path / "baseline_metrics_v11_re50.json").write_text(
        json.dumps({"rows": [{"tau": 0.668, "cd_mean": 1.637,
                              "cl_rms": 0.03, "cl_peak_amp": 0.034,
                              "st": 0.1376, "st_lit": 0.1206,
                              "st_rel_err": 0.142}]}),
        encoding="utf-8")
    new = [{"re": 75.0, "tau": 0.612, "cd_mean": 1.5, "cl_rms": 0.2,
            "cl_peak_amp": 0.2, "st": 0.16, "st_lit": 0.151,
            "st_rel_err": 0.06, "wall_min": 3.0, "src": "v1.2"}]
    merged = exp.write_sweep(new)
    assert [r["re"] for r in merged] == [50.0, 75.0]
    assert merged[0]["src"] == "v1.1"
    assert merged[1]["src"] == "v1.2"
    assert (tmp_path / "v12_sweep_rows.csv").exists()
