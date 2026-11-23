"""v1.8 低 τ 校准单测：只锁自适应步数与 ν→Re 误差换算。

生产扫描是 8 档剪切波（约 18 min），不进 pytest；缓存结果的一致性由
``results/v18_lowtau.json`` 与本报告 §23.6 的数字核对。
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
def lowtau():
    spec = importlib.util.spec_from_file_location(
        "v18_lowtau", ROOT / "scripts" / "17_v18_lowtau.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_steps_for_scales_inverse_nu(lowtau):
    """τ→0.5 时 ν→0，步数必须单调增；高 τ 档保底 4000 步。"""
    s = [lowtau.steps_for(t) for t in (0.505, 0.51, 0.52, 0.55, 0.8)]
    assert all(a > b for a, b in zip(s, s[1:]))
    assert s[-1] == 4000
    assert all(x % 20 == 0 for x in s)


def test_steps_for_rejects_nonphysical_tau(lowtau):
    with pytest.raises(ValueError):
        lowtau.steps_for(0.5)
    with pytest.raises(ValueError):
        lowtau.steps_for(0.49)


def test_re_error_budget_mapping_is_exact(lowtau):
    rows = [{"tau": 0.51, "rel_err_pct": 1.0},
            {"tau": 0.8, "rel_err_pct": 0.2}]
    b = lowtau.re_error_budget(rows, re_target=400.0)
    assert b["tau_nearest"] == 0.51
    assert b["nu_rel_err_pct"] == pytest.approx(1.0)
    assert b["re_eff"] == pytest.approx(400.0 / 1.01)
    assert b["re_shift_pct"] == pytest.approx(-100.0 / 101.0)


def test_cached_scan_stays_below_one_percent(lowtau):
    """缓存扫描的硬上界：低 τ 段不得出现 >1% 的 ν 反测误差。"""
    p = ROOT / "results" / "v18_lowtau.json"
    if not p.exists():
        pytest.skip("低 τ 扫描尚未生成")
    import json
    d = json.loads(p.read_text(encoding="utf-8"))
    assert len(d["rows"]) == len(lowtau.TAUS)
    assert all(r["rel_err_pct"] < 1.0 for r in d["rows"])
    assert d["re_budget"]["tau_nearest"] == 0.51
    assert abs(d["re_budget"]["re_shift_pct"]) < 1.0
    assert np.isfinite(d["re_budget"]["re_eff"])
