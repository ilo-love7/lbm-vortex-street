"""v1.10 纯盒长分离单测：锁跨盒长可比对判据与 4D↔8D 同波长前提。

40T+10T 长窗每档约 60–120 min，不进 pytest；生产脚本用 ``--smoke`` 单独验。
这里守四类易错口径：① m_peak 跨盒长不可比（λ_z=Lz/m）；② 谱重心/模式带/
相关长度跨盒长可比；③ 4D↔8D 必须同为 λ_A=4D 才是纯盒长对；④ 缺档给
incomplete 而不静默出判据。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbm.solver3d import mode_band_energy, representable_wavelengths  # noqa: E402


@pytest.fixture(scope="module")
def v110():
    spec = importlib.util.spec_from_file_location(
        "v110_boxlength", ROOT / "scripts" / "19_v110_boxlength.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _point(*, nz, st, cd, cl, lam_c, m_peak, lam_peak, band_a, lc, wall=60.0):
    return {
        "re": 250.0, "ny": 80, "nz": nz, "st": st, "cd_mean": cd,
        "cl_rms": cl, "uz_rel": 0.05, "wall_min": wall,
        "spectrum": {"nz": nz, "d": 16.0, "stations": [{
            "p3d_rel": 1e-3, "m_peak": m_peak,
            "lambda_peak_over_d": lam_peak,
            # power 首项是 m=0（直流，被 spectral_centroid 切掉）；
            # 只在峰位放能量 ⇒ 重心恰等于峰位，便于断言
            "power": [0.0] + [0.0] * (m_peak - 1) + [1.0],
            "bands": {"bands": {"A": {"frac_of_p3d": band_a},
                                "B": {"frac_of_p3d": 0.2}}},
            "corr": {"Lc_over_d": lc},
        }]},
    }


# ---------------------------------------------------------------- 可表示波长前提
def test_mode_a_wavelength_4d_and_8d_both_4d():
    """纯盒长对的前提：4D 与 8D 的 A 带可表示波长必须都含 4D。"""
    a4 = representable_wavelengths(64, 16.0)
    a6 = representable_wavelengths(96, 16.0)
    a8 = representable_wavelengths(128, 16.0)
    assert any(abs(x - 4.0) < 1e-9 for x in a4)
    assert any(abs(x - 3.0) < 1e-9 for x in a6)
    assert any(abs(x - 4.0) < 1e-9 for x in a8)
    # 6D 的 A 带只有 3D（没有 4D）
    assert not any(abs(x - 4.0) < 1e-9 for x in a6)


def test_mode_peak_not_cross_box_comparable():
    """同一物理 λ=4D 在 4D 盒是 m=1、8D 盒是 m=2 —— 峰位编号不可比。"""
    a4 = representable_wavelengths(64, 16.0)
    a8 = representable_wavelengths(128, 16.0)
    m4 = int(np.argmin(np.abs(a4 - 4.0)))
    m8 = int(np.argmin(np.abs(a8 - 4.0)))
    # representable_wavelengths 从 m=1 起，索引 0 对应 m=1
    assert m4 + 1 != m8 + 1


# ---------------------------------------------------------------- 跨盒长读数
def test_cross_box_readout_separates_comparable_and_not(v110):
    r = _point(nz=128, st=0.2263, cd=1.5776, cl=0.5145,
               lam_c=4.0, m_peak=2, lam_peak=4.0, band_a=0.5, lc=0.4)
    rd = v110.cross_box_readout(r)
    assert rd["lambda_centroid_over_d"] == pytest.approx(4.0, rel=1e-6)
    assert rd["band_A_frac"] == pytest.approx(0.5)
    assert rd["Lc_over_d"] == pytest.approx(0.4)
    assert rd["m_peak"] == 2          # 单列，不参与可比性
    assert rd["lambda_peak_over_d"] == pytest.approx(4.0)


def test_centroid_cuts_dc_term(v110):
    """直流 m=0 必须切除，否则重心被钉在 0。"""
    s0 = {"power": [1e6, 0, 1, 0]}    # 直流极大，真信号在 m=2
    cen = v110.v16.spectral_centroid(s0, 96, 16.0)
    assert cen["m_centroid"] == pytest.approx(2.0)
    assert cen["lambda_centroid_over_d"] == pytest.approx(3.0)


# ---------------------------------------------------------------- 判据
def test_verdict_incomplete_when_missing_box(v110):
    pts = {64: _point(nz=64, st=0.2263, cd=1.5776, cl=0.5145,
                      lam_c=4, m_peak=1, lam_peak=4, band_a=0.5, lc=0.4)}
    v = v110.box_verdict(pts)
    assert v["label"] == "incomplete"
    assert "8D" in v["missing"] and "6D" in v["missing"]


def test_verdict_wavelength_dominant(v110):
    """4D≈8D（同 λ_A）而 6D 明显不同 ⇒ 差异主因是允许波长。"""
    pts = {
        64: _point(nz=64, st=0.2263, cd=1.5776, cl=0.5140,
                   lam_c=4, m_peak=1, lam_peak=4, band_a=0.5, lc=0.4),
        96: _point(nz=96, st=0.2205, cd=1.5887, cl=0.5596,
                   lam_c=3, m_peak=2, lam_peak=3, band_a=0.42, lc=0.30),
        128: _point(nz=128, st=0.2263, cd=1.5776, cl=0.5145,
                    lam_c=4, m_peak=2, lam_peak=4, band_a=0.5, lc=0.4),
    }
    v = v110.box_verdict(pts)
    assert v["label"] == "wavelength_dominant"
    assert v["pairs"]["4D_vs_8D_pure_box"]["pure_box"] is True
    assert v["pairs"]["6D_vs_8D_box_and_wavelength"]["pure_box"] is False
    assert v["metrics"]["ratio"] < 0.5


def test_verdict_box_length_dominant(v110):
    """4D 与 8D 差很大、与 6D 接近 ⇒ 盒长本身是主因。"""
    pts = {
        64: _point(nz=64, st=0.2100, cd=1.60, cl=0.60,
                   lam_c=3.5, m_peak=1, lam_peak=4, band_a=0.3, lc=0.2),
        96: _point(nz=96, st=0.2250, cd=1.58, cl=0.51,
                   lam_c=3, m_peak=2, lam_peak=3, band_a=0.5, lc=0.4),
        128: _point(nz=128, st=0.2260, cd=1.577, cl=0.514,
                    lam_c=4, m_peak=2, lam_peak=4, band_a=0.5, lc=0.4),
    }
    v = v110.box_verdict(pts)
    assert v["label"] == "box_length_dominant"


def test_verdict_mixed_when_comparable(v110):
    pts = {
        64: _point(nz=64, st=0.218, cd=1.582, cl=0.525,
                   lam_c=3.5, m_peak=1, lam_peak=4, band_a=0.4, lc=0.35),
        96: _point(nz=96, st=0.220, cd=1.585, cl=0.530,
                   lam_c=3, m_peak=2, lam_peak=3, band_a=0.45, lc=0.33),
        128: _point(nz=128, st=0.226, cd=1.577, cl=0.514,
                    lam_c=4, m_peak=2, lam_peak=4, band_a=0.5, lc=0.4),
    }
    v = v110.box_verdict(pts)
    assert v["label"] == "mixed_or_unresolved"


def test_verdict_pure_function_no_mutation(v110):
    pts = {
        64: _point(nz=64, st=0.2263, cd=1.5776, cl=0.5140,
                   lam_c=4, m_peak=1, lam_peak=4, band_a=0.5, lc=0.4),
        96: _point(nz=96, st=0.2205, cd=1.5887, cl=0.5596,
                   lam_c=3, m_peak=2, lam_peak=3, band_a=0.42, lc=0.30),
        128: _point(nz=128, st=0.2263, cd=1.5776, cl=0.5145,
                    lam_c=4, m_peak=2, lam_peak=4, band_a=0.5, lc=0.4),
    }
    import copy
    before = copy.deepcopy(pts)
    v110.box_verdict(pts)
    assert pts == before


# ---------------------------------------------------------------- 缓存接入
def test_4d_long_window_peak_leaves_mode_a_band(v110):
    """反读：4D 短窗 m=1/λ=4D，长窗主导峰移到 λ=2D（A 带外）。

    这是本轮最关键的负面发现——"可表示 λ_A=4D"不等于"长窗选中"，
    纯盒长对在实测模式态层面不成立。缺 4D 缓存时跳过（档未跑）。
    """
    p = v110.cache_path(64)
    r = v110._load(p)
    if r is None:
        pytest.skip("4D 长窗档未跑")
    s0 = r["spectrum"]["stations"][0]
    # 可表示 λ_A=4D 的 m=1 确实存在，但主导峰不在 A 带（3-4D）
    assert s0["lambda_peak_over_d"] < 3.0 or s0["m_peak"] != 1
    assert r["spectrum"]["nz"] == 64
    assert abs(float(r["re"]) - 250.0) < 1e-9


def test_6d_reuse_requires_cached_40t(v110):
    """6D 长窗必须来自 v1.6 t40 缓存，缺档返回 None（不造数）。"""
    r = v110.load_6d_from_v16()
    assert r is not None, "v16_3d_re250_nz96_t40.json 缺失"
    assert int(r["nz"]) == 96
    assert abs(float(r["re"]) - 250.0) < 1e-9
    s0 = r["spectrum"]["stations"][0]
    # 长窗 6D 已进入 m=2 / λ=3D（短窗是 m=1 / λ=6D）
    assert s0["m_peak"] == 2
