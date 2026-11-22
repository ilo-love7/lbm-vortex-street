"""v1.6 高 Re 三维单测：展向两点相关、模式带内能量、饱和律拟合。

设计原则同 v1.5：只锁**不变量**与**可解析核对的量**，生产数值归
``scripts/14_v16_highre.py``。

本文件里最重要的是三条：
- ``test_correlation_of_pure_mode``：循环相关必须与解析式 ``cos(2πmr/nz)`` 逐点
  相符（含 ``r0 = nz/4m``）——否则 ``L_c`` 只是一串无意义的数；
- ``test_mean_removal_kills_dc``：不减去展向均值的相关函数会被 m=0 直流项钉死
  在 1，``r0`` 直接跑到半个盒长，把"二维流动"读成"超长相关"；
- ``test_bands_disjoint_at_6d_but_not_at_1d``：Lz=1D 时 Mode A 带里**没有任何
  可表示的 m**，此时"带内能量为 0"是盒子造成的（v1.4 空判据的同类陷阱）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbm.kernels3d import compile_step3d
from lbm.lattice3d import equilibrium3d
from lbm.solver3d import (Case3D, Solver3D, mode_band_energy,
                          representable_wavelengths)

GEO = dict(d=16.0, cx_c=80.0, u_in=0.1, sponge_width=32)
NX_S, NY_S = 140, 40
NZ6, D = 96, 16.0          # Lz = 6D


@pytest.fixture(scope="module", autouse=True)
def _compile():
    compile_step3d()


def _solver_with(uzz: np.ndarray, nz: int, u_inplane: float = 0.05):
    case = Case3D(nx=NX_S, ny=NY_S, nz=nz, re=100.0, **GEO)
    sim = Solver3D(case)
    sim.f = equilibrium3d(np.ones_like(uzz), np.full_like(uzz, u_inplane),
                          np.zeros_like(uzz), uzz)
    return sim


def _mode_field(nz: int, m: int, amp: float = 1e-3, dc: float = 0.0):
    z = np.arange(nz)
    uzz = np.zeros((nz, NY_S, NX_S))
    uzz += (dc + amp * np.sin(2 * np.pi * m * z / nz))[:, None, None]
    return uzz


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "v16", ROOT / "scripts" / "14_v16_highre.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 两点相关
def test_correlation_of_pure_mode():
    """纯正弦：R(0)=1、R(λ/2)=−1、r0=λ/4（离散解析式逐点核对）。"""
    nz, m = 96, 3
    sim = _solver_with(_mode_field(nz, m), nz)
    cc = sim.spanwise_correlation(xs=[120])["stations"][0]
    assert cc["valid"] is True
    R = np.asarray(cc["R"], float)
    r = np.arange(R.size)
    expect = np.cos(2 * np.pi * m * r / nz)
    # 循环相关对整数周期正弦是精确的；只允许浮点舍入
    assert np.allclose(R, expect, atol=1e-9)
    assert R[0] == pytest.approx(1.0, abs=1e-12)
    assert R[nz // (2 * m)] == pytest.approx(-1.0, abs=1e-9)
    assert cc["r0"] == nz // (4 * m)


def test_correlation_length_decreases_with_mode():
    """积分尺度随模式数单调下降，且 r0 精确等于 nz/(4m)。"""
    Lc = []
    for m in (1, 2, 4):
        sim = _solver_with(_mode_field(96, m), 96)
        cc = sim.spanwise_correlation(xs=[120])["stations"][0]
        assert cc["r0"] == 96 // (4 * m)
        Lc.append(cc["Lc_over_d"])
    assert Lc[0] > Lc[1] > Lc[2] > 0
    # 解析值 Σ_{r=0}^{r0−1} cos(2πmr/nz)，单位 D
    assert Lc[0] == pytest.approx(
        sum(np.cos(2 * np.pi * r / 96) for r in range(24)) / D, rel=1e-9)


def test_mean_removal_kills_dc():
    """带直流偏置的正弦：不减均值会把 r0 推到半盒长，减了才回到 λ/4。"""
    nz, m = 96, 3
    sim = _solver_with(_mode_field(nz, m, dc=5.0), nz)
    cc = sim.spanwise_correlation(xs=[120])["stations"][0]
    assert cc["r0"] == nz // (4 * m)          # 不是 nz//2+1
    # 手工复算：不减均值时首个过零点应在半盒长之外
    rho, ux, uy, uz = sim.macros()
    u = np.asarray(uz[:, 1:-1, 120], float)
    var_raw = float(np.mean(u * u))
    R_raw = [float(np.mean(u * np.roll(u, -r, axis=0))) / var_raw
             for r in range(nz // 2 + 1)]
    assert all(v > 0.0 for v in R_raw[:40])


def test_uniform_spanwise_field_is_invalid():
    """uz 在 z 上恒定（二维流动）⇒ R 无定义，valid=False 且 Lc=0。"""
    sim = _solver_with(np.zeros((96, NY_S, NX_S)), 96, u_inplane=0.05)
    cc = sim.spanwise_correlation(xs=[120])["stations"][0]
    assert cc["valid"] is False
    assert cc["Lc_over_d"] == 0.0 and cc["r0"] == 0


# ---------------------------------------------------------------- 模式带
def test_band_energy_selects_mode_a():
    """λ=3D（m=2, Lz=6D）⇒ 能量全在 Mode A 带，B 带为机器零。"""
    sim = _solver_with(_mode_field(NZ6, 2), NZ6)
    full = sim.spanwise_spectrum(xs=[120])
    sp = full["stations"][0]
    b = mode_band_energy(sp["power"], full["nz"], full["d"])["bands"]
    assert b["A"]["m"] == [2]
    assert b["A"]["frac_of_p3d"] > 0.99
    assert b["B"]["frac_of_p3d"] < 1e-20


def test_band_energy_selects_mode_b():
    """λ=1D（m=6, Lz=6D）⇒ 能量全在 Mode B 带，A 带为机器零。"""
    sim = _solver_with(_mode_field(NZ6, 6), NZ6)
    full = sim.spanwise_spectrum(xs=[120])
    sp = full["stations"][0]
    b = mode_band_energy(sp["power"], full["nz"], full["d"])["bands"]
    # Lz=6D 时 λ=1.2/1.0/0.857D 分别落 m=5/6/7（m=5 恰在带边 1.2D）
    assert b["B"]["m"] == [5, 6, 7]
    assert b["B"]["frac_of_p3d"] > 0.99
    assert b["A"]["frac_of_p3d"] < 1e-20


def test_bands_disjoint_at_6d_but_not_at_1d():
    """Lz=6D 两支都离开带边；Lz=1D 时 Mode A 带里没有可表示的 m。"""
    p = np.ones(49)
    b6 = mode_band_energy(p, 96, D)
    assert b6["bands"]["A"]["m"] and b6["bands"]["B"]["m"]
    assert not set(b6["bands"]["A"]["m"]) & set(b6["bands"]["B"]["m"])
    assert b6["bands"]["A"]["repr_ok"] and b6["bands"]["B"]["repr_ok"]

    b1 = mode_band_energy(np.ones(9), 16, D)     # Lz = 1D
    assert b1["bands"]["A"]["repr_ok"] is False   # 3–4D 装不进 1D 盒
    assert b1["bands"]["B"]["m"] == [1]           # 1D 只有 m=1
    # 空判据的判别：repr_ok=False 时 frac=0 是盒子造成的，不是物理结论
    assert b1["bands"]["A"]["frac_of_p3d"] == 0.0


def test_representable_wavelengths_include_both_bands_at_6d():
    lam = representable_wavelengths(NZ6, D)
    assert np.any((lam >= 3.0) & (lam <= 4.0))
    assert np.any((lam >= 0.8) & (lam <= 1.2))
    assert np.allclose(np.sort(lam)[::-1], lam)   # m 递增 ⇒ λ 递减


# ---------------------------------------------------------------- τ 与稳定
def test_tau_decreases_with_re():
    """固定 D、U ⇒ ν 反比于 Re，τ 随 Re 单调下降且仍高于 0.5001。"""
    taus = [Case3D(ny=80, nz=8, re=re, **GEO).tau
            for re in (250.0, 300.0, 350.0, 400.0)]
    assert all(b < a for a, b in zip(taus, taus[1:]))
    assert taus[-1] > 0.5 and taus[0] < 0.53
    assert Case3D(ny=80, nz=8, re=400.0, **GEO).tau == pytest.approx(
        3 * (0.1 * 16.0 / 400.0) + 0.5, rel=1e-12)


def test_re400_short_run_stays_finite():
    """τ=0.512 的小盒短跑必须保持有限、密度为正（不替代生产跑的稳定性结论）。"""
    case = Case3D(ny=40, nz=8, re=400.0, **GEO)
    sim = Solver3D(case)
    for _ in range(300):
        sim.step()
    rho, ux, uy, uz = sim.macros()
    rf = rho[:, sim.fluid]
    assert np.isfinite(rf).all() and np.isfinite(uz).all()
    assert rf.min() > 0.0 and rf.max() < 2.0
    assert np.abs(ux).max() < 0.5


# ---------------------------------------------------------------- 饱和律拟合
def test_power_law_fit_recovers_exponent():
    """y = 2.5 x^0.5 上取四点，拟合必须回到 b=0.5、a=2.5、R²=1。"""
    m = _load_script()
    x = np.array([0.545, 1.0, 1.455, 1.909])       # (Re−190)/110
    y = 2.5 * x ** 0.5
    f = m.fit_power(x, y)
    assert f["b"] == pytest.approx(0.5, abs=1e-8)
    assert f["a"] == pytest.approx(2.5, rel=1e-9)
    assert f["r2"] == pytest.approx(1.0, abs=1e-9)
    assert f["n"] == 4


def test_saturated_series_gives_zero_exponent():
    """常数序列（真饱和）⇒ b≈0；线性序列 ⇒ b≈1。线性拟合对常数无判别力。"""
    m = _load_script()
    x = np.array([0.545, 1.0, 1.455, 1.909])
    flat = m.fit_power(x, np.full(4, 3.0))
    assert abs(flat["b"]) < 1e-9
    lin = m.fit_power(x, 3.0 * x)
    assert lin["b"] == pytest.approx(1.0, abs=1e-9)


def test_saturation_analysis_ratio_is_model_free():
    """E(400)/E(250)：饱和≈1、线性≈3.5，都不依赖任何拟合。"""
    m = _load_script()
    rows = [{"re": 250.0, "dst_pct": -0.3, "dcd_pct": -3.0,
             "dcl_rms_pct": -22.0},
            {"re": 300.0, "dst_pct": -0.4, "dcd_pct": -3.5,
             "dcl_rms_pct": -22.5},
            {"re": 350.0, "dst_pct": -0.5, "dcd_pct": -4.0,
             "dcl_rms_pct": -23.0},
            {"re": 400.0, "dst_pct": -0.6, "dcd_pct": -4.5,
             "dcl_rms_pct": -23.5}]
    sa = m.saturation_analysis(rows)
    assert sa["ratio_400_250"] == pytest.approx(23.5 / 22.0)
    assert sa["x_norm"][0] == pytest.approx((250 - 190) / 110)
    assert sa["E_cl_rms"][0] == pytest.approx(22.0)   # 取绝对值
