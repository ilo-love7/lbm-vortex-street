"""v1.5 三维失稳区单测：展向谱解析判据、盒长可表示性、Re=250 稳定性。

设计原则与 v1.4 相同：只锁**不变量**与**可解析核对的量**，生产数值归
``scripts/11_v15_3d_unstable.py`` / ``12_v15_domain20d.py``。

本文件里最重要的是 ``test_uniform_flow_has_zero_3d_fraction`` 与
``test_mode_a_not_representable_in_short_box``：前者守住"展向谱不会把二维
流动谎报成三维"，后者守住"Lz=1D 的 nz 扫描本来就没法检验 Mode A"——
v1.4 §14.4 的空判据正是被这两条同时漏掉才发生的。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm.kernels3d import compile_step3d
from lbm.lattice3d import equilibrium3d
from lbm.postprocess import literature_st
from lbm.solver3d import (Case3D, Solver3D, representable_wavelengths)

GEO = dict(d=16.0, cx_c=80.0, u_in=0.1, sponge_width=32)
MODE_A_BAND = (3.0, 4.0)          # 文献 Mode A 展向波长区间（以 D 计）
NX_S, NY_S = 140, 40              # 谱测试用小网格（站位 x=120 仍在下游）


@pytest.fixture(scope="module", autouse=True)
def _compile():
    compile_step3d()


def _synthetic(nz: int, uzz: np.ndarray, xs=(120,), u_inplane: float = 0.05):
    """把给定 u_z 场塞进一个静止的 solver，返回展向谱。

    ``u_inplane`` 给一个均匀的 x 向背景流：``p3d_rel`` 的分母是面内动能，
    没有背景流时分母为零、比值无意义（这正是生产算例里必须避开的除零）。
    """
    case = Case3D(nx=NX_S, ny=NY_S, nz=nz, re=100.0, **GEO)
    sim = Solver3D(case)
    rho = np.ones_like(uzz)
    uxx = np.full_like(uzz, u_inplane)
    sim.f = equilibrium3d(rho, uxx, np.zeros_like(uzz), uzz)
    return sim.spanwise_spectrum(xs=list(xs))


# ---------------------------------------------------------------- 谱方法
def test_single_spanwise_mode_is_recovered():
    """uz = A sin(2π m z/nz) ⇒ 峰在 m、λ_z = Lz/m，三维能量占比可测。"""
    nz, d = 16, 16.0
    z = np.arange(nz)
    for m in (1, 2, 4):
        uzz = np.zeros((nz, NY_S, NX_S))
        uzz += (1e-3 * np.sin(2 * np.pi * m * z / nz))[:, None, None]
        s = _synthetic(nz, uzz)["stations"][0]
        assert s["m_peak"] == m
        assert s["lambda_peak_over_d"] == pytest.approx(nz / m / d, rel=1e-12)
        assert s["frac_3d"] == pytest.approx(1.0, abs=1e-12)
        assert s["p3d_rel"] > 1e-5          # 5e-7 / 0.05²
        assert s["is_3d"]


def test_machine_zero_is_not_reported_as_three_dimensional():
    """uz ~ 1e-16（机器零）时 frac_3d 仍可能是 0.5，但 p3d_rel 必须判为二维。

    这是 v1.5 首轮的真实坑：Lz=0.5D、Re=250 的算例 uz 已衰减到 2e-15，
    谱却给出 ``frac_3d=0.73`` —— 因为分母 p0 也只剩 1e-33。判据必须把三维
    能量放到**面内动能**的标尺上（``p3d_rel``），否则"占比"会在零上放大噪声。
    """
    nz = 16
    rng = np.random.default_rng(3)
    uzz = 1e-16 * rng.standard_normal((nz, NY_S, NX_S))
    s = _synthetic(nz, uzz)["stations"][0]
    assert s["p3d_rel"] < 1e-25
    assert not s["is_3d"]


def test_uniform_flow_has_zero_3d_fraction():
    """z 均匀场（二维流动）必须给 frac_3d=0、p3d=0。

    这条是"空判据守门"：v1.4 的 Re=100 展向扫描里流动就是 z 均匀的，
    谱方法必须在没有噪声时读出**严格零**，而不是让有限振幅冒充三维。
    """
    nz = 16
    uzz = np.full((nz, NY_S, NX_S), 0.07)
    s = _synthetic(nz, uzz)["stations"][0]
    assert s["p3d"] == pytest.approx(0.0, abs=1e-24)
    assert s["frac_3d"] == pytest.approx(0.0, abs=1e-12)
    assert s["p0"] > 0.0                    # m=0 分量确实非零


def test_spectrum_power_matches_variance():
    """Σ_m P_m = mean_{z,y}(u_z²)（Parseval，按 nz 归一后相对 1e-10）。

    注意 y 范围要与 ``spanwise_spectrum`` 一致（``lo=1, hi=ny-1``，丢掉壁面
    两行）——首版拿全场均值对账，差了 8.5%，误以为是归一化错，实则是抽样域
    不一致：**对账口径必须先对齐，再谈误差**。
    """
    nz, x = 16, 120
    rng = np.random.default_rng(7)
    uzz = 1e-3 * rng.standard_normal((nz, NY_S, NX_S))
    s = _synthetic(nz, uzz, xs=(x,))["stations"][0]
    p = np.asarray(s["power"], float)
    # rfft 的 m=0 与 Nyquist 项不折叠，其余乘 2
    full = p[0] + p[-1] + 2.0 * p[1:-1].sum()
    ref = float(np.mean(uzz[:, 1:-1, x] ** 2))
    assert full == pytest.approx(ref, rel=1e-10)


# ---------------------------------------------------------------- 盒长
def test_mode_a_not_representable_in_short_box():
    """Lz=0.5D/1D/2D 都装不下 Mode A（3–4D），Lz=4D 才装得下。

    这是 E22 的前提：若没有这条，nz=8/16/24 的"结果一致"会被误读成
    展向分辨率收敛（v1.4 §14.4 的错误）。
    """
    for nz in (8, 16, 32):
        lam = representable_wavelengths(nz, 16.0)
        assert lam.max() <= 2.0 + 1e-12
        assert not ((lam >= MODE_A_BAND[0]) & (lam <= MODE_A_BAND[1])).any()
    lam64 = representable_wavelengths(64, 16.0)
    assert ((lam64 >= MODE_A_BAND[0]) & (lam64 <= MODE_A_BAND[1])).any()
    assert lam64[0] == pytest.approx(4.0)


def test_representable_wavelengths_are_harmonics():
    lam = representable_wavelengths(64, 16.0)
    assert lam.shape == (32,)
    assert np.allclose(lam * np.arange(1, 33), 4.0)


# ---------------------------------------------------------------- 协议
def test_re250_protocol_is_stable_and_has_shorter_period():
    """τ(Re=250) 落在稳定区间；同一个 D/U 下 T(250) < T(100)。"""
    c250 = Case3D(nx=160, ny=40, nz=8, re=250.0, **GEO)
    c100 = Case3D(nx=160, ny=40, nz=8, re=100.0, **GEO)
    assert 0.5001 < c250.tau < 0.6
    assert c250.tau == pytest.approx(0.5 + 3 * 0.1 * 16.0 / 250.0)
    t = lambda c: c.d / (c.u_in * literature_st(c.re))   # noqa: E731
    assert t(c250) < t(c100)
    assert 0.19 < literature_st(250.0) < 0.20


def test_re250_runs_without_blowup():
    """Re=250、边界全开、300 步不发散（生产跑的前置检查）。"""
    case = Case3D(nx=160, ny=40, nz=8, re=250.0, **GEO)
    sim = Solver3D(case)
    for _ in range(300):
        sim.step()
    rho, ux, uy, uz = sim.macros()
    assert np.isfinite(rho).all() and np.isfinite(ux).all()
    assert rho[:, sim.fluid].min() > 0.5
    assert np.abs(ux).max() < 1.0


def test_domain_20d_only_changes_nx():
    """20D 与 15D 只差 nx：D、cx_c、海绵、ny 全部相同（隔离域长变量）。"""
    c15 = Case3D(nx=320, ny=80, nz=8, re=100.0, **GEO)
    c20 = Case3D(nx=400, ny=80, nz=8, re=100.0, **GEO)
    assert (c15.d, c15.cx_c, c15.sponge_width) == (c20.d, c20.cx_c,
                                                   c20.sponge_width)
    assert c20.nx - c20.cx_c > c15.nx - c15.cx_c
