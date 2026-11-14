"""E15 单测：隐式 Smagorinsky / 持续入口扰动 / 新旧内核一致性。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm import (CylinderCase, Solver, apply_smagorinsky_numpy, equilibrium,
                 inlet_base, inlet_uy, nu_t_over_nu0, pi_neq_norm,
                 smagorinsky_tau)
from lbm import solver as solver_mod
from lbm.kernels import HAVE_NUMBA

SMALL = dict(nx=100, ny=60, d=10.0, cx_c=28.0, u_in=0.1, sponge_width=6)


# ---------- 隐式 Smagorinsky 二次方程 ----------
def test_cs_zero_is_exact_bgk():
    for tau0 in (0.508, 0.53, 0.62, 1.2):
        for q in (0.0, 1e-6, 0.4, 12.0):
            assert smagorinsky_tau(tau0, q, 0.0) == pytest.approx(tau0, abs=1e-14)


def test_tau_total_satisfies_quadratic_and_is_monotone():
    tau0, rho, cs = 0.53, 1.07, 0.16
    qs = np.array([0.0, 1e-4, 1e-3, 1e-2, 0.1])
    tt = smagorinsky_tau(tau0, qs, cs, rho=rho)
    assert np.all(tt >= tau0 - 1e-14)
    assert np.all(np.diff(tt) >= -1e-14)
    # τ² − τ₀τ − 9C_s²Q/(2ρ) = 0
    resid = tt * tt - tau0 * tt - 9.0 * cs ** 2 * qs / (2.0 * rho)
    assert np.allclose(resid, 0.0, atol=1e-12)
    # 单调随 C_s 增大
    assert (smagorinsky_tau(tau0, 0.05, 0.30)
            > smagorinsky_tau(tau0, 0.05, 0.10)
            > smagorinsky_tau(tau0, 0.05, 0.0))


def test_zero_density_does_not_blow_up():
    tau0 = 0.53
    q = np.array([0.0, 1e-3])
    rho = np.array([0.0, 1.0])
    tt = smagorinsky_tau(tau0, q, 0.16, rho=rho)
    assert np.all(np.isfinite(tt))
    assert tt[0] == pytest.approx(tau0, abs=1e-14)
    assert tt[1] > tau0


def test_pi_neq_norm_is_zero_at_equilibrium_and_positive_off():
    rho = np.full((5, 4), 1.0)
    ux = np.full((5, 4), 0.07)
    uy = np.full((5, 4), -0.02)
    f = equilibrium(rho, ux, uy)
    assert np.max(pi_neq_norm(f)) < 1e-12
    g = f + 1e-3 * np.sin(np.arange(180).reshape(5, 4, 9))
    q = pi_neq_norm(g)
    assert q.shape == (5, 4)
    assert np.all(q >= 0) and q.max() > 0


def test_apply_smagorinsky_numpy_matches_pure_function():
    rng = np.random.default_rng(7)
    rho = 1.0 + 0.01 * rng.standard_normal((6, 5))
    ux = 0.08 + 0.01 * rng.standard_normal((6, 5))
    uy = 0.01 * rng.standard_normal((6, 5))
    f = equilibrium(rho, ux, uy) + 1e-4 * rng.standard_normal((6, 5, 9))
    from lbm import macros
    tau0, cs = 0.55, 0.16
    out, tt = apply_smagorinsky_numpy(f, tau0, cs)
    # 参考实现内部由 f 的一阶矩自算宏观量（加了扰动的 f 与构造时的 rho/ux/uy
    # 并不相同），因此对照也必须从 f 反算，否则 feq 口径不一致。
    r2, u2, v2 = macros(f)
    q = pi_neq_norm(f, r2, u2, v2)
    expect = f - (1.0 / smagorinsky_tau(tau0, q, cs, rho=r2))[..., None] \
        * (f - equilibrium(r2, u2, v2))
    assert np.allclose(out, expect, atol=1e-14)
    assert np.allclose(tt, smagorinsky_tau(tau0, q, cs, rho=r2))


def test_nu_t_over_nu0_definition():
    tau0 = 0.56              # ν₀ = (τ₀−0.5)/3
    tt = np.array([tau0, tau0 + 0.03, tau0 + 0.12])
    r = nu_t_over_nu0(tt, tau0)
    assert r[0] == 0.0
    assert np.allclose(r, (tt - tau0) / (tau0 - 0.5))
    assert np.isinf(nu_t_over_nu0(0.6, 0.5))


# ---------- 入口扰动 ----------
def test_inlet_base_is_exactly_zero_mean():
    for ny in (60, 140, 280):
        for m in (1, 3, 7):
            b = inlet_base(ny, 0.05, 0.1, m)
            assert b.shape == (ny,)
            assert abs(b.mean()) < 1e-14
            assert np.isclose(np.abs(b).max(), 0.005, atol=1e-12)


def test_inlet_uy_is_periodic_and_zero_when_eps_zero():
    base = inlet_base(60, 0.0, 0.1, 3)
    assert np.all(inlet_uy(base, 123, 500.0) == 0.0)
    base = inlet_base(60, 0.04, 0.1, 3)
    P = 400.0
    a = inlet_uy(base, 0, P)
    assert np.allclose(a, np.zeros_like(a), atol=1e-14)
    assert np.allclose(inlet_uy(base, 100, P), base)       # sin(π/2)=1
    assert np.allclose(inlet_uy(base, 300, P), -base)      # sin(3π/2)=−1
    assert np.allclose(inlet_uy(base, 400, P), inlet_uy(base, 0, P), atol=1e-14)
    assert np.allclose(inlet_uy(base, 10, 0.0), 0.0)


# ---------- 求解器集成 ----------
def test_les_relaxes_empirical_tau_floor():
    kw = dict(**SMALL, re=800.0)
    assert CylinderCase(re=800.0, **SMALL).tau < 0.51
    with pytest.raises(ValueError):
        Solver(CylinderCase(**kw))
    s = Solver(CylinderCase(les=True, c_smag=0.16, **kw))
    assert s.case.les and np.isfinite(s.f).all()


def test_les_with_mrt_is_rejected():
    with pytest.raises(ValueError, match="BGK"):
        Solver(CylinderCase(collision="mrt", les=True, **SMALL, re=100.0))


def test_les_cs_zero_equals_bgk_bitwise():
    """C_s=0 时 LES 分支必须逐位退化到 BGK（否则 baselines 不可比）。"""
    kw = dict(**SMALL, re=100.0)
    a = Solver(CylinderCase(**kw))
    b = Solver(CylinderCase(les=True, c_smag=0.0, **kw))
    for _ in range(25):
        a.step()
        b.step()
    assert np.array_equal(a.f, b.f)
    assert a.force() == b.force()


def test_les_adds_viscosity_and_keeps_flow_finite():
    # Re=300 时 τ=0.510 恰好撞上非 LES 的经验下限 0.51，改用 Re=200（τ=0.515）
    kw = dict(**SMALL, re=200.0)
    a = Solver(CylinderCase(**kw))
    b = Solver(CylinderCase(les=True, c_smag=0.30, **kw))
    for _ in range(40):
        a.step()
        b.step()
    assert np.isfinite(b.f).all()
    d = b.les_diagnostics()
    assert d is None or d["nu_t_over_nu0_median"] >= 0.0
    assert a.les_diagnostics() is None
    assert not np.array_equal(a.f, b.f)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba 不可用")
def test_numba_and_numpy_paths_agree_with_les_and_inlet():
    """LES + 入口扰动下，numba 内核与 numpy 回退路径必须一致（防两条实现漂移）。"""
    kw = dict(**SMALL, re=200.0, les=True, c_smag=0.22,
              inlet_eps=0.03, inlet_period_ratio=1.7)
    a = Solver(CylinderCase(**kw))
    for _ in range(20):
        a.step()                      # numba 路径
    old = solver_mod.HAVE_NUMBA
    try:
        solver_mod.HAVE_NUMBA = False
        b = Solver(CylinderCase(**kw))
        for _ in range(20):
            b.step()                  # numpy 回退路径
    finally:
        solver_mod.HAVE_NUMBA = old
    assert np.allclose(a.f, b.f, atol=1e-10, rtol=1e-10)
    assert a.force() == pytest.approx(b.force(), abs=1e-9)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba 不可用")
def test_numba_and_numpy_paths_agree_with_inlet_only():
    kw = dict(**SMALL, re=100.0, inlet_eps=0.05, inlet_period_ratio=2.37)
    a = Solver(CylinderCase(**kw))
    for _ in range(20):
        a.step()
    old = solver_mod.HAVE_NUMBA
    try:
        solver_mod.HAVE_NUMBA = False
        b = Solver(CylinderCase(**kw))
        for _ in range(20):
            b.step()
    finally:
        solver_mod.HAVE_NUMBA = old
    assert np.allclose(a.f, b.f, atol=1e-12, rtol=1e-12)
