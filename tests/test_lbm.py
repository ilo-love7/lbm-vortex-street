"""LBM 核心单测：格子恒等式 / 求解器行为 / 边界 / 测力 / 后处理。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm import CylinderCase, Solver, strouhal
from lbm.lattice import CX, CY, NL, OPP, W, equilibrium, macros


# ---------- lattice ----------
def test_lattice_identities():
    assert NL == 9
    assert abs(W.sum() - 1.0) < 1e-12
    assert np.all(CX[OPP] == -CX) and np.all(CY[OPP] == -CY)
    assert abs((W * CX * CX).sum() - 1 / 3) < 1e-12


def test_equilibrium_consistency():
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.1 * rng.standard_normal((5, 4))
    ux = 0.05 * rng.standard_normal((5, 4))
    uy = 0.05 * rng.standard_normal((5, 4))
    feq = equilibrium(rho, ux, uy)
    r, x, y = macros(feq)
    assert np.allclose(r, rho, atol=1e-12)
    assert np.allclose(x, ux, atol=1e-10)
    assert np.allclose(y, uy, atol=1e-10)


# ---------- solver ----------
@pytest.fixture
def small_case():
    return CylinderCase(nx=120, ny=60, d=8.0, cx_c=30.0, u_in=0.1, re=60.0)


def test_solver_no_nan(small_case):
    s = Solver(small_case)
    for _ in range(30):
        s.step()
    assert np.isfinite(s.f).all()
    assert (s.f[s.solid] == 0.0).all()


def test_inlet_velocity_bc(small_case):
    s = Solver(small_case)
    for _ in range(50):
        s.step()
    _, ux, uy = s.macros_masked()
    col = ux[:, 1]  # 入口内侧一列
    assert np.allclose(col, small_case.u_in, atol=0.02)


def test_force_symmetry(small_case):
    """对称流场（初始无扰动）下升力均值应接近 0、阻力为正。"""
    c = CylinderCase(nx=120, ny=60, d=8.0, cx_c=30.0, u_in=0.1, re=20.0, trip=0.0)
    s = Solver(c)
    fx_sum = fy_sum = 0.0
    for _ in range(30):
        s.step()
        fx, fy = s.force()
        fx_sum += fx
        fy_sum += fy
    assert fx_sum > 0
    assert abs(fy_sum) / fx_sum < 0.05


# ---------- postprocess ----------
def test_strouhal_sine():
    t = np.arange(20000)
    f0 = 0.00066  # St = f·D/U = 0.165，落在物理频带内
    cl = np.sin(2 * np.pi * f0 * t)
    out = strouhal(cl, d=20.0, u_in=0.08)
    st_true = f0 * 20.0 / 0.08
    assert abs(out["st"] - st_true) / st_true < 0.02
