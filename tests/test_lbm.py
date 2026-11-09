"""LBM 核心单测：格子恒等式 / 求解器行为 / 边界 / 测力 / 后处理。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm import CylinderCase, Solver, strouhal
from lbm.bouzidi import cylinder_link_q
from lbm.kernels import HAVE_NUMBA
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


# ---------- bouzidi ----------
def test_cylinder_link_q_analytic():
    # 圆心 (10.5, 10.5) R=4；节点 (15,10) 沿 -x 指向圆：壁在 x=10.5+sqrt(16-0.25)
    q = cylinder_link_q(np.array([10.0]), np.array([15.0]), 7,
                        10.5, 10.5, 4.0)
    wall_x = 10.5 + np.sqrt(16.0 - 0.25)
    assert abs(q[0] - (15.0 - wall_x)) < 1e-12
    assert 0.0 < q[0] <= 1.0


def test_bouzidi_halfway_equivalence():
    """bouzidi=False（q≡0.5）时力口径应等于旧式 2 Σ e_i f_i*。"""
    s = Solver(CylinderCase(nx=120, ny=60, d=8.0, cx_c=30.0,
                            u_in=0.1, re=60.0, bouzidi=False))
    for _ in range(20):
        s.step()
    fx, fy = s.force()
    fx2 = fy2 = 0.0
    for k in range(s._table["ys"].size):
        y, x, i = int(s._table["ys"][k]), int(s._table["xs"][k]), int(s._table["dirs"][k])
        fi_star = s.f[y, x, OPP[i]]
        fx2 += 2.0 * CX[i] * fi_star
        fy2 += 2.0 * CY[i] * fi_star
    assert abs(fx - fx2) < 1e-9 * max(1.0, abs(fx))
    assert abs(fy - fy2) < 1e-9 * max(1.0, abs(fy))


def test_bouzidi_runs_stable():
    """Bouzidi 打开：无 nan、力为正、固体保持清零。"""
    s = Solver(CylinderCase(nx=120, ny=60, d=8.0, cx_c=30.0,
                            u_in=0.1, re=60.0, bouzidi=True))
    for _ in range(30):
        s.step()
    assert np.isfinite(s.f).all()
    fx, fy = s.force()
    assert fx > 0
    assert (s.f[s.solid] == 0.0).all()


def test_numba_matches_numpy():
    """numba 与 numpy 路径逐位一致（含 Bouzidi+海绵）。"""
    if not HAVE_NUMBA:
        pytest.skip("numba 不可用")
    kw = dict(nx=120, ny=60, d=8.0, cx_c=30.0, u_in=0.1, re=60.0)
    s1 = Solver(CylinderCase(**kw, bouzidi=True))
    s2 = Solver(CylinderCase(**kw, bouzidi=True))
    for _ in range(25):
        s1.step()          # numba
        s2._step_numpy()   # numpy
    assert np.allclose(s1.f, s2.f, rtol=0, atol=1e-12)
    f1, f2 = s1.force(), s2.force()
    assert abs(f1[0] - f2[0]) < 1e-9 and abs(f1[1] - f2[1]) < 1e-9


def test_sponge_damps_perturbation():
    """海绵层内密度扰动应快速衰减（出口慢波吸收）。"""
    c = CylinderCase(nx=120, ny=60, d=8.0, cx_c=30.0, u_in=0.1, re=60.0,
                     sponge_width=20, sponge_sigma=0.2)
    s = Solver(c)
    # 在海绵区注入密度扰动
    s.f[:, -10:, :] *= 1.01
    rho0, _, _ = s.macros_masked()
    amp0 = float(np.nanmax(np.abs(rho0[:, -10:] - c.rho0)))
    for _ in range(60):
        s.step()
    rho1, _, _ = s.macros_masked()
    amp1 = float(np.nanmax(np.abs(rho1[:, -10:] - c.rho0)))
    assert amp1 < 0.25 * amp0
