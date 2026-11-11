"""v1.2 单测：MRT 碰撞算子。"""
import numpy as np
import pytest

from lbm.kernels import CX as KCX, CY as KCY
from lbm.lattice import equilibrium
from lbm.mrt import mrt_collide, mrt_matrices, mrt_meq, mrt_relax_rates


def test_matrix_rows_match_velocity_order():
    M, Minv = mrt_matrices()
    # jx 行必须等于 CX，jy 行等于 CY（置换正确性的锚点）
    assert np.allclose(M[3], KCX.astype(float))
    assert np.allclose(M[5], KCY.astype(float))
    # ρ 行全 1
    assert np.allclose(M[0], 1.0)


def test_minv_is_inverse():
    M, Minv = mrt_matrices()
    assert np.allclose(M @ Minv, np.eye(9), atol=1e-12)


def test_meq_consistent_with_M_feq():
    M, _ = mrt_matrices()
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.01 * rng.standard_normal(20)
    ux = 0.05 * rng.standard_normal(20)
    uy = 0.05 * rng.standard_normal(20)
    f = equilibrium(rho, ux, uy)          # (20, 9)
    m_feq = f @ M.T
    m_analytic = mrt_meq(rho * 1.0, rho * ux, rho * uy)
    assert np.allclose(m_feq, m_analytic, atol=1e-11)


def test_mrt_reduces_to_bgk():
    # 全 s=1/τ 时 MRT 碰撞 ≡ BGK 碰撞（机器精度）
    rng = np.random.default_rng(1)
    f = 0.3 + 0.01 * rng.standard_normal((6, 7, 9))
    tau = 0.9
    M, Minv = mrt_matrices()
    S = mrt_relax_rates(1.0 / tau, preset="bgk")
    out = mrt_collide(f, M, Minv, S, tau)
    rho, ux, uy = f.sum(-1), (f * KCX).sum(-1) / f.sum(-1), (f * KCY).sum(-1) / f.sum(-1)
    feq = equilibrium(rho, ux, uy)
    bgk = f + -(1.0 / tau) * (f - feq)
    assert np.allclose(out, bgk, atol=1e-11)


def test_relax_rate_presets():
    s_bgk = mrt_relax_rates(0.6, "bgk")
    assert np.allclose(s_bgk, 0.6)
    s_magic = mrt_relax_rates(0.6, "magic")
    assert s_magic[0] == 0.6 and s_magic[3] == 0.6   # ρ/jx 不弛豫
    assert s_magic[1] == 1.1 and s_magic[2] == 1.1   # s_e=s_ε
    assert s_magic[4] == 1.2 and s_magic[6] == 1.2   # s_q
    assert s_magic[7] == 0.6 and s_magic[8] == 0.6   # s_ν=1/τ
    with pytest.raises(ValueError):
        mrt_relax_rates(0.6, "nope")


def test_unknown_and_shape():
    M, Minv = mrt_matrices()
    f = np.full((3, 3, 9), 0.1)
    S = mrt_relax_rates(1.0, "magic")
    out = mrt_collide(f, M, Minv, S, 1.0)
    assert out.shape == f.shape


def test_solver_accepts_mrt():
    from lbm import CylinderCase, Solver
    case = CylinderCase(nx=60, ny=40, d=8.0, cx_c=15.0, u_in=0.1, re=80.0,
                        collision="mrt", mrt_preset="magic")
    s = Solver(case)
    for _ in range(20):
        s.step()
    rho, ux, uy = s.macros_masked()
    assert np.all(np.isfinite(ux)) and float(np.nanmax(np.abs(ux))) < 1.0
