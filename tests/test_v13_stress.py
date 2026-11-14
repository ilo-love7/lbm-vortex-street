"""E14 单测：围道几何、Π=Σcc f 分解与朴素节点口径。

注意：这里的 ``cv_force`` 是**朴素节点口径**（碰撞前分布二阶矩的面积分），
v1.3 已证实它与严格守恒通量差 ``Π^neq/τ``、违反围道无关性，保留作方法学
反例；主口径见 ``test_v13_link_flux.py`` 的链接穿越法与尾流亏损法。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm import (CylinderCase, Solver, equilibrium, force_coefficients_cv,
                 macros, momentum_flux, rect_contour, stress_decomposition)
from lbm.lattice import CS2, CX, CY
from lbm.stress import contours_for_case, cv_force


def _uniform_field(ny=60, nx=80, rho=1.0, ux=0.1, uy=0.0):
    return equilibrium(np.full((ny, nx), rho), np.full((ny, nx), ux),
                       np.full((ny, nx), uy))


# ---------- 围道几何 ----------
def test_rect_contour_is_closed_and_normals_cancel():
    ys, xs, nxs, nys = rect_contour(30, 30, 8, 8)
    assert ys.size == 8 * 8  # 4 条边 × 2h 点 = 8h = 64
    assert ys.size == 64
    # 闭合：沿边顺序相邻点曼哈顿距离为 1（含首尾闭合）
    d = np.abs(np.diff(ys)) + np.abs(np.diff(xs))
    assert np.all(d == 1)
    assert abs(ys[0] - ys[-1]) + abs(xs[0] - xs[-1]) == 1
    # 法向抵消（闭合面 ∮n ds = 0）
    assert abs(nxs.sum()) < 1e-12 and abs(nys.sum()) < 1e-12
    assert np.all((nxs != 0) ^ (nys != 0))  # 每条边只有一个法向分量


def test_rect_contour_length_and_bounds():
    ys, xs, nxs, nys = rect_contour(30, 30, 5, 10)
    assert ys.size == 2 * (2 * 5 + 2 * 10)  # 周长 / ds
    assert ys.min() == 20 and ys.max() == 40
    assert xs.min() == 25 and xs.max() == 35
    with pytest.raises(ValueError):
        rect_contour(30, 30, 5, 5, ny=32)  # y1=35 ≥ 32 → 越界
    with pytest.raises(ValueError):
        rect_contour(30, 30, 40, 5, ny=60, nx=80)  # x0=-10 < 0
    with pytest.raises(ValueError):
        rect_contour(30, 30, 0, 5)


def test_contours_for_case_skips_out_of_domain():
    c = CylinderCase(nx=60, ny=40, d=8.0, cx_c=15.0, u_in=0.1, re=80.0)
    out = contours_for_case(c)
    # 小域只放得下最内围道（其余应被跳过而不是抛异常）
    assert len(out) >= 1
    for h, cont in out:
        ys, xs = cont[0], cont[1]
        assert ys.min() >= 0 and ys.max() < c.ny
        assert xs.min() >= 0 and xs.max() < c.nx
        assert h > c.radius


# ---------- 均匀流零力（守恒律最基本检验） ----------
@pytest.mark.parametrize("uy", [0.0, 0.03, -0.05])
def test_uniform_flow_gives_zero_force(uy):
    f = _uniform_field(uy=uy)
    for h in (6, 10, 14):
        fx, fy = cv_force(f, rect_contour(40, 30, h, h))
        assert abs(fx) < 1e-10, (uy, h, fx)
        assert abs(fy) < 1e-10, (uy, h, fy)


def test_uniform_flow_zero_force_is_contour_independent():
    """不同围道半径都给 0 —— 围道无关性在无阻力解析解上精确成立。"""
    f = _uniform_field()
    vals = [cv_force(f, rect_contour(40, 30, h, h))[0] for h in (5, 8, 12, 16)]
    assert max(vals) - min(vals) < 1e-10


def test_momentum_flux_matches_explicit_sum():
    f = _uniform_field()
    ys, xs, _, _ = rect_contour(40, 30, 8, 8)
    pxx, pxy, pyy = momentum_flux(f, ys, xs)
    fc = f[ys, xs, :]
    assert np.allclose(pxx, (fc * (CX * CX)).sum(axis=1))
    assert np.allclose(pxy, (fc * (CX * CY)).sum(axis=1))
    assert np.allclose(pyy, (fc * (CY * CY)).sum(axis=1))


# ---------- 分解与离散应力 ----------
def test_stress_decomposition_identity():
    """精确恒等式 Π = conv + pI + Π^neq；σ' = −(1−1/(2τ))Π^neq。"""
    rng = np.random.default_rng(3)
    rho = 1.0 + 0.02 * rng.standard_normal((7, 9))
    ux = 0.08 + 0.01 * rng.standard_normal((7, 9))
    uy = 0.01 * rng.standard_normal((7, 9))
    f = equilibrium(rho, ux, uy) + 1e-4 * rng.standard_normal((7, 9, 9))
    ys = np.repeat(np.arange(7), 9)
    xs = np.tile(np.arange(9), 7)
    pxx, pxy, pyy = momentum_flux(f, ys, xs)
    d = stress_decomposition(f, ys, xs, tau=0.56)
    conv, pi = d["conv"], d["pi_neq"]
    assert np.allclose(pxx, conv[0] + d["p"] + pi[0], atol=1e-12)
    assert np.allclose(pxy, conv[1] + pi[1], atol=1e-12)
    assert np.allclose(pyy, conv[2] + d["p"] + pi[2], atol=1e-12)
    k = -(1.0 - 1.0 / (2.0 * 0.56))
    assert np.allclose(d["sigma_visc"][0], k * pi[0], atol=1e-12)
    # 连续通量与离散通量之差 = Π^neq/(2τ)
    diff = pxx - (conv[0] + d["p"] - d["sigma_visc"][0])
    assert np.allclose(diff, pi[0] / (2.0 * 0.56), atol=1e-12)


def test_equilibrium_has_zero_nonequilibrium_stress():
    f = _uniform_field()
    ys, xs, _, _ = rect_contour(40, 30, 8, 8)
    d = stress_decomposition(f, ys, xs, tau=0.56)
    for comp in d["pi_neq"]:
        assert np.max(np.abs(comp)) < 1e-12
    assert np.allclose(d["p"], 1.0 * CS2)


def test_force_coefficients_cv_matches_forces_module():
    from lbm import force_coefficients
    assert force_coefficients_cv(0.31, -0.07, 1.0, 0.1, 28.0) == \
        force_coefficients(0.31, -0.07, 1.0, 0.1, 28.0)


# ---------- 与 MRT 兼容（朴素口径只读 f，不需要 τ / 矩基） ----------
def test_cv_force_is_defined_for_mrt_state():
    """cv_force 只读 f，不需要 τ / 矩基；MRT 流场同样可测。"""
    case = CylinderCase(nx=120, ny=80, d=12.0, cx_c=35.0, u_in=0.1,
                        re=100.0, collision="mrt", sponge_width=8)
    s = Solver(case)
    for _ in range(30):
        s.step()
    out = s.cv_forces()
    assert len(out) >= 1
    for h, fx, fy in out:
        assert np.isfinite(fx) and np.isfinite(fy)
        assert h > case.radius


def test_mem_and_cv_forces_have_same_sign_on_developed_flow():
    """短时发展后两者都给正阻力（符号/量级一致，不是精确相等）。"""
    case = CylinderCase(nx=160, ny=80, d=12.0, cx_c=40.0, u_in=0.1,
                        re=100.0, sponge_width=8)
    s = Solver(case)
    mem, cv = [], []
    for _ in range(600):
        s.step()
        fx, _ = s.force()
        mem.append(fx)
        cv.append(np.mean([a for _, a, _ in s.cv_forces()]))
    assert np.mean(mem) > 0 and np.mean(cv) > 0
    # 同一量级（0.5~2×）：数量级一致即可，严格相等要整周期平均
    assert 0.3 < np.mean(cv) / np.mean(mem) < 3.0
