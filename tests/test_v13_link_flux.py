"""E14 (v1.3) 单测：严格离散守恒的链接穿越法 + 远场尾流动量亏损。

核心命题（v1.3 实测发现，故用单测锁死）：

* 链接穿越法扣除 ``−d/dt∫_V ρu`` 后，是离散动量守恒律的**恒等式**，
  因此不同围道逐步逐位一致，且精确复现动量交换法（MEM）。
  ⇒ 面积分并不独立于 MEM，不能用来判定 MRT 的 Cd 偏移是不是测力伪影。
* 围道上碰撞前分布的二阶矩 ``Σcc f``（朴素节点口径）少了 ``Π^neq/τ``，
  近场系统性高估且违反围道无关性 —— 作为方法学反例保留。
* 只有远场尾流动量亏损（只读守恒矩 ρ/ρu）是真正独立的判据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm import (CylinderCase, Solver, box_mask, box_momentum,
                 build_link_flux, cv_force, cv_force_links, equilibrium,
                 link_boxes_for_case, macros, rect_contour,
                 station_consistency, wake_force, wake_ref_x, wake_stations)
from lbm.lattice import CS2, CX, CY, NL

SMALL = dict(nx=140, ny=80, d=12.0, cx_c=40.0, u_in=0.1,
             re=100.0, sponge_width=8)


def _uniform(ny=60, nx=80, rho=1.0, ux=0.1, uy=0.0):
    return equilibrium(np.full((ny, nx), rho), np.full((ny, nx), ux),
                       np.full((ny, nx), uy))


def _proto_links(f, box):
    """逐方向全场切片的朴素实现（慢，只作正确性对照）。"""
    ny, nx, _ = f.shape
    fx = fy = 0.0
    for i in range(NL):
        cx, cy = int(CX[i]), int(CY[i])
        sy0, sy1 = max(0, -cy), min(ny, ny - cy)
        sx0, sx1 = max(0, -cx), min(nx, nx - cx)
        dy0, dy1 = sy0 + cy, sy1 + cy
        dx0, dx1 = sx0 + cx, sx1 + cx
        in_dst = box[dy0:dy1, dx0:dx1]
        in_src = box[sy0:sy1, sx0:sx1]
        vals = f[dy0:dy1, dx0:dx1, i]
        w = np.where(in_dst & ~in_src, 1.0,
                     np.where(~in_dst & in_src, -1.0, 0.0)) * vals
        fx += cx * float(w.sum())
        fy += cy * float(w.sum())
    return fx, fy


# ---------- 几何与建表 ----------
def test_box_mask_matches_rect_contour_geometry():
    b = box_mask(30, 30, 8, 8, ny=60, nx=80)
    assert b.shape == (60, 80)
    assert b.sum() == 17 * 17
    ys, xs, _, _ = rect_contour(30, 30, 8, 8)
    assert np.all(b[ys, xs])                      # 围道节点都在 V 内
    assert not b[29 - 9, 30] and not b[30, 30 + 9]  # V 外一层为 False


def test_box_mask_rejects_out_of_domain():
    with pytest.raises(ValueError):
        box_mask(30, 30, 40, 5, ny=60, nx=80)
    with pytest.raises(ValueError):
        box_mask(30, 30, 0, 5, ny=60, nx=80)


def test_build_link_flux_enter_exit_balance_per_direction():
    b = box_mask(30, 30, 8, 8, ny=60, nx=80)
    fl = build_link_flux(b)
    assert fl.dst.size == fl.dirs.size == fl.signs.size
    assert fl.nodes.size == 17 * 17
    for i in range(NL):
        n_in = int(np.sum((fl.dirs == i) & (fl.signs > 0)))
        n_out = int(np.sum((fl.dirs == i) & (fl.signs < 0)))
        assert n_in == n_out, f"方向 {i} 入/出链接数不平衡"
        if i != 0:                       # 静止方向不跨界，不产生链接
            assert n_in > 0


def test_build_link_flux_rejects_full_domain_box():
    with pytest.raises(ValueError):
        build_link_flux(np.ones((20, 30), dtype=bool))


def test_cv_force_links_matches_explicit_direction_loop():
    f = _uniform()
    # 叠加一点非平衡扰动，确保不是靠对称性蒙混过关
    rng = np.random.default_rng(0)
    f = f + 1e-4 * rng.standard_normal(f.shape)
    b = box_mask(30, 30, 8, 8, ny=60, nx=80)
    fl = build_link_flux(b)
    assert np.allclose(cv_force_links(f, fl), _proto_links(f, b), atol=1e-12)
    assert np.allclose(cv_force_links(f, b), _proto_links(f, b), atol=1e-12)
    assert np.allclose(fl(f), _proto_links(f, b), atol=1e-12)


def test_cv_force_links_uniform_flow_is_zero_for_every_box_size():
    f = _uniform()
    for h in (5, 8, 12, 18):
        fl = build_link_flux(box_mask(30, 30, h, h, ny=60, nx=80))
        fx, fy = cv_force_links(f, fl)
        assert abs(fx) < 1e-10 and abs(fy) < 1e-10


def test_cv_force_links_rejects_shape_mismatch():
    fl = build_link_flux(box_mask(30, 30, 8, 8, ny=60, nx=80))
    with pytest.raises(ValueError):
        cv_force_links(np.zeros((30, 30, NL)), fl)


def test_box_momentum_is_exactly_sum_of_rho_u():
    f = _uniform(rho=1.2, ux=0.07, uy=0.02)
    fl = build_link_flux(box_mask(30, 30, 8, 8, ny=60, nx=80))
    rho, ux, uy = macros(f)
    b = np.zeros((60, 80), dtype=bool)
    b[22:39, 22:39] = True
    assert box_momentum(f, fl) == pytest.approx(
        (float(np.sum((rho * ux)[b])), float(np.sum((rho * uy)[b]))),
        abs=1e-12)


# ---------- 核心：严格守恒恒等式 ----------
def test_links_minus_accumulation_equals_mem_and_is_contour_independent():
    """逐步扣除非定常累积项后，三围道逐位一致且逐位等于 MEM。"""
    case = CylinderCase(**SMALL)
    s = Solver(case)
    for _ in range(300):
        s.step()
    boxes = link_boxes_for_case(case)
    fluxes = {h: build_link_flux(b) for h, b in boxes}
    assert len(fluxes) >= 2
    q = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    raw = {h: [] for h in fluxes}
    corr = {h: [] for h in fluxes}
    mem = []
    for _ in range(120):
        p0 = {h: box_momentum(s.f, fluxes[h]) for h in fluxes}
        s.step()
        mem.append(s.force()[0] / q)
        for h, fl in fluxes.items():
            lx = cv_force_links(s.f, fl)[0]
            p1 = box_momentum(s.f, fl)
            raw[h].append(lx / q)
            corr[h].append((lx - (p1[0] - p0[h][0])) / q)
    mem = np.asarray(mem)
    for h in fluxes:
        c = np.asarray(corr[h])
        assert np.max(np.abs(c - mem)) < 1e-10, f"h={h} 未逐位复现 MEM"
    hs = list(fluxes)
    for a, b in zip(hs[:-1], hs[1:]):
        assert np.max(np.abs(np.asarray(corr[a]) - np.asarray(corr[b]))) < 1e-10
    # 未修正的链接口径确实带非定常相位残差（说明修正不是空操作）
    assert max(np.mean(raw[h]) for h in fluxes) - \
        min(np.mean(raw[h]) for h in fluxes) > 1e-4


def test_solver_helpers_expose_four_methods():
    case = CylinderCase(**SMALL)
    s = Solver(case)
    s.step()
    assert len(s.cv_forces()) == len(s.link_forces()) == len(s._fluxes)
    assert set(s.box_momenta()) == {h for h, _ in s._fluxes}
    wf = s.wake_forces()
    assert wf and all(isinstance(w, dict) and "force" in w for _, w in wf)
    assert all(np.isfinite(w["force"]) for _, w in wf)


def test_naive_node_pi_violates_contour_independence_but_links_do_not():
    """方法学反例：Σcc f 面积分随围道漂移，链接法（修正后）不漂。"""
    case = CylinderCase(**SMALL)
    s = Solver(case)
    for _ in range(300):
        s.step()
    from lbm import contours_for_case
    contours = {h: c for h, c in contours_for_case(case)}
    fluxes = {h: build_link_flux(b) for h, b in link_boxes_for_case(case)}
    node = {h: [] for h in contours}
    link = {h: [] for h in fluxes}
    for _ in range(120):
        p0 = {h: box_momentum(s.f, fluxes[h]) for h in fluxes}
        s.step()
        p1 = {h: box_momentum(s.f, fluxes[h]) for h in fluxes}
        for h, c in contours.items():
            node[h].append(cv_force(s.f, c)[0])
        for h, fl in fluxes.items():
            link[h].append(cv_force_links(s.f, fl)[0]
                           - (p1[h][0] - p0[h][0]))
    node_mean = {h: float(np.mean(v)) for h, v in node.items()}
    link_mean = {h: float(np.mean(v)) for h, v in link.items()}
    nspread = (max(node_mean.values()) - min(node_mean.values())) \
        / abs(np.mean(list(node_mean.values())))
    lspread = (max(link_mean.values()) - min(link_mean.values())) \
        / abs(np.mean(list(link_mean.values())))
    assert nspread > 0.01, f"朴素口径应明显违反围道无关性，实测 {nspread:.4%}"
    assert lspread < 1e-8, f"链接法应严格围道无关，实测 {lspread:.4%}"


# ---------- 远场尾流动量亏损 ----------
def test_wake_force_uniform_flow_is_zero_both_modes():
    f = _uniform(ny=60, nx=80, rho=1.0, ux=0.1)
    for ref in (None, 5):
        r = wake_force(f, 40, 1.0, 0.1, x_ref=ref)
        assert r["force"] == pytest.approx(0.0, abs=1e-10)
        assert r["momentum"] == pytest.approx(0.0, abs=1e-10)
        assert r["pressure"] == pytest.approx(0.0, abs=1e-10)
        assert r["viscous"] == 0.0
        assert r["mode"] == ("two_section" if ref is not None else "classic")


def test_two_section_mode_equals_explicit_flux_difference():
    """严格两截面口径 = Σ_ref(ρu²+p) − Σ_x(ρu²+p)，逐项可复核。"""
    rng = np.random.default_rng(3)
    f = _uniform(ny=60, nx=80, rho=1.0, ux=0.1) \
        + 1e-4 * rng.standard_normal((60, 80, NL))
    from lbm.lattice import CS2 as _cs2
    def flux(x):
        fc = f[1:-1, x, :]
        rho = fc.sum(axis=1)
        ux = (fc @ CX.astype(np.float64)) / np.where(rho > 0, rho, 1.0)
        return float(np.sum(rho * ux * ux + rho * _cs2))
    r = wake_force(f, 40, 1.0, 0.1, x_ref=5)
    assert r["force"] == pytest.approx(flux(5) - flux(40), abs=1e-10)
    assert r["ref_flux"] == pytest.approx(flux(5), abs=1e-10)
    assert r["station_flux"] == pytest.approx(flux(40), abs=1e-10)
    assert np.isfinite(r["mass_flux_ref"])


def test_wake_ref_x_is_after_west_sponge():
    assert wake_ref_x(20) == 22
    assert wake_ref_x(32, margin=3) == 35


def test_wake_force_viscous_branch_requires_valid_tau():
    f = _uniform()
    with pytest.raises(ValueError):
        wake_force(f, 40, 1.0, 0.1, include_viscous=True)
    with pytest.raises(ValueError):
        wake_force(f, 40, 1.0, 0.1, include_viscous=True, tau=0.5)
    r = wake_force(f, 40, 1.0, 0.1, include_viscous=True, tau=0.6)
    assert np.isfinite(r["viscous"])


def test_wake_stations_skip_sponge_and_scale_with_d():
    st = wake_stations(170.0, 28.0, 840, stations=(3, 6, 10, 40),
                       downstream_guard=42)
    assert [s for s, _ in st] == [3, 6, 10]
    assert all(0 < x < 798 for _, x in st)
    assert st[1][1] - st[0][1] == 3 * 28


def test_station_consistency_matches_definition():
    r = station_consistency({"3": 1.4, "6": 1.41, "10": 1.39})
    assert r["n"] == 3
    assert r["spread"] == pytest.approx(0.02)
    assert r["rel_spread"] == pytest.approx(0.02 / 1.4)
    assert np.isnan(station_consistency({})["spread"])
