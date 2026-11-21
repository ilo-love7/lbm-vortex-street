"""v1.4 三维校验单测：D3Q19 晶格性质、守恒、入口/壁面、三重解析验证。

这些测试刻意做成"小网格 + 短时窗"，只锁**不变量**与**解析判据**，
生产数值由 ``scripts/08_3d_validation.py`` 负责（带缓存与统计窗口）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm.kernels3d import compile_step3d
from lbm.lattice3d import (CS2, CX3, CY3, CZ3, NL3, OPP3, SPEC_Y3, W3,
                           equilibrium3d, lattice3d_checks, macros3d)
from lbm.solver3d import (Case3D, Solver3D, channel_poiseuille,
                          shear_wave_decay, taylor_green)


@pytest.fixture(scope="module", autouse=True)
def _compile():
    compile_step3d()


# ---------------------------------------------------------------- 晶格
def test_lattice3d_properties():
    r = lattice3d_checks()
    assert r["mass"] < 1e-14
    assert r["momentum"] < 1e-14
    assert r["isotropy"] < 1e-14
    assert r["quad_moment"] < 1e-14
    assert r["n_unique"] == 19
    assert r["opposite"] < 1e-14
    assert W3.sum() == pytest.approx(1.0)


def test_opposite_is_involution():
    assert all(OPP3[OPP3[i]] == i for i in range(NL3))
    assert all(OPP3[i] != i for i in range(1, NL3))


def test_specular_map_is_involution_and_flips_only_cy():
    """镜面映射必须只翻转 cy；三维里它与对向表不同（16/19 个方向不同）。"""
    for i in range(NL3):
        j = SPEC_Y3[i]
        assert CX3[j] == CX3[i]
        assert CY3[j] == -CY3[i]
        assert CZ3[j] == CZ3[i]
        assert SPEC_Y3[j] == i
    assert sum(SPEC_Y3[i] != OPP3[i] for i in range(NL3)) == 16


def test_equilibrium_roundtrip():
    rng = np.random.default_rng(0)
    shape = (4, 5, 6)
    ux = rng.uniform(-0.05, 0.05, shape)
    uy = rng.uniform(-0.05, 0.05, shape)
    uz = rng.uniform(-0.05, 0.05, shape)
    rho = 1.0 + rng.uniform(-0.01, 0.01, shape)
    f = equilibrium3d(rho, ux, uy, uz)
    assert f.shape == shape + (NL3,)
    r, a, b, c = macros3d(f)
    assert np.max(np.abs(r - rho)) < 1e-13
    assert np.max(np.abs(a - ux)) < 1e-13
    assert np.max(np.abs(b - uy)) < 1e-13
    assert np.max(np.abs(c - uz)) < 1e-13


def test_cs2_is_one_third():
    assert CS2 == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------- 守恒/边界
def _uniform_case(bc: str) -> Solver3D:
    case = Case3D(nx=12, ny=12, nz=6, d=0.0, u_in=0.1, re=1.0,
                  top_bottom=bc, inlet="periodic", outlet="periodic",
                  sponge_width=0, sponge_sigma=0.0, trip=0.0, tau_direct=0.6)
    sim = Solver3D(case)
    rho = np.ones((6, 12, 12))
    sim.f = equilibrium3d(rho, np.full_like(rho, 0.1),
                          np.zeros_like(rho), np.zeros_like(rho))
    return sim


@pytest.mark.parametrize("bc", ["periodic", "free_slip"])
def test_uniform_flow_preserved(bc):
    """均匀流必须逐位保持；自由滑移壁不得吃掉切向动量（v1.4 曾因此崩）。"""
    sim = _uniform_case(bc)
    for _ in range(200):
        sim.step()
    rho, ux, uy, uz = sim.macros()
    assert np.max(np.abs(ux - 0.1)) < 1e-12
    assert np.max(np.abs(uy)) < 1e-12
    assert np.max(np.abs(rho - 1.0)) < 1e-12


def test_inlet_prescribes_velocity_and_density():
    """3D Zou/He：入口平面应给出 ux≈U、rho≈1（一般化质量守恒式的功能检验）。"""
    case = Case3D(nx=64, ny=24, nz=8, d=0.0, u_in=0.1, re=1.0,
                  top_bottom="free_slip", inlet="zouhe",
                  outlet="convective", sponge_width=16, trip=0.0,
                  tau_direct=0.6)
    sim = Solver3D(case)
    for _ in range(400):
        sim.step()
    rho, ux, uy, uz = sim.macros()
    assert abs(float(ux[:, :, 0].mean()) - 0.1) < 2e-3
    assert abs(float(rho[:, :, 0].mean()) - 1.0) < 5e-3
    assert np.all(np.isfinite(sim.f))


# ---------------------------------------------------------------- 解析验证
def test_poiseuille_second_order():
    # steps 默认 None ⇒ 8H²/ν（H 越大越自动加步）：固定 40000 步在 H=63
    # 只有 6.6 个扩散时间，残余瞬变会把二阶收敛误判成零阶（v1.4 首轮实测
    # ny=65 的 L2 反而回升、收敛阶 −0.13）。
    coarse = channel_poiseuille(ny=9, nz=3, nx=3, tau=0.7, g=1e-6)
    fine = channel_poiseuille(ny=33, nz=3, nx=3, tau=0.7, g=1e-6)
    assert coarse["u_max_num"] / coarse["u_max_exact"] > 0.95
    assert fine["u_max_num"] / fine["u_max_exact"] > 0.995
    order = np.log(coarse["l2_rel"] / fine["l2_rel"]) / np.log((33 - 2) / (9 - 2))
    assert order > 1.6          # 半程反弹 + Guo 力应为二阶
    assert fine["drift"] < 1e-5     # 末段 1000 步幅值漂移，确已到稳态
    assert fine["steps"] >= 15000   # 自动步数按 12·H²/(π²ν) 缩放（H=31）


def test_shear_wave_viscosity_recovery():
    r = shear_wave_decay(n=24, nz=3, tau=0.6, amp=1e-3)
    assert r["rel_err"] < 0.03
    assert r["nu_fit"] == pytest.approx(r["nu_input"], rel=0.03)
    assert r["n_fit"] > 20          # 相对幅值窗口有足够拟合点


def test_shear_wave_large_tau_noise_floor():
    """τ=1.2 高衰减档：旧实现（绝对阈值 1e-16 + 1200 步）ν 低估 25%。

    幅值在 ~840 步内就跌到 double 舍入噪声，末段"有效点"全是噪声。
    相对幅值窗口 [1e-3, 0.6] 修复后误差应与低 τ 档同量级。
    """
    r = shear_wave_decay(n=16, nz=3, tau=1.2, amp=1e-3)
    assert r["rel_err"] < 0.03, f"nu_fit={r['nu_fit']:.5f}"
    assert r["fit_lo"] > 1e-3       # 噪声地板被截掉


@pytest.mark.parametrize("tau", [0.55, 0.8])
def test_taylor_green_initial_decay(tau):
    """3D TG 初始斜率 = −6νk_l²，k_l = 2π/n（格子单位，不是物理单位）。"""
    r = taylor_green(n=32, tau=tau, amp=1e-3, steps=500, sample=10)
    assert r["rel_err"] < 0.05
    assert r["div_max"] < 5e-3


def test_taylor_green_needs_lattice_wavenumber():
    """口径回归：若把 k_l 当成 1（物理单位误用），会得到约 2π 倍的假偏差。"""
    r = taylor_green(n=32, tau=0.55, amp=1e-3, steps=300, sample=10)
    assert r["k_lattice"] == pytest.approx(2 * np.pi / 32)
    wrong = -6.0 * r["nu"]
    assert abs(r["slope0"] / wrong) < 0.3     # 与错口径相差 25 倍以上


# ---------------------------------------------------------------- 几何/测力
def test_link_table_bouzidi_geometry():
    """D3Q19 对角方向的 q 必须等于同 (y,x) 面内方向的 q（圆柱沿 z 挤出）。"""
    case = Case3D(nx=80, ny=80, nz=4, d=16.0, cx_c=40.0, u_in=0.1, re=100.0)
    sim = Solver3D(case)
    t = sim._tab
    assert t["ly"].size > 100
    assert np.all(t["lq"] > 0.0) and np.all(t["lq"] <= 1.0)
    assert np.all(sim.fluid[t["ly"], t["lx"]])
    # dir 11 = (1,0,1) 与 dir 1 = (1,0,0) 在相同节点的 q 相同
    idx1 = np.flatnonzero(t["ld"] == 1)
    idx11 = np.flatnonzero(t["ld"] == 11)
    m1 = {(int(t["ly"][k]), int(t["lx"][k])): t["lq"][k] for k in idx1}
    common = [(k, m1[(int(t["ly"][k]), int(t["lx"][k]))])
              for k in idx11
              if (int(t["ly"][k]), int(t["lx"][k])) in m1]
    assert common
    for k, q1 in common:
        assert t["lq"][k] == pytest.approx(q1, abs=1e-12)


def test_cylinder_force_matches_2d():
    """同 D/同 β/同 Re：3D 单位深度阻力应与 2D 一致（Re=100 展向不变）。"""
    from lbm import CylinderCase, Solver

    geo = dict(nx=160, d=16.0, cx_c=40.0, u_in=0.1, re=100.0)
    c2 = CylinderCase(ny=80, sponge_width=16, **geo)
    s2 = Solver(c2)
    for _ in range(1200):
        s2.step()
    cd2 = np.mean([s2.force()[0] for _ in range(200)])
    c3 = Case3D(ny=80, nz=8, sponge_width=16, **geo)
    s3 = Solver3D(c3)
    for _ in range(1200):
        s3.step()
    cd3 = np.mean([s3.force()[0] for _ in range(200)])
    q = 0.5 * geo["u_in"] ** 2 * geo["d"]
    assert abs(cd3 / q / (cd2 / q) - 1.0) < 0.15
    assert abs(s3.force3()[2]) < 1e-6 * max(abs(cd3), 1e-12) + 1e-9


def test_spanwise_diagnostics_zero_for_2d_flow():
    """Re=100 无三维不稳定性：u_z 只能来自启动瞬变，且必须衰减掉。

    判据是**衰减**而不是"一上来就机器零"。v1.4 首版单测取 600 步、
    断言 ``<1e-5`` 失败（实测 1.53e-4），诊断（nx=120/ny=64/nz=8）给出
    uz_rel 随步数单调下降：100→6.7e-4、300→3.1e-4、600→1.5e-4、
    1200→8.5e-6、2400→3.5e-6，fz 同步从 7.7e-6 掉到 4.4e-8。
    所以 600 步抓到的是初始均匀平衡态与圆柱不相容产生的声学瞬变尾巴，
    不是展向失稳；生产算例（nx=320、6000+ 步）为 1.5e-7。
    """
    case = Case3D(nx=120, ny=64, nz=8, d=16.0, cx_c=30.0, u_in=0.1, re=100.0,
                  sponge_width=12)
    sim = Solver3D(case)
    for _ in range(300):
        sim.step()
    uz_early = sim.spanwise_diagnostics()["rel"]
    for _ in range(900):
        sim.step()
    uz_late = sim.spanwise_diagnostics()["rel"]
    _, _, fz = sim.force3()
    assert uz_late < 3e-5                 # 瞬变尾巴已掉到 1e-5 量级
    assert uz_late < uz_early / 5.0       # 必须衰减，不能增长（锁失稳）
    assert abs(fz) < 1e-5                 # 展向净力同步消失
