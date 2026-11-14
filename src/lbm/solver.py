"""D2Q9 + BGK LBM 求解器：圆柱绕流（Bouzidi 次网格反弹）+ Zou/He 入口 + 对流出口。

实现要点（相对 Mocz 教学版的严谨化）：
- Bouzidi 次网格反弹（q 解析壁距；q≡0.5 时即半程反弹 f_ī(x_f)=f_i*(x_f)）
- 动量交换测力：F = Σ e_i [f_i* + f_ī_new]（反弹阶段累积，半程时 = 2 Σ e_i f_i*）
- Zou/He 速度入口（西，ux=U, uy=0）；一阶对流出口（东）；顶底自由滑移（镜面）或无滑移
- 海绵层（东西两端）：向目标态 feq(ρ0,U,0) 弛豫，吸收出/入口反射的慢变压力波，
  消除高 Re 长时间力测量的慢波污染（v1.0 实测 τ≲0.55 时边界组合泵慢波）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bouzidi import apply_bb, build_link_table
from .kernels import HAVE_NUMBA, _step_kernel
from .lattice import CX, CY, NL, OPP, W, equilibrium, macros
from .les import (inlet_base, inlet_uy, nu_t_over_nu0, pi_neq_norm,
                  smagorinsky_tau)
from .mrt import mrt_collide, mrt_matrices, mrt_relax_rates
from .postprocess import literature_st
from .stress import (box_momentum, contours_for_case, cv_force,
                     cv_force_links, link_fluxes_for_case,
                     wake_stations_for_case)
from .wake import wake_force, wake_ref_x


@dataclass
class CylinderCase:
    """圆柱绕流几何与物性（全为格子单位）。"""

    nx: int = 400
    ny: int = 200
    d: float = 20.0            # 圆柱直径 [lu]
    cx_c: float = 100.0        # 圆柱中心 x [lu]
    cy_c: float | None = None  # 中心 y，默认 ny/2
    u_in: float = 0.08         # 入口速度 [lu/step]
    re: float = 100.0          # 雷诺数 Re = U D / ν
    rho0: float = 1.0
    top_bottom: str = "free_slip"   # free_slip | no_slip
    outlet: str = "convective"      # convective | extrapolate
    trip: float = 1e-3         # 初始横向扰动幅度（促发涡脱落）
    bouzidi: bool = True       # Bouzidi 次网格反弹（False → 半程反弹）
    sponge_width: int = 32     # 东西海绵层宽度 [lu]（0 关闭）
    sponge_sigma: float = 0.08  # 海绵层最大弛豫率
    collision: str = "bgk"     # bgk | mrt（v1.2）
    mrt_preset: str = "magic"  # magic | bulk | bgk（见 mrt.mrt_relax_rates）
    les: bool = False          # v1.3：Smagorinsky 亚格子涡黏（仅 BGK）
    c_smag: float = 0.16       # v1.3：Smagorinsky 常数 C_s
    inlet_eps: float = 0.0     # v1.3：持续入口横向扰动幅值 ε（相对 U）
    inlet_m: int = 3           # v1.3：扰动剖面沿 y 的波数（整数 ⇒ 零均值）
    inlet_period_ratio: float = 2.37  # v1.3：扰动周期 / 自然脱落周期 T（非整数避锁频）

    @property
    def radius(self) -> float:
        return self.d / 2.0

    @property
    def cy(self) -> float:
        return self.ny / 2.0 if self.cy_c is None else self.cy_c

    @property
    def nu(self) -> float:
        return self.u_in * self.d / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5


class Solver:
    def __init__(self, case: CylinderCase):
        self.case = case
        c = case
        # v1.0 的 0.51 下限是「半程反弹 + 无海绵」口径下的经验红线；
        # Bouzidi+海绵已把它压到 τ≈0.524（v1.2 E10 实测），LES 又额外增黏，
        # 因此 LES 工况只要求 τ₀>0.5（ν₀>0），其余保持原判据不变。
        lo = 0.5001 if c.les else 0.51
        if not (lo < c.tau < 3.0):
            raise ValueError(f"tau={c.tau:.3f} 超出稳定区间，调整 U/D/Re")
        if c.les and c.collision != "bgk":
            raise ValueError("Smagorinsky LES 仅支持 BGK 碰撞（MRT 各矩弛豫率不同，隐式解不成立）")
        yy, xx = np.mgrid[0 : c.ny, 0 : c.nx]
        self.cyl = (xx - c.cx_c) ** 2 + (yy - c.cy) ** 2 < c.radius**2
        self.solid = self.cyl.copy()
        if c.top_bottom == "no_slip":
            self.solid[0, :] = True
            self.solid[-1, :] = True
        self.fluid = ~self.solid

        rho = np.full((c.ny, c.nx), c.rho0, dtype=np.float64)
        ux = np.full((c.ny, c.nx), c.u_in, dtype=np.float64)
        uy = np.zeros((c.ny, c.nx), dtype=np.float64)
        rng = np.random.default_rng(42)
        uy += c.trip * rng.standard_normal(uy.shape)
        self.f = equilibrium(rho, ux, uy)
        self.f[self.solid] = 0.0

        # 反弹/测力链接表（仅圆柱链接，避免壁面力污染升阻力）
        self._table = build_link_table(self.fluid, self.cyl,
                                       c.cx_c, c.cy, c.radius,
                                       bouzidi=c.bouzidi)
        # 诊断用掩码（与链接表同源）
        self._link = [np.zeros((c.ny, c.nx), dtype=bool) for _ in range(NL)]
        for k in range(self._table["ys"].size):
            self._link[int(self._table["dirs"][k])][
                int(self._table["ys"][k]), int(self._table["xs"][k])] = True
        self._bb = self._link  # 兼容旧名

        # 海绵层剖面：sigma(x) 两端 smoothstep 升到 sponge_sigma
        self._sigma_x = np.zeros(c.nx, dtype=np.float64)
        w = int(c.sponge_width)
        if w > 0 and c.sponge_sigma > 0.0:
            xi = np.linspace(0.0, 1.0, w)
            ramp = xi * xi * (3.0 - 2.0 * xi)  # smoothstep
            self._sigma_x[:w] = c.sponge_sigma * ramp[::-1]
            self._sigma_x[-w:] = c.sponge_sigma * ramp
        # 海绵目标态 feq(ρ0, U, 0)
        cu = CX * c.u_in
        self._ftarget = W * c.rho0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * c.u_in**2)

        self._cyl_u1 = self.cyl.astype(np.bool_)
        self._solid_u1 = self.solid.astype(np.bool_)
        self._free_slip = case.top_bottom == "free_slip"
        self._convective = case.outlet == "convective"
        self._fx = 0.0
        self._fy = 0.0

        # MRT 矩阵（v1.2）：use_mrt=0 时内核走 BGK
        self._use_mrt = case.collision == "mrt"
        if self._use_mrt:
            M, Minv = mrt_matrices()
            S = mrt_relax_rates(1.0 / case.tau, preset=case.mrt_preset)
            self._M = np.ascontiguousarray(M)
            self._Minv = np.ascontiguousarray(Minv)
            self._Sdiag = np.ascontiguousarray(S)
        else:
            self._M = np.zeros((9, 9))
            self._Minv = np.zeros((9, 9))
            self._Sdiag = np.zeros(9)

        # v1.3：LES 与持续入口扰动
        self._les_on = bool(c.les)
        self._cs_smag_sq = float(c.c_smag) ** 2
        T_nat = c.d / (c.u_in * literature_st(c.re))
        self._inlet_period = max(float(c.inlet_period_ratio) * T_nat, 1.0)
        self._inlet_base = inlet_base(c.ny, c.inlet_eps, c.u_in, c.inlet_m)
        self._it = 0
        # 控制体测力围道（E14）：3 个同心方形，全部落在流体里
        _cv_ok = c.radius + 14 < min(c.ny / 2, c.nx)
        self._contours = contours_for_case(c) if _cv_ok else []
        # 严格守恒的链接穿越表（与上面围道同几何）+ 远场尾流站位
        self._fluxes = link_fluxes_for_case(c) if _cv_ok else []
        self._wake_stations = wake_stations_for_case(c)
        self._wake_ref_x = wake_ref_x(c.sponge_width)

        # numba 内核参数（扁平 int64/float64 数组）
        t = self._table
        self._l_y = np.ascontiguousarray(t["ys"], dtype=np.int64)
        self._l_x = np.ascontiguousarray(t["xs"], dtype=np.int64)
        self._l_i = np.ascontiguousarray(t["dirs"], dtype=np.int64)
        self._l_q = np.ascontiguousarray(t["q"], dtype=np.float64)
        self._l_sy = np.ascontiguousarray(t["sys"], dtype=np.int64)
        self._l_sx = np.ascontiguousarray(t["sxs"], dtype=np.int64)
        self._sigma_x = np.ascontiguousarray(self._sigma_x, dtype=np.float64)
        self._ftarget = np.ascontiguousarray(self._ftarget, dtype=np.float64)

    # ---------- 单步 ----------
    def step(self):
        inlet = inlet_uy(self._inlet_base, self._it, self._inlet_period)
        self._it += 1
        if HAVE_NUMBA:
            fx, fy = _step_kernel(
                self.f, self._solid_u1, self.case.tau, self.case.u_in,
                self._free_slip, self._convective,
                self._l_y, self._l_x, self._l_i, self._l_q,
                self._l_sy, self._l_sx, self._sigma_x, self._ftarget,
                self._use_mrt, self._M, self._Minv, self._Sdiag,
                self._les_on, self._cs_smag_sq, inlet)
            self._fx, self._fy = float(fx), float(fy)
            return
        self._step_numpy(inlet)

    def _step_numpy(self, inlet=None):
        c = self.case
        self._inlet_cur = (np.zeros(c.ny) if inlet is None
                           else np.asarray(inlet, dtype=np.float64))
        rho, ux, uy = macros(self.f)
        if self._use_mrt:
            self.f = mrt_collide(self.f, self._M, self._Minv, self._Sdiag, c.tau)
        else:
            feq = equilibrium(rho, ux, uy)
            if self._les_on:
                q = pi_neq_norm(self.f, rho, ux, uy)
                tau_loc = smagorinsky_tau(c.tau, q, c.c_smag, rho=rho)
                omega = 1.0 / np.asarray(tau_loc)[..., None]
            else:
                omega = 1.0 / c.tau
            self.f += -omega * (self.f - feq)
        # 海绵层
        if self._sigma_x.max() > 0.0:
            sig = self._sigma_x[None, :, None]
            self.f += sig * (self._ftarget[None, None, :] - self.f)
        f_post = self.f
        f_new = np.empty_like(self.f)
        for i in range(NL):
            f_new[:, :, i] = np.roll(f_post[:, :, i], shift=(CY[i], CX[i]), axis=(0, 1))
        # Bouzidi 反弹 + 动量交换力
        self._fx, self._fy = apply_bb(f_post, f_new, self._table)
        # 顶底自由滑移：镜面反射（cx 不变、cy 反向）
        if c.top_bottom == "free_slip":
            # 底边 y=0：错误槽位 cy>0（1,2,8）← 镜像 cy<0 的碰撞后值
            f_new[0, :, 1] = f_post[0, :, 5]
            f_new[0, :, 2] = f_post[0, :, 4]
            f_new[0, :, 8] = f_post[0, :, 6]
            # 顶边 y=ny-1：错误槽位 cy<0（4,5,6）← 镜像 cy>0
            f_new[-1, :, 4] = f_post[-1, :, 2]
            f_new[-1, :, 5] = f_post[-1, :, 1]
            f_new[-1, :, 6] = f_post[-1, :, 8]
        self.f = f_new
        self._apply_inlet()
        self._apply_outlet()
        self.f[self.solid] = 0.0

    # ---------- 边界 ----------
    def _apply_inlet(self):
        """西边界 Zou/He 速度入口：ux=U, uy=uy_b(y)（idx 2,3,4 为未知）。

        uy_b≡0 时即 v1.2 原式；非零时仅在 f2/f4 上多 ∓0.5ρ·uy_b，
        密度公式与 x 动量条件不受影响（见 les.py 推导）。
        """
        ux_b = self.case.u_in
        uy_b = getattr(self, "_inlet_cur", np.zeros(self.case.ny))
        f = self.f[:, 0, :]
        f0, f1, f5 = f[:, 0], f[:, 1], f[:, 5]
        f6, f7, f8 = f[:, 6], f[:, 7], f[:, 8]
        rho = (f0 + f1 + f5 + 2.0 * (f6 + f7 + f8)) / (1.0 - ux_b)
        f[:, 3] = f7 + (2.0 / 3.0) * rho * ux_b
        f[:, 2] = (f6 + 0.5 * (f5 - f1) + (1.0 / 6.0) * rho * ux_b
                   + 0.5 * rho * uy_b)
        f[:, 4] = (f8 + 0.5 * (f1 - f5) + (1.0 / 6.0) * rho * ux_b
                   - 0.5 * rho * uy_b)

    def _apply_outlet(self):
        """东边界：一阶对流 ∂f/∂t + U_c ∂f/∂x = 0，或零阶外推。"""
        c = self.case
        if c.outlet == "extrapolate":
            self.f[:, -1, :] = self.f[:, -2, :]
            return
        u_c = c.u_in
        self.f[:, -1, :] -= u_c * (self.f[:, -1, :] - self.f[:, -2, :])

    # ---------- 观测 ----------
    def macros_masked(self):
        rho, ux, uy = macros(self.f)
        ux = np.where(self.fluid, ux, 0.0)
        uy = np.where(self.fluid, uy, 0.0)
        return rho, ux, uy

    def vorticity(self) -> np.ndarray:
        _, ux, uy = self.macros_masked()
        dvdx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / 2.0
        dudy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / 2.0
        w = dvdx - dudy
        return np.where(self.fluid, w, np.nan)

    def force(self) -> tuple[float, float]:
        """动量交换力（格子单位，单位深度；取最近一次 step 的反弹值）。

        F = Σ_{圆柱链接} e_i [f_i*(x_f) + f_ī_new(x_f)]；
        半程反弹时恒等于 2 Σ e_i f_i*（旧口径）。
        """
        return self._fx, self._fy

    def cv_forces(self):
        """E14 **朴素节点口径**（方法学反例）：围道上碰撞前 Π=Σcc f 的面积分。

        与严格守恒通量差 ``Π^neq/τ``，近场系统性高估、违反围道无关性，
        保留仅用于量化该偏差。返回 [(半边长, Fx, Fy), ...]。
        """
        return [(h, *cv_force(self.f, c)) for h, c in self._contours]

    def link_forces(self):
        """E14 **主口径**：链接穿越法，严格离散守恒，返回 [(h, Fx, Fy), ...]。

        不读 τ/feq/反弹链接/碰撞算子；配合 ``box_momenta()`` 逐步扣除
        ``−d/dt∫_V ρu`` 后即为逐步精确的物体受力（围道无关）。
        """
        return [(h, *cv_force_links(self.f, fl)) for h, fl in self._fluxes]

    def box_momenta(self):
        """各控制体内总动量 ``∫ρu dV``，用于逐步扣除非定常累积项。"""
        return {h: box_momentum(self.f, fl) for h, fl in self._fluxes}

    def wake_forces(self, classic=False):
        """E14 独立主判据：远场尾流动量亏损，返回 [(x/D, dict), ...]。

        默认用 v1.3 **严格两截面口径**（参考截面在西侧海绵之后），只读
        守恒矩 ρ/ρu，BGK 与 MRT 的弛豫率对这些矩恒为 0。
        ``classic=True`` 退回经典 wake survey（实测低估 53%），供方法学对照。
        """
        c = self.case
        ref = None if classic else self._wake_ref_x
        return [(s, wake_force(self.f, x, c.rho0, c.u_in, x_ref=ref))
                for s, x in self._wake_stations]

    def les_diagnostics(self):
        """v1.3 E15：全场 τ_total / ν_t 占比的分位数（LES 关闭时返回 None）。

        只看流体节点（固体 f≡0 会污染分位数）。
        """
        if not self._les_on:
            return None
        c = self.case
        rho, ux, uy = macros(self.f)
        q = pi_neq_norm(self.f, rho, ux, uy)
        tt = smagorinsky_tau(c.tau, q, c.c_smag, rho=rho)
        vals = nu_t_over_nu0(tt, c.tau)[self.fluid]
        return {
            "nu_t_over_nu0_median": float(np.median(vals)),
            "nu_t_over_nu0_p90": float(np.percentile(vals, 90)),
            "nu_t_over_nu0_max": float(np.max(vals)),
            "tau_total_median": float(np.median(np.asarray(tt)[self.fluid])),
        }
