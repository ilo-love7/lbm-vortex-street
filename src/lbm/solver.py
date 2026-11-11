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
from .mrt import mrt_collide, mrt_matrices, mrt_relax_rates


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
        if not (0.51 < c.tau < 3.0):
            raise ValueError(f"tau={c.tau:.3f} 超出稳定区间，调整 U/D/Re")
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
        if HAVE_NUMBA:
            fx, fy = _step_kernel(
                self.f, self._solid_u1, self.case.tau, self.case.u_in,
                self._free_slip, self._convective,
                self._l_y, self._l_x, self._l_i, self._l_q,
                self._l_sy, self._l_sx, self._sigma_x, self._ftarget,
                self._use_mrt, self._M, self._Minv, self._Sdiag)
            self._fx, self._fy = float(fx), float(fy)
            return
        self._step_numpy()

    def _step_numpy(self):
        c = self.case
        rho, ux, uy = macros(self.f)
        if self._use_mrt:
            self.f = mrt_collide(self.f, self._M, self._Minv, self._Sdiag, c.tau)
        else:
            feq = equilibrium(rho, ux, uy)
            self.f += -(1.0 / c.tau) * (self.f - feq)
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
        """西边界 Zou/He 速度入口：ux=U, uy=0（idx 2,3,4 为未知）。"""
        ux_b = self.case.u_in
        f = self.f[:, 0, :]
        f0, f1, f5 = f[:, 0], f[:, 1], f[:, 5]
        f6, f7, f8 = f[:, 6], f[:, 7], f[:, 8]
        rho = (f0 + f1 + f5 + 2.0 * (f6 + f7 + f8)) / (1.0 - ux_b)
        f[:, 3] = f7 + (2.0 / 3.0) * rho * ux_b
        f[:, 2] = f6 + 0.5 * (f5 - f1) + (1.0 / 6.0) * rho * ux_b
        f[:, 4] = f8 + 0.5 * (f1 - f5) + (1.0 / 6.0) * rho * ux_b

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
