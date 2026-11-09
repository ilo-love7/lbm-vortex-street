"""D2Q9 + BGK LBM 求解器：圆柱绕流（半程反弹）+ Zou/He 入口 + 对流出口。

实现要点（相对 Mocz 教学版的严谨化）：
- 半程反弹（link-wise）：f_ī(x_f, t+1) = f_i*(x_f, t)，x_f+e_i ∈ solid
- 动量交换测力：F = 2 Σ_{cylinder links} e_i f_i*（静壁；f_i* 反弹时暂存于 ī 槽位）
- Zou/He 速度入口（西，ux=U, uy=0）；一阶对流出口（东）；顶底自由滑移（镜面）或无滑移
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kernels import HAVE_NUMBA, _force_kernel, _step_kernel
from .lattice import CX, CY, NL, OPP, equilibrium, macros


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

        # 反弹/测力掩码（仅圆柱链接，避免壁面力污染升阻力）
        self._bb = []
        self._link = []
        ys, xs, dirs = [], [], []
        for i in range(NL):
            nb = np.roll(self.cyl, shift=(-CY[i], -CX[i]), axis=(0, 1))
            m = self.fluid & nb
            self._bb.append(m)
            self._link.append(m)
            if i > 0:
                yy_i, xx_i = np.nonzero(m)
                ys.append(yy_i)
                xs.append(xx_i)
                dirs.append(np.full(yy_i.size, i, dtype=np.int64))
        self._link_y = np.concatenate(ys) if ys else np.zeros(0, dtype=np.int64)
        self._link_x = np.concatenate(xs) if xs else np.zeros(0, dtype=np.int64)
        self._link_i = np.concatenate(dirs) if dirs else np.zeros(0, dtype=np.int64)
        self._cyl_u1 = self.cyl.astype(np.bool_)
        self._fluid_u1 = self.fluid.astype(np.bool_)
        self._free_slip = case.top_bottom == "free_slip"
        self._convective = case.outlet == "convective"

    # ---------- 单步 ----------
    def step(self):
        if HAVE_NUMBA:
            _step_kernel(self.f, self._cyl_u1, self._fluid_u1,
                         self.case.tau, self.case.u_in,
                         self._free_slip, self._convective)
            return
        self._step_numpy()

    def _step_numpy(self):
        c = self.case
        rho, ux, uy = macros(self.f)
        feq = equilibrium(rho, ux, uy)
        self.f += -(1.0 / c.tau) * (self.f - feq)
        f_post = self.f
        f_new = np.empty_like(self.f)
        for i in range(NL):
            f_new[:, :, i] = np.roll(f_post[:, :, i], shift=(CY[i], CX[i]), axis=(0, 1))
        # 半程反弹：f_ī(x_f) ← f_i*(x_f)
        for i in range(1, NL):
            m = self._bb[i]
            if m.any():
                f_new[:, :, OPP[i]][m] = f_post[:, :, i][m]
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
        """动量交换力（格子单位，单位深度）：F = 2 Σ_{圆柱链接} e_i f_i*。"""
        if HAVE_NUMBA:
            fx, fy = _force_kernel(self.f, self._link_y, self._link_x, self._link_i)
            return float(fx), float(fy)
        fx = 0.0
        fy = 0.0
        for i in range(1, NL):
            m = self._link[i]
            if not m.any():
                continue
            # 反弹后 ī 槽位存的是 f_i*
            fi_star = self.f[:, :, OPP[i]][m]
            fx += 2.0 * CX[i] * float(fi_star.sum())
            fy += 2.0 * CY[i] * float(fi_star.sum())
        return fx, fy
