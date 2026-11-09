"""numba JIT 核函数：单步 LBM（碰撞→迁移→反弹→边界）。

物理与 Solver 的 numpy 路径一致；numba 不可用时 Solver 自动回退 numpy。
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn

        return wrap


CX = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
CY = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])
OPP = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4], dtype=np.int64)


@njit(cache=True)
def _step_kernel(f, cyl, fluid, tau, u_in, free_slip, convective):
    ny, nx, _ = f.shape
    # 1) 碰撞
    for y in range(ny):
        for x in range(nx):
            rho = 0.0
            mx = 0.0
            my = 0.0
            for i in range(9):
                v = f[y, x, i]
                rho += v
                mx += CX[i] * v
                my += CY[i] * v
            if rho > 0.0:
                ux = mx / rho
                uy = my / rho
            else:
                ux = 0.0
                uy = 0.0
            usq = ux * ux + uy * uy
            for i in range(9):
                cu = CX[i] * ux + CY[i] * uy
                feq = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
                f[y, x, i] += -(1.0 / tau) * (f[y, x, i] - feq)
    # 2) 迁移（pull，周期）到临时场
    f_new = np.empty_like(f)
    for y in range(ny):
        for x in range(nx):
            for i in range(9):
                sy = y - CY[i]
                sx = x - CX[i]
                if sy < 0:
                    sy += ny
                elif sy >= ny:
                    sy -= ny
                if sx < 0:
                    sx += nx
                elif sx >= nx:
                    sx -= nx
                f_new[y, x, i] = f[sy, sx, i]
    # 3) 半程反弹：f_ī(x_f) ← f_i*(x_f)
    for y in range(ny):
        for x in range(nx):
            if not fluid[y, x]:
                continue
            for i in range(1, 9):
                nyb = y + CY[i]
                nxb = x + CX[i]
                if nyb < 0 or nyb >= ny or nxb < 0 or nxb >= nx:
                    continue
                if cyl[nyb, nxb]:
                    f_new[y, x, OPP[i]] = f[y, x, i]
    # 4) 顶底自由滑移：镜面反射
    if free_slip:
        for x in range(nx):
            # 底边：cy>0 槽位 ← 镜像 cy<0
            f_new[0, x, 1] = f[0, x, 5]
            f_new[0, x, 2] = f[0, x, 4]
            f_new[0, x, 8] = f[0, x, 6]
            # 顶边：cy<0 槽位 ← 镜像 cy>0
            f_new[ny - 1, x, 4] = f[ny - 1, x, 2]
            f_new[ny - 1, x, 5] = f[ny - 1, x, 1]
            f_new[ny - 1, x, 6] = f[ny - 1, x, 8]
    # 5) 西入口 Zou/He（ux=u_in, uy=0）
    for y in range(ny):
        f0 = f_new[y, 0, 0]
        f1 = f_new[y, 0, 1]
        f5 = f_new[y, 0, 5]
        f6 = f_new[y, 0, 6]
        f7 = f_new[y, 0, 7]
        f8 = f_new[y, 0, 8]
        rho = (f0 + f1 + f5 + 2.0 * (f6 + f7 + f8)) / (1.0 - u_in)
        f_new[y, 0, 3] = f7 + (2.0 / 3.0) * rho * u_in
        f_new[y, 0, 2] = f6 + 0.5 * (f5 - f1) + (1.0 / 6.0) * rho * u_in
        f_new[y, 0, 4] = f8 + 0.5 * (f1 - f5) + (1.0 / 6.0) * rho * u_in
    # 6) 东出口
    for y in range(ny):
        for i in range(9):
            if convective:
                f_new[y, nx - 1, i] -= u_in * (f_new[y, nx - 1, i] - f_new[y, nx - 2, i])
            else:
                f_new[y, nx - 1, i] = f_new[y, nx - 2, i]
    # 7) 固体清零
    for y in range(ny):
        for x in range(nx):
            if not fluid[y, x]:
                for i in range(9):
                    f_new[y, x, i] = 0.0
    # 写回
    for y in range(ny):
        for x in range(nx):
            for i in range(9):
                f[y, x, i] = f_new[y, x, i]


@njit(cache=True)
def _force_kernel(f, link_y, link_x, link_i):
    """动量交换：F = 2 Σ e_i f_i*（f_i* 存于 ī 槽位）。link_y/x/i 为边界链接坐标与方向。"""
    fx = 0.0
    fy = 0.0
    n = link_y.size
    for k in range(n):
        y = link_y[k]
        x = link_x[k]
        i = link_i[k]
        fi_star = f[y, x, OPP[i]]
        fx += 2.0 * CX[i] * fi_star
        fy += 2.0 * CY[i] * fi_star
    return fx, fy


def make_force_fn():
    return _force_kernel


def compile_step():
    """触发 JIT 编译（测试用）。"""
    f = np.zeros((4, 4, 9))
    _step_kernel(f, np.zeros((4, 4), dtype=np.bool_), np.ones((4, 4), dtype=np.bool_),
                 0.8, 0.05, True, True)
