"""numba JIT 核函数：单步 LBM（碰撞→海绵层→迁移→Bouzidi 反弹→边界）。

物理与 Solver 的 numpy 路径一致；numba 不可用时 Solver 自动回退 numpy。
反弹统一走链接表（q≡0.5 时即半程反弹）；力在反弹阶段按动量交换累积：
F = Σ_links e_i [f_i*(x_f) + f_ī_new(x_f)]。
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
def _step_kernel(f, solid, tau, u_in, free_slip, convective,
                 link_y, link_x, link_i, link_q, link_sy, link_sx,
                 sigma_x, ftarget,
                 use_mrt, Mm, Minv, Sdiag,
                 les_on, cs_smag_sq, inlet_uy):
    """单步：碰撞(BGK/MRT/LES) → 海绵 → 迁移 → Bouzidi 反弹/测力 → 边界。

    v1.3 新增两个可选开关（关闭时与原 v1.2 内核逐位一致）：
    - ``les_on`` + ``cs_smag_sq=C_s²``：Smagorinsky 隐式局部 τ_total；
    - ``inlet_uy``：西边界逐 y 的横向速度 uy_b（零均值即退化为原 Zou/He）。
    """
    ny, nx, _ = f.shape
    m = np.empty(9, dtype=np.float64)
    meq = np.empty(9, dtype=np.float64)
    feq = np.empty(9, dtype=np.float64)
    # 1) 碰撞（BGK 或 MRT，全场含固体——固体节点仅作缓冲，随后清零）
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
            if use_mrt:
                # m = M·f；meq；m += S(meq−m)；f = Minv·m
                for r in range(9):
                    acc = 0.0
                    for cidx in range(9):
                        acc += Mm[r, cidx] * f[y, x, cidx]
                    m[r] = acc
                usq = ux * ux + uy * uy
                rho_m = m[0]
                meq[0] = rho_m
                meq[1] = -2.0 * rho_m + 3.0 * rho_m * usq
                meq[2] = rho_m - 3.0 * rho_m * usq
                meq[3] = rho_m * ux
                meq[4] = -rho_m * ux
                meq[5] = rho_m * uy
                meq[6] = -rho_m * uy
                meq[7] = rho_m * (ux * ux - uy * uy)
                meq[8] = rho_m * ux * uy
                for r in range(9):
                    m[r] += Sdiag[r] * (meq[r] - m[r])
                for cidx in range(9):
                    acc = 0.0
                    for r in range(9):
                        acc += Minv[cidx, r] * m[r]
                    f[y, x, cidx] = acc
            else:
                usq = ux * ux + uy * uy
                pxx = 0.0
                pyy = 0.0
                pxy = 0.0
                for i in range(9):
                    cu = CX[i] * ux + CY[i] * uy
                    feq[i] = W[i] * rho * (1.0 + 3.0 * cu
                                           + 4.5 * cu * cu - 1.5 * usq)
                    if les_on:
                        fn = f[y, x, i] - feq[i]
                        pxx += CX[i] * CX[i] * fn
                        pyy += CY[i] * CY[i] * fn
                        pxy += CX[i] * CY[i] * fn
                if les_on:
                    # Q = sqrt(2 Π^neq:Π^neq)；隐式 Smagorinsky 二次方程正根
                    q = np.sqrt(2.0 * max(pxx * pxx + pyy * pyy
                                          + 2.0 * pxy * pxy, 0.0))
                    if rho > 1e-12:
                        disc = tau * tau + 18.0 * cs_smag_sq * q / rho
                        tau_loc = 0.5 * (tau + np.sqrt(max(disc, 0.0)))
                    else:
                        tau_loc = tau  # 固体节点 f≡0，避免 0/0
                    omega = 1.0 / tau_loc
                else:
                    omega = 1.0 / tau
                for i in range(9):
                    f[y, x, i] += -omega * (f[y, x, i] - feq[i])
    # 1b) 海绵层：向目标态 feq(ρ0, U, 0) 弛豫（吸收慢变压力波，防出口/入口反射）
    for y in range(ny):
        for x in range(nx):
            sig = sigma_x[x]
            if sig > 0.0:
                for i in range(9):
                    f[y, x, i] += sig * (ftarget[i] - f[y, x, i])
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
    # 3) Bouzidi 反弹（q≡0.5 退化半程）+ 动量交换力
    fx = 0.0
    fy = 0.0
    nlink = link_y.size
    for k in range(nlink):
        y = link_y[k]
        x = link_x[k]
        i = link_i[k]
        q = link_q[k]
        fi = f[y, x, i]
        if q < 0.5:
            bb = 2.0 * q * fi + (1.0 - 2.0 * q) * f[link_sy[k], link_sx[k], i]
        else:
            inv = 1.0 / (2.0 * q)
            bb = inv * fi + (2.0 * q - 1.0) * inv * f[y, x, OPP[i]]
        f_new[y, x, OPP[i]] = bb
        fx += CX[i] * (fi + bb)
        fy += CY[i] * (fi + bb)
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
    # 5) 西入口 Zou/He（ux=u_in, uy=uy_b(y)；uy_b≡0 即 v1.2 原式）
    for y in range(ny):
        f0 = f_new[y, 0, 0]
        f1 = f_new[y, 0, 1]
        f5 = f_new[y, 0, 5]
        f6 = f_new[y, 0, 6]
        f7 = f_new[y, 0, 7]
        f8 = f_new[y, 0, 8]
        uy_b = inlet_uy[y]
        rho = (f0 + f1 + f5 + 2.0 * (f6 + f7 + f8)) / (1.0 - u_in)
        f_new[y, 0, 3] = f7 + (2.0 / 3.0) * rho * u_in
        f_new[y, 0, 2] = (f6 + 0.5 * (f5 - f1) + (1.0 / 6.0) * rho * u_in
                          + 0.5 * rho * uy_b)
        f_new[y, 0, 4] = (f8 + 0.5 * (f1 - f5) + (1.0 / 6.0) * rho * u_in
                          - 0.5 * rho * uy_b)
    # 6) 东出口
    for y in range(ny):
        for i in range(9):
            if convective:
                f_new[y, nx - 1, i] -= u_in * (f_new[y, nx - 1, i] - f_new[y, nx - 2, i])
            else:
                f_new[y, nx - 1, i] = f_new[y, nx - 2, i]
    # 7) 固体清零（Bouzidi 不依赖固体节点值，可安全清零）
    for y in range(ny):
        for x in range(nx):
            if solid[y, x]:
                for i in range(9):
                    f_new[y, x, i] = 0.0
    # 写回
    for y in range(ny):
        for x in range(nx):
            for i in range(9):
                f[y, x, i] = f_new[y, x, i]
    return fx, fy


@njit(cache=True)
def _force_kernel(f, link_y, link_x, link_i):
    """（诊断用）半程反弹口径的力：F = 2 Σ e_i f_i*（f_i* 存于 ī 槽位）。"""
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
    li = np.zeros(0, dtype=np.int64)
    lq = np.zeros(0)
    _step_kernel(f, np.zeros((4, 4), dtype=np.bool_), 0.8, 0.05, True, True,
                 li, li, li, lq, li, li, np.zeros(4), np.zeros(9),
                 False, np.zeros((9, 9)), np.zeros((9, 9)), np.zeros(9),
                 False, 0.0, np.zeros(4))
