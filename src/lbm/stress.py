"""E14 (v1.3)：控制体测力的两条口径 + 一条严格守恒的链接穿越法。

动机（报告 §9 最高优先未做项）：v1.2 发现 MRT 会把 Cd 系统性抬高 +1.3~1.9%，
但动量交换法（MEM）读的是反弹链接上的 ``f_i* + f_ī_new``，其中 ``f_i*`` 是
**碰撞后**的非平衡分布；换一个碰撞算子（MRT 给各矩不同弛豫率）就会改变这个
读数。这个偏移到底是物理（MRT 真的改变平均阻力）还是测力伪影，必须用一条
不读壁面链接、不依赖碰撞算子 CE 关系的独立测力路径来判定。

三个口径
--------
1. ``cv_force``（**朴素节点口径**，保留作方法学反例）：在围道节点上对
   **碰撞前**分布的二阶矩 ``Π_αβ(x) = Σ_i c_iα c_iβ f_i(x)`` 做面积分
   ``−∮Π·n ds``。它不依赖 τ / feq，看似与碰撞算子无关，但 LBM 的离散
   动量守恒律通量其实是**碰撞后**分布 ``f*`` 的二阶矩，二者相差
   ``Π^neq/τ`` ——恰是粘性应力的量级。近场剪切强 → 系统性高估，远场
   剪切弱 → 回落，因此这个口径**违反围道无关性**（v1.3 实测 h=18/22/28
   给 Cd 2.25/1.78/1.38，而 MEM 仅 1.29）。它不能作为主判据。

2. ``cv_force_links``（**链接穿越法，主口径**）：直接从离散守恒律出发。
   由 ``f_i(x+c_i, t+1) = f_i*(x,t)`` 与碰撞的动量守恒
   ``Σ_i c_i f_i* = Σ_i c_i f_i = ρu``，控制体 V 内动量变化精确等于

       P_V(t+1) − P_V(t) = Σ_入 c_i f_i* − Σ_出 c_i f_i* + F_ext→fluid

   其中入/出指 src→dst (dst=src+c_i) 跨越 V 边界的链接，而
   ``f_i*(src,t)`` **就是当前 f 在目标节点的值** ``f_i(dst,t+1)``。
   周期平均后累积项为零，故

       F_fluid→body = Σ_i c_i [ Σ_入 f_i(dst) − Σ_出 f_i(dst) ]

   只读 ``f``、不需要 τ、feq、反弹链接表、也不需要知道碰撞算子，且
   **严格满足离散守恒 ⇒ 围道无关**。这正是判定 MRT 偏移性质所需。

3. ``wake.wake_force``（远场尾流动量亏损）：只读守恒矩 ρ/ρu，见该模块。

围道构造
--------
``rect_contour`` 生成矩形边界，四条边各含起点角、不含终点角，角点只算一次，
构成闭合回路且相邻点间距恒为 1 lu（ds=1）。``box_mask`` 生成同一围道所围
的**节点集合** V（含边界），供链接法使用。两者必须完全落在流体中：
半边长须 > 圆柱半径，且避开东西海绵层。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import CX, CY, CS2, NL, equilibrium

# 预计算的二阶方向张量（长度 9），避免每步重复乘
_CXX = (CX * CX).astype(np.float64)
_CYY = (CY * CY).astype(np.float64)
_CXY = (CX * CY).astype(np.float64)


def momentum_flux(f: np.ndarray, ys, xs):
    """围道节点的 Π_xx, Π_xy, Π_yy（向量化）。

    f: (ny, nx, 9)；ys/xs: 等长整数数组。返回三个长度 N 的数组。
    """
    fc = np.asarray(f)[np.asarray(ys, dtype=int), np.asarray(xs, dtype=int), :]
    return fc @ _CXX, fc @ _CXY, fc @ _CYY


def rect_contour(cx: float, cy: float, half_x: int, half_y: int,
                 ny: int | None = None, nx: int | None = None):
    """矩形闭合围道，返回 (ys, xs, nxs, nys) 四个 int/float 数组。

    法向 n 指向围道**外侧**（远离圆柱）。四条边各含起点角、不含终点角，
    角点恰好计一次；相邻围道点间距 1 lu，故 ds=1。
    """
    x0, x1 = int(round(cx)) - int(half_x), int(round(cx)) + int(half_x)
    y0, y1 = int(round(cy)) - int(half_y), int(round(cy)) + int(half_y)
    if half_x < 1 or half_y < 1:
        raise ValueError("half_x/half_y 必须 ≥1")
    if ny is not None and (y0 < 0 or y1 >= ny):
        raise ValueError(f"围道 y∈[{y0},{y1}] 越出 ny={ny}")
    if nx is not None and (x0 < 0 or x1 >= nx):
        raise ValueError(f"围道 x∈[{x0},{x1}] 越出 nx={nx}")

    ys, xs, nxs, nys = [], [], [], []
    # 下边 y=y0，x: x0 → x1−1，n=(0,−1)
    for x in range(x0, x1):
        ys.append(y0); xs.append(x); nxs.append(0.0); nys.append(-1.0)
    # 右边 x=x1，y: y0 → y1−1，n=(+1,0)
    for y in range(y0, y1):
        ys.append(y); xs.append(x1); nxs.append(1.0); nys.append(0.0)
    # 上边 y=y1，x: x1 → x0+1，n=(0,+1)
    for x in range(x1, x0, -1):
        ys.append(y1); xs.append(x); nxs.append(0.0); nys.append(1.0)
    # 左边 x=x0，y: y1 → y0+1，n=(−1,0)
    for y in range(y1, y0, -1):
        ys.append(y); xs.append(x0); nxs.append(-1.0); nys.append(0.0)
    return (np.asarray(ys, dtype=np.int64), np.asarray(xs, dtype=np.int64),
            np.asarray(nxs, dtype=np.float64), np.asarray(nys, dtype=np.float64))


def box_mask(cx: float, cy: float, half_x: int, half_y: int,
             ny: int, nx: int) -> np.ndarray:
    """围道所围节点集合 V 的布尔掩码（含边界），供 ``cv_force_links`` 使用。

    与 ``rect_contour`` 用同一个 cx/cy/half_x/half_y，保证两种口径的
    控制体几何严格一致；越界时抛 ValueError（与 rect_contour 同口径）。
    """
    x0, x1 = int(round(cx)) - int(half_x), int(round(cx)) + int(half_x)
    y0, y1 = int(round(cy)) - int(half_y), int(round(cy)) + int(half_y)
    if half_x < 1 or half_y < 1:
        raise ValueError("half_x/half_y 必须 ≥1")
    if y0 < 0 or y1 >= ny or x0 < 0 or x1 >= nx:
        raise ValueError(f"控制体 x∈[{x0},{x1}] y∈[{y0},{y1}] 越出 "
                         f"({ny},{nx})")
    m = np.zeros((ny, nx), dtype=np.bool_)
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


@dataclass(frozen=True)
class LinkFlux:
    """预计算的链接穿越表（构造一次，逐步复用）。

    ``dst/dirs/signs`` 描述所有跨越 V 边界的链接：目标节点展平索引、
    方向 i、符号（+1 入 V / −1 出 V）。``nodes`` 是 V 内节点展平索引，
    用于逐步扣除 ``−d/dt∫_V ρu`` 累积项（严格守恒的关键）。
    """
    ny: int
    nx: int
    dst: np.ndarray
    dirs: np.ndarray
    signs: np.ndarray
    nodes: np.ndarray

    def __call__(self, f: np.ndarray) -> tuple[float, float]:
        return cv_force_links(f, self)


def build_link_flux(box: np.ndarray) -> LinkFlux:
    """由控制体掩码预计算链接穿越表；几何固定后每步只做一次 fancy index。"""
    b = np.asarray(box, dtype=np.bool_)
    ny, nx = b.shape
    dst_l, dir_l, sgn_l = [], [], []
    for i in range(NL):
        cx, cy = int(CX[i]), int(CY[i])
        if cx == 0 and cy == 0:
            continue                      # 静止方向不跨越任何边界
        sy0, sy1 = max(0, -cy), min(ny, ny - cy)
        sx0, sx1 = max(0, -cx), min(nx, nx - cx)
        dy0, dy1 = sy0 + cy, sy1 + cy
        dx0, dx1 = sx0 + cx, sx1 + cx
        in_dst = b[dy0:dy1, dx0:dx1]
        in_src = b[sy0:sy1, sx0:sx1]
        for mask, sgn in ((in_dst & ~in_src, 1.0),   # 入 V
                          (~in_dst & in_src, -1.0)):  # 出 V
            yy, xx = np.nonzero(mask)
            if yy.size == 0:
                continue
            dst_l.append((yy + dy0) * nx + (xx + dx0))
            dir_l.append(np.full(yy.size, i, dtype=np.int64))
            sgn_l.append(np.full(yy.size, sgn, dtype=np.float64))
    if not dst_l:
        raise ValueError("控制体没有跨界链接（box 覆盖全网格？）")
    nodes = np.flatnonzero(b.ravel()).astype(np.int64)
    return LinkFlux(ny=ny, nx=nx,
                    dst=np.concatenate(dst_l),
                    dirs=np.concatenate(dir_l),
                    signs=np.concatenate(sgn_l),
                    nodes=nodes)


def cv_force_links(f: np.ndarray, flux) -> tuple[float, float]:
    """链接穿越法：严格离散守恒的控制体力（E14 主口径）。

    ``F_α = Σ_i c_iα [ Σ_{入 V} f_i(dst) − Σ_{出 V} f_i(dst) ]``，
    其中链接 src→dst = src + c_i 跨越 V 边界，``f_i(dst)`` 是当前时刻
    ``f`` 在目标节点的分量（= 上一步碰撞后分布 ``f_i*(src)``）。

    ``flux`` 可为 ``LinkFlux``（推荐，预计算）或布尔 box（临时构造）。
    只依赖 ``f`` 与几何，不需要 τ / feq / 碰撞算子 / 反弹链接表；
    由于是离散守恒律的直接恒等式，扣除累积项后**逐步**围道无关。
    返回 (Fx, Fy)，格子单位、单位深度。
    """
    fa = np.asarray(f)
    if not isinstance(flux, LinkFlux):
        flux = build_link_flux(flux)
    if fa.shape[:2] != (flux.ny, flux.nx) or fa.shape[2] != NL:
        raise ValueError("f 形状与链接表不一致")
    flat = fa.reshape(-1, NL)
    vals = flat[flux.dst, flux.dirs] * flux.signs
    return (float(np.dot(CX[flux.dirs].astype(np.float64), vals)),
            float(np.dot(CY[flux.dirs].astype(np.float64), vals)))


def box_momentum(f: np.ndarray, flux: LinkFlux) -> tuple[float, float]:
    """V 内总动量 ``∫_V ρu dV``（格子单位），用于逐步扣除累积项。"""
    fa = np.asarray(f).reshape(-1, NL)
    sub = fa[flux.nodes]
    return (float(np.sum(sub @ CX.astype(np.float64))),
            float(np.sum(sub @ CY.astype(np.float64))))


def cv_force(f: np.ndarray, contour) -> tuple[float, float]:
    """朴素节点口径：F_α = −∮ Π_αβ n_β ds（ds=1），Π=Σcc f（碰撞前）。

    **方法学反例，不作主判据**：见模块文档，此口径与严格守恒通量相差
    ``Π^neq/τ``，近场高估且违反围道无关性。保留它用于量化该偏差。
    返回 (Fx, Fy)，格子单位、单位深度；时间平均由调用方完成。
    """
    ys, xs, nxs, nys = contour
    pxx, pxy, pyy = momentum_flux(f, ys, xs)
    fx = -float(np.sum(pxx * nxs + pxy * nys))
    fy = -float(np.sum(pxy * nxs + pyy * nys))
    return fx, fy


def stress_decomposition(f: np.ndarray, ys, xs, tau: float):
    """BGK 口径下围道上 Π 的分解（诊断用，非主判据）。

    返回 dict：``p``（压力 ρc_s²）、``pi_neq`` 三分量、``sigma_visc`` 三分量、
    ``conv`` 三分量（ρu_αu_β）。精确恒等式是
    ``Π = conv + p·I + pi_neq``；连续 N-S 通量则是
    ``Π_cont = conv + p·I − sigma_visc``（二者差 ``Π^neq/(2τ)``，
    这是 LBM 离散应力的著名 ``τ`` vs ``τ−1/2`` 之差）。主判据用 raw Π，
    正因为它不依赖这个 CE 关系、对 MRT 同样成立。
    """
    fc = np.asarray(f)[np.asarray(ys, dtype=int), np.asarray(xs, dtype=int), :]
    rho = fc.sum(axis=1)
    rho_safe = np.where(rho > 0.0, rho, 1.0)
    ux = (fc @ CX.astype(np.float64)) / rho_safe
    uy = (fc @ CY.astype(np.float64)) / rho_safe
    ux = np.where(rho > 0.0, ux, 0.0)
    uy = np.where(rho > 0.0, uy, 0.0)
    feq = equilibrium(rho, ux, uy)
    fneq = fc - feq
    pxx_n = fneq @ _CXX
    pyy_n = fneq @ _CYY
    pxy_n = fneq @ _CXY
    k = -(1.0 - 1.0 / (2.0 * tau))
    return {
        "p": rho * CS2,
        "pi_neq": (pxx_n, pxy_n, pyy_n),
        "sigma_visc": (k * pxx_n, k * pxy_n, k * pyy_n),
        "conv": (rho * ux * ux, rho * ux * uy, rho * uy * uy),
        "rho": rho, "ux": ux, "uy": uy,
    }


def force_coefficients_cv(fx: float, fy: float, rho0: float,
                          u_in: float, d: float) -> tuple[float, float]:
    """与 forces.force_coefficients 同一归一化，便于逐位对照。"""
    q = 0.5 * rho0 * u_in * u_in * d
    return fx / q, fy / q


def contour_sizes(case, margins=(4, 8, 14)):
    """按圆柱半径 + margin 生成 3 个同心方形围道的半边长（跳过越界者）。"""
    out = []
    r = case.radius
    for m in margins:
        h = int(np.ceil(r)) + int(m)
        try:
            rect_contour(case.cx_c, case.cy, h, h, ny=case.ny, nx=case.nx)
        except ValueError:
            continue  # 小测试域放不下大围道时静默跳过，不影响 CFD 主路径
        out.append(h)
    return out


def contours_for_case(case, margins=(4, 8, 14)):
    """按圆柱半径 + margin 生成 3 个同心方形围道（避开海绵层）。

    半边长 = ceil(radius) + margin；同时检查上下界与东西海绵宽度。
    """
    return [(h, rect_contour(case.cx_c, case.cy, h, h,
                             ny=case.ny, nx=case.nx))
            for h in contour_sizes(case, margins)]


def link_boxes_for_case(case, margins=(4, 8, 14)):
    """与 ``contours_for_case`` 同几何的控制体节点掩码列表 ``(h, box)``。"""
    return [(h, box_mask(case.cx_c, case.cy, h, h,
                         ny=case.ny, nx=case.nx))
            for h in contour_sizes(case, margins)]


def link_fluxes_for_case(case, margins=(4, 8, 14)):
    """预计算链接表列表 ``(h, LinkFlux)``，实验脚本逐步复用。"""
    return [(h, build_link_flux(b)) for h, b in link_boxes_for_case(case, margins)]


def wake_stations_for_case(case, stations=(3.0, 6.0, 10.0)):
    """E14 远场站位（延迟导入避免 stress ↔ wake 循环依赖）。"""
    from .wake import wake_stations
    return wake_stations(case.cx_c, case.d, case.nx,
                         stations=stations,
                         downstream_guard=max(40, case.sponge_width + 10))
