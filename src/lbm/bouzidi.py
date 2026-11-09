"""Bouzidi-Lallemand-Luo 次网格反弹（curved boundary，q 加权）。

线性两点插值（Bouzidi 2000；Lallemand & Luo 2003 式 (12)/(14) 的两点极限），
对每条流体→固体链接按壁距 q∈(0,1] 插值（i 为指向固体的方向，ī=OPP[i]）：

  q < 1/2: f_ī(x_f,t+1) = 2q·f_i*(x_f,t) + (1−2q)·f_i*(x_f−e_i,t)
  q ≥ 1/2: f_ī(x_f,t+1) = (1/(2q))·f_i*(x_f,t) + ((2q−1)/(2q))·f_ī*(x_f,t)

q=0.5 时两式均退化为半程反弹 f_ī = f_i*。

**两式都只读流体节点的碰撞后值**（x_f、x_f−e_i、x_f 的 ī 槽位），
不依赖固体节点分布——固体可照常清零（v1.0 备用实现曾用 f_i*(x_s)，是错的）。

q 由圆柱解析面 |x_f + q e_i − C| = R 求得（二次方程小根）。
静壁动量交换力：F = Σ_links e_i [f_i*(x_f) + f_ī_new(x_f)]，
半程反弹时恒等于 2 Σ e_i f_i*。
"""
from __future__ import annotations

import numpy as np

from .lattice import CX, CY, OPP


def cylinder_link_q(cyl_y: np.ndarray, cyl_x: np.ndarray, direction: int,
                    cx_c: float, cy_c: float, radius: float) -> np.ndarray:
    """每个链接的壁距 q（沿 e_i 从流体节点到圆柱面的距离比例，0<q≤1）。"""
    ex, ey = float(CX[direction]), float(CY[direction])
    dx = cyl_x - cx_c
    dy = cyl_y - cy_c
    # |d + q e|² = R² → q² + 2q(d·e) + |d|² − R² = 0
    b = dx * ex + dy * ey
    c = dx * dx + dy * dy - radius * radius
    disc = b * b - c
    disc = np.maximum(disc, 0.0)
    # 近侧交点：q = −b − √(b²−c)（d 在圆外 c≥0，小根即壁距比例）
    q = -b - np.sqrt(disc)
    return np.clip(q, 1e-6, 1.0)


def build_link_table(fluid: np.ndarray, cyl: np.ndarray,
                     cx_c: float, cy_c: float, radius: float,
                     bouzidi: bool = True):
    """构建圆柱反弹链接表（扁平数组，供 numpy/numba 共用）。

    返回 dict: ys/xs（流体节点坐标）、dirs（指向固体的方向）、
    q（壁距比例，bouzidi=False 时恒 0.5）、sys/sxs（x_f−e_i 源节点，q<0.5 用）。
    """
    ys_all, xs_all, dirs_all, q_all, sys_all, sxs_all = [], [], [], [], [], []
    ny, nx = cyl.shape
    for i in range(1, 9):
        nb = np.roll(cyl, shift=(-CY[i], -CX[i]), axis=(0, 1))
        m = fluid & nb
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        if bouzidi:
            q = cylinder_link_q(ys.astype(np.float64), xs.astype(np.float64),
                                i, cx_c, cy_c, radius)
        else:
            q = np.full(ys.size, 0.5)
        sy = ys - CY[i]
        sx = xs - CX[i]
        if sy.min() < 0 or sy.max() >= ny or sx.min() < 0 or sx.max() >= nx:
            raise RuntimeError("反弹链接源节点越界（圆柱应远离边界）")
        if not fluid[sy, sx].all():
            raise RuntimeError("q<0.5 源节点落在固体上（凸体不应发生）")
        ys_all.append(ys)
        xs_all.append(xs)
        dirs_all.append(np.full(ys.size, i, dtype=np.int64))
        q_all.append(q)
        sys_all.append(sy)
        sxs_all.append(sx)
    if not ys_all:
        empty_i = np.zeros(0, dtype=np.int64)
        return {"ys": empty_i, "xs": empty_i, "dirs": empty_i,
                "q": np.zeros(0), "sys": empty_i, "sxs": empty_i}
    return {
        "ys": np.concatenate(ys_all).astype(np.int64),
        "xs": np.concatenate(xs_all).astype(np.int64),
        "dirs": np.concatenate(dirs_all).astype(np.int64),
        "q": np.concatenate(q_all),
        "sys": np.concatenate(sys_all).astype(np.int64),
        "sxs": np.concatenate(sxs_all).astype(np.int64),
    }


def apply_bb(f_post: np.ndarray, f_new: np.ndarray, table: dict):
    """numpy 路径：Bouzidi 反弹写入 f_new，并返回动量交换力 (fx, fy)。

    F = Σ_links e_i [f_i*(x_f) + f_ī_new(x_f)]；q≡0.5 时退化为 2 Σ e_i f_i*。
    """
    ys, xs, dirs = table["ys"], table["xs"], table["dirs"]
    q, sys_, sxs = table["q"], table["sys"], table["sxs"]
    fx = 0.0
    fy = 0.0
    for k in range(ys.size):
        y, x, i = int(ys[k]), int(xs[k]), int(dirs[k])
        qq = float(q[k])
        fi = f_post[y, x, i]
        if qq < 0.5:
            bb = 2.0 * qq * fi + (1.0 - 2.0 * qq) * f_post[int(sys_[k]), int(sxs[k]), i]
        else:
            inv = 1.0 / (2.0 * qq)
            bb = inv * fi + (2.0 * qq - 1.0) * inv * f_post[y, x, OPP[i]]
        f_new[y, x, OPP[i]] = bb
        fx += CX[i] * (fi + bb)
        fy += CY[i] * (fi + bb)
    return fx, fy
