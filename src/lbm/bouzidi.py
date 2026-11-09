"""Bouzidi-Lallemand-Luo 次网格反弹（curved boundary，q 加权）。

对每条流体→固体链接按壁距 q∈(0,1] 插值：
  q≥0.5: f_ī(x_f,t+1) = 2q·f_i*(x_f,t) + (1−2q)·f_i*(x_f+e_i,t)
  q<0.5: f_ī(x_f,t+1) = (1/(2q))·f_i*(x_f,t) + ((2q−1)/(2q))·f_ī(x_f,t)
q 由圆柱解析面 |x_f + q e_i − C| = R 求得（二次方程小根）。
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
    # 近侧交点：q = −b − √(b²−c)（d 在圆外 c>0，小根>0 即壁距比例）
    q = -b - np.sqrt(disc)
    return np.clip(q, 1e-6, 1.0)


class BouzidiBB:
    """预计算每方向的 (流体节点, q, 反弹源节点) 索引表。"""

    def __init__(self, solid: np.ndarray, fluid: np.ndarray,
                 cx_c: float, cy_c: float, radius: float):
        ny, nx = solid.shape
        self.entries = []  # 每方向: (ys, xs, q, src_ys, src_xs, near)
        for i in range(1, 9):
            nb = np.roll(solid, shift=(-CY[i], -CX[i]), axis=(0, 1))
            m = fluid & nb
            ys, xs = np.nonzero(m)
            if ys.size == 0:
                self.entries.append(None)
                continue
            q = cylinder_link_q(ys.astype(float), xs.astype(float), i, cx_c, cy_c, radius)
            s_ys = ys + CY[i]  # x_f + e_i（固体侧）
            s_xs = xs + CX[i]
            s_ys = np.clip(s_ys, 0, ny - 1)
            s_xs = np.clip(s_xs, 0, nx - 1)
            self.entries.append((ys, xs, q, s_ys, s_xs))

    def apply(self, f_post: np.ndarray, f_new: np.ndarray):
        """f_post: 碰撞后；f_new: 迁移后（含固体侧垃圾）。"""
        for i in range(1, 9):
            e = self.entries[i]
            if e is None:
                continue
            ys, xs, q, s_ys, s_xs = e
            opp = OPP[i]
            fi_star = f_post[ys, xs, i]
            fi_star_s = f_post[s_ys, s_xs, i]      # 固体侧同向（若无碰撞则为 0）
            fopp_new = f_new[ys, xs, opp]          # 迁移带来的（q<0.5 时作二次反弹源）
            near = q >= 0.5
            out = np.empty_like(fi_star)
            out[near] = 2 * q[near] * fi_star[near] + (1 - 2 * q[near]) * fi_star_s[near]
            inv = 1.0 / (2 * q[~near])
            out[~near] = inv * fi_star[~near] + (2 * q[~near] - 1) * inv * fopp_new[~near]
            f_new[ys, xs, opp] = out
