"""D2Q9 格子：速度集、权、反对称映射、平衡分布（Mocz 2020 顺序）。

i:   0  1  2  3  4  5  6  7  8
cx:  0  0  1  1  1  0 -1 -1 -1
cy:  0  1  1  0 -1 -1 -1  0  1
opp: 0  5  6  7  8  1  2  3  4
"""
from __future__ import annotations

import numpy as np

NL = 9
CX = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
CY = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])
OPP = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4], dtype=np.int64)
CS2 = 1.0 / 3.0


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """f_eq = w_i ρ (1 + 3 c·u + 4.5 (c·u)² − 1.5 u²)。返回 (ny, nx, 9)。"""
    f = np.empty(rho.shape + (NL,), dtype=rho.dtype)
    usq = ux * ux + uy * uy
    for i in range(NL):
        cu = CX[i] * ux + CY[i] * uy
        f[..., i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    return f


def macros(f: np.ndarray):
    """由分布得 ρ, ux, uy（固体节点 ρ=0 时返回零速度，避免 nan）。"""
    rho = f.sum(axis=2)
    rho_safe = np.where(rho > 0.0, rho, 1.0)
    ux = (f * CX).sum(axis=2) / rho_safe
    uy = (f * CY).sum(axis=2) / rho_safe
    ux = np.where(rho > 0.0, ux, 0.0)
    uy = np.where(rho > 0.0, uy, 0.0)
    return rho, ux, uy
