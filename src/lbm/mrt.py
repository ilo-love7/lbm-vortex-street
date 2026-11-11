"""D2Q9 MRT 碰撞算子（v1.2）：纯函数 + 可注入求解器。

矩定义按 d'Humières/Lallemand-Luo (2002)，列序按本包 kernels.CX/CY
（i0:(0,0) i1:(0,1) i2:(1,1) i3:(1,0) i4:(1,-1) i5:(0,-1) i6:(-1,-1) i7:(-1,0) i8:(-1,1)）。
S = diag(0, s_e, s_ε, 0, s_q, 0, s_q, s_ν, s_ν)，s_ν = 1/τ。
当所有 s_* = s_ν 时 MRT ≡ BGK（单测机器精度验证）。
"""
from __future__ import annotations

import numpy as np

# 速度序（与 kernels.py 一致）：(cx, cy)
CX = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1], dtype=np.float64)
CY = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1], dtype=np.float64)

# Lallemand-Luo 原始列序 (0,0)(1,0)(0,1)(-1,0)(0,-1)(1,1)(-1,1)(-1,-1)(1,-1)
# 到本包列序的置换：LL_k -> ours[PERM[k]]
_PERM = np.array([0, 3, 1, 7, 5, 2, 8, 6, 4], dtype=np.int64)

# Lallemand-Luo 2002 变换矩阵（行=矩，列=LL 速度序）
_M_LL = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],          # rho
    [-4, -1, -1, -1, -1, 2, 2, 2, 2],     # e
    [4, -2, -2, -2, -2, 1, 1, 1, 1],      # eps
    [0, 1, 0, -1, 0, 1, -1, -1, 1],       # jx
    [0, -2, 0, 2, 0, 1, -1, -1, 1],       # qx
    [0, 0, 1, 0, -1, 1, 1, -1, -1],       # jy
    [0, 0, -2, 0, 2, 1, 1, -1, -1],       # qy
    [0, 1, -1, 1, -1, 0, 0, 0, 0],        # pxx
    [0, 0, 0, 0, 0, 1, -1, 1, -1],        # pxy
], dtype=np.float64)


def mrt_matrices() -> tuple[np.ndarray, np.ndarray]:
    """返回 (M, Minv)：m = M @ f，f = Minv @ m（本包速度列序）。"""
    M = np.empty_like(_M_LL)
    M[:, _PERM] = _M_LL
    Minv = np.linalg.inv(M)
    return M, Minv


def mrt_meq(rho: np.ndarray, jx: np.ndarray, jy: np.ndarray) -> np.ndarray:
    """平衡态矩向量（最后一维=9）。

    meq = [ρ, -2ρ+3|j|²/ρ, ρ-3|j|²/ρ, jx, -jx, jy, -jy, (jx²-jy²)/ρ, jx·jy/ρ]
    """
    rho = np.asarray(rho, dtype=np.float64)
    jx = np.asarray(jx, dtype=np.float64)
    jy = np.asarray(jy, dtype=np.float64)
    safe = np.where(rho > 1e-30, rho, 1.0)
    usq = (jx * jx + jy * jy) / safe
    meq = np.stack([
        rho,
        -2.0 * rho + 3.0 * usq,
        rho - 3.0 * usq,
        jx,
        -jx,
        jy,
        -jy,
        (jx * jx - jy * jy) / safe,
        jx * jy / safe,
    ], axis=-1)
    return meq


def mrt_relax_rates(s_nu: float, preset: str = "magic") -> np.ndarray:
    """松弛率对角向量 S（9,）。

    preset:
      bgk    全部 = s_nu（严格退化为 BGK）
      magic  s_e = s_ε = 1.1，s_q = 1.2（常见 D2Q9 低噪声参数化）
      bulk   s_e = s_ε = 1.0，s_q = 1.0（体积/声模更快弛豫）
    """
    s = np.full(9, float(s_nu))
    if preset == "bgk":
        return s
    if preset == "magic":
        s[1] = 1.1   # s_e
        s[2] = 1.1   # s_eps
        s[4] = 1.2   # s_q
        s[6] = 1.2   # s_q
        return s
    if preset == "bulk":
        s[1] = 1.0
        s[2] = 1.0
        s[4] = 1.0
        s[6] = 1.0
        return s
    raise ValueError(f"未知 MRT preset: {preset}")


def mrt_collide(f: np.ndarray, M: np.ndarray, Minv: np.ndarray,
                S: np.ndarray, tau: float) -> np.ndarray:
    """向量化 MRT 碰撞（f 形状 (..., 9)），返回碰撞后副本。

    全 s=1/tau 时与 BGK 碰撞机器精度一致（等价性由单测锁定）。
    """
    f = np.asarray(f, dtype=np.float64)
    m = f @ M.T
    rho = m[..., 0]
    jx = m[..., 3]
    jy = m[..., 5]
    meq = mrt_meq(rho, jx, jy)
    s = np.asarray(S, dtype=np.float64)
    if s.ndim == 1:
        m_post = m + s * (meq - m)
    else:
        m_post = m + s * (meq - m)
    return m_post @ Minv.T
