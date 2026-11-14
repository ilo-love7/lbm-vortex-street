"""E15 (v1.3)：Smagorinsky 亚格子模型 + 持续入口横向扰动。

Smagorinsky LBM（隐式解）
-------------------------
局部非平衡二阶矩 ``Π^neq = Σ c_i c_i (f_i − f_i^eq)`` 给出应变率

    S_αβ = −3 Π^neq_αβ / (2 ρ τ_total)          (c_s² = 1/3，格子单位)

定义 ``Q = sqrt(2 Π^neq:Π^neq) = sqrt(2(Πxx²+Πyy²+2Πxy²))``，则
``|S| = sqrt(2 S:S) = 3Q/(2ρτ_total)``。Smagorinsky 涡黏
``ν_t = (C_s Δ)² |S|``，且 ``τ_total = 0.5 + 3(ν₀ + ν_t) = τ₀ + 3ν_t``，
代入后得到关于 τ_total 的二次方程

    τ_total² − τ₀ τ_total − 9 C_s² Δ² Q / (2ρ) = 0
    τ_total = (τ₀ + sqrt(τ₀² + 18 C_s² Δ² Q / ρ)) / 2     （取正根）

这是 Krafczyk/Hou 等的经典隐式解，无需迭代、天然 τ_total ≥ τ₀ > 0.5。
本模块把它做成纯函数，便于单测；内核只多一次开方。

**关键诊断量**：``nu_t_over_nu0 = (τ_total − τ₀)/(τ₀ − 0.5) = ν_t/ν₀``。
在 D=28 lu 的 2D 网格上 |S| ~ U/D ~ 3.6e-3，ν_t/ν₀ 通常只有百分之几——
即亚格子耗散相对分子耗散几乎可忽略。这决定了 2D LES 能（或不能）带来什么。

入口扰动
--------
v1.0–v1.2 的 ``trip`` 只是**初始**横向随机扰动，会被流场遗忘；真实来流
湍流度是持续的。这里给西边界 Zou/He 速度条件加一个**零均值、随时间变化**
的横向速度剖面

    uy_b(y,t) = ε U · sin(2π m y / ny) · sin(2π t / T_f)

零均值（对 y 与 t 都是），不注入净横向动量；T_f 取自然脱落周期 T 的非整数
倍以避免锁频。Zou/He 在 uy_b≠0 时的解析式见 kernels._step_kernel（仅在
f2/f4 上多加 ∓0.5 ρ uy_b）。
"""
from __future__ import annotations

import numpy as np

from .lattice import CS2, CX, CY, equilibrium


def pi_neq_norm(f: np.ndarray, rho=None, ux=None, uy=None) -> np.ndarray:
    """``Q = sqrt(2 Π^neq:Π^neq)``，返回与节点数同形的数组。

    f 可为 (9,)、(N,9) 或 (ny,nx,9)；rho/ux/uy 省略时由 f 的矩自算。
    """
    fa = np.asarray(f, dtype=np.float64)
    shape = fa.shape[:-1]
    fc = fa.reshape(-1, 9)
    if rho is None:
        rho = fc.sum(axis=1)
    if ux is None:
        ux = (fc @ CX.astype(np.float64)) / np.where(rho > 0, rho, 1.0)
    if uy is None:
        uy = (fc @ CY.astype(np.float64)) / np.where(rho > 0, rho, 1.0)
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    ux = np.asarray(ux, dtype=np.float64).reshape(-1)
    uy = np.asarray(uy, dtype=np.float64).reshape(-1)
    ux = np.where(rho > 0, ux, 0.0)
    uy = np.where(rho > 0, uy, 0.0)
    fneq = fc - equilibrium(rho, ux, uy)
    pxx = fneq @ (CX * CX).astype(np.float64)
    pyy = fneq @ (CY * CY).astype(np.float64)
    pxy = fneq @ (CX * CY).astype(np.float64)
    q = np.sqrt(2.0 * np.maximum(pxx * pxx + pyy * pyy + 2.0 * pxy * pxy, 0.0))
    return q.reshape(shape)


def smagorinsky_tau(tau0: float, q_neq, cs_smag: float,
                    rho=1.0, dx: float = 1.0):
    """隐式 Smagorinsky 总弛豫时间（标量或数组均可）。

    参数
    ----
    tau0 : 分子弛豫时间 = 0.5 + 3ν₀
    q_neq : ``pi_neq_norm`` 的输出
    cs_smag : Smagorinsky 常数 C_s（0 即退化为 BGK）
    rho : 局部密度（标量或数组）
    dx : 格子间距 Δ（默认 1 lu）
    """
    q = np.asarray(q_neq, dtype=np.float64)
    r = np.asarray(rho, dtype=np.float64)
    rs = np.where(r > 1e-12, r, 1.0)
    # τ_total² − τ₀τ_total − 9C_s²Δ²Q/(2ρ) = 0
    # ⇒ 判别式 τ₀² + 18C_s²Δ²Q/ρ（注意不是 4×18，推导见模块 docstring）
    disc = (float(tau0) ** 2
            + 18.0 * (float(cs_smag) ** 2) * (float(dx) ** 2) * q / rs)
    disc = np.where(r > 1e-12, disc, float(tau0) ** 2)
    tau_tot = 0.5 * (float(tau0) + np.sqrt(np.maximum(disc, 0.0)))
    if np.isscalar(q_neq) or q.ndim == 0:
        return float(np.asarray(tau_tot))
    return tau_tot


def nu_t_over_nu0(tau_total, tau0: float):
    """ν_t/ν₀ = (τ_total − τ₀)/(τ₀ − 0.5)；τ₀=0.5 时返回 inf（无分子黏性）。"""
    denom = float(tau0) - 0.5
    if denom <= 0.0:
        return np.inf
    return (np.asarray(tau_total, dtype=np.float64) - float(tau0)) / denom


def inlet_base(ny: int, eps: float, u_in: float, m: int = 3) -> np.ndarray:
    """入口横向扰动的空间剖面 ε U sin(2π m y / ny)，对 y 严格零均值。

    m 取整数时离散求和 Σ_y sin(2π m y/ny) = 0（精确），因此不注入净横向动量。
    """
    y = np.arange(int(ny), dtype=np.float64)
    return float(eps) * float(u_in) * np.sin(2.0 * np.pi * int(m) * y / int(ny))


def inlet_uy(base: np.ndarray, step: int, period_steps: float) -> np.ndarray:
    """某一步的入口 uy_b 数组 = base · sin(2π t / T_f)。"""
    if period_steps <= 0.0:
        return np.zeros_like(base)
    return base * np.sin(2.0 * np.pi * float(step) / float(period_steps))


def apply_smagorinsky_numpy(f: np.ndarray, tau0: float, cs_smag: float):
    """numpy 参考实现：一步 BGK+LES 碰撞（单测用于对照 numba 内核）。"""
    from .lattice import macros
    rho, ux, uy = macros(f)
    q = pi_neq_norm(f, rho, ux, uy)
    tau_loc = smagorinsky_tau(tau0, q, cs_smag, rho=rho)
    feq = equilibrium(rho, ux, uy)
    omega = 1.0 / np.asarray(tau_loc)[..., None]
    return f - omega * (f - feq), tau_loc
