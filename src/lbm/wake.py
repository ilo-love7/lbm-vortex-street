"""E14 独立主判据：远场尾流动量亏损（wake survey，v1.3 严格两截面口径）。

为什么需要它
------------
``stress.cv_force_links`` 虽然不读壁面链接，但扣除逐步动量累积项后是离散
动量守恒律的**恒等式**，实测逐位复现 MEM（maxdiff < 1e-12），因此对
「MRT 把 Cd 抬高 +1.3~1.9% 是不是动量交换法的读数伪影」**没有判别力**。
要判定这一点，需要一条只读**守恒矩**（ρ / ρu）的独立路径：BGK 与 MRT
对守恒矩的弛豫率都恒为 0，若两算子的平均流场相同，远场观测就应相同。

公式（严格两截面口径，v1.3 定案）
----------------------------------
取参考截面 ``x_ref`` 落在**西侧海绵层之后、圆柱之前**的干净均匀流区，
下游截面 ``x`` 落在东侧海绵之前。对控制体 [x_ref, x] × [1, ny−2] 写
稳态 x 动量守恒（上下自由滑移壁无 x 向应力、无质量穿过）：

    F_D = Σ_{x_ref}(ρ u_x² + p − σ'_xx) − Σ_x(ρ u_x² + p − σ'_xx)

其中 ``p = ρ c_s²``，``σ'_xx = −(1−1/(2τ)) Π^neq_xx``（默认忽略，远场
剪切弱，实测贡献 < 1e-4；可用 ``include_viscous`` 打开作稳健性对照）。
**不做质量守恒化简、不假设 p∞=ρ₀c_s²**——这是 v1.3 的关键修正。

为什么不能用经典 wake survey（v1.3 发现的反例，保留作对照）
-----------------------------------------------------------
经典写法 ``D = ∫ρu(U−u)dy + ∫(p∞−p)dy`` 用 ``∫ρU²dy = U∫ρu dy`` 把
上游通量化成 ``ρ₀U²H``，并假设上游压力为 ``p∞``。本项目西侧入口带
**目标态海绵层**（体积力源），x=1 截面已被弛豫，两个假设同时失效：
v1.3 实测经典口径只给 Cd≈0.61，而 MEM / 严格两截面口径给 1.34~1.40，
**低估 53%**；且下游截面 ``Σρu² > UΣρu``（10% 阻塞使两侧加速区未恢复），
连 ``momentum`` 分项都是负的。``x_ref=None`` 时本模块退回该经典口径，
仅供报告量化这一偏差，不作判据。

非定常口径
----------
对**瞬时**被积函数做时间平均，而非对平均速度套用稳态公式——后者会丢掉
周期性涡脱落的雷诺应力项 ⟨ρu'_x²⟩。每次调用返回瞬时值，由实验脚本在
整数个脱落周期上取平均。上下自由滑移壁节点经过镜面反射处理，密度可能
带边界伪影，故积分跳过 y=0 与 y=ny−1 两行（在报告中如实标注）。
"""
from __future__ import annotations

import numpy as np

from .lattice import CS2, CX, CY, equilibrium

_CX = CX.astype(np.float64)
_CY = CY.astype(np.float64)
_CXX = (CX * CX).astype(np.float64)


def wake_stations(cx_c: float, d: float, nx: int,
                  stations=(3.0, 6.0, 10.0), downstream_guard: int = 40):
    """下游站位 x/D ∈ stations；自动剔除落在出口海绵层内的站位。"""
    out = []
    for s in stations:
        x = int(round(float(cx_c) + float(s) * float(d)))
        if 0 < x < int(nx) - downstream_guard:
            out.append((float(s), x))
    return out


def wake_ref_x(sponge_width: int, margin: int = 2) -> int:
    """参考截面：西侧海绵层之后 margin 格（须仍在圆柱上游的干净均匀流区）。"""
    return int(sponge_width) + int(margin)


def _section(f, x: int, drop_wall_rows: bool = True):
    """单截面守恒矩：ρ, ρu_x, ρu_y, Π^neq_xx。"""
    fc = np.asarray(f)[:, int(x), :]
    if drop_wall_rows and fc.shape[0] > 2:
        fc = fc[1:-1, :]
    rho = fc.sum(axis=1)
    rs = np.where(rho > 0.0, rho, 1.0)
    ux = (fc @ _CX) / rs
    uy = (fc @ _CY) / rs
    ux = np.where(rho > 0.0, ux, 0.0)
    uy = np.where(rho > 0.0, uy, 0.0)
    return rho, ux, uy, fc


def _flux_xx(f, x, drop_wall_rows=True):
    """截面 x 上的 ``Σ(ρu_x² + p)`` 与 ``Σσ'_xx``（后者需 τ）。"""
    rho, ux, uy, fc = _section(f, x, drop_wall_rows)
    conv_p = float(np.sum(rho * ux * ux + rho * CS2))
    return conv_p, (rho, ux, uy, fc)


def _sigma_xx(rho, ux, uy, fc, tau):
    fneq = fc - equilibrium(rho, ux, uy)
    pxx = fneq @ _CXX
    return float(np.sum(-(1.0 - 1.0 / (2.0 * float(tau))) * pxx))


def wake_force(f: np.ndarray, x: int, rho0: float, u_in: float,
               cs2: float = CS2, x_ref: int | None = None,
               include_viscous: bool = False, tau: float | None = None,
               drop_wall_rows: bool = True) -> dict:
    """单个下游截面上的瞬时阻力（格子单位、单位深度）。

    ``x_ref`` 非空 → **v1.3 严格两截面口径**（主判据）；
    ``x_ref=None`` → 经典 wake survey（方法学反例，实测低估 53%）。

    返回 dict：``force`` 总力；``momentum`` / ``pressure`` / ``viscous``
    分项；``mass_flux`` 与 ``mass_flux_ref`` 诊断质量守恒；``mode`` 口径标记。
    """
    if include_viscous and (tau is None or tau <= 0.5):
        raise ValueError("include_viscous 需要 BGK 弛豫时间 tau>0.5")
    fa = np.asarray(f)
    a, (rho, ux, uy, fc) = _flux_xx(fa, x, drop_wall_rows)
    mass = float(np.sum(rho * ux))
    viscous = 0.0
    if include_viscous:
        viscous = _sigma_xx(rho, ux, uy, fc, tau)

    if x_ref is not None:
        b, (rho_r, ux_r, uy_r, fc_r) = _flux_xx(fa, int(x_ref),
                                                 drop_wall_rows)
        mass_ref = float(np.sum(rho_r * ux_r))
        # 分解为对流项差与压力项差（逐项分别作差，避免大数相减丢精度）
        conv = float(np.sum(rho_r * ux_r * ux_r) - np.sum(rho * ux * ux))
        pres = float(cs2 * (np.sum(rho_r) - np.sum(rho)))
        if include_viscous:
            viscous = viscous - _sigma_xx(rho_r, ux_r, uy_r, fc_r, tau)
        return {"force": conv + pres + viscous,
                "momentum": conv, "pressure": pres, "viscous": viscous,
                "mass_flux": mass, "mass_flux_ref": mass_ref,
                "ref_flux": b, "station_flux": a,
                "mode": "two_section"}

    # 经典口径：p∞=ρ₀c_s²，上游 ρ₀U²H 由 U·(站位质量通量) 化简
    momentum = float(np.sum(rho * ux * (float(u_in) - ux)))
    p_inf = float(rho0) * float(cs2)
    pressure = float(np.sum(p_inf - rho * float(cs2)))
    return {"force": momentum + pressure + viscous,
            "momentum": momentum, "pressure": pressure,
            "viscous": viscous, "mass_flux": mass,
            "mass_flux_ref": float("nan"), "ref_flux": float("nan"),
            "station_flux": a, "mode": "classic"}


def wake_cd(f: np.ndarray, x: int, rho0: float, u_in: float, d: float,
            **kw) -> dict:
    """归一化到 Cd = F/(0.5ρ₀U²D)；分项也一并返回（便于诊断）。"""
    q = 0.5 * float(rho0) * float(u_in) ** 2 * float(d)
    r = wake_force(f, x, rho0, u_in, **kw)
    raw = ("mass_flux", "mass_flux_ref", "mode")
    return {**{k: v / q for k, v in r.items() if k not in raw},
            **{k: r[k] for k in raw if k in r}}


def station_consistency(cd_by_station: dict) -> dict:
    """站位无关性：不同 x/D 给出的平均阻力分散度（守恒律检验）。"""
    vals = np.asarray([float(v) for v in cd_by_station.values()], dtype=float)
    if vals.size == 0:
        return {"spread": float("nan"), "rel_spread": float("nan"), "n": 0}
    mean = float(vals.mean())
    spread = float(vals.max() - vals.min())
    return {"spread": spread,
            "rel_spread": spread / abs(mean) if mean != 0 else float("nan"),
            "mean": mean, "n": int(vals.size)}
