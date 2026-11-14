"""v1.3 后处理纯函数（E13 阻塞比外推 / E14 双口径 / E15 起振与耗散）。

单独成模块是为了让实验脚本与单测共用同一套口径：所有会影响结论的
统计量（整周期平均、β→0 外推、起振判据、围道一致性）都能被单测直接锁定。
"""
from __future__ import annotations

import numpy as np


def blockage_ratio(d: float, ny: int) -> float:
    """阻塞比 β = D/H，H 为上下（自由滑移）壁间距 = ny。"""
    return float(d) / float(ny)


def period_average(x, period_steps: float, min_periods: int = 1) -> dict:
    """按**整数个脱落周期**取尾部平均，消除非定常体积分项的相位残留。

    控制体面积分含 −d/dt∫ρu dV，只有整周期平均才严格为零；
    全段平均会留下 O(1/N_period) 的相位偏差（E14 报告会同时给出两者）。
    """
    a = np.asarray(x, dtype=float)
    T = float(period_steps)
    k = int(np.floor(a.size / T)) if T > 0 else 0
    full = float(a.mean())
    if k >= int(min_periods):
        n = int(round(k * T))
        return {"mean": float(a[-n:].mean()), "periods": k,
                "full_mean": full, "phase_residual": float(a[-n:].mean() - full)}
    return {"mean": full, "periods": k, "full_mean": full,
            "phase_residual": 0.0}


def fit_extrapolate_zero(beta, values, degree: int = 1) -> dict:
    """把 St/Cd 对阻塞比 β 做多项式最小二乘，外推到 β→0（无限域）。

    线性外推是主口径（物理上阻塞修正首项 ∝β）；二次外推作为稳健性对照，
    只有当点数 ≥ degree+1 时才给。返回截距、斜率、R² 与相对残差。
    """
    b = np.asarray(beta, dtype=float)
    v = np.asarray(values, dtype=float)
    if b.size != v.size or b.size < 2:
        raise ValueError("beta 与 values 等长且至少 2 点")
    coef = np.polyfit(b, v, int(degree))
    pred = np.polyval(coef, b)
    ss_res = float(np.sum((v - pred) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "intercept": float(np.polyval(coef, 0.0)),
        "slope": float(coef[-2]) if int(degree) >= 1 else 0.0,
        "r2": float(r2),
        "degree": int(degree),
        "rmse": float(np.sqrt(ss_res / b.size)),
        "beta_range": [float(b.min()), float(b.max())],
    }


def contour_consistency(cd_by_radius: dict) -> dict:
    """围道无关性：不同半边长给出的平均阻力分散度。

    控制体守恒律要求围道位置不影响 ⟨F⟩；分散度大说明海绵层/边界/泄漏污染。
    """
    vals = np.asarray([float(v) for v in cd_by_radius.values()], dtype=float)
    if vals.size == 0:
        return {"spread": float("nan"), "rel_spread": float("nan"), "n": 0}
    mean = float(vals.mean())
    spread = float(vals.max() - vals.min())
    return {"spread": spread,
            "rel_spread": spread / abs(mean) if mean != 0 else float("nan"),
            "mean": mean, "n": int(vals.size)}


def method_gap(cd_mem: float, cd_cv: float) -> dict:
    """双口径差：Δ = Cd_MEM − Cd_CV，及其相对量。"""
    return {"delta": float(cd_mem) - float(cd_cv),
            "rel_delta": (float(cd_mem) - float(cd_cv)) / abs(float(cd_cv))
            if cd_cv != 0 else float("nan")}


def onset_time(cl, period_steps: float, frac: float = 0.9,
               win_frac: float = 0.25) -> dict:
    """涡脱落起振时刻：滑动窗 RMS 首次达到「饱和 RMS」的 frac 比例。

    饱和 RMS 取末 1/4 时间窗内滑动 RMS 的中位数；返回起振步数与
    「以脱落周期计」的无量纲时间。全段都未低于阈值时返回 0（瞬时起振）。
    """
    x = np.asarray(cl, dtype=float)
    T = float(period_steps)
    w = max(int(round(T * float(win_frac))), 8)
    if x.size < 2 * w:
        return {"onset_steps": 0, "onset_periods": 0.0,
                "sat_rms": float(np.sqrt(np.mean(x ** 2)))}
    # 滑动窗 RMS（cumsum 精确）
    c = np.cumsum(np.r_[0.0, x * x])
    rms = np.sqrt((c[w:] - c[:-w]) / w)
    tail = rms[max(rms.size // 4 * 3, 1):]
    sat = float(np.median(tail)) if tail.size else float(np.median(rms))
    thr = float(frac) * sat
    hit = np.flatnonzero(rms >= thr)
    onset = int(hit[0]) if hit.size else 0
    return {"onset_steps": onset,
            "onset_periods": onset / T if T > 0 else 0.0,
            "sat_rms": sat, "threshold": thr}


def st_residual(st: float, st_lit: float) -> dict:
    return {"rel_err": (float(st) - float(st_lit)) / float(st_lit),
            "abs_err": float(st) - float(st_lit)}


def nu_ratio_summary(ratios) -> dict:
    """ν_t/ν₀ 场的分位数（LES 耗散强度诊断）。"""
    a = np.asarray(ratios, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}
