"""涡量/速度探针频谱分析（v1.2）：纯函数。

- probe_st：任意探针时程的 St（Hann+抛物线插值，与 postprocess.strouhal 同口径）
- dual_st：升力 St vs 探针 St 双测法一致性
- convection_delay：两探针互谱相位 → 涡对流时滞/速度
- harmonic_ratio：二次谐波/基波幅值比（脱落非正弦度）
"""
from __future__ import annotations

import numpy as np


def _peak_st(x: np.ndarray, d: float, u_in: float, dt: float,
             band: tuple[float, float]) -> dict:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 16:
        raise ValueError("时程太短")
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=dt)
    st_arr = freqs * d / u_in
    lo, hi = band
    mask = (st_arr > lo) & (st_arr < hi)
    if not mask.any():
        raise ValueError("频带内无谱线")
    k = int(np.argmax(spec[mask]))
    idx = int(np.flatnonzero(mask)[k])
    delta = 0.0
    if 0 < idx < spec.size - 1:
        y0, y1, y2 = spec[idx - 1], spec[idx], spec[idx + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
    f_peak = freqs[idx] + delta * (freqs[1] - freqs[0])
    return {"st": float(f_peak * d / u_in), "f_peak_lu": float(f_peak),
            "spec": spec, "freqs": freqs, "idx": idx}


def probe_st(signal: np.ndarray, d: float, u_in: float, dt: float = 1.0,
             band: tuple[float, float] = (0.02, 0.5)) -> dict:
    """任意单探针时程（uy、ω 等）的主频 St。"""
    return _peak_st(signal, d, u_in, dt, band)


def dual_st(cl: np.ndarray, probe: np.ndarray, d: float, u_in: float,
            dt: float = 1.0) -> dict:
    """升力与探针双测 St 一致性（v1.1 双测法口径扩展）。"""
    a = _peak_st(cl, d, u_in, dt, (0.02, 0.5))
    b = _peak_st(probe, d, u_in, dt, (0.02, 0.5))
    rel = abs(a["st"] - b["st"]) / max(a["st"], 1e-12)
    return {"st_force": a["st"], "st_probe": b["st"], "rel_diff": float(rel)}


def convection_delay(sig_up: np.ndarray, sig_down: np.ndarray, d: float,
                     u_in: float, dx: float, dt: float = 1.0) -> dict:
    """上下游探针互谱在基频处的相位差 → 时滞 → 对流速度 U_c = dx / delay。

    相位主值折叠到 (-π, π]，时滞取最小正解（假设 U_c ∈ (0.2, 3)U）。
    """
    a = np.asarray(sig_up, dtype=float)
    b = np.asarray(sig_down, dtype=float)
    n = min(a.size, b.size)
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    win = np.hanning(n)
    fa = np.fft.rfft(a * win)
    fb = np.fft.rfft(b * win)
    freqs = np.fft.rfftfreq(n, d=dt)
    st_arr = freqs * d / u_in
    mask = (st_arr > 0.02) & (st_arr < 0.5)
    if not mask.any():
        raise ValueError("频带内无谱线")
    csd = fa[mask] * np.conj(fb[mask])
    k = int(np.argmax(np.abs(csd)))
    idx = int(np.flatnonzero(mask)[k])
    f_peak = freqs[idx]
    phase = float(np.angle(csd[k]))          # arg(Fa·conj(Fb)) = φa−φb
    # u 在上游、b 在下游：b 滞后 a → φa−φb = +2πf·delay（mod 2π）
    delay = phase / (2.0 * np.pi * f_peak)
    # 折叠到 (0, 1/f_peak)
    period = 1.0 / f_peak
    while delay < 0:
        delay += period
    while delay >= period:
        delay -= period
    u_c = dx / delay if delay > 1e-12 else float("inf")
    return {"delay_steps": float(delay), "u_conv": float(u_c),
            "u_conv_over_u": float(u_c / u_in), "f_peak_lu": float(f_peak),
            "phase_rad": phase}


def harmonic_ratio(signal: np.ndarray, d: float, u_in: float, dt: float = 1.0,
                   n_harm: int = 2, half_width: int = 3) -> dict:
    """n 次谐波/基波幅值比（Hann 窗 2|X|/Σw 口径）。

    谐波峰在整数 bin 上取样会系统性低估（主瓣宽 2 bins，频率略偏时精确 2×idx
    处幅值近零）。改为在 n_harm×idx ± half_width 邻域取最大，并返回实际 bin。
    """
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    n = x.size
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=dt)
    st_arr = freqs * d / u_in
    mask = (st_arr > 0.02) & (st_arr < 0.5)
    k = int(np.argmax(spec[mask]))
    idx = int(np.flatnonzero(mask)[k])
    amp1 = 2.0 * spec[idx] / win.sum()
    j = min(idx * n_harm, spec.size - 1)
    # 下界须避开基波 Hann 主瓣（半宽 2 bins），否则低频时邻域会把基波尾巴当成谐波
    lo = max(idx + 3, j - half_width)
    hi = min(spec.size, j + half_width + 1)
    jj = lo + int(np.argmax(spec[lo:hi]))
    amp_h = 2.0 * spec[jj] / win.sum()
    return {"amp_fund": float(amp1), "amp_harm": float(amp_h),
            "fund_idx": idx, "harm_idx": jj,
            "ratio": float(amp_h / amp1) if amp1 > 0 else float("nan")}
