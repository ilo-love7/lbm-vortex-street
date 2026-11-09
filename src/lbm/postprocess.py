"""后处理：时均 Cd、Cl RMS、Strouhal 数（升力谱主频）。"""
from __future__ import annotations

import numpy as np


def time_stats(cd: np.ndarray, cl: np.ndarray) -> dict:
    return {
        "cd_mean": float(np.mean(cd)),
        "cd_std": float(np.std(cd)),
        "cl_rms": float(np.sqrt(np.mean(cl**2))),
        "cl_mean": float(np.mean(cl)),
    }


def strouhal(cl: np.ndarray, d: float, u_in: float, dt: float = 1.0) -> dict:
    """升力时程 FFT 主频 → St = f·D/U。

    Hann 窗 + 抛物线插值细化峰值频率；dt 为采样间隔（格子步）。
    注意：降采样必须先低通，否则高频力噪声混叠入带内（实测教训）。
    """
    x = np.asarray(cl, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 16:
        raise ValueError("时程太短，无法估计 St")
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=dt)  # [1/格子步]
    # 只看 f·D/U ∈ [0.02, 0.5] 的物理区间
    st_arr = freqs * d / u_in
    band = (st_arr > 0.02) & (st_arr < 0.5)
    if not band.any():
        raise ValueError("频带内无谱线")
    k = np.argmax(spec[band])
    idx = np.flatnonzero(band)[k]
    # 抛物线插值细化
    if 0 < idx < spec.size - 1:
        y0, y1, y2 = spec[idx - 1], spec[idx], spec[idx + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    f_peak = freqs[idx] + delta * (freqs[1] - freqs[0])
    st = f_peak * d / u_in
    return {"st": float(st), "f_peak_lu": float(f_peak), "spectrum": (freqs, spec)}


def literature_st(re: float) -> float:
    """文献经验式 St ≈ 0.2124 − 4.592/Re（圆柱绕流，Re ≳ 50）。"""
    return 0.2124 - 4.592 / re


def coherent_cl(cl: np.ndarray, d: float, u_in: float, dt: float = 1.0) -> dict:
    """升力相干峰幅（Hann 窗单边正弦幅值 2|X|/Σw）+ 带内 rms。

    相干幅不受宽带力噪声膨胀影响，比 total rms 更公平（v1.1 收口口径）。
    """
    x = np.asarray(cl, dtype=float)
    n = x.size
    win = np.hanning(n)
    xw = (x - x.mean()) * win
    spec = np.abs(np.fft.rfft(xw))
    freqs = np.fft.rfftfreq(n, d=dt)
    st_arr = freqs * d / u_in
    band = (st_arr > 0.02) & (st_arr < 0.5)
    k = int(np.argmax(spec[band]))
    idx = int(np.flatnonzero(band)[k])
    amp = 2.0 * spec[idx] / win.sum()
    # 带内 rms（主峰 ±2 谱线带通逆变换）
    phase = np.angle(np.fft.rfft(xw))
    mask = np.zeros_like(spec)
    mask[max(idx - 2, 0):min(idx + 3, spec.size)] = 1.0
    xw2 = np.fft.irfft(spec * mask * np.exp(1j * phase), n)
    return {"cl_peak_amp": float(amp), "st_peak": float(st_arr[idx]),
            "cl_rms_total": float(np.sqrt((x ** 2).mean())),
            "cl_rms_band": float(np.sqrt((xw2 ** 2).mean()))}
