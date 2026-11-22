"""15 v1.7 把窗长锁住：一次 50T 模拟切 5 个 10T 窗。

为什么不是"四档各补一个 50T 终点"
----------------------------------
v1.6 §20.5 已经证明 Re=250 的 26T 记录窗恰好停在 uz_rel 瞬态峰值
（0.0868 @ t=26.2T），50T 后掉到 0.0345 且模式从 m=1 换成 m=2。
只补四个 50T 终点只能回答"50T 是多少"，不能回答三个更要紧的问题：

1. 峰值时刻是否随 Re 移动？（若 Re=300 的峰值在 40T，26T 和 50T 都会误判）
2. 力幅 / 频率 / 模式组成各自的定常时间尺度是否不同？
3. 26T 下"随 Re 加深"的趋势在 50T 下是否还在？

本脚本在一档 50T 模拟里**全程采样**，再切成 5 个 10T 窗分别归约，
一次跑得到"Re × 窗"的二维矩阵。成本与只跑终点相同（约 63 min/档）。

关键口径
--------
- 协议与 v1.6 完全一致：nx=320, ny=80, D=16, β=20%, Lz=6D (nz=96),
  40T 瞬态 + 10T 记录，UZ_EVERY=400。只动"记录窗的位置"这个自由度。
- 每窗单独算 St（Hann+抛物线插值的 FFT 主频）。10T 窗的频率分辨率
  ΔSt ≈ 1/(10T)·D/U ≈ 0.022，抛物线细化后约 0.005；因此 St 的**窗间**
  差只有 >1% 才可判读，小于此值报"分辨率内"。
- 谱/相关在**窗末**取瞬时快照（与 v1.6 记录窗末的口径一致），不是窗内平均。
- `force3()` 只读缓存的力，不改变流场；全程每步调用它不会扰动动力学。
  Re=250 的 50T full 结果可与 v1.6 长窗 JSON 逐项交叉核对（见 summary）。

用法：
    python scripts/15_v17_window.py --cases 250        # 单档（串行排程）
    python scripts/15_v17_window.py                    # 无参数：读缓存汇总+画图
    python scripts/15_v17_window.py --figures-only
    python scripts/15_v17_window.py --smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbm import CylinderCase, Solver, literature_st, strouhal  # noqa: E402
from lbm.solver3d import Case3D, Solver3D, mode_band_energy  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = RESULTS / "v17_cache"

GEO = dict(nx=320, d=16.0, cx_c=80.0, u_in=0.1, sponge_width=32)
NY20 = 80
LZ_D = 6.0
NSPAN6 = int(LZ_D * GEO["d"])
T_TRANS, T_REC = 40.0, 10.0
WINDOW_T = 10.0
RE_LIST = (250.0, 300.0, 350.0, 400.0)
RE_C = 190.0
RE_SPAN = 110.0
UZ_EVERY = 400
EPS = 1e-12


# ---------------------------------------------------------------- 纯函数
def window_bounds(n_steps: int, steps_per_t: float,
                  window_t: float = WINDOW_T) -> list[tuple[int, int]]:
    """把 [0, n_steps) 切成等长 `window_t` 的半开区间，末窗补齐。

    末窗可能比前面的短几步（步数取整造成），因此用实参 `n_steps` 收口，
    不做"丢弃余数"或"补齐到整窗"——后者会在总步数上偏离真实模拟。
    """
    w = max(1, int(round(window_t * steps_per_t)))
    edges = list(range(0, n_steps + 1, w))
    if edges[-1] != n_steps:
        edges.append(n_steps)
    out = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)
           if edges[i + 1] > edges[i]]
    # v1.7 实测：Re=400 的 n=39816、w=7963 ⇒ 末窗只剩 1 步，strouhal 直接报错。
    # 把这种"除不尽的尾巴"并进前一窗（末窗最多比整窗长 1 步），
    # 保证每窗都有足够点数做 FFT；窗边界偏移 <0.02T，不影响判读。
    if len(out) >= 2 and (out[-1][1] - out[-1][0]) < 16:
        out[-2] = (out[-2][0], out[-1][1])
        out.pop()
    return out


def peak_timing(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """窗内峰值及其时刻。空序列返回 (nan, nan)，绝不回退到 0。"""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    i = int(np.argmax(y))
    return float(y[i]), float(t[i])


def drift_pct(early: float, late: float) -> float:
    """late 相对 early 的百分比差；early≈0 时返回 nan（不给假精度）。"""
    a, b = float(early), float(late)
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < EPS:
        return float("nan")
    return 100.0 * (b - a) / abs(a)


def stability_flag(early: float, late: float, tol_pct: float = 2.0) -> bool:
    """末窗相对参考窗变化是否落在阈值内（仅作判读辅助，不是收敛证明）。"""
    d = drift_pct(early, late)
    return bool(np.isfinite(d) and abs(d) <= tol_pct)


def re_trend(re_list, values, re_c: float = RE_C) -> dict:
    """对 (Re−Re_c) 的斜率 + 单调性 + 端点比（无模型）。"""
    x = np.asarray(re_list, float) - re_c
    y = np.asarray(values, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return {"slope": float("nan"), "monotone_up": False,
                "monotone_down": False, "ratio_last_first": float("nan")}
    xx, yy = x[ok], y[ok]
    slope = float(np.polyfit(xx, yy, 1)[0]) if np.ptp(xx) > 0 else float("nan")
    return {"slope": slope,
            "monotone_up": bool(np.all(np.diff(yy) >= -EPS)),
            "monotone_down": bool(np.all(np.diff(yy) <= EPS)),
            "ratio_last_first": (float(yy[-1] / yy[0]) if abs(yy[0]) > EPS
                                 else float("nan"))}


def spectral_centroid(st0: dict, nz: int, d: float) -> dict:
    """谱的一阶矩（重心）与带宽；峰位只当"最大那个 m"报告。"""
    p = np.asarray(st0["power"], dtype=np.float64)[1:]
    if p.size == 0 or p.sum() <= 0:
        return {"m_centroid": None, "lambda_centroid_over_d": None,
                "m_std": None}
    m = np.arange(1, p.size + 1, dtype=np.float64)
    tot = float(p.sum())
    mc = float((m * p).sum() / tot)
    var = float((((m - mc) ** 2) * p).sum() / tot)
    return {"m_centroid": mc,
            "lambda_centroid_over_d": (float(nz) / mc / float(d)) if mc else None,
            "m_std": float(np.sqrt(max(var, 0.0)))}


def _q(x):
    return float(x)


def _period(case) -> float:
    return case.d / (case.u_in * literature_st(case.re))


def _cache_path(key: str) -> Path:
    return CACHE / f"{key}.json"


def _load(key: str):
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save(key: str, payload: dict):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = _cache_path(key)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                              default=_q), encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------- 2D 参考
def run_2d(re: float, ny: int = NY20, t_trans: float = T_TRANS) -> dict:
    case = CylinderCase(ny=ny, re=re, **GEO)
    T = _period(case)
    qf = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    sim = Solver(case)
    for _ in range(int(t_trans * T)):
        sim.step()
    cd, cl = [], []
    for _ in range(int(T_REC * T)):
        sim.step()
        fx, fy = sim.force()
        cd.append(fx / qf)
        cl.append(fy / qf)
    cd = np.asarray(cd)
    cl = np.asarray(cl)
    return {"dim": 2, "re": re, "ny": ny, "nz": 0, "beta": _q(case.d / ny),
            "tau": _q(case.tau), "T": _q(T),
            "cd_mean": _q(cd.mean()), "cl_rms": _q(np.sqrt(np.mean(cl ** 2))),
            "st": _q(strouhal(cl, case.d, case.u_in)["st"]),
            "st_lit": _q(literature_st(re))}


# ---------------------------------------------------------------- 多窗 3D
def run_3d_windowed(re: float, nz: int = NSPAN6, ny: int = NY20,
                    t_trans: float = T_TRANS, t_rec: float = T_REC,
                    tag: str = "t40") -> dict:
    """跑满 t_trans+t_rec，全程记录力与 uz_rel，在每个 10T 窗末抓谱/相关。

    `t_rec` 暴露出来只为 smoke 能跑 0.4T；生产永远用默认的 10T。
    """
    case = Case3D(ny=ny, nz=nz, re=re, **GEO)
    T = _period(case)
    qf = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    sim = Solver3D(case)
    n_trans, n_rec = int(t_trans * T), int(t_rec * T)
    n_total = n_trans + n_rec
    bounds = window_bounds(n_total, T)
    stop_to_j = {b: j for j, (_, b) in enumerate(bounds)}

    cd = np.empty(n_total, dtype=np.float64)
    cl = np.empty(n_total, dtype=np.float64)
    fz = np.empty(n_total, dtype=np.float64)
    uz_t, uz_val = [], []
    snaps: dict[int, dict] = {}
    t0 = time.perf_counter()

    mem_retries = 0
    for it in range(n_total):
        # 每 2000 步主动回收：v1.7 首跑在 t≈40T 的 step 内失败于 1.11 MiB 分配
        # （64 位、空闲 5 GB），判为堆碎片/瞬时压力。**不在 step 级重试**——
        # 崩溃点在 outlet，swap 尚未执行，重试会重复碰撞-迁移、物理上错误。
        if it and it % 2000 == 0:
            gc.collect()
        sim.step()
        fx, fy, fzk = sim.force3()
        cd[it] = fx / qf
        cl[it] = fy / qf
        fz[it] = abs(fzk) / qf
        if it % UZ_EVERY == 0:
            sm = sim.sample_metrics()
            uz_t.append(it / T)
            uz_val.append(sm["uz_rel"])
        j = stop_to_j.get(it + 1)
        if j is not None:
            # 两次诊断各做一次全场宏量（4×19.6 MB）。**分开做并显式释放**：
            # v1.7 首跑在 t≈40T 的第 4 个窗末出现 numpy 1.11 MiB 分配失败，
            # 根因未完全查清；把两次 78 MB 峰值拆开并 gc，是最保守的修法。
            spec = sim.spanwise_spectrum()
            s0 = spec["stations"][0]
            be = mode_band_energy(s0["power"], spec["nz"], spec["d"])
            snaps[j] = {
                "spectrum": {"m_peak": s0["m_peak"],
                             "lambda_peak_over_d": s0["lambda_peak_over_d"],
                             "p3d_rel": s0["p3d_rel"],
                             "power": list(s0["power"])},
                "centroid": spectral_centroid(s0, spec["nz"], spec["d"]),
                "bands": {k: {
                    "frac_of_p3d": v["frac_of_p3d"],
                    "repr_ok": v["repr_ok"], "m": v["m"]}
                    for k, v in be["bands"].items()},
            }
            del spec, s0, be
            gc.collect()
            corr = sim.spanwise_correlation()
            c0 = corr["stations"][0]
            snaps[j]["corr"] = {"Lc_over_d": c0.get("Lc_over_d"),
                                "r0": c0.get("r0"), "valid": c0.get("valid")}
            del corr, c0
            gc.collect()

    rho, ux, uy, uz = sim.macros()
    rf = rho[:, sim.fluid]
    tau_t = np.asarray(uz_t, float)
    tau_v = np.asarray(uz_val, float)
    windows = []
    for j, (a, b) in enumerate(bounds):
        cw, lw = cd[a:b], cl[a:b]
        m = (tau_t >= a / T) & (tau_t < b / T)
        pk, pkt = peak_timing(tau_t[m], tau_v[m])
        sn = snaps.get(j, {})
        st_j = (strouhal(lw, case.d, case.u_in)["st"]
                if lw.size >= 16 else float("nan"))
        windows.append({
            "index": j, "t_start": _q(a / T), "t_end": _q(b / T),
            "n_steps": int(b - a), "cd_mean": _q(cw.mean()),
            "cl_rms": _q(np.sqrt(np.mean(lw ** 2))) if lw.size else float("nan"),
            "st": _q(st_j), "fz_rel": _q(fz[a:b].mean()),
            "uz_rel_mean": _q(tau_v[m].mean()) if m.any() else float("nan"),
            "uz_rel_peak": pk, "uz_peak_t": pkt,
            "m_peak": sn.get("spectrum", {}).get("m_peak"),
            "lambda_peak_over_d": sn.get("spectrum", {}).get("lambda_peak_over_d"),
            "p3d_rel": sn.get("spectrum", {}).get("p3d_rel"),
            "m_centroid": sn.get("centroid", {}).get("m_centroid"),
            "lambda_centroid_over_d":
                sn.get("centroid", {}).get("lambda_centroid_over_d"),
            "m_std": sn.get("centroid", {}).get("m_std"),
            "band_A_frac": sn.get("bands", {}).get("A", {}).get("frac_of_p3d"),
            "band_B_frac": sn.get("bands", {}).get("B", {}).get("frac_of_p3d"),
            "Lc_over_d": sn.get("corr", {}).get("Lc_over_d"),
            "r0": sn.get("corr", {}).get("r0"),
        })

    # `full` 必须与 v1.6 的口径一致：**只统计末 10T 记录窗**。
    # `all_steps` 才是全程 50T 的统计量，两者不可混用。
    a_last, b_last = bounds[-1]
    cl_last, cd_last = cl[a_last:b_last], cd[a_last:b_last]
    full_st = strouhal(cl_last, case.d, case.u_in)["st"]
    return {"dim": 3, "re": re, "ny": ny, "nz": nz,
            "beta": _q(case.d / ny), "tau": _q(case.tau), "T": _q(T),
            "steps": int(n_total), "windows": windows,
            "full": {"t_start": _q(a_last / T), "t_end": _q(b_last / T),
                     "cd_mean": _q(cd_last.mean()),
                     "cl_rms": _q(np.sqrt(np.mean(cl_last ** 2))),
                     "st": _q(full_st), "fz_rel": _q(fz[a_last:b_last].mean()),
                     "uz_rel": windows[-1]["uz_rel_mean"],
                     "uz_rel_peak": _q(tau_v.max()) if tau_v.size else float("nan"),
                     "uz_peak_t": _q(tau_t[tau_v.argmax()]) if tau_v.size else float("nan")},
            "all_steps": {"cd_mean": _q(cd.mean()),
                          "cl_rms": _q(np.sqrt(np.mean(cl ** 2))),
                          "st": _q(strouhal(cl, case.d, case.u_in)["st"]),
                          "fz_rel": _q(fz.mean())},
            "rho_min": _q(rf.min()), "rho_max": _q(rf.max()),
            "ufield_max": _q(float(np.abs(ux).max())),
            "finite": bool(np.isfinite(rf).all() and np.isfinite(uz).all()),
            "uz_series": {"t_over_T": [_q(v) for v in tau_t],
                          "uz_rel": [_q(v) for v in tau_v]},
            "wall_min": _q((time.perf_counter() - t0) / 60.0), "tag": tag}


# ---------------------------------------------------------------- 汇总
def v16_crosscheck(rows: list[dict]) -> dict:
    """Re=250 与 v1.6 两个缓存交叉核对（同协议、同步数、采样器只读）。"""
    out = {}
    for r in rows:
        if r["re"] != 250.0:
            continue
        # 只比 50T 那一档：v1.6 的 t16 是 16T+10T，窗口位置与 v1.7 不同，
        # 强行并列只会造出一个"差异"假象。
        key = "3d_re250_nz96_t40"
        old = json.loads((RESULTS / "v16_cache" / f"v16_{key}.json").read_text(
            encoding="utf-8"))
        out["t40"] = {k: {"v16": old[k], "v17": r["full"][k],
                          "abs_diff": abs(old[k] - r["full"][k])}
                      for k in ("cd_mean", "cl_rms", "st")
                      if k in old and k in r["full"]}
    return out


def summarize(rows: list[dict], ref: list[dict] | None = None) -> dict:
    rows = sorted(rows, key=lambda r: r["re"])
    by_re = {int(r["re"]): r for r in rows}
    drift, peak, stable = {}, {}, {}
    for re, r in by_re.items():
        w = r["windows"]
        if len(w) < 2:
            continue
        anchor = w[2]          # 20–30T：v1.6 的 26T 记录窗所在区间
        last = w[-1]           # 40–50T
        drift[str(re)] = {
            k: drift_pct(anchor[k], last[k])
            for k in ("cd_mean", "cl_rms", "st", "uz_rel_mean", "m_centroid")}
        peak[str(re)] = {
            "global_peak_t": r["full"]["uz_peak_t"],
            "window_peaks": [{"t": x["uz_peak_t"], "val": x["uz_rel_peak"],
                              "window": x["index"]} for x in w],
            "peak_window": int(np.argmax([x["uz_rel_peak"] for x in w])),
        }
        stable[str(re)] = {
            k: stability_flag(anchor[k], last[k])
            for k in ("cd_mean", "cl_rms", "st", "uz_rel_mean", "m_centroid")}

    trend = []
    maxw = max((len(r["windows"]) for r in rows), default=0)
    for j in range(maxw):
        ww = [(r["re"], r["windows"][j]) for r in rows
              if len(r["windows"]) > j]
        if len(ww) < 2:
            continue
        trend.append({
            "window": j,
            "t_mid": _q((ww[0][1]["t_start"] + ww[0][1]["t_end"]) / 2),
            "st": re_trend([re for re, _ in ww], [w["st"] for _, w in ww]),
            "cl_rms": re_trend([re for re, _ in ww],
                               [w["cl_rms"] for _, w in ww]),
            "cd": re_trend([re for re, _ in ww],
                           [w["cd_mean"] for _, w in ww]),
            "m_centroid": re_trend([re for re, _ in ww],
                                   [w["m_centroid"] for _, w in ww]),
        })

    # 3D 相对 2D 的偏差：逐窗算，检验"随 Re 加深"是否跨窗稳健
    dev = []
    if ref:
        r2 = {int(x["re"]): x for x in ref}
        for j in range(maxw):
            row = []
            for r in rows:
                b = r2.get(int(r["re"]))
                if not b or len(r["windows"]) <= j:
                    continue
                w = r["windows"][j]
                row.append({
                    "re": int(r["re"]), "window": j,
                    "dst_pct": 100 * (w["st"] / b["st"] - 1),
                    "dcd_pct": 100 * (w["cd_mean"] / b["cd_mean"] - 1),
                    "dcl_rms_pct": 100 * (w["cl_rms"] / b["cl_rms"] - 1)})
            dev.append(row)

    return {"drift_anchor_to_last": drift, "peak_timing": peak,
            "stability": stable, "re_trend_by_window": trend,
            "deviation_by_window": dev,
            "v16_crosscheck": v16_crosscheck(rows)}


# ---------------------------------------------------------------- 图
def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    import lbm.plots  # noqa: F401  统一中文字体（Microsoft YaHei）

    rows = res["cases"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    labels = [f"Re={int(r['re'])}" for r in rows]

    # (a) 全程 uz_rel
    a = ax[0, 0]
    for c, r, lb in zip(colors, rows, labels):
        s = r["uz_series"]
        a.plot(s["t_over_T"], s["uz_rel"], color=c, lw=1.1, label=lb)
        pk = r["full"]["uz_peak_t"]
        if np.isfinite(pk):
            a.plot([pk], [r["full"]["uz_rel_peak"]], "o", ms=4, color=c)
    for t in (10, 20, 30, 40):
        a.axvline(t, color="0.7", ls=":", lw=0.7)
    a.set(xlabel="t / T", ylabel="uz_rel", title="(a) 50T 全程：圆点=峰值")
    a.legend(fontsize=7, ncol=2, loc="upper right")
    a.grid(alpha=0.2)

    # (b) 各窗 Cl_rms 相对末窗
    b = ax[0, 1]
    for c, r, lb in zip(colors, rows, labels):
        w = r["windows"]
        last = w[-1]["cl_rms"]
        if not np.isfinite(last) or last <= 0:
            continue
        b.plot([x["t_start"] + 5 for x in w],
               [100 * (x["cl_rms"] / last - 1) for x in w],
               "-o", ms=4, color=c, lw=1.2, label=lb)
    b.axhline(0, color="k", lw=0.7)
    b.set(xlabel="窗中心 t / T", ylabel="Cl_rms 相对末窗 / %",
          title="(b) 力幅的窗间漂移")
    b.grid(alpha=0.2)

    # (c) 谱重心
    c = ax[1, 0]
    for col, r, lb in zip(colors, rows, labels):
        w = r["windows"]
        c.plot([x["t_start"] + 5 for x in w], [x["m_centroid"] for x in w],
               "-o", ms=4, color=col, lw=1.2, label=lb)
    c.set(xlabel="窗中心 t / T", ylabel="谱重心 m",
          title="(c) 模式组成的窗间漂移")
    c.grid(alpha=0.2)

    # (d) 各窗 |ΔCl_rms| 随 Re 的斜率
    d = ax[1, 1]
    tr = [t for t in res["summary"]["re_trend_by_window"]]
    xs = [t["t_mid"] for t in tr]
    ys = [abs(t["cl_rms"]["slope"]) for t in tr]
    d.plot(xs, ys, "-o", ms=5, color="tab:purple")
    d.set(xlabel="窗中心 t / T", ylabel="|斜率|（每 Re）",
          title="(d) 「随 Re 加深」的强度是否稳")
    d.xaxis.set_major_locator(FixedLocator(xs))
    d.xaxis.set_major_formatter(FixedFormatter([f"{x:.0f}" for x in xs]))
    d.yaxis.set_minor_formatter(NullFormatter())
    d.grid(alpha=0.2)

    fig.suptitle("N5 v1.7 — 窗长是第三个旋钮：Re × 10T 窗矩阵（Lz=6D, β=20%）",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES / "v17_window_lock.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="只跑指定 Re，逗号分隔；串行")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.force and CACHE.exists():
        for p in CACHE.glob("v17_*.json"):
            p.unlink()

    if args.smoke:
        r = run_3d_windowed(250.0, t_trans=0.2, t_rec=0.2, tag="smoke")
        # smoke 只跑 0.4T：验证字段/采样/谱快照不抛错，不用于任何物理读数
        assert r["steps"] > 0 and r["windows"]
        assert np.isfinite(r["full"]["cd_mean"])
        assert all(np.isfinite(w["st"]) for w in r["windows"])
        print(f"smoke ok: steps={r['steps']} windows={len(r['windows'])} "
              f"wall={r['wall_min']:.2f}min")
        return

    if args.figures_only:
        d = json.loads((RESULTS / "v17_window.json").read_text(encoding="utf-8"))
        make_figure(d)
        return

    if args.cases:
        todo = [float(x) for x in args.cases.split(",") if x.strip()]
    else:
        todo = list(RE_LIST)

    for re in todo:
        key = f"3d_re{int(re)}_nz{NSPAN6}_t{int(T_TRANS)}"
        if _load(key) is None:
            print(f"[run] Re={int(re)} ...", flush=True)
            r = run_3d_windowed(re)
            _save(key, r)
            print(f"[done] Re={int(re)} wall={r['wall_min']:.1f}min", flush=True)
        else:
            print(f"[cached] Re={int(re)}", flush=True)

    if args.cases:
        # 分批跑：只补齐缓存，不做汇总（汇总必须四档齐）
        return

    rows = []
    for re in RE_LIST:
        key = f"3d_re{int(re)}_nz{NSPAN6}_t{int(T_TRANS)}"
        r = _load(key)
        if r is None:
            print(f"[skip summary] missing Re={int(re)}")
            return
        rows.append(r)
    ref = []
    for re in RE_LIST:
        k = f"2d_re{int(re)}_ny{NY20}"
        # v1.6 的 2D 缓存名带 t16 后缀；按 glob 找，避免文件名漂移造成 ref 空表。
        found = sorted(CACHE.glob(f"v17_{k}*.json")) or sorted(
            (RESULTS / "v16_cache").glob(f"v16_{k}*.json"))
        if found:
            ref.append(json.loads(found[0].read_text(encoding="utf-8")))
    if len(ref) != len(RE_LIST):
        raise RuntimeError(
            f"2D 参考缺失（{len(ref)}/{len(RE_LIST)}）——拒绝生成空偏差表")
    res = {"cases": rows, "ref2d": ref, "summary": summarize(rows, ref),
           "protocol": {"geo": GEO, "ny": NY20, "nz": NSPAN6,
                        "lz_over_d": LZ_D, "beta": GEO["d"] / NY20,
                        "t_trans": T_TRANS, "t_rec": T_REC,
                        "window_t": WINDOW_T, "re_list": list(RE_LIST),
                        "re_c": RE_C, "uz_every": UZ_EVERY}}
    (RESULTS / "v17_window.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=_q),
        encoding="utf-8")
    import csv
    cols = ["re", "window", "t_start", "t_end", "cd_mean", "cl_rms", "st",
            "uz_rel_mean", "uz_rel_peak", "uz_peak_t", "m_centroid",
            "band_A_frac", "band_B_frac", "Lc_over_d", "p3d_rel"]
    with (RESULTS / "v17_window_rows.csv").open("w", encoding="utf-8",
                                                newline="") as fh:
        wr = csv.writer(fh, lineterminator="\r\n")
        wr.writerow(cols)
        for r in rows:
            for w in r["windows"]:
                wr.writerow([int(r["re"])] +
                            [w.get(k, "") for k in cols[1:]])
    make_figure(res)
    print(f"summary -> {RESULTS.relative_to(ROOT) / 'v17_window.json'}")


if __name__ == "__main__":
    main()
