"""18 v1.9 强迫/边界消融：高 Re 的非定常究竟是"被强迫维持"还是"自持"。

v1.8 用 100T 证明 Re=400 的 uz_rel 峰值后移到 75.8T、末窗仍在峰值的
85.9%（persistent_high）。但那只回答了"没有衰减"，没回答"是谁在维持"：

- v1.3/v1.5 的 ``trip`` 是 **t=0 一次性初场扰动**（幅度 1e-3、种子固定 42），
  之后没有任何持续随机强迫；入口是定常 Zou/He，出口是海绵+对流。
- 因此两种解释都能写出同样的 uz_rel 曲线：
  (a) 初始扰动只是"点火"，之后是流场自持的三维非定常吸引子；
  (b) 非定常被某个边界/初始细节持续供给，换个初始 realization 就塌掉。

判据（必须同时报数，不能只报标签）
------------------------------------
1. **幅度消融**：trip=1e-4 / 1e-2 与基准 1e-3 的 70–100T 平均 uz_rel
   若在 ±25% 内，而峰值时刻明显移动 ⇒ 幅度只改"点火早晚"，不改饱和幅值。
2. ** realization 消融**：幅度相同、种子 43 的 late 幅值与基准接近而峰值时刻
   不同 ⇒ 动力学不依赖特定初场细节。
3. **零扰动对照**：trip=0 从完全 z 对称初场出发，应停留在二维（p3d_rel
   机器零）；证明没有"边界自发三维源"。
4. **壁面消融**：no_slip（滑移长度 ∞→0）若仍持续高位 ⇒ 自由滑移边界
   不是维持非定常的必要条件（但有效阻塞比改变，不与 2D 基准直接相减）。

所有对照都固定 Re=400、nx=320、ny=80、D=16、Lz=6D(nz=96)、40T 瞬态 +
60T 记录，切成 10 个 10T 窗。基准档与 v1.8 缓存逐窗硬核对（差 0）后才汇总。

用法：
    python scripts/18_v19_forcing.py --smoke
    python scripts/18_v19_forcing.py --cases base,trip_lo,trip_hi,seed43,noslip,notrip
    python scripts/18_v19_forcing.py                 # 读缓存汇总+画图
    python scripts/18_v19_forcing.py --figures-only
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "v17_window", ROOT / "scripts" / "15_v17_window.py")
v17 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v17)

_SPEC18 = importlib.util.spec_from_file_location(
    "v18_longrun", ROOT / "scripts" / "16_v18_longrun.py")
v18 = importlib.util.module_from_spec(_SPEC18)
_SPEC18.loader.exec_module(v18)

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = RESULTS / "v19_cache"

RE_BASE = 400.0
T_TRANS, T_REC = 40.0, 60.0
WINDOW_T = 10.0
N_LATE = 3                       # 70–100T 三个窗
AMP_LO, AMP_HI = 0.75, 1.25      # 饱和幅值"不敏感"带
PEAK_SHIFT_T = 4.0               # 峰值时刻"明显移动"阈值（10T 窗分辨率）
EPS = 1e-14

# 消融矩阵：key -> (trip, trip_seed, top_bottom, t_rec)
CASES: dict[str, dict] = {
    "base":    dict(trip=1e-3, trip_seed=42, top_bottom="free_slip", t_rec=60.0),
    "trip_lo": dict(trip=1e-4, trip_seed=42, top_bottom="free_slip", t_rec=60.0),
    "trip_hi": dict(trip=1e-2, trip_seed=42, top_bottom="free_slip", t_rec=60.0),
    "seed43":  dict(trip=1e-3, trip_seed=43, top_bottom="free_slip", t_rec=60.0),
    "noslip":  dict(trip=1e-3, trip_seed=42, top_bottom="no_slip",  t_rec=60.0),
    "notrip":  dict(trip=0.0,  trip_seed=42, top_bottom="free_slip", t_rec=10.0),
}


# ---------------------------------------------------------------- 纯函数
def cache_path(key: str, spec: dict) -> Path:
    total = int(T_TRANS + spec["t_rec"])
    return CACHE / f"v19_re{int(RE_BASE)}_{key}_t{total}.json"


def _f(v) -> float:
    """None/字符串/非数一律 nan；短窗 smoke 缺谱量时不能抛 float(None)。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def late_stats(row: dict, n: int = N_LATE, keys=("uz_rel_mean", "uz_rel_peak",
                                                "cd_mean", "cl_rms", "st",
                                                "m_centroid", "p3d_rel")) -> dict:
    """末 n 个窗的均值/斜率；窗口不足时给 nan，不造假精度。"""
    w = row.get("windows", [])[-n:] if n > 0 else []
    out: dict = {"n": len(w), "t_start": float("nan")}
    for k in keys:
        vals = np.array([_f(x.get(k)) for x in w], dtype=float)
        ok = np.isfinite(vals)
        out[k + "_mean"] = float(vals[ok].mean()) if ok.any() else float("nan")
        out[k + "_slope"] = (float(np.polyfit(np.flatnonzero(ok), vals[ok], 1)[0])
                             if ok.sum() >= 2 else float("nan"))
    if w:
        out["t_start"] = float(w[0]["t_start"])
    return out


def peak_info(row: dict) -> dict:
    f = row.get("full", {})
    return {"uz_rel_peak": float(f.get("uz_rel_peak", np.nan)),
            "peak_t": float(f.get("uz_peak_t", np.nan)),
            "regime": v18.regime_label(row)["label"]}


def growth_rate(row: dict, t_lo: float = 2.0, t_hi: float = 25.0) -> dict:
    """起振段 log(uz_rel) 的指数增长率（1/T）；只对正值点做最小二乘。

    trip 幅度差 100× 时，纯线性增长会被 Initial 值带偏；指数率是幅值无关的
    判据。零扰动档 uz_rel 在机器零附近，返回 nan 而不是负的"衰减率"。
    """
    s = row.get("uz_series", {})
    t = np.asarray(s.get("t_over_T", []), dtype=float)
    y = np.asarray(s.get("uz_rel", []), dtype=float)
    ok = (t >= t_lo) & (t <= t_hi) & np.isfinite(t) & (y > EPS)
    if ok.sum() < 8:
        return {"rate_per_T": float("nan"), "doubling_T": float("nan"),
                "n_points": int(ok.sum())}
    p = np.polyfit(t[ok], np.log(y[ok]), 1)
    rate = float(p[0])
    return {"rate_per_T": rate,
            "doubling_T": float(np.log(2.0) / rate) if rate > 0 else float("nan"),
            "n_points": int(ok.sum())}


def e_fold_delay(row: dict, amp_ratio: float) -> dict:
    """初场幅度差 amp_ratio 倍时，纯指数起振应带来的峰值时刻平移 Δt=ln(r)/g。

    这是"点火变晚"与"幅值被强迫决定"的分水岭：若观测到的峰值平移与
    ln(幅度比)/增长率同量级，就不能把晚期幅值偏低当成"强迫维持"。
    """
    g = growth_rate(row)["rate_per_T"]
    if not (np.isfinite(g) and g > 0 and amp_ratio > 0):
        return {"delay_t": float("nan"), "growth_per_T": g}
    return {"delay_t": float(np.log(amp_ratio) / g), "growth_per_T": float(g)}


def is_2d_control(row: dict, eps: float = 1e-8) -> dict:
    """trip=0 档必须真处于二维机器零：uz_rel 与 p3d_rel 双读数。"""
    la = late_stats(row, n=2, keys=("uz_rel_mean", "p3d_rel"))
    u, p = la["uz_rel_mean_mean"], la["p3d_rel_mean"]
    ok = bool(np.isfinite(u) and np.isfinite(p) and u < eps and p < eps)
    return {"is_2d": ok, "uz_rel_mean": u, "p3d_rel_mean": p,
            "threshold": eps}


# 只有这三类属于"初场 realization 消融"：**零扰动对照与壁面消融不参与幅值判据**。
# 把 trip=0 的机器零 uz 当成 1e-16 倍幅度，会造出假"amplitude_dependent"。
AMPLITUDE_KEYS = ("trip_lo", "trip_hi", "seed43")


def compare_to_base(rows: dict[str, dict], base_key: str = "base") -> dict:
    """各消融档相对基准的"幅值/峰值/力幅"差；缺档给 nan，不静默出空表。"""
    if base_key not in rows:
        raise KeyError(f"缺基准档 {base_key}，无法做消融对比")
    b_late = late_stats(rows[base_key])
    b_peak = peak_info(rows[base_key])
    out = {}
    for key, row in rows.items():
        if key == base_key:
            continue
        la, pk = late_stats(row), peak_info(row)
        d = {}
        for k in ("uz_rel_mean", "cd_mean", "cl_rms", "st"):
            a, b = la[k + "_mean"], b_late[k + "_mean"]
            d[k + "_ratio"] = (float(a / b) if np.isfinite(a) and np.isfinite(b)
                               and abs(b) > EPS else float("nan"))
        d["peak_shift_t"] = (float(pk["peak_t"] - b_peak["peak_t"])
                             if np.isfinite(pk["peak_t"]) and
                             np.isfinite(b_peak["peak_t"]) else float("nan"))
        d["regime"] = pk["regime"]
        d["late_window_start_t"] = la["t_start"]
        d["in_amplitude_verdict"] = key in AMPLITUDE_KEYS
        out[key] = d
    return out


def attribution_verdict(cmp: dict, ctrl: dict | None = None,
                        delays: dict | None = None) -> dict:
    """阈值判据，标签与数字同时返回；不把"未在窗内看到"说成"证明无源"。"""
    ratios = [v["uz_rel_mean_ratio"] for k, v in cmp.items()
              if k in AMPLITUDE_KEYS and np.isfinite(v["uz_rel_mean_ratio"])]
    shifts = [abs(v["peak_shift_t"]) for k, v in cmp.items()
              if k in AMPLITUDE_KEYS and np.isfinite(v["peak_shift_t"])]
    amp_insensitive = bool(ratios and all(AMP_LO <= r <= AMP_HI for r in ratios))
    phase_shifted = bool(shifts and max(shifts) >= PEAK_SHIFT_T)
    control_2d = bool(ctrl and ctrl.get("is_2d"))
    # 幅度比是否可由"指数起振延迟"解释（不放宽成任意慢＝任意幅度）
    delay_ok, delay_detail = False, {}
    if delays:
        for k, dd in delays.items():
            if k not in cmp or not np.isfinite(dd.get("delay_t", np.nan)):
                continue
            observed = cmp[k]["peak_shift_t"]
            pred = dd["delay_t"]
            rel = (abs(observed - pred) / abs(pred)
                   if abs(pred) > EPS else abs(observed))
            delay_detail[k] = {"predicted_t": float(pred),
                               "observed_t": float(observed),
                               "rel_err": float(rel)}
        delay_ok = bool(delay_detail and all(
            v["rel_err"] <= 0.6 or abs(v["observed_t"] - v["predicted_t"])
            <= WINDOW_T for v in delay_detail.values()))
    if amp_insensitive and phase_shifted and control_2d:
        label = "self_sustained_attractor"
    elif amp_insensitive and control_2d:
        label = "same_state_no_phase_resolution"
    elif delay_ok and control_2d:
        label = "delayed_ignition_not_forced"
    elif ratios and max(ratios) / min(ratios) >= 1.5:
        label = "amplitude_dependent"
    else:
        label = "unresolved"
    return {"label": label, "amp_insensitive": amp_insensitive,
            "phase_shifted": phase_shifted, "zero_trip_is_2d": control_2d,
            "e_fold_delay_consistent": delay_ok, "e_fold_detail": delay_detail,
            "amp_ratio_range": [min(ratios), max(ratios)] if ratios else None,
            "max_peak_shift_t": max(shifts) if shifts else None,
            "thresholds": {"amp_lo": AMP_LO, "amp_hi": AMP_HI,
                           "peak_shift_t": PEAK_SHIFT_T}}


def baseline_crosscheck(new: dict, old: dict, atol: float = 1e-12) -> dict:
    """v1.9 base 档与 v1.8 Re=400 100T 缓存逐窗核对（同协议必须差 0）。"""
    keys = ("cd_mean", "cl_rms", "st", "uz_rel_mean", "uz_rel_peak",
            "uz_peak_t", "m_centroid", "p3d_rel")
    out = {}
    for k in keys:
        nd = [float(w.get(k, np.nan)) for w in new["windows"]]
        od = [float(w.get(k, np.nan)) for w in old["windows"]]
        n = min(len(nd), len(od))
        diff = max((abs(nd[i] - od[i]) for i in range(n)
                    if np.isfinite(nd[i]) and np.isfinite(od[i])), default=np.nan)
        out[k] = {"n": n, "max_abs_diff": float(diff),
                  "identical": bool(np.isfinite(diff) and diff <= atol)}
    for k in ("cd_mean", "cl_rms", "st"):
        out["full_" + k] = {"v18": float(old["full"][k]),
                            "v19": float(new["full"][k]),
                            "abs_diff": abs(float(old["full"][k]) -
                                            float(new["full"][k]))}
    return out


# ---------------------------------------------------------------- 生产运行
def _save(payload: dict, path: Path) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                              default=v17._q), encoding="utf-8")
    tmp.replace(path)


def run_case(key: str) -> dict:
    spec = CASES[key]
    extra = {k: spec[k] for k in ("trip", "trip_seed", "top_bottom")}
    row = v17.run_3d_windowed(
        RE_BASE, nz=v17.NSPAN6, ny=v17.NY20, t_trans=T_TRANS,
        t_rec=spec["t_rec"], tag="v19_" + key, case_extra=extra)
    row["case"] = key
    row.update(extra)
    _save(row, cache_path(key, spec))
    print(f"[done] {key}: steps={row['steps']} windows={len(row['windows'])} "
          f"wall={row['wall_min']:.1f}min", flush=True)
    return row


def summarize(rows: dict[str, dict]) -> dict:
    cmp = compare_to_base(rows)
    ctrl = is_2d_control(rows["notrip"]) if "notrip" in rows else None
    delays = {}
    if "trip_lo" in rows:
        delays["trip_lo"] = e_fold_delay(rows["trip_lo"], 10.0)
    if "trip_hi" in rows:
        delays["trip_hi"] = e_fold_delay(rows["trip_hi"], 0.1)
    return {"cases": rows, "late": {k: late_stats(v) for k, v in rows.items()},
            "peak": {k: peak_info(v) for k, v in rows.items()},
            "growth": {k: growth_rate(v) for k, v in rows.items()},
            "compare": cmp, "zero_trip_control": ctrl, "e_fold_delays": delays,
            "verdict": attribution_verdict(cmp, ctrl, delays),
            "protocol": {"re": RE_BASE, "geo": v17.GEO, "ny": v17.NY20,
                         "nz": v17.NSPAN6, "lz_over_d": v17.LZ_D,
                         "t_trans": T_TRANS, "t_rec": T_REC,
                         "window_t": WINDOW_T, "n_late": N_LATE,
                         "cases": CASES}}


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lbm import plots  # noqa: F401  中文字体/样式副作用

    rows = res["cases"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    styles = {"base": ("k", "-"), "trip_lo": ("tab:blue", "--"),
              "trip_hi": ("tab:red", "-."), "seed43": ("tab:green", ":"),
              "noslip": ("tab:purple", "-"), "notrip": ("0.55", ":")}
    for key, r in rows.items():
        c, ls = styles.get(key, ("k", "-"))
        s = r.get("uz_series", {})
        ax[0, 0].plot(s.get("t_over_T", []), s.get("uz_rel", []),
                      color=c, ls=ls, lw=1.1, label=key)
        w = r["windows"]
        mid = [0.5 * (x["t_start"] + x["t_end"]) for x in w]
        ax[0, 1].plot(mid, [x["uz_rel_mean"] for x in w], marker="o", ms=3,
                      color=c, linestyle=ls, lw=1.1, label=key)
    ax[0, 0].set(xlabel="t / T", ylabel="uz_rel", title="(a) 100T series")
    ax[0, 1].set(xlabel="window center t / T", ylabel="uz_rel mean",
                 title="(b) 10T window means")
    keys = [k for k in CASES if k in res["late"]]
    late = [res["late"][k]["uz_rel_mean_mean"] for k in keys]
    peaks = [res["peak"][k]["peak_t"] for k in keys]
    ax[1, 0].bar(keys, late, color=["0.3" if k == "base" else "tab:blue"
                                     for k in keys])
    ax[1, 0].set(ylabel="late uz_rel mean (70-100T)",
                 title="(c) saturated amplitude")
    ax[1, 1].bar(keys, peaks, color=["0.3" if k == "base" else "tab:red"
                                      for k in keys])
    ax[1, 1].set(ylabel="global peak t / T",
                 title="(d) global peak (not robust across amplitudes)")
    for a in ax.ravel():
        a.grid(alpha=0.2)
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].legend(fontsize=8)
    fig.suptitle("N5 v1.9 - forcing/boundary ablation at Re=400 (Lz=6D)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES / "v19_forcing_ablation.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resummarize", action="store_true",
                    help="只从缓存重算判据/图，不重跑 CFD")
    args = ap.parse_args()

    if args.smoke:
        for key in ("base", "notrip"):
            spec = CASES[key]
            r = v17.run_3d_windowed(
                RE_BASE, nz=v17.NSPAN6, ny=v17.NY20, t_trans=0.2, t_rec=0.3,
                tag="v19_smoke",
                case_extra={k: spec[k] for k in ("trip", "trip_seed",
                                                 "top_bottom")})
            assert r["steps"] > 0 and r["windows"]
            assert np.isfinite(r["full"]["cd_mean"])
            assert late_stats(r)["n"] >= 1
        print("smoke ok")
        return

    if args.figures_only:
        make_figure(json.loads((RESULTS / "v19_forcing.json").read_text(
            encoding="utf-8")))
        return

    todo = ([] if args.resummarize else
            [x.strip() for x in args.cases.split(",") if x.strip()])
    for key in todo:
        if key not in CASES:
            raise ValueError(f"未知消融档 {key}")
        if cache_path(key, CASES[key]).exists():
            print(f"[cached] {key}", flush=True)
        else:
            run_case(key)

    rows = {}
    for key in CASES:
        p = cache_path(key, CASES[key])
        if not p.exists():
            # 基准档缺失是硬错误；消融档可按算力分轮补跑，不许静默出空表。
            if key == "base":
                raise FileNotFoundError(f"缺基准档 {p.name}，先跑 base")
            print(f"[skip summary] missing {p.name}")
            continue
        rows[key] = json.loads(p.read_text(encoding="utf-8"))

    # 基准硬核对：v1.9 只改了"可选 trip_seed"，默认值必须逐位复现 v1.8。
    old = json.loads((RESULTS / "v18_cache" /
                      f"v18_3d_re{int(RE_BASE)}_nz{v17.NSPAN6}_t100.json"
                      ).read_text(encoding="utf-8"))
    ck = baseline_crosscheck(rows["base"], old)
    for k, v in ck.items():
        ok = v.get("identical", abs(v.get("abs_diff", 0.0)) <= 1e-12)
        if not ok:
            raise RuntimeError(f"base 档与 v1.8 不一致: {k} -> {v}")

    res = summarize(rows)
    res["baseline_crosscheck"] = ck
    (RESULTS / "v19_forcing.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=v17._q),
        encoding="utf-8")
    cols = ["case", "trip", "trip_seed", "top_bottom", "window", "t_start",
            "t_end", "cd_mean", "cl_rms", "st", "uz_rel_mean", "uz_rel_peak",
            "uz_peak_t", "m_centroid", "p3d_rel"]
    with (RESULTS / "v19_forcing_rows.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\r\n")
        wr.writerow(cols)
        for key, r in rows.items():
            for w in r["windows"]:
                wr.writerow([key, r["trip"], r["trip_seed"], r["top_bottom"]]
                            + [w.get(k, "") for k in cols[4:]])
    make_figure(res)
    print("verdict ->", res["verdict"]["label"])


if __name__ == "__main__":
    main()
