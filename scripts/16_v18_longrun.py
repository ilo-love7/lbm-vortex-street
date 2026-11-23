"""16 v1.8 把高 Re 瞬态走完：Re=350/400 各 100T，切 10 个 10T 窗。

v1.7 只到 50T，留下一个未决问题：高 Re 的 uz_rel 是"衰减得很慢"还是
"根本不衰减"？50T 末窗 Re=350/400 仍停在 0.081/0.086，Re=350 的峰值
甚至落在最后一个窗（48.8T），所以再补一个 50T 终点没有意义。

本脚本
------
1. 用与 v1.7 **完全相同**的几何、瞬态、采样协议，只把记录窗从 10T 延长
   到 60T（总 100T），仍切成 10T 窗。前 5 个窗与 v1.7 的 5 个窗在物理上
   是同一段模拟，必须逐位相同——这是本轮最硬的交叉核对。
2. 用"尾窗斜率 + 末窗/全局峰值比"两个独立读数判 regime，不靠单点阈值。
3. 比较 60–80T 与 80–100T 两个 20T 块的均值，判断后半程是否仍在漂移。

注意：只跑 Re=350/400 两档；4 档全跑需要额外 5 h，且 v1.7 已证明
Re=250/300 在 50T 内完成起振—衰减，不是本轮的瓶颈。
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

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = RESULTS / "v18_cache"

RE_LONG = (350.0, 400.0)
T_TRANS, T_REC = 40.0, 60.0
T_TOTAL = 100.0
WINDOW_T = 10.0
V17_SPLIT = 50.0
EPS = 1e-12


# ------------------------------------------------------------- 纯归约函数
def tail_stats(windows: list[dict], key: str, n_tail: int = 3) -> dict:
    """末 n_tail 个窗的斜率/首尾漂移/均值；点数不足返回 nan，不造假精度。"""
    tail = windows[-n_tail:] if n_tail > 0 else windows
    vals = np.array([float(w.get(key, np.nan)) for w in tail], dtype=float)
    ok = np.isfinite(vals)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "slope_per_window": float("nan"),
                "drift_pct": float("nan"), "mean": float("nan"),
                "first": float("nan"), "last": float("nan")}
    v = vals[ok]
    out = {"n": n, "first": float(v[0]), "last": float(v[-1]),
           "mean": float(v.mean()),
           "slope_per_window": float("nan"),
           "drift_pct": float("nan")}
    if n >= 2:
        x = np.flatnonzero(ok).astype(float)
        out["slope_per_window"] = float(np.polyfit(x, v, 1)[0])
        if abs(v[0]) > EPS:
            out["drift_pct"] = 100.0 * (v[-1] - v[0]) / abs(v[0])
    return out


def regime_label(row: dict, low_ratio: float = 0.6, high_ratio: float = 0.8,
                 flat_pct: float = 5.0, slope_eps: float = 1e-5) -> dict:
    """两套读数联合判"衰减 / 持续高位 / 仍在增长"；单看末窗不可判。"""
    w = row["windows"]
    tail = tail_stats(w, "uz_rel_mean", 3)
    peak = float(row["full"]["uz_rel_peak"])
    last = tail["last"]
    ratio = float("nan") if not np.isfinite(peak) or peak <= EPS else last / peak
    slope, pct = tail["slope_per_window"], tail["drift_pct"]
    label = "unresolved"
    if not (np.isfinite(ratio) and np.isfinite(slope) and np.isfinite(pct)):
        pass
    elif ratio <= low_ratio and abs(pct) <= flat_pct:
        label = "settled_low"
    elif ratio <= low_ratio and slope < -slope_eps:
        label = "settling"
    elif ratio >= high_ratio and abs(pct) <= flat_pct:
        label = "persistent_high"
    elif ratio >= high_ratio and slope > slope_eps:
        label = "still_growing"
    elif slope < -slope_eps:
        label = "decaying"
    elif slope > slope_eps:
        label = "growing"
    else:
        label = "flat_mid"
    return {"label": label, "last_over_peak": ratio,
            "tail_slope_per_window": slope, "tail_drift_pct": pct,
            "peak_t": float(row["full"]["uz_peak_t"]),
            "threshold_low_ratio": low_ratio,
            "threshold_high_ratio": high_ratio, "threshold_flat_pct": flat_pct}


def split_crosscheck(new: dict, old: dict, keys=("cd_mean", "cl_rms", "st",
                                                "uz_rel_peak", "m_centroid",
                                                "p3d_rel")) -> dict:
    """v1.8 前 5 窗 vs v1.7 5 窗。

    **前 4 窗（0–40T）必须逐位相同**：同协议、同种子、同边界。
    第 5 窗（40–50T）不能要求逐位相同：v1.7 把 1 步尾窗并进了它，
    v1.8 延长后尾窗移到末尾，因此 v1.7 的窗4 比 v1.8 多 1–2 步。
    这一窗只做数值容差核对，并在报告里如实标注。
    """
    n = min(len(new["windows"]), len(old["windows"]))
    out = {}
    for k in keys:
        diffs = []
        for j in range(n):
            a, b = new["windows"][j].get(k), old["windows"][j].get(k)
            if a is None or b is None:
                continue
            a, b = float(a), float(b)
            if np.isfinite(a) and np.isfinite(b):
                diffs.append(abs(a - b))
        out[k] = {"n_windows": len(diffs),
                  "max_abs_diff": max(diffs) if diffs else float("nan"),
                  "bitwise_identical": bool(diffs[:4] and max(diffs[:4]) == 0.0),
                  "first4_max_abs_diff": (max(diffs[:4]) if diffs[:4]
                                          else float("nan")),
                  "window4_abs_diff": (diffs[4] if len(diffs) > 4
                                       else float("nan"))}
    # 50T 记录窗：v1.7 的 full 对应 v1.8 的第 5 窗（索引 4），只做容差核对
    for k in ("cd_mean", "cl_rms", "st"):
        a, b = old["full"].get(k), new["windows"][4].get(k)
        a, b = float(a), float(b)
        out["full50_" + k] = {"v17": a, "v18_window4": b, "abs_diff": abs(a - b),
                              "rel_pct": (100 * abs(a - b) / abs(a)
                                          if abs(a) > EPS else float("nan"))}
    return out


def block_shift(row: dict, keys=("cd_mean", "cl_rms", "st", "uz_rel_mean",
                                 "m_centroid")) -> dict:
    """60–80T 与 80–100T 两个 20T 块的均值差；>2 个块才可判后半程漂移。"""
    w = row["windows"]
    if len(w) < 10:
        return {}
    a, b = w[6:8], w[8:10]
    out = {}
    for k in keys:
        va = np.array([float(x.get(k, np.nan)) for x in a])
        vb = np.array([float(x.get(k, np.nan)) for x in b])
        if not (np.isfinite(va).all() and np.isfinite(vb).all()):
            out[k] = float("nan")
        else:
            out[k] = v17.drift_pct(float(va.mean()), float(vb.mean()))
    return out


def deviation_by_window(rows: list[dict], ref2d: list[dict]) -> list[dict]:
    """逐窗 3D−2D 偏差；缺 2D 参考时返回空表（调用方必须检查）。"""
    refs = {int(r["re"]): r for r in ref2d}
    out = []
    for r in rows:
        b = refs.get(int(r["re"]))
        if not b:
            continue
        for w in r["windows"]:
            out.append({
                "re": int(r["re"]), "window": w["index"],
                "t_mid": 0.5 * (w["t_start"] + w["t_end"]),
                "dcd_pct": 100 * (w["cd_mean"] / b["cd_mean"] - 1),
                "dcl_rms_pct": 100 * (w["cl_rms"] / b["cl_rms"] - 1),
                "dst_pct": 100 * (w["st"] / b["st"] - 1)})
    return out


def make_summary(rows: list[dict], old_rows: list[dict],
                 ref2d: list[dict]) -> dict:
    old = {int(r["re"]): r for r in old_rows}
    tail, regime, cross, shift, peak = {}, {}, {}, {}, {}
    for r in rows:
        key = str(int(r["re"]))
        tail[key] = {k: tail_stats(r["windows"], k, 3)
                     for k in ("uz_rel_mean", "cd_mean", "cl_rms", "st",
                               "m_centroid")}
        regime[key] = regime_label(r)
        o = old.get(int(r["re"]))
        cross[key] = split_crosscheck(r, o) if o else {}
        shift[key] = block_shift(r)
        w = r["windows"]
        peak[key] = {
            "global_peak_t": r["full"]["uz_peak_t"],
            "global_peak_val": r["full"]["uz_rel_peak"],
            "after50_peak_t": max(w[5:], key=lambda x: x["uz_rel_peak"])["uz_peak_t"],
            "after50_peak_val": max(x["uz_rel_peak"] for x in w[5:]),
            "window_mean": [ {"t_mid": 0.5 * (x["t_start"] + x["t_end"]),
                              "uz_rel_mean": x["uz_rel_mean"],
                              "cl_rms": x["cl_rms"], "st": x["st"],
                              "m_centroid": x["m_centroid"]} for x in w]}
    return {"regime": regime, "tail_stats": tail, "crosscheck_v17": cross,
            "block_shift_60_80_vs_80_100": shift, "peak_timing": peak,
            "deviation_by_window": deviation_by_window(rows, ref2d)}


# ------------------------------------------------------------------ 运行
def run_case(re: float) -> dict:
    print(f"[run] Re={int(re)} total={T_TOTAL:.0f}T "
          f"({T_TRANS:.0f}T trans + {T_REC:.0f}T rec) ...", flush=True)
    r = v17.run_3d_windowed(re, t_trans=T_TRANS, t_rec=T_REC, tag="t100")
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"v18_3d_re{int(re)}_nz{v17.NSPAN6}_t100.json"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=v17._q),
                   encoding="utf-8")
    tmp.replace(p)
    print(f"[done] Re={int(re)} windows={len(r['windows'])} "
          f"wall={r['wall_min']:.1f}min", flush=True)
    return r


def _load_ref2d() -> list[dict]:
    refs = []
    for re in v17.RE_LIST:
        pat = f"2d_re{int(re)}_ny{v17.NY20}*.json"
        found = sorted(v17.CACHE.glob(pat)) or sorted(
            (RESULTS / "v16_cache").glob(f"v16_{pat}"))
        if not found:
            raise RuntimeError(f"2D 参考缺失：Re={int(re)}")
        refs.append(json.loads(found[0].read_text(encoding="utf-8")))
    return refs


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, S = res["cases"], res["summary"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.8))
    colors = {350: "tab:blue", 400: "tab:red"}
    for r in rows:
        re = int(r["re"])
        c = colors.get(re, "tab:gray")
        s = r["uz_series"]
        ax[0, 0].plot(s["t_over_T"], s["uz_rel"], color=c, lw=1.0,
                      label=f"Re={re}")
        ax[0, 0].plot([r["full"]["uz_peak_t"]], [r["full"]["uz_rel_peak"]],
                      "o", ms=5, color=c)
        w = r["windows"]
        mid = [0.5 * (x["t_start"] + x["t_end"]) for x in w]
        ax[0, 1].plot(mid, [x["uz_rel_mean"] for x in w], "-o", ms=4,
                      color=c, lw=1.3, label=f"Re={re}")
        ax[1, 0].plot(mid, [x["m_centroid"] for x in w], "-o", ms=4,
                      color=c, lw=1.3, label=f"Re={re}")
        dev = [d for d in S["deviation_by_window"] if d["re"] == re]
        ax[1, 1].plot([d["t_mid"] for d in dev],
                      [d["dcl_rms_pct"] for d in dev], "-o", ms=4,
                      color=c, lw=1.3, label=f"Re={re}")
    ax[0, 0].axvline(V17_SPLIT, color="0.45", ls="--", lw=1.0)
    ax[0, 0].text(V17_SPLIT + 1, 0.005, "v1.7 limit", fontsize=8)
    ax[0, 0].set(xlabel="t / T", ylabel="uz_rel",
                 title="(a) 100T series, circle = global peak")
    ax[0, 1].axvline(V17_SPLIT, color="0.45", ls="--", lw=1.0)
    ax[0, 1].set(xlabel="window center t / T", ylabel="uz_rel mean",
                 title="(b) 10T window means")
    ax[1, 0].axvline(V17_SPLIT, color="0.45", ls="--", lw=1.0)
    ax[1, 0].set(xlabel="window center t / T", ylabel="spectral centroid m",
                 title="(c) mode composition")
    ax[1, 1].axhline(0, color="k", lw=0.7)
    ax[1, 1].axvline(V17_SPLIT, color="0.45", ls="--", lw=1.0)
    ax[1, 1].set(xlabel="window center t / T", ylabel="Cl_rms 3D-2D / %",
                 title="(d) force-amplitude deviation")
    for a in ax.ravel():
        a.grid(alpha=0.2)
        a.legend(fontsize=8)
    fig.suptitle("N5 v1.8 - 100T long windows: Re=350/400 (Lz=6D, beta=20%)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES / "v18_longrun_100t.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(str(int(x)) for x in RE_LONG))
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        r = v17.run_3d_windowed(250.0, t_trans=0.2, t_rec=0.3, tag="v18_smoke")
        assert r["steps"] > 0 and r["windows"]
        assert np.isfinite(r["full"]["cd_mean"])
        x = tail_stats(r["windows"], "uz_rel_mean", 3)
        assert x["n"] >= 1 and np.isnan(x["slope_per_window"])
        print(f"smoke ok: steps={r['steps']} windows={len(r['windows'])} "
              f"wall={r['wall_min']:.2f}min")
        return

    if args.figures_only:
        d = json.loads((RESULTS / "v18_longrun.json").read_text(encoding="utf-8"))
        make_figure(d)
        return

    todo = [float(x) for x in args.cases.split(",") if x.strip()]
    for re in todo:
        key = f"v18_3d_re{int(re)}_nz{v17.NSPAN6}_t100.json"
        if (CACHE / key).exists():
            print(f"[cached] Re={int(re)}")
        else:
            run_case(re)

    rows, old_rows = [], []
    for re in RE_LONG:
        p = CACHE / f"v18_3d_re{int(re)}_nz{v17.NSPAN6}_t100.json"
        if not p.exists():
            print(f"[skip summary] missing {p.name}")
            return
        rows.append(json.loads(p.read_text(encoding="utf-8")))
        op = v17.CACHE / f"3d_re{int(re)}_nz{v17.NSPAN6}_t{int(v17.T_TRANS)}.json"
        old_rows.append(json.loads(op.read_text(encoding="utf-8")))

    # 硬核对：前 40T 是同一段模拟，Cd/Cl_rms/St 必须逐位相同；否则停。
    # 40–50T 那窗因 v1.7 的 1 步尾窗并入而差 1–2 步，只做容差核对（报告标注）。
    for r, o in zip(rows, old_rows):
        ck = split_crosscheck(r, o)
        for k in ("cd_mean", "cl_rms", "st"):
            if not ck[k]["bitwise_identical"]:
                raise RuntimeError(
                    f"Re={int(r['re'])} 前 40T 的 {k} 与 v1.7 不一致 "
                    f"(first4 diff={ck[k]['first4_max_abs_diff']})，停止汇总")
        for k in ("cd_mean", "cl_rms", "st"):
            rel = ck["full50_" + k]["rel_pct"]
            if not np.isfinite(rel) or rel > 1.0:
                raise RuntimeError(
                    f"Re={int(r['re'])} 40–50T 的 {k} 与 v1.7 差 {rel}% > 1%，停止汇总")

    ref2d = _load_ref2d()
    res = {"cases": rows, "ref2d": ref2d,
           "summary": make_summary(rows, old_rows, ref2d),
           "protocol": {"geo": v17.GEO, "ny": v17.NY20, "nz": v17.NSPAN6,
                        "lz_over_d": v17.LZ_D, "beta": v17.GEO["d"] / v17.NY20,
                        "t_trans": T_TRANS, "t_rec": T_REC, "t_total": T_TOTAL,
                        "window_t": WINDOW_T, "re_list": list(RE_LONG),
                        "re_c": v17.RE_C, "uz_every": v17.UZ_EVERY}}
    (RESULTS / "v18_longrun.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=v17._q),
        encoding="utf-8")
    cols = ["re", "window", "t_start", "t_end", "cd_mean", "cl_rms", "st",
            "uz_rel_mean", "uz_rel_peak", "uz_peak_t", "m_centroid",
            "band_A_frac", "band_B_frac", "p3d_rel"]
    with (RESULTS / "v18_longrun_rows.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\r\n")
        wr.writerow(cols)
        for r in rows:
            for w in r["windows"]:
                wr.writerow([int(r["re"])] + [w.get(k, "") for k in cols[1:]])
    make_figure(res)
    print(f"summary -> {(RESULTS / 'v18_longrun.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
