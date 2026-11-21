"""12 v1.5 域长口径：E24 20D 下游 β 系列与截距对账。

为什么做这一轮
--------------
v1.4 §14.5 留了一条没有收口的账：把柱径偏置修正后，本轮 β→0 截距为
**+1.05%**，而 v1.3 报的是 **−4.06%**，仍差 **5.1pp**。候选来源有三个
（域长、瞬态长度、柱径），v1.5 已用 E23 单独处理维度，这里处理**域长**：

- v1.4 生产域：`nx=320, cx_c=80, D=16` ⇒ 下游 15D；
- 本轮换 `nx=400` ⇒ 下游 20D，其余（D=16、U=0.1、β 点集、16T/10T、
  海绵 32）**逐项不变**，唯一变量是域长；
- 同时补 β=14.3%/6.7% 两点，凑成 v1.3 的**同一点集**（5 点），
  用来回答另一个悬置问题：线性与二次外推差 6.4pp，是点集造成还是
  曲线本身凸？

读数：β→0 线性/二次截距、两点集之差、以及与 v1.3/v1.4 的对账。

用法：
    python scripts/12_v15_domain20d.py
    python scripts/12_v15_domain20d.py --figures-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbm import CylinderCase, Solver, literature_st, strouhal  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = RESULTS / "v15_cache"

# 与 v1.4 生产跑同 D/U/海绵/瞬态，只把下游从 15D 拉到 20D
GEO20 = dict(nx=400, d=16.0, cx_c=80.0, u_in=0.1, sponge_width=32)
GEO15 = dict(nx=320, d=16.0, cx_c=80.0, u_in=0.1, sponge_width=32)
NY3 = (80, 160, 320)                 # β = 20 / 10 / 5 %（v1.4 的三点）
NY5 = (80, 112, 160, 240, 320)       # + 14.3 / 6.7 %（v1.3 的五点）
T_TRANS, T_REC = 16, 10
RE = 100.0


def _q(x):
    return float(x)


def _load(key: str):
    p = CACHE / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save(key: str, payload: dict):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{key}.json"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                              default=_q), encoding="utf-8")
    tmp.replace(p)


def run_2d(nx: int, ny: int, re: float = RE) -> dict:
    geo = GEO20 if nx == 400 else GEO15
    case = CylinderCase(ny=ny, re=re, **geo)
    T = case.d / (case.u_in * literature_st(re))
    qf = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    sim = Solver(case)
    t0 = time.perf_counter()
    for _ in range(int(T_TRANS * T)):
        sim.step()
    cd, cl = [], []
    wake = {str(s): [] for s, _ in sim._wake_stations}
    for _ in range(int(T_REC * T)):
        sim.step()
        fx, fy = sim.force()
        cd.append(fx / qf)
        cl.append(fy / qf)
        for s, w in sim.wake_forces():
            wake[str(s)].append(w["force"] / qf)
    cd = np.asarray(cd)
    cl = np.asarray(cl)
    st = strouhal(cl, case.d, case.u_in)
    wk = {k: _q(np.mean(v)) for k, v in wake.items()}
    return {"dim": 2, "re": re, "nx": nx, "ny": ny, "beta": _q(case.d / ny),
            "downstream_over_d": _q((nx - case.cx_c - case.radius) / case.d),
            "tau": _q(case.tau), "st": _q(st["st"]),
            "st_lit": _q(literature_st(re)),
            "cd_mean": _q(cd.mean()), "cl_rms": _q(np.sqrt(np.mean(cl ** 2))),
            "wake_cd_mean": _q(np.mean(list(wk.values()))),
            "wall_min": _q((time.perf_counter() - t0) / 60.0)}


def fit(rows: list[dict]) -> dict:
    """与 v1.3/v1.4 同口径：β→0 的线性与二次截距（二次仅 5 点可用）。"""
    b = np.array([r["beta"] for r in rows], float)
    st = np.array([r["st"] for r in rows], float)
    lit = float(rows[0]["st_lit"])
    lin = np.polyfit(b, st, 1)
    out = {"n": int(b.size), "beta": b.tolist(), "st": st.tolist(),
           "st_lit": lit,
           "linear": {"intercept": _q(lin[1]),
                      "rel_pct": _q(100 * (lin[1] / lit - 1)),
                      "r2": _q(1 - np.sum((np.polyval(lin, b) - st) ** 2)
                               / np.sum((st - st.mean()) ** 2))}}
    if b.size >= 4:
        quad = np.polyfit(b, st, 2)
        out["quadratic"] = {"intercept": _q(quad[2]),
                            "rel_pct": _q(100 * (quad[2] / lit - 1))}
        out["lin_minus_quad_pp"] = _q(100 * (lin[1] - quad[2]) / lit)
        out["resid_quad"] = _q(float(np.max(np.abs(np.polyval(quad, b) - st))))
    return out


def load_v14_15d() -> dict:
    """v1.4 的 15D 三点（直接读生产 JSON，不重跑）。"""
    p = RESULTS / "v14_blockage3d.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = [r for r in d["rows"] if r["dim"] == 2 and r["nz"] == 0]
    rows = sorted(rows, key=lambda r: -r["beta"])
    return fit(rows)


def make_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    ax = axes[0]
    for tag, mk in (("20d5", "o-"), ("20d3", "s--")):
        f = res[tag]
        bb = np.array(f["beta"])
        ax.plot(bb * 100, f["st"], mk, ms=4,
                label=f"20D {f['n']}-point St")
        xs = np.linspace(0, max(bb) * 1.05, 60)
        ax.plot(xs * 100, np.polyval(np.polyfit(bb, f["st"], 1), xs), ":",
                lw=1)
    ax.axhline(res["20d3"]["st_lit"], color="k", ls=":", lw=1,
               label="St_lit")
    ax.set_xlabel("blockage beta [%]"); ax.set_ylabel("St")
    ax.set_title("E24 St(beta) at 20D downstream", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 截距对账：v1.3 / v1.4-15D / 本轮-20D
    ax = axes[1]
    labels, vals, cols = [], [], []
    for lab, v, c in (("v1.3 D=20\n22.5D,6T", -4.06, "tab:gray"),
                      ("v1.4 D=16\n15D,16T", res["v14_15d"].get(
                          "linear", {}).get("rel_pct", np.nan), "tab:blue"),
                      ("v1.4 20D 3pt", res["20d3"]["linear"]["rel_pct"],
                       "tab:orange"),
                      ("v1.4 20D 5pt", res["20d5"]["linear"]["rel_pct"],
                       "tab:green")):
        labels.append(lab); vals.append(v); cols.append(c)
    ax.bar(range(len(vals)), vals, color=cols, alpha=0.8)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.2f}%", (i, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("beta->0 intercept vs St_lit [%]")
    ax.set_title("intercept reconciliation", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # 两个点集的外推口径分歧
    ax = axes[2]
    for tag, mk in (("20d3", "s--"), ("20d5", "o-")):
        f = res[tag]
        ax.plot(f["beta"], f["st"], mk, ms=4, label=f"20D {f['n']}-pt")
    for tag, ls in (("20d3", ":"), ("20d5", "-.")):
        f = res[tag]
        b = np.array(f["beta"]); s = np.array(f["st"])
        xs = np.linspace(0, max(b) * 1.05, 60)
        ax.plot(xs, np.polyval(np.polyfit(b, s, 1), xs), ls, lw=1,
                label=f"{tag} linear")
        if "quadratic" in f:
            ax.plot(xs, np.polyval(np.polyfit(b, s, 2), xs), ls, lw=1,
                    alpha=0.6, label=f"{tag} quad")
    ax.axhline(res["20d3"]["st_lit"], color="k", ls=":", lw=1)
    ax.set_xlabel("beta"); ax.set_ylabel("St")
    ax.set_title("point-set sensitivity of extrapolation", fontsize=9)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "v15_domain20d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_p = RESULTS / "v15_domain20d.json"

    if args.figures_only:
        res = json.loads(out_p.read_text(encoding="utf-8"))
        make_figure(res)
        print("仅重画图完成")
        return
    if args.force:
        for p in CACHE.glob("v15_2d_nx400_*.json"):
            p.unlink()

    rows = {}
    for ny in NY5:
        key = f"v15_2d_nx400_ny{ny}_t16"
        r = _load(key)
        if r is None:
            r = run_2d(400, ny)
            _save(key, r)
        rows[ny] = r
        print(f"  20D beta={r['beta'] * 100:5.1f}% St={r['st']:.4f} "
              f"Cd={r['cd_mean']:.3f} Cl_rms={r['cl_rms']:.4f} "
              f"({r['wall_min']:.1f} min)", flush=True)

    r3 = [rows[ny] for ny in NY3]
    r5 = [rows[ny] for ny in NY5]
    res = {"rows": list(rows.values()),
           "20d3": fit(r3), "20d5": fit(r5),
           "v14_15d": load_v14_15d(),
           "protocol": {"geo20": GEO20, "geo15": GEO15, "ny3": list(NY3),
                        "ny5": list(NY5), "t_trans": T_TRANS,
                        "t_rec": T_REC, "re": RE}}
    out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_q),
                     encoding="utf-8")

    import pandas as pd
    pd.DataFrame(res["rows"]).to_csv(RESULTS / "v15_domain20d_rows.csv",
                                     index=False)
    make_figure(res)

    print(f"  20D 3pt: lin {res['20d3']['linear']['rel_pct']:+.2f}% "
          f"(R2={res['20d3']['linear']['r2']:.3f})")
    print(f"  20D 5pt: lin {res['20d5']['linear']['rel_pct']:+.2f}% "
          f"quad {res['20d5']['quadratic']['rel_pct']:+.2f}% "
          f"(Δ{res['20d5']['lin_minus_quad_pp']:.2f}pp)")
    v14 = res["v14_15d"].get("linear", {}).get("rel_pct")
    if v14 is not None:
        print(f"  15D(3pt) {v14:+.2f}% → 20D(3pt) "
              f"{res['20d3']['linear']['rel_pct']:+.2f}%  "
              f"Δ={res['20d3']['linear']['rel_pct'] - v14:+.2f}pp")
    print("完成：results/v15_domain20d.json + figures/v15_domain20d.png")


if __name__ == "__main__":
    main()
