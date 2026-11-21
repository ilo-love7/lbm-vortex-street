"""13 v1.5 缺口审计：把 v1.3 的 −4.06% 一步步搬到 v1.4 的 +2.08%。

为什么做这一轮
--------------
v1.4 §14.5 留了最后一条没对上的账：β→0 线性外推的截距，v1.3 是 **−4.06%**
（D=20 / nx=560 / 6T / 五点），v1.4 是 **+2.08%**（D=16 / nx=320 / 16T /
三点），差 **6.1pp**；修正柱径偏置后仍差 5.1pp。v1.4 只猜"主因是域长与
瞬态"，没做单旋钮拆解 —— 而 E24 已经证明 15D→20D 只改 −0.1%，说明域长
**不是**主因。本脚本用**逐级替换**（每步只动一个旋钮）把 6.1pp 完整分配：

  A0  v1.3 原样        D=20, 下游 21.5D, 6T, 五点   ← 应复现 −4.06%
  A1  瞬态 6T→16T      （其余同 A0）
  A2  柱径 D=20→16     （其余同 A1，仍 21.5D）
  A3  点集 五点→三点   （A2 的子集，零成本）
  A4  域长 21.5D→15D   （= v1.4 生产几何，取 v14_blockage3d.json）

链式替换的性质：各步差值之和**恒等于**总缺口，不需要任何拟合，也不存在
"归因自由度"。这正是 v1.4 缺的那一步。

用法：
    python scripts/13_v15_gap_audit.py
    python scripts/13_v15_gap_audit.py --figures-only
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
RE = 100.0
T_REC = 10

# 五条链（每步只动一个旋钮）。ny 由 β 与 D 推出，保持点集相同。
# ``desc`` 进 JSON（中文可读），``desc_en`` 只用于图上标签——matplotlib 的
# 默认 DejaVu Sans 没有中文字形，中文标签会变成豆腐块（v1.2 起的老坑）。
STEPS = (
    {"tag": "A0_v13", "nx": 560, "cx_c": 120.0, "d": 20.0, "t_trans": 6,
     "betas": (0.20, 1 / 7, 0.10, 1 / 15, 0.05), "desc": "v1.3 原样",
     "desc_en": "v1.3 as-is"},
    {"tag": "A1_t16", "nx": 560, "cx_c": 120.0, "d": 20.0, "t_trans": 16,
     "betas": (0.20, 1 / 7, 0.10, 1 / 15, 0.05), "desc": "瞬态 6T→16T",
     "desc_en": "transient 6T->16T"},
    {"tag": "A2_d16", "nx": 424, "cx_c": 80.0, "d": 16.0, "t_trans": 16,
     "betas": (0.20, 1 / 7, 0.10, 1 / 15, 0.05), "desc": "柱径 20→16（仍 21.5D）",
     "desc_en": "cylinder D 20->16"},
    {"tag": "A3_3pt", "nx": 424, "cx_c": 80.0, "d": 16.0, "t_trans": 16,
     "betas": (0.20, 0.10, 0.05), "desc": "点集 五点→三点（复用 A2）",
     "desc_en": "points 5->3"},
    {"tag": "A4_15d", "nx": 320, "cx_c": 80.0, "d": 16.0, "t_trans": 16,
     "betas": (0.20, 0.10, 0.05), "desc": "域长 21.5D→15D（v1.4 生产）",
     "desc_en": "domain 21.5D->15D"},
)


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


def ny_for(beta: float, d: float) -> int:
    """由目标 β 反推域高：β 用 d/ny 定义（与 v1.3 §11.1 的表格一致）。"""
    return int(round(d / beta))


def run_2d(nx: int, cx_c: float, d: float, ny: int, t_trans: float) -> dict:
    geo = dict(nx=nx, cx_c=cx_c, d=d, u_in=0.1, sponge_width=32)
    case = CylinderCase(ny=ny, re=RE, **geo)
    T = case.d / (case.u_in * literature_st(RE))
    qf = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    sim = Solver(case)
    t0 = time.perf_counter()
    for _ in range(int(t_trans * T)):
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
    return {"nx": nx, "cx_c": cx_c, "d": d, "ny": ny, "t_trans": t_trans,
            "beta": _q(d / ny), "T": _q(T),
            "downstream_over_d": _q((nx - cx_c - case.radius) / d),
            "upstream_over_d": _q((cx_c - case.radius) / d),
            "st": _q(st["st"]), "st_lit": _q(literature_st(RE)),
            "cd_mean": _q(cd.mean()), "cl_rms": _q(np.sqrt(np.mean(cl ** 2))),
            "wake_cd_mean": _q(np.mean(list(wk.values()))),
            "wall_min": _q((time.perf_counter() - t0) / 60.0)}


def fit(rows: list[dict]) -> dict:
    b = np.array([r["beta"] for r in rows], float)
    st = np.array([r["st"] for r in rows], float)
    lit = float(rows[0]["st_lit"])
    lin = np.polyfit(b, st, 1)
    out = {"n": int(b.size), "beta": b.tolist(), "st": st.tolist(),
           "st_lit": lit,
           "linear": {"intercept": _q(lin[1]),
                      "rel_pct": _q(100 * (lin[1] / lit - 1)),
                      "slope": _q(lin[0]),
                      "r2": _q(1 - np.sum((np.polyval(lin, b) - st) ** 2)
                               / np.sum((st - st.mean()) ** 2))}}
    if b.size >= 4:
        quad = np.polyfit(b, st, 2)
        out["quadratic"] = {"intercept": _q(quad[2]),
                            "rel_pct": _q(100 * (quad[2] / lit - 1))}
        out["lin_minus_quad_pp"] = _q(100 * (lin[1] - quad[2]) / lit)
    return out


def make_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = res["steps"]
    labels = [s["tag"] for s in steps]
    vals = [s["fit"]["linear"]["rel_pct"] for s in steps]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))

    ax = axes[0]
    ax.plot(range(len(vals)), vals, "o-", color="tab:blue")
    for i, (lab, v) in enumerate(zip(labels, vals)):
        ax.annotate(f"{v:+.2f}%", (i, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([s["tag"].replace("_", "\n") for s in steps],
                       fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("beta->0 linear intercept vs St_lit [%]")
    ax.set_title("E25 one-knob-at-a-time chain (v1.3 -> v1.4)", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    deltas = [steps[i]["fit"]["linear"]["rel_pct"]
              - steps[i - 1]["fit"]["linear"]["rel_pct"]
              for i in range(1, len(steps))]
    names = [f"{steps[i]['tag']}\n{steps[i]['desc_en']}" for i in range(1,
                                                                     len(steps))]
    cols = ["tab:orange" if abs(d) > 1.0 else "tab:gray" for d in deltas]
    ax.barh(range(len(deltas)), deltas, color=cols, alpha=0.85)
    xmax = max(max(deltas), 0.0) or 1.0
    for i, d in enumerate(deltas):
        if abs(d) < 0.3 * xmax:          # 小步长：贴着 0 放到右侧，避免压轴
            ax.annotate(f"{d:+.2f}pp", (0.0, i), va="center", ha="left",
                        xytext=(4, 0), textcoords="offset points", fontsize=8)
        else:
            ax.annotate(f"{d:+.2f}pp", (d, i), va="center",
                        ha="left" if d >= 0 else "right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("change of intercept [pp]")
    ax.set_title("gap decomposition (sum = total gap)", fontsize=9)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "v15_gap_audit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    out_p = RESULTS / "v15_gap_audit.json"
    if args.figures_only:
        make_figure(json.loads(out_p.read_text(encoding="utf-8")))
        print("仅重画图完成")
        return

    steps_out = []
    for s in STEPS:
        if s["tag"] == "A4_15d":
            v14 = json.loads((RESULTS / "v14_blockage3d.json")
                             .read_text(encoding="utf-8"))
            rows = [r for r in v14["rows"]
                    if r["dim"] == 2 and r["nz"] == 0]
            rows = sorted(rows, key=lambda r: -r["beta"])
            steps_out.append({"tag": s["tag"], "desc": s["desc"],
                              "desc_en": s["desc_en"],
                              "rows": rows, "fit": fit(rows),
                              "source": "v14_blockage3d.json"})
            print(f"  [A4_15d] 复用 v1.4 三点："
                  f"{steps_out[-1]['fit']['linear']['rel_pct']:+.2f}%",
                  flush=True)
            continue
        rows = []
        for beta in s["betas"]:
            ny = ny_for(beta, s["d"])
            key = (f"v15_audit_{s['tag']}_nx{s['nx']}_ny{ny}"
                   f"_t{int(s['t_trans'])}")
            r = _load(key)
            if r is None:
                r = run_2d(s["nx"], s["cx_c"], s["d"], ny, s["t_trans"])
                _save(key, r)
            rows.append(r)
            print(f"  [{s['tag']}] beta={r['beta'] * 100:5.1f}% "
                  f"St={r['st']:.4f} Cd={r['cd_mean']:.3f} "
                  f"({r['wall_min']:.1f} min)", flush=True)
        steps_out.append({"tag": s["tag"], "desc": s["desc"],
                          "desc_en": s["desc_en"], "rows": rows,
                          "fit": fit(rows), "source": "run"})
        print(f"  [{s['tag']}] 截距 {steps_out[-1]['fit']['linear']['rel_pct']:+.2f}% "
              f"(R2={steps_out[-1]['fit']['linear']['r2']:.3f})", flush=True)

    deltas = []
    for i in range(1, len(steps_out)):
        deltas.append({"from": steps_out[i - 1]["tag"],
                       "to": steps_out[i]["tag"],
                       "desc": steps_out[i]["desc"],
                       "pp": _q(steps_out[i]["fit"]["linear"]["rel_pct"]
                                - steps_out[i - 1]["fit"]["linear"]["rel_pct"])})
    total = _q(steps_out[-1]["fit"]["linear"]["rel_pct"]
               - steps_out[0]["fit"]["linear"]["rel_pct"])

    # 三点链：全部步骤只用 {20,10,5}%，与 v1.4 的点集一致。v1.4 的 6.1pp 是
    # 用三点算的，所以"把缺口归因"必须用同一点集，否则点集变化会串到别的步里。
    sel3 = lambda rs: [r for r in rs if min(abs(r["beta"] - b)                # noqa: E731
                                           for b in (0.20, 0.10, 0.05)) < 1e-9]
    chain3 = []
    for s in steps_out:
        f = fit(sel3(s["rows"]))
        chain3.append({"tag": s["tag"], "desc": s["desc"], "fit": f})
    deltas3 = [{"from": chain3[i - 1]["tag"], "to": chain3[i]["tag"],
                "desc": chain3[i]["desc"],
                "pp": _q(chain3[i]["fit"]["linear"]["rel_pct"]
                         - chain3[i - 1]["fit"]["linear"]["rel_pct"])}
               for i in range(1, len(chain3))]
    total3 = _q(chain3[-1]["fit"]["linear"]["rel_pct"]
                - chain3[0]["fit"]["linear"]["rel_pct"])

    res = {"steps": steps_out, "deltas": deltas, "total_pp": total,
           "chain3": chain3, "deltas3": deltas3, "total3_pp": total3,
           "protocol": {"re": RE, "t_rec": T_REC,
                        "steps": [{k: v for k, v in s.items()
                                   if k != "betas"} for s in STEPS]}}
    out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_q),
                     encoding="utf-8")

    import pandas as pd
    flat = []
    for s in steps_out:
        for r in s["rows"]:
            d = dict(r)
            d["chain"] = s["tag"]
            flat.append(d)
    pd.DataFrame(flat).to_csv(RESULTS / "v15_gap_audit_rows.csv", index=False)
    make_figure(res)

    print("  缺口分解：")
    for d in deltas:
        print(f"    {d['from']} → {d['to']}: {d['pp']:+.2f}pp  ({d['desc']})")
    print(f"    合计 {total:+.2f}pp")
    print("  缺口分解（统一三点 {20,10,5}%，与 v1.4 同点集）：")
    for d in deltas3:
        print(f"    {d['from']} → {d['to']}: {d['pp']:+.2f}pp  ({d['desc']})")
    print(f"    合计 {total3:+.2f}pp")
    print("完成：results/v15_gap_audit.json + figures/v15_gap_audit.png")


if __name__ == "__main__":
    main()
