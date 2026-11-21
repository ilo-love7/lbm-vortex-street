"""08 v1.4 三维校验：E19 内核三重解析验证 / E20 3D-2D 阻比对照 / E21 3D β→0 外推。

为什么做这一轮
--------------
v1.3 报告 §12 第 1 条把「3D 校验」列为**最高优先未做项**，并明确
「优先级高于继续加大二维计算域」。v1.3 的 E13 给出 β→0 外推
线性 −4.06% / 二次 +2.35%（两者差 6.4pp），Re=250 残余 +4.71% 仍随 Re
增长；这些残余是「2D 约束误差」还是「拟合口径噪声」，只能由三维裁决。

E19 内核验证（三个解析判据 + 一个守恒判据）
  a) 平面 Poiseuille：壁面半程反弹 + Guo 体积力，``u_max=gH²/(8ν)``，
     ny=9/17/33/65 收敛阶；
  b) 剪切波反测黏度：``A(t)=A0 e^{−νk_l²t}``，k_l=2π/n（**格子波数**），
     4 个 τ × 2 个分辨率；
  c) 3D Taylor-Green：``d lnE/dt|₀ = −6νk_l²``（|k|²=3 个方向），
     自适应拟合窗口；
  d) 均匀流不变性：周期盒与自由滑移盒都应逐位保持到机器精度
     （这一条抓出了首版两个真 bug：prange 下共享 feq 缓冲区的数据竞争，
     以及把三维"镜面"误写成"对向"导致壁面吃掉 92% 切向动量）。

E20 3D-2D 阻比对（同 D=16、同 U、同 Re、同海绵）
  3D：z 周期、y 自由滑移、Bouzidi 圆柱；2D 用仓库既有已验证求解器。
  同时做展向分辨率 nz=8/16/24 收敛性。

E21 β→0 外推裁决
  3D 补 β=5%，用与 2D 完全相同的三点做线性/二次外推，比较截距与残差。

输出：results/v14_validation.json / v14_blockage3d.json
      + figures/v14_*.png + results/v14_case_rows.csv

用法：
    python scripts/08_3d_validation.py            # 增量（命中缓存即跳过）
    python scripts/08_3d_validation.py --force    # 清缓存全量重算
    python scripts/08_3d_validation.py --figures-only
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

from lbm import CylinderCase, Solver, literature_st, strouhal, time_stats  # noqa: E402
from lbm.solver3d import (Case3D, Solver3D, channel_poiseuille,  # noqa: E402
                          shear_wave_decay, taylor_green)

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = RESULTS / "v14_cache"

# ---- 统一协议（3D 与 2D 共用，唯一变量是维度与阻塞比）----
GEO = dict(nx=320, d=16.0, cx_c=80.0, u_in=0.1, re=100.0, sponge_width=32)
NZ = 16
BLK_NY = (80, 160, 320)          # β = 20% / 10% / 5%
# **瞬态必须 16T（v1.4 实测修正）**：6T→12T→16T 在 2D ny=160（β=10%）上给
# St 0.17950→0.18107→0.18109、Cd 1.4789→1.5335→1.5348、Cl_rms
# 0.1966→0.2535→0.2550；16/20/24T 完全持平（St 0.18109、Cd 1.5347–1.5350、
# wake 1.5414）。即 6T 低估 **Cl_rms 23%、Cd 3.6%、St 0.9%**，且这个偏差
# 与 3D 无关（2D 同样偏）。6T 下 3D 与 2D 的 β=10% 差 2.0%，看起来像"维度
# 效应"，实际是两者都没起振完 —— 换 16T 后差 0.01%。故主协议取 16T。
T_TRANS, T_REC = 16, 10          # 以文献 St 定义的周期数
T_TRANS_SPAN = 6                 # 展向扫描沿用 6T（见 make_figures 注：逐位相同）
WAKE_EVERY = 20                  # 尾流/展向诊断抽样间隔（全场宏量的成本约束）
_TTAG = f"t{int(T_TRANS)}"       # 缓存 key 带瞬态长度，6T 旧缓存保留作对照


def _q(x):
    return float(x)


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


# ---------------------------------------------------------------- E19
def e19_poiseuille() -> list[dict]:
    out = []
    for ny in (9, 17, 33, 65):
        # steps=None ⇒ 12·t_diff = 12H²/(π²ν)，保证残余瞬变 ≪ 离散误差
        # （固定 40000 步在 H=63 只有 6.6 t_diff，不够）
        r = channel_poiseuille(ny=ny, nz=3, nx=3, tau=0.7, g=1e-6,
                               steps=None)
        out.append({"ny": ny, "H": ny - 2, "steps": r["steps"],
                    "t_diff": _q(r["t_diff"]), "drift": _q(r["drift"]),
                    "u_max_num": _q(r["u_max_num"]),
                    "u_max_exact": _q(r["u_max_exact"]),
                    "ratio": _q(r["u_max_num"] / r["u_max_exact"]),
                    "l2_rel": _q(r["l2_rel"]), "nu": _q(r["nu"])})
    # 收敛阶：相邻两档 L2 误差比
    for i in range(1, len(out)):
        out[i]["order"] = _q(np.log(out[i - 1]["l2_rel"] / out[i]["l2_rel"])
                             / np.log((out[i]["H"]) / out[i - 1]["H"]))
    return out


def e19_shear() -> list[dict]:
    out = []
    for n in (16, 24):
        for tau in (0.55, 0.60, 0.80, 1.20):
            # steps=4000 + 相对幅值拟合窗口（旧 1200 步/绝对阈值在 τ=1.2 失真）
            r = shear_wave_decay(n=n, nz=3, tau=tau, amp=1e-3, steps=4000)
            out.append({"n": n, "tau": tau, "k_lattice": _q(r["k"]),
                        "nu_input": _q(r["nu_input"]),
                        "nu_fit": _q(r["nu_fit"]),
                        "n_fit": r["n_fit"],
                        "rel_err_pct": _q(r["rel_err"] * 100)})
    return out


def e19_tg() -> list[dict]:
    out = []
    for n in (32, 48):
        for tau in (0.55, 0.80):
            r = taylor_green(n=n, tau=tau, amp=1e-3, steps=800, sample=10)
            out.append({"n": n, "tau": tau, "k_lattice": _q(r["k_lattice"]),
                        "slope": _q(r["slope0"]),
                        "slope_exact": _q(r["slope_exact"]),
                        "rel_err_pct": _q(r["rel_err"] * 100),
                        "div_rel": _q(r["div_max"]),
                        "fit_lo": _q(r["fit_window"][0]),
                        "fit_hi": _q(r["fit_window"][1])})
    return out


def e19_uniform() -> list[dict]:
    out = []
    for bc in ("periodic", "free_slip"):
        case = Case3D(nx=16, ny=16, nz=8, d=0.0, u_in=0.1, re=1.0,
                      top_bottom=bc, inlet="periodic", outlet="periodic",
                      sponge_width=0, sponge_sigma=0.0, trip=0.0,
                      tau_direct=0.6)
        sim = Solver3D(case)
        from lbm.lattice3d import equilibrium3d
        rho = np.ones((8, 16, 16))
        sim.f = equilibrium3d(rho, np.full_like(rho, 0.1),
                              np.zeros_like(rho), np.zeros_like(rho))
        for _ in range(400):
            sim.step()
        r, ux, uy, uz = sim.macros()
        out.append({"bc": bc,
                    "ux_dev": _q(np.max(np.abs(ux - 0.1))),
                    "uy_max": _q(np.max(np.abs(uy))),
                    "rho_dev": _q(np.max(np.abs(r - 1.0)))})
    return out


# ---------------------------------------------------------------- 2D 参考
def run_2d(ny: int, t_trans: float | None = None) -> dict:
    tt = T_TRANS if t_trans is None else t_trans
    case = CylinderCase(ny=ny, **GEO)
    T = case.d / (case.u_in * literature_st(case.re))
    qf = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    sim = Solver(case)
    t0 = time.perf_counter()
    for _ in range(int(tt * T)):
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
    return {"dim": 2, "ny": ny, "nz": 0, "beta": _q(case.d / ny),
            "tau": _q(case.tau), "T": _q(T),
            "cd_mean": _q(cd.mean()), "cl_rms": _q(np.sqrt(np.mean(cl ** 2))),
            "st": _q(st["st"]), "st_lit": _q(literature_st(case.re)),
            "wake_cd": wk, "wake_cd_mean": _q(np.mean(list(wk.values()))),
            "uz_rel": 0.0, "fz": 0.0,
            "wall_min": _q((time.perf_counter() - t0) / 60.0)}


# ---------------------------------------------------------------- 3D
def run_3d(ny: int, nz: int = NZ, t_trans: float | None = None) -> dict:
    tt = T_TRANS if t_trans is None else t_trans
    case = Case3D(ny=ny, nz=nz, **GEO)
    T = case.d / (case.u_in * literature_st(case.re))
    qf = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    sim = Solver3D(case)
    t0 = time.perf_counter()
    for _ in range(int(tt * T)):
        sim.step()
    cd, cl, uzr = [], [], []
    wake = {str(s): [] for s, _ in sim._wake_stations}
    for i in range(int(T_REC * T)):
        sim.step()
        fx, fy = sim.force()
        cd.append(fx / qf)
        cl.append(fy / qf)
        if i % WAKE_EVERY == 0:
            sm = sim.sample_metrics()
            for k, v in sm["wake"].items():
                wake[k].append(v["force"] / qf)
            uzr.append(sm["uz_rel"])
    cd = np.asarray(cd)
    cl = np.asarray(cl)
    st = strouhal(cl, case.d, case.u_in)
    wk = {k: _q(np.mean(v)) for k, v in wake.items()}
    _, _, fz = sim.force3()
    return {"dim": 3, "ny": ny, "nz": nz, "beta": _q(case.d / ny),
            "tau": _q(case.tau), "T": _q(T),
            "cd_mean": _q(cd.mean()), "cl_rms": _q(np.sqrt(np.mean(cl ** 2))),
            "st": _q(st["st"]), "st_lit": _q(literature_st(case.re)),
            "wake_cd": wk, "wake_cd_mean": _q(np.mean(list(wk.values()))),
            "uz_rel": _q(np.mean(uzr)) if uzr else 0.0,
            "fz": _q(abs(fz)),
            "wall_min": _q((time.perf_counter() - t0) / 60.0)}


# ---------------------------------------------------------------- 外推
def fit_extrap(rows: list[dict]) -> dict:
    """对 St(β) 做线性/二次外推（与 v1.3 E13 同口径）。"""
    b = np.array([r["beta"] for r in rows], dtype=float)
    st = np.array([r["st"] for r in rows], dtype=float)
    st_lit = float(rows[0]["st_lit"])
    lin = np.polyfit(b, st, 1)
    out = {"beta": b.tolist(), "st": st.tolist(),
           "st_lit": st_lit,
           "linear": {"intercept": _q(lin[1]), "slope": _q(lin[0]),
                      "rel_pct": _q((lin[1] / st_lit - 1) * 100)},
           "resid_linear": _q(float(np.max(np.abs(np.polyval(lin, b) - st))))}
    if b.size >= 3:
        quad = np.polyfit(b, st, 2)
        out["quadratic"] = {"intercept": _q(quad[2]),
                            "rel_pct": _q((quad[2] / st_lit - 1) * 100)}
        out["resid_quadratic"] = _q(float(np.max(np.abs(
            np.polyval(quad, b) - st))))
        out["lin_minus_quad_pp"] = _q(100 * (lin[1] - quad[2]) / st_lit)
    return out


# ---------------------------------------------------------------- 图
def make_figures(val, blk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    ax = axes[0, 0]
    p = val["poiseuille"]
    ny = [d["ny"] for d in p]
    ax.plot(ny, [d["l2_rel"] for d in p], "o-", label="L2 rel. error")
    ref = p[0]["l2_rel"] * (np.array(ny, float) / ny[0]) ** -2
    ax.plot(ny, ref, "k--", lw=1, label="slope -2 ref")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("channel ny"); ax.set_ylabel("relative L2 error")
    ax.set_title("A1 Poiseuille: 2nd-order convergence", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    for n in sorted({d["n"] for d in val["shear"]}):
        ss = [d for d in val["shear"] if d["n"] == n]
        ax.semilogy([d["tau"] for d in ss],
                    [max(d["rel_err_pct"], 1e-3) for d in ss], "o-",
                    label=f"n={n}")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="1% line")
    ax.set_xlabel("tau"); ax.set_ylabel("|nu_fit/nu_in - 1| [%]")
    ax.set_title("A2 shear-wave viscosity recovery", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    for d in val["tg"]:
        ax.plot(d["n"], d["rel_err_pct"], "o")
        ax.annotate(f"tau={d['tau']}", (d["n"], d["rel_err_pct"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("n (box = 2*pi, k_lattice = 2*pi/n)")
    ax.set_ylabel("|slope/slope_exact - 1| [%]")
    ax.set_title("A3 3D Taylor-Green initial decay", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    rows = blk["rows"]
    b = sorted({r["beta"] for r in rows})
    for dim, mk in ((2, "s--"), (3, "o-")):
        ss = [r for r in rows if r["dim"] == dim and r["nz"] in (0, NZ)]
        ss = sorted(ss, key=lambda r: r["beta"])
        if ss:
            ax.plot([r["beta"] * 100 for r in ss], [r["st"] for r in ss], mk,
                    label=f"{dim}D St")
    ax.axhline(rows[0]["st_lit"], color="k", ls=":", lw=1,
               label="literature St")
    ax.set_xlabel("blockage beta [%]"); ax.set_ylabel("St")
    ax.set_title("E20/E21 St(beta): 2D vs 3D", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "v14_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 图 2：展向收敛 + 尾流一致性 + 外推
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    sp = blk["spanwise"]
    nzs = [d["nz"] for d in sp]
    ref = next(d for d in sp if d["nz"] == NZ)
    # 三档 St/Cd 只差 1e-6 量级；画绝对偏差（单位 1e-6），否则自动缩放会把
    # 噪声放大成"趋势"，正好是本节要避免的误读。
    ax.plot(nzs, [(d["st"] - ref["st"]) / ref["st"] * 1e6 for d in sp], "o-",
            label="St")
    ax.plot(nzs, [(d["cd_mean"] - ref["cd_mean"]) / ref["cd_mean"] * 1e6
            for d in sp], "s--", label="Cd")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("spanwise nz")
    ax.set_ylabel("relative deviation from nz=16 [ppm]")
    ax.set_title("E20 spanwise nz: flat = no 3D structure", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for dim, mk in ((2, "s"), (3, "o")):
        ss = [r for r in rows if r["dim"] == dim and r["nz"] in (0, NZ)]
        ax.plot([r["cd_mean"] for r in ss], [r["wake_cd_mean"] for r in ss],
                mk, label=f"{dim}D")
    lim = [1.0, 2.2]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Cd (momentum exchange)"); ax.set_ylabel("Cd (wake deficit)")
    ax.set_title("E20 force-defect consistency", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    ex = blk["extrapolation"]
    for dim, mk in (("2d", "s--"), ("3d", "o-")):
        e = ex[dim]
        bb = np.array(e["beta"])
        ax.plot(bb * 100, e["st"], mk, label=f"{dim}D")
        xs = np.linspace(0, max(bb) * 1.05, 50)
        lin = np.polyfit(bb, e["st"], 1)
        ax.plot(xs * 100, np.polyval(lin, xs), ":", lw=1)
    ax.axhline(ex["2d"]["st_lit"], color="k", ls=":", lw=1)
    ax.set_xlabel("blockage beta [%]"); ax.set_ylabel("St")
    ax.set_title("E21 beta->0 extrapolation", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "v14_blockage3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    if args.force and CACHE.exists():
        for p in CACHE.glob("*.json"):
            p.unlink()

    val_p = RESULTS / "v14_validation.json"
    blk_p = RESULTS / "v14_blockage3d.json"
    if args.figures_only:
        val = json.loads(val_p.read_text(encoding="utf-8"))
        blk = json.loads(blk_p.read_text(encoding="utf-8"))
        make_figures(val, blk)
        print("仅重画图完成")
        return

    # ---- E19
    cached = _load("e19")
    if cached:
        val = cached
        print("E19 命中缓存")
    else:
        print("E19 内核验证 ...", flush=True)
        t0 = time.time()
        val = {"poiseuille": e19_poiseuille(),
               "shear": e19_shear(),
               "tg": e19_tg(),
               "uniform": e19_uniform()}
        val["wall_min"] = round((time.time() - t0) / 60, 2)
        _save("e19", val)
    for d in val["poiseuille"]:
        print(f"  Poiseuille ny={d['ny']:2d} steps={d['steps']:6d} "
              f"ratio={d['ratio']:.4f} L2={d['l2_rel']:.2e} "
              f"drift={d['drift']:.1e} order="
              f"{d.get('order', float('nan')):.2f}")
    print(f"  shear max err {max(d['rel_err_pct'] for d in val['shear']):.2f}%"
          f" | TG max err {max(d['rel_err_pct'] for d in val['tg']):.2f}%")
    print(f"  uniform dev {max(d['ux_dev'] for d in val['uniform']):.2e}")

    # ---- E20/E21 3D 与 2D 全 beta
    rows = []
    for ny in BLK_NY:
        key3 = f"d3_ny{ny}_nz{NZ}_{_TTAG}"
        r3 = _load(key3)
        if r3 is None:
            t0 = time.time()
            r3 = run_3d(ny, NZ)
            r3["wall_min"] = round((time.time() - t0) / 60, 2)
            _save(key3, r3)
        rows.append(r3)
        print(f"  3D beta={r3['beta']:.2f} St={r3['st']:.4f} "
              f"Cd={r3['cd_mean']:.3f} wake={r3['wake_cd_mean']:.3f} "
              f"uz_rel={r3['uz_rel']:.2e} ({r3['wall_min']:.1f} min)",
              flush=True)
        key2 = f"d2_ny{ny}_{_TTAG}"
        r2 = _load(key2)
        if r2 is None:
            r2 = run_2d(ny)
            _save(key2, r2)
        rows.append(r2)
        print(f"  2D beta={r2['beta']:.2f} St={r2['st']:.4f} "
              f"Cd={r2['cd_mean']:.3f}", flush=True)

    # 展向收敛（β=20%）：沿用 6T 协议 —— 三个 nz 的 St/Cd **逐位相同**，
    # 已足以证明展向严格解耦（z 均匀 ⇒ 各 z 平面互不耦合，盒长只影响内存）。
    # 用 6T 而非 16T 是为省 23 min，该结论与瞬态长度无关。
    sp = []
    for nz in (8, 16, 24):
        key = f"d3_span_nz{nz}"
        r = (_load(f"d3_ny80_nz{NZ}") if nz == NZ      # 6T 旧缓存（无 t 后缀）
             else _load(key))
        if r is None:
            t0 = time.time()
            r = run_3d(80, nz, t_trans=T_TRANS_SPAN)
            r["wall_min"] = round((time.time() - t0) / 60, 2)
            _save(key, r)
        sp.append({k: r[k] for k in ("nz", "st", "cd_mean", "wake_cd_mean",
                                     "uz_rel", "fz")})
        print(f"  span nz={nz:2d} St={r['st']:.4f} Cd={r['cd_mean']:.3f}",
              flush=True)

    ex = {}
    for dim, tag in ((2, "2d"), (3, "3d")):
        sub = [r for r in rows if r["dim"] == dim]
        sub = sorted(sub, key=lambda r: -r["beta"])
        ex[tag] = fit_extrap(sub)
        e = ex[tag]
        print(f"  {dim}D extrap: lin {e['linear']['rel_pct']:+.2f}% "
              f"quad {e['quadratic']['rel_pct']:+.2f}% "
              f"(Δ{e['lin_minus_quad_pp']:.2f}pp)")

    # 维度差：同 β 同分辨率
    contrast = []
    for ny in BLK_NY:
        r3 = next(r for r in rows if r["dim"] == 3 and r["ny"] == ny)
        r2 = next(r for r in rows if r["dim"] == 2 and r["ny"] == ny)
        contrast.append({"beta": r3["beta"],
                         "st3": r3["st"], "st2": r2["st"],
                         "dst_pct": _q((r3["st"] / r2["st"] - 1) * 100),
                         "cd3": r3["cd_mean"], "cd2": r2["cd_mean"],
                         "dcd_pct": _q((r3["cd_mean"] / r2["cd_mean"] - 1)
                                       * 100)})

    blk = {"rows": rows, "spanwise": sp, "extrapolation": ex,
           "contrast": contrast,
           "protocol": {"geo": GEO, "nz": NZ, "blk_ny": list(BLK_NY),
                        "t_trans": T_TRANS, "t_rec": T_REC}}
    blk_p.write_text(json.dumps(blk, ensure_ascii=False, indent=1, default=_q),
                     encoding="utf-8")
    val_p.write_text(json.dumps(val, ensure_ascii=False, indent=1, default=_q),
                     encoding="utf-8")

    import pandas as pd
    flat = [{k: v for k, v in r.items() if k != "wake_cd"} for r in rows]
    pd.DataFrame(flat).to_csv(RESULTS / "v14_case_rows.csv", index=False)

    make_figures(val, blk)
    print("完成：results/v14_*.json + figures/v14_*.png")


if __name__ == "__main__":
    main()
