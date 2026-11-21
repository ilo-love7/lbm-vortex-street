"""11 v1.5 三维失稳区：E22 展向盒长扫描 / E23 失稳 Re 阈值区间。

为什么做这一轮
--------------
v1.4 §15 把「Re=250 三维」列为最高优先未做项；§14.4 又证明 Re=100 的展向
nz 扫描是**空判据**（没有三维结构可收敛）。这两件事是同一个问题的两面：
展向自由度在 Re=100 上没有任何可分辨的结构。要检验展向分辨率，必须先进入
三维失稳区；而进入失稳区的第一件事是**给够盒长**——Lz=1D 连 Mode A 的波长
（文献 3–4D）都装不下，nz=16 的"无三维结构"因此可能是盒子逼出来的。

E22 展向盒长扫描（Re=250，β=20%，16T+10T）
  Lz/D ∈ {0.5, 1, 2, 4}（nz = 8/16/32/64，dz 恒为 1 lu ⇒ 分辨率不变，只改盒长）。
  读数：末态 uz_rel（全局）、展向谱 frac_3d 与峰位 m、展向力 fz、St/Cd/Cl_rms。
  判别：若 Lz=0.5D/1D 无三维结构而 4D 有 ⇒ v1.4 的"nz 空判据"其实是**盒长
  不足**，必须改写；若 4D 仍无 ⇒ Re=250 在本离散下确实不触发三维失稳。

E22-control 长盒对照（Re=100，nz=64）
  同一条 4D 长盒、同分辨率，只把 Re 降回 100。若这里 uz_rel 仍到 1e-11，
  则"4D 盒子自己长出三维结构"这个数值伪影解释被排除——这是让 Re=250 的
  正结果可信的**独立对照**（v1.4 的教训：判据必须独立，不能自证）。

E23 失稳阈值区间（nz=64）
  Re ∈ {190, 220, 250}，与同几何 2D 逐项配对。文献 Mode A 起始 Re≈190，
  Mode B≈230–260；本网格 dz=0.25D 只能解析 Mode A 的波长，Mode B 欠解析，
  这个限制必须与结论同报。

用法：
    python scripts/11_v15_3d_unstable.py            # 增量（命中缓存即跳过）
    python scripts/11_v15_3d_unstable.py --smoke    # 小算例冒烟（不写生产产物）
    python scripts/11_v15_3d_unstable.py --figures-only
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
from lbm.solver3d import Case3D, Solver3D  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = RESULTS / "v15_cache"

# ---- 协议：与 v1.4 生产跑逐项一致，唯一新增的自由度是 Re 与 Lz --------------
GEO = dict(nx=320, d=16.0, cx_c=80.0, u_in=0.1, sponge_width=32)
NY20 = 80                        # β = D/ny = 20%
T_TRANS, T_REC = 16, 10          # v1.4 §14.5：16T 起统计量饱和
SPAN_NZ = (8, 16, 32, 64)        # Lz/D = 0.5 / 1 / 2 / 4（dz = 1 lu）
NSPAN = 64                       # 失稳区主分辨率（Lz = 4D）
SPAN_RE = 250.0                  # 盒长扫描的 Re（v1.5 的目标工况）
ONSET_RE = (190.0, 220.0, 250.0)
CTRL_RE = 100.0                  # 长盒对照
UZ_EVERY = 400                   # 瞬态展向诊断抽样间隔（全场宏量有成本）
WAKE_EVERY = 20


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


def _period(case) -> float:
    return case.d / (case.u_in * literature_st(case.re))


def decay_audit(r: dict) -> dict:
    """判断「残余 uz」是黏性记忆、数值地板，还是失稳。

    三个互补读数（缺一不可）
    ------------------------
    1. ``slope_per_step``：``ln(uz_rel)`` 对步数的拟合斜率（t≥1T，避开 trip 建立段）；
       与最长允许模式的黏性率 ``-ν(2π/nz)²`` 之比 ``ratio``。注意**初场是宽带的**：
       高 m 衰减更快，所以早段斜率天然比最长模式更陡（ratio<1 是正常的），
       不能用它单独判失稳。
    2. ``late_level``：末段 10% 采样的中位数——真正的「停在哪」。
    3. ``excess``：``late_level / (uz_rel_start · e^{-νk²t})``。分母是**上限**估计
       （假设全部初始能量都在最慢模式上），因此
       ``excess ≤ 1`` ⇒ 完全可以用黏性解释（二维）；
       ``excess ≫ 1`` ⇒ 有非黏性来源（数值地板或失稳），再看它随 Re 是否变化：
       与 Re 无关 ⇒ 数值地板；随 Re 抬升并饱和到 1e-2 ⇒ 失稳。
    """
    s = r.get("uz_series") or {}
    if not s:
        return {}
    T = float(r["T"])
    t = np.asarray(s["t_over_T"], float) * T
    a = np.asarray(s["uz_rel"], float)
    if a.size < 8:
        return {"n": 0}
    nu = (float(r["tau"]) - 0.5) / 3.0
    k = 2.0 * np.pi / float(r["nz"])
    visc = -nu * k * k

    m = (a > 1e-10) & (t > T)
    if m.sum() < 4:
        m = a > 1e-14
    slope = float(np.polyfit(t[m], np.log(a[m]), 1)[0]) if m.sum() >= 2 else 0.0

    n = a.size
    late = np.arange(n) >= int(0.7 * n)
    late_level = float(np.median(a[late])) if late.any() else float(a[-1])
    late_slope = 0.0
    if late.sum() >= 3:
        late_slope = float(np.polyfit(t[late], np.log(np.maximum(a[late],
                                                              1e-300)), 1)[0])
    t_end = float(t[-1])
    pred = float(a[0]) * float(np.exp(visc * t_end))
    return {"slope_per_step": slope, "visc_per_step": visc,
            "ratio": (slope / visc) if visc else float("nan"),
            "n_fit": int(m.sum()),
            "window_T": [_q(t[m].min() / T), _q(t[m].max() / T)],
            "decay_per_T_pct": _q(100 * (1 - np.exp(slope * T))),
            "late_level": late_level,
            "late_slope_per_step": late_slope,
            "late_growing": bool(late_slope > 0.0),
            "machine_zero": bool(late_level < 1e-13),
            "visc_pred_level": pred,
            "excess": (late_level / pred) if pred > 0 else float("inf"),
            "t_end_over_T": _q(t_end / T)}


def _post(r: dict) -> dict:
    """给（可能来自缓存的）算例补上派生的衰减审计。"""
    r["decay"] = decay_audit(r)
    return r


# ---------------------------------------------------------------- 2D 参考
def run_2d(re: float, ny: int = NY20, t_trans: float = T_TRANS) -> dict:
    case = CylinderCase(ny=ny, re=re, **GEO)
    T = _period(case)
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
    return {"dim": 2, "re": re, "ny": ny, "nz": 0, "beta": _q(case.d / ny),
            "tau": _q(case.tau), "T": _q(T),
            "cd_mean": _q(cd.mean()), "cl_rms": _q(np.sqrt(np.mean(cl ** 2))),
            "st": _q(st["st"]), "st_lit": _q(literature_st(re)),
            "wake_cd_mean": _q(np.mean(list(wk.values()))),
            "wall_min": _q((time.perf_counter() - t0) / 60.0)}


# ---------------------------------------------------------------- 3D 主跑
def run_3d(re: float, nz: int, ny: int = NY20,
           t_trans: float = T_TRANS, tag: str = "t16") -> dict:
    """16T 瞬态 + 10T 记录；瞬态期按 UZ_EVERY 抽展向诊断，末态算展向谱。

    展向诊断（uz_rel 时程）是本实验的主证据：起点由 ``trip=1e-3`` 的随机初场
    给出，终点只有两种命运——衰减到 1e-10 量级（稳定）或饱和到 1e-1 量级
    （不稳定）。两者差 9 个量级，不需要拟合增长率就能判读，因此比"拟合一个
    σ"更稳健（Re=250 的基础流本身不定常，拟合会被脱落振荡污染）。
    """
    case = Case3D(ny=ny, nz=nz, re=re, **GEO)
    T = _period(case)
    qf = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    sim = Solver3D(case)
    n_trans, n_rec = int(t_trans * T), int(T_REC * T)
    t0 = time.perf_counter()

    uz_t, uz_val = [], []
    for it in range(n_trans):
        sim.step()
        if it % UZ_EVERY == 0:
            sm = sim.sample_metrics()
            uz_t.append(it / T)
            uz_val.append(sm["uz_rel"])

    cd, cl, uzr, fzr = [], [], [], []
    wake = {str(s): [] for s, _ in sim._wake_stations}
    for i in range(n_rec):
        sim.step()
        fx, fy, fz = sim.force3()
        cd.append(fx / qf)
        cl.append(fy / qf)
        fzr.append(abs(fz) / qf)
        if i % WAKE_EVERY == 0:
            sm = sim.sample_metrics()
            for k, v in sm["wake"].items():
                wake[k].append(v["force"] / qf)
            uzr.append(sm["uz_rel"])
            uz_t.append((n_trans + i) / T)
            uz_val.append(sm["uz_rel"])

    xs = [x for _, x in sim._wake_stations] or [int(case.cx_c + 2 * case.d)]
    spec = sim.spanwise_spectrum(xs=xs)

    cd = np.asarray(cd)
    cl = np.asarray(cl)
    st = strouhal(cl, case.d, case.u_in)
    wk = {k: _q(np.mean(v)) for k, v in wake.items()}
    uz_end = float(np.mean(uzr))
    return {"dim": 3, "re": re, "ny": ny, "nz": nz, "beta": _q(case.d / ny),
            "tau": _q(case.tau), "T": _q(T),
            "steps": int(n_trans + n_rec),
            "cd_mean": _q(cd.mean()), "cl_rms": _q(np.sqrt(np.mean(cl ** 2))),
            "st": _q(st["st"]), "st_lit": _q(literature_st(re)),
            "wake_cd_mean": _q(np.mean(list(wk.values()))),
            "uz_rel": _q(uz_end),
            "uz_rel_start": _q(uz_val[0]) if uz_val else 0.0,
            "uz_rel_peak": _q(max(uz_val)) if uz_val else 0.0,
            "gamma": _q(uz_end / uz_val[0]) if uz_val and uz_val[0] > 0 else 0.0,
            "fz_rel": _q(np.mean(fzr)) if fzr else 0.0,
            "spectrum": spec,
            "uz_series": {"t_over_T": [_q(v) for v in uz_t],
                          "uz_rel": [_q(v) for v in uz_val]},
            "wall_min": _q((time.perf_counter() - t0) / 60.0),
            "tag": tag}


# ---------------------------------------------------------------- 图
def make_figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6))

    # (a) uz_rel 时程：衰减 vs 饱和，一眼可分
    ax = axes[0, 0]
    for r in sorted(res["span"], key=lambda r: r["nz"]):
        s = r["uz_series"]
        rt = r.get("decay", {}).get("ratio", float("nan"))
        ax.semilogy(s["t_over_T"], np.maximum(s["uz_rel"], 1e-17), "-",
                    lw=1.1,
                    label=f"Re={r['re']:.0f} Lz={r['nz'] / 16:.2f}D "
                          f"(dec/visc={rt:.2f})")
    ct = res["control"]
    crt = ct.get("decay", {}).get("ratio", float("nan"))
    ax.semilogy(ct["uz_series"]["t_over_T"],
                np.maximum(ct["uz_series"]["uz_rel"], 1e-17), "k--", lw=1.2,
                label=f"Re=100 Lz=4D control (dec/visc={crt:.2f})")
    ax.axhline(1e-3, color="gray", ls=":", lw=1)
    ax.set_xlabel("t / T (T = D/(U St_lit))")
    ax.set_ylabel("spanwise velocity ratio uz_rel")
    ax.set_title("E22 Re=250: spanwise box Lz/D sweep", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=0.3, which="both")

    # (b) 末态 uz_rel vs 盒长（含 Re=100 长盒对照）
    ax = axes[0, 1]
    L = [r["nz"] / 16 for r in res["span"]]
    ax.semilogy(L, [max(r["uz_rel"], 1e-16) for r in res["span"]], "o-",
                label="Re=250")
    ax.semilogy([ct["nz"] / 16], [max(ct["uz_rel"], 1e-16)], "ks",
                label="Re=100 control")
    ax.axhline(1e-3, color="gray", ls=":", lw=1, label="initial trip 1e-3")
    ax.set_xlabel("spanwise box length Lz/D"); ax.set_ylabel("uz_rel (end)")
    ax.set_title("end state: decay or saturate", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    # (c) 展向谱（Lz=4D，三个尾流站位）：纵轴是「模式能量 / 面内动能」
    ax = axes[1, 0]
    r64 = next((r for r in res["span"] if r["nz"] == NSPAN), None)
    if r64:
        for st_d, spec in zip((3, 6, 10), r64["spectrum"]["stations"]):
            p = np.asarray(spec["power"], float)
            inpl = max(spec["inplane"], 1e-300)
            ax.semilogy(np.arange(p.size), np.maximum(p / inpl, 1e-30), "o-",
                        ms=3, lw=1, label=f"x/D={st_d}")
        ax.axhline(1e-8, color="r", ls="--", lw=1, label="3D threshold 1e-8")
        ax.set_xlabel("spanwise mode m (lambda_z = Lz/m)")
        ax.set_ylabel("P_m / in-plane kinetic energy")
        ax.set_title("E22 spanwise spectrum at Re=250, Lz=4D", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    # (d) 2D vs 3D 随 Re：St 与 uz_rel 双面板拼一张（左右轴）
    ax = axes[1, 1]
    on = res["onset"]
    res_re = [r["re"] for r in on]
    ax.plot(res_re, [r["st"] for r in on], "o-", label="3D St (Lz=4D)")
    ax.plot(res_re, [r["st_2d"] for r in on], "s--", label="2D St")
    ax.plot(res_re, [r["st_lit"] for r in on], "k:", lw=1, label="St_lit")
    ax.set_xlabel("Re"); ax.set_ylabel("St")
    ax2 = ax.twinx()
    ax2.semilogy(res_re, [max(r["uz_rel"], 1e-16) for r in on], "^:",
                 color="tab:red", lw=1, label="uz_rel (right)")
    ax2.set_ylabel("uz_rel (end)", color="tab:red")
    ax2.tick_params(axis="y", colors="tab:red")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="lower left")
    ax.set_title("E23 onset bracket: 3D vs 2D over Re", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "v15_3d_unstable.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 冒烟
def smoke():
    """小算例：确认 τ=0.5192（Re=250）在本离散下不炸，谱方法读到正确 m。"""
    print("smoke: Re=250 nz=16 ny=80, 600 步 ...", flush=True)
    case = Case3D(ny=80, nz=16, re=250.0, **GEO)
    sim = Solver3D(case)
    for _ in range(600):
        sim.step()
    rho, ux, uy, uz = sim.macros()
    print(f"  tau={case.tau:.4f} rho∈[{rho.min():.4f},{rho.max():.4f}] "
          f"|u|max={np.abs(ux).max():.4f} finite={np.isfinite(rho).all()}")
    # 合成谱自检：uz = A sin(2π·2z/16) ⇒ 峰在 m=2
    nz = 16
    z = np.arange(nz)
    from lbm.lattice3d import equilibrium3d
    uzz = np.zeros((nz, 80, 320))
    uzz += (1e-3 * np.sin(2 * np.pi * 2 * z / nz))[:, None, None]
    sim.f = equilibrium3d(np.ones_like(uzz), np.full_like(uzz, 0.05),
                          np.zeros_like(uzz), uzz)
    sp = sim.spanwise_spectrum(xs=[128])
    s = sp["stations"][0]
    print(f"  synthetic m=2 -> m_peak={s['m_peak']} "
          f"lambda/D={s['lambda_peak_over_d']:.3f} "
          f"p3d_rel={s['p3d_rel']:.2e} is_3d={s['is_3d']}")


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    if args.force and CACHE.exists():
        for p in CACHE.glob("*.json"):
            p.unlink()

    out_p = RESULTS / "v15_3d_unstable.json"
    if args.figures_only:
        res = json.loads(out_p.read_text(encoding="utf-8"))
        make_figures(res)
        print("仅重画图完成")
        return

    # ---- E22 展向盒长扫描
    span = []
    for nz in SPAN_NZ:
        key = f"v15_3d_re{int(SPAN_RE)}_nz{nz}_t16"
        r = _load(key)
        if r is None:
            r = run_3d(SPAN_RE, nz)
            _save(key, r)
        r = _post(r)
        span.append(r)
        s = next((d for d in r["spectrum"]["stations"]), None)
        dc = r["decay"]
        print(f"  [span] Lz={nz / 16:.2f}D nz={nz:2d} St={r['st']:.4f} "
              f"Cd={r['cd_mean']:.3f} Cl_rms={r['cl_rms']:.4f} "
              f"uz_rel={r['uz_rel']:.2e} (start {r['uz_rel_start']:.1e}) "
              f"fz={r['fz_rel']:.1e} "
              f"p3d_rel={s['p3d_rel'] if s else float('nan'):.2e} "
              f"m_peak={s['m_peak'] if s else '-'} "
              f"decay/visc={dc.get('ratio', float('nan')):.2f} "
              f"[{r['wall_min']:.1f} min]", flush=True)

    # ---- E22-control：同一条 4D 长盒、Re=100
    key = f"v15_3d_re{int(CTRL_RE)}_nz{NSPAN}_t16"
    ctrl = _load(key)
    if ctrl is None:
        ctrl = run_3d(CTRL_RE, NSPAN)
        _save(key, ctrl)
    ctrl = _post(ctrl)
    print(f"  [control] Re=100 Lz=4D uz_rel={ctrl['uz_rel']:.2e} "
          f"St={ctrl['st']:.4f} "
          f"decay/visc={ctrl['decay'].get('ratio', float('nan')):.2f} "
          f"[{ctrl['wall_min']:.1f} min]", flush=True)

    # ---- E23 失稳阈值区间
    onset = []
    for re in ONSET_RE:
        key3 = f"v15_3d_re{int(re)}_nz{NSPAN}_t16"
        r3 = _load(key3)
        if r3 is None:
            r3 = run_3d(re, NSPAN)
            _save(key3, r3)
        r3 = _post(r3)
        key2 = f"v15_2d_re{int(re)}_ny{NY20}_t16"
        r2 = _load(key2)
        if r2 is None:
            r2 = run_2d(re, NY20)
            _save(key2, r2)
        onset.append({"re": re, "st": r3["st"], "st_2d": r2["st"],
                      "st_lit": r3["st_lit"],
                      "dst_pct": _q(100 * (r3["st"] / r2["st"] - 1)),
                      "cd": r3["cd_mean"], "cd_2d": r2["cd_mean"],
                      "dcd_pct": _q(100 * (r3["cd_mean"] / r2["cd_mean"] - 1)),
                      "cl_rms": r3["cl_rms"], "cl_rms_2d": r2["cl_rms"],
                      "uz_rel": r3["uz_rel"], "uz_rel_2d": 0.0,
                      "fz_rel": r3["fz_rel"],
                      "wall_min": r3["wall_min"]})
        print(f"  [onset] Re={re:.0f} 3D St={r3['st']:.4f} "
              f"(2D {r2['st']:.4f}, {onset[-1]['dst_pct']:+.2f}%) "
              f"Cld={onset[-1]['dcd_pct']:+.2f}% "
              f"Cl_rms {r3['cl_rms']:.4f}/{r2['cl_rms']:.4f} "
              f"uz_rel={r3['uz_rel']:.2e} [{r3['wall_min']:.1f} min]",
              flush=True)

    res = {"span": span, "control": ctrl, "onset": onset,
           "protocol": {"geo": GEO, "ny": NY20, "beta": 0.2,
                        "t_trans": T_TRANS, "t_rec": T_REC,
                        "span_nz": list(SPAN_NZ), "nspan": NSPAN,
                        "span_re": SPAN_RE, "onset_re": list(ONSET_RE),
                        "control_re": CTRL_RE, "uz_every": UZ_EVERY}}
    out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_q),
                     encoding="utf-8")

    import pandas as pd
    flat = []
    for r in span + [ctrl]:
        d = {k: v for k, v in r.items()
             if k not in ("spectrum", "uz_series")}
        d["case"] = "control" if r is ctrl else "span"
        flat.append(d)
    for o in onset:
        flat.append({"case": "onset", "dim": 3, "re": o["re"], "nz": NSPAN,
                     "st": o["st"], "cd_mean": o["cd"], "cl_rms": o["cl_rms"],
                     "uz_rel": o["uz_rel"], "wall_min": o["wall_min"]})
    pd.DataFrame(flat).to_csv(RESULTS / "v15_case_rows.csv", index=False)
    make_figures(res)
    print("完成：results/v15_3d_unstable.json + figures/v15_3d_unstable.png")


if __name__ == "__main__":
    main()
