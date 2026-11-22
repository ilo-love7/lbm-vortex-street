"""14 v1.6 高 Re 三维：3D 效应随 Re 的走向（饱和还是继续长）。

为什么做这一轮
--------------
v1.5 §17.4 只测到 Re=250 一个点：3D 失稳确实发生（Mode A，阈值 (190, 220]），
且它改的是**力幅**（Cl_rms −22.2%、Cd −3.05%）而不是 St（−0.34%）。但一个点
不能区分两种完全不同的物理：

- **饱和**：Mode A 幅值在阈值之上很快饱和 ⇒ 3D 修正量在高 Re 趋于常数；
- **继续增长**：3D 结构与 2D 脱落相互调制 ⇒ 修正量随 Re 单调加大。

这个区别直接决定 v1.3「2D 约束误差随 Re 增长」的旧假设是否在阈值之上**兑现**。
v1.3 §11.1 曾把 Re=250 残余 +4.71% 归给 2D 约束；v1.4 用 Re=100 的三维否掉了
它（维度只值 0.04pp），但 Re>190 的三维从未测过 —— 本轮补上。

协议（与 v1.5 逐项一致，只动 Re 与盒长）
----------------------------------------
Lz = 6D（nz=96，dz=1 lu），Re ∈ {250, 300, 350, 400}，β=20%，16T+10T。
**为什么把盒长从 4D 拉到 6D**：周期盒只允许 λ_z = Lz/m。
- Mode A（3–4D）在 Lz=6D 落 m=2（λ=3D），在 Lz=4D 落 m=1（λ=4D，带边）；
- Mode B（0.8–1D）在 Lz=6D 落 m=6/7（1D / 0.857D），在 Lz=4D 落 m=4/5（带边）。
6D 是两支都**离开带边**的最小整数盒长；同时 Re=250 在 nz=96 与 v1.5 的 nz=64
构成天然的**盒长桥接**（同 Re 不同盒长，直接检验 v1.5 的饱和值是否盒长无关）。

新增两个读数（v1.5 只有谱峰 m，换盒子就不可比）
- `mode_band_energy`：把功率按**物理波长带**汇总，与盒子无关；
- `spanwise_correlation`：物理空间的两点相关与积分尺度 L_c，拓扑独立。

用法：
    python scripts/14_v16_highre.py --cases 250,300    # 只跑指定 Re（并行分批）
    python scripts/14_v16_highre.py                    # 增量（命中缓存即跳过）
    python scripts/14_v16_highre.py --long250           # Re=250 的 40T+10T 定常性对照
    python scripts/14_v16_highre.py --box-extra         # 只跑 Re=250 的 Lz=8D（本轮抢核未跑）
    python scripts/14_v16_highre.py --smoke
    python scripts/14_v16_highre.py --figures-only

未做项（如实标注）：原设计的 `Lz=8D`（nz=128）档用于把 Mode A 的 λ 还原成 4D、
从而把"盒长"与"允许波长"分开。它与主批、2D 配对三进程并行时 `nz=128` 掉到约
2.3 步/秒（预计 2.7 h）且把主批拖到 73 min，跑到 56.5 min 无产出后取消。因此
4D↔6D 的差异只能报**合计**；`--box-extra` 保留给算力宽裕时补这一格。
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
from lbm.solver3d import (Case3D, Solver3D, mode_band_energy,  # noqa: E402
                          representable_wavelengths)

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = RESULTS / "v16_cache"

GEO = dict(nx=320, d=16.0, cx_c=80.0, u_in=0.1, sponge_width=32)
NY20 = 80
T_TRANS, T_REC = 16, 10
LZ_D = 6.0
NSPAN6 = int(LZ_D * GEO["d"])          # 96：dz = 1 lu，分辨率与 v1.5 相同
NSPAN8 = 128                            # Lz = 8D（可选，用于盒长/波长分离）
LONG_T_TRANS = 40                       # 长窗对照：Re=250 在 6D 盒里的定常性检查
RE_LIST = (250.0, 300.0, 350.0, 400.0)
RE_C = 190.0                            # v1.5 实测阈值区间下界
RE_SPAN = 110.0                         # (Re - Re_c) 的归一化尺度（=400-190-100）
UZ_EVERY = 400
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
def run_3d(re: float, nz: int = NSPAN6, ny: int = NY20,
           t_trans: float = T_TRANS, tag: str = "t16") -> dict:
    case = Case3D(ny=ny, nz=nz, re=re, **GEO)
    T = _period(case)
    qf = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    sim = Solver3D(case)
    n_trans, n_rec = int(t_trans * T), int(T_REC * T)
    t0 = time.perf_counter()

    uz_t, uz_val, rho_lo, rho_hi = [], [], [], []
    for it in range(n_trans):
        sim.step()
        if it % UZ_EVERY == 0:
            sm = sim.sample_metrics()
            uz_t.append(it / T)
            uz_val.append(sm["uz_rel"])
            rho_lo.append(sm.get("rho_min", float("nan")))
            rho_hi.append(sm.get("rho_max", float("nan")))

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
            rho_lo.append(sm.get("rho_min", float("nan")))
            rho_hi.append(sm.get("rho_max", float("nan")))

    xs = [x for _, x in sim._wake_stations] or [int(case.cx_c + 2 * case.d)]
    spec = sim.spanwise_spectrum(xs=xs)
    corr = sim.spanwise_correlation(xs=xs)
    rho, ux, uy, uz = sim.macros()
    fluid = sim.fluid
    rf = rho[:, fluid]
    for st, cc in zip(spec["stations"], corr["stations"]):
        st["bands"] = mode_band_energy(st["power"], spec["nz"], spec["d"])
        st["corr"] = {k: cc[k] for k in ("valid", "r0", "r0_over_d",
                                        "Lc_over_d", "R", "r_over_d")}

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
            "fz_rel": _q(np.mean(fzr)) if fzr else 0.0,
            "spectrum": spec, "corr": corr,
            "rho_min": _q(rf.min()), "rho_max": _q(rf.max()),
            "rho_min_seq": [_q(v) for v in rho_lo],
            "ufield_max": _q(float(np.abs(ux).max())),
            "finite": bool(np.isfinite(rf).all() and np.isfinite(uz).all()),
            "uz_series": {"t_over_T": [_q(v) for v in uz_t],
                          "uz_rel": [_q(v) for v in uz_val]},
            "wall_min": _q((time.perf_counter() - t0) / 60.0),
            "tag": tag}


# ---------------------------------------------------------------- 饱和律
def fit_power(x, y):
    """``y = a·x^b`` 的最小二乘（在 log-log 上线性），返回 a/b/R²。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pos = (x > 0) & (y > 0)
    if pos.sum() < 2:
        return None
    b, lna = np.polyfit(np.log(x[pos]), np.log(y[pos]), 1)
    a = float(np.exp(lna))
    pred = a * x ** b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"a": a, "b": float(b),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "pred": pred.tolist(), "n": int(pos.sum())}


def fit_linear(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return None
    k, c = np.polyfit(x, y, 1)
    pred = k * x + c
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"slope": float(k), "intercept": float(c),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "pred": pred.tolist(), "n": int(x.size)}


def spectral_centroid(st0: dict, nz: int, d: float) -> dict:
    """谱的一阶矩（重心）与二阶矩（带宽），比单一峰位稳健。

    v1.6 实测教训：宽带谱上 ``m_peak`` 会在相邻模式间跳（Re=250 与 400 都跳到
    m=1，Re=350 跳到 m=4），而带内占比与重心是连续量。峰位只能当"最大那个
    模式"报告，不能当"特征波长"。
    """
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


def saturation_analysis(rows):
    """把「3D 相对 2D 的偏差」按 Re 汇总，做幂律 vs 线性两种拟合。

    目标量取 ``E(Re) = |ΔCl_rms| / Cl_rms_2D × 100``（力幅偏差，v1.5 证明它是
    3D 效应最灵敏的读数，St 只有它的 1/10）。同时报无模型读数
    ``E(400)/E(250)``：饱和 ⇒ 比值 ≈1；线性增长 ⇒ 比值 ≈ (400−190)/(250−190)=3.5。
    """
    re = np.asarray([r["re"] for r in rows], float)
    e_cl = np.asarray([abs(r["dcl_rms_pct"]) for r in rows], float)
    e_cd = np.asarray([abs(r["dcd_pct"]) for r in rows], float)
    e_st = np.asarray([abs(r["dst_pct"]) for r in rows], float)
    x = re - RE_C
    out = {"re": re.tolist(), "E_cl_rms": e_cl.tolist(),
           "E_cd": e_cd.tolist(), "E_st": e_st.tolist(),
           "x_norm": (x / RE_SPAN).tolist(),
           "power_cl_rms": fit_power(x / RE_SPAN, e_cl),
           "linear_cl_rms": fit_linear(x / RE_SPAN, e_cl),
           "power_cd": fit_power(x / RE_SPAN, e_cd),
           "power_st": fit_power(x / RE_SPAN, e_st),
           "ratio_400_250": (float(e_cl[-1] / e_cl[0]) if e_cl[0] else None),
           "ratio_350_250": (float(e_cl[-2] / e_cl[0]) if e_cl[0] else None)}
    return out


# ---------------------------------------------------------------- 盒长分离
def _box_entry(r: dict, lz_over_d: float, source: str) -> dict:
    s0 = r["spectrum"]["stations"][0]
    return {"nz": r["nz"], "lz_over_d": lz_over_d,
            "st": r["st"], "cd": r["cd_mean"], "cl_rms": r["cl_rms"],
            "uz_rel": r["uz_rel"], "p3d_rel": s0["p3d_rel"],
            "m_peak": s0["m_peak"],
            "lambda_peak_over_d": s0["lambda_peak_over_d"],
            "Lc_over_d": (s0.get("corr") or {}).get("Lc_over_d"),
            "source": source}


def load_v15_box(nz_target: int = 64):
    """从 v1.5 的 JSON 取同协议（nx=320/ny=80/D=16/16T+10T）的 Lz=4D 结果。

    直接复用而不是重跑：v1.5 的缓存目录与协议常量与本轮逐项一致，
    重跑 33 min 只为拿一份已有数字不划算；但**必须核对协议**（见报告 §19.5），
    否则就成了跨协议拼数（v1.5 E25 的教训）。
    """
    p = RESULTS / "v15_3d_unstable.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    r = next((x for x in d.get("span", []) if x.get("nz") == nz_target), None)
    if r is None:
        return None
    return _box_entry(r, nz_target / GEO["d"], "v1.5 JSON (Lz=4D)")


def box_compare(r250: dict, r8: dict | None) -> list:
    """Re=250 的盒长三点：4D（λ_A=4D）/ 6D（λ_A=3D）/ 8D（λ_A=4D）。

    6D 相对 4D 的差异**同时**含「盒子变长」与「Mode A 的离散波长 4D→3D」；
    8D 把 λ_A 还原成 4D，于是 4D↔8D 只差盒长、4D↔6D 差盒长+波长。
    没有 8D 这一档，Re=250 的盒长桥接无法归因（v1.5 教训：判据必须独立）。
    """
    out = []
    r4 = load_v15_box(64)
    if r4:
        out.append(r4)
    out.append(_box_entry(r250, LZ_D, "v1.6 run (Lz=6D)"))
    if r8 is not None:
        out.append(_box_entry(r8, NSPAN8 / GEO["d"], "v1.6 run (Lz=8D)"))
    return out


def make_box_figure(box):
    """盒长三点对比：St / Cd / Cl_rms 随 Lz/D。

    用**类别轴**（0/1/2）而不是数值轴：Lz/D 只有 4/6/8 三个离散取值，
    数值轴会挤出 5/7 这种不存在的刻度，还会因为数值范围极窄而触发
    matplotlib 的 offset 记法（`1e-5+2.262e-1`），读数多一步换算。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not box or len(box) < 2:
        return
    xpos = np.arange(len(box))
    labels = []
    for b in box:
        lam = b.get("lambda_peak_over_d")
        lam_s = (f"{lam:.2f}D" if lam not in (None, float("inf")) else "-")
        labels.append(f"Lz={b['lz_over_d']:.0f}D\n" + "m=" + str(b["m_peak"]) +
                      "\n" + "lam=" + lam_s)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    for ax, key, lab in zip(axes, ("st", "cd", "cl_rms"),
                            ("St", "Cd", "Cl_rms")):
        v = np.asarray([b[key] for b in box], float)
        ax.plot(xpos, v, "o-")
        for x, y in zip(xpos, v):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), fontsize=6.5, ha="center")
        span = float(v.max() - v.min())
        pad = max(span * 0.7, abs(float(v.mean())) * 2e-3)
        ax.set_ylim(v.min() - pad, v.max() + pad)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=6.2)
        ax.set_xlim(-0.5, len(box) - 0.5)
        ax.set_title(f"Re=250  {lab}", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("box length vs allowed Mode A wavelength "
                 "(Re=250, Lz=4D/6D; 8D cancelled)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "v16_box_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图
def make_figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["cases"]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6))

    # (a) uz_rel 时程：四个 Re 全部饱和
    ax = axes[0, 0]
    for r in rows:
        s = r["uz_series"]
        ax.semilogy(s["t_over_T"], np.maximum(s["uz_rel"], 1e-17), lw=1.1,
                    label=f"Re={r['re']:.0f} (uz_rel={r['uz_rel']:.1e})")
    _st = (res.get("stationarity") or {}).get("run")
    if _st:
        _s = _st["uz_series"]
        ax.semilogy(_s["t_over_T"], np.maximum(_s["uz_rel"], 1e-17), "k--",
                    lw=1.2, label=f"Re=250 {LONG_T_TRANS}T window")
    ax.axhline(1e-3, color="gray", ls=":", lw=1, label="initial trip 1e-3")
    ax.axhline(1e-8, color="r", ls="--", lw=1, label="3D threshold 1e-8")
    ax.set_xlabel("t / T   (T = D/(U St_lit))")
    ax.set_ylabel("spanwise velocity ratio  uz_rel")
    # 标题不写"已饱和"：Re=250 在 26T 窗内仍在爬，是否定常由 §20.5 的长窗单独回答
    ax.set_title("(a) spanwise velocity ratio, Lz=6D (nz=96)", fontsize=9)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3, which="both")

    # (b) 谱按物理波长铺开（跨盒长可比）：阴影是 Mode A / Mode B 带
    ax = axes[0, 1]
    for r in rows:
        st0 = r["spectrum"]["stations"][0]
        lam = np.asarray(st0["lambda_over_d"], float)[1:]
        p = np.asarray(st0["power"], float)[1:]
        inpl = max(st0["inplane"], 1e-300)
        ax.loglog(lam, np.maximum(p / inpl, 1e-30), "o-", ms=2.6, lw=0.9,
                  label=f"Re={r['re']:.0f}")
    ax.axvspan(3.0, 4.0, color="tab:blue", alpha=0.10)
    ax.axvspan(0.8, 1.2, color="tab:orange", alpha=0.14)
    # 50T 的谱不画：与 26T 的差别主要在带内占比（A 0.20→0.42），峰高差小于
    # 宽带谱的视觉分辨力，画上去只增杂乱；模式切换用 §20.5 的表说明。
    ax.axhline(1e-8, color="r", ls="--", lw=1, label="3D threshold 1e-8")
    # 带名放底部空白区（谱在高 m 端已到机器零），避免压在曲线上/出画布
    ax.set_ylim(bottom=1e-15)          # 给带名留一条底边，否则文字掉到轴外
    ax.text(3.5, 1.4e-15, "Mode A\n3-4D", fontsize=6.5, ha="center",
            va="bottom", color="tab:blue")
    ax.text(1.0, 1.4e-15, "Mode B\n0.8-1.2D", fontsize=6.5, ha="center",
            va="bottom", color="tab:orange")
    # 对数轴上必须同时换掉 locator 与 formatter，否则默认 LogFormatter 会把
    # 2×10^-1 / 3×10^-1 这类标签叠到自己设的刻度上（首版就是这副样子）
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.125, 0.25, 0.5, 1.0, 2.0,
                                             3.0, 4.0, 6.0]))
    ax.xaxis.set_major_formatter(FixedFormatter(
        ["0.125", "0.25", "0.5", "1", "2", "3", "4", "6"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.11, 7.2)
    ax.set_xlabel(r"spanwise wavelength  $\lambda_z/D$")
    ax.set_ylabel(r"$P_m$ / in-plane kinetic energy")
    ax.set_title("(b) spectrum vs physical wavelength (x/D=3)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.3, which="both")

    # (c) 三个偏差量随 Re
    ax = axes[1, 0]
    re = [r["re"] for r in rows]
    ax.plot(re, [r["dst_pct"] for r in rows], "o-", label="dSt/St_2D")
    ax.plot(re, [r["dcd_pct"] for r in rows], "s-", label="dCd/Cd_2D")
    ax.plot(re, [r["dcl_rms_pct"] for r in rows], "^-", label="dCl_rms/Cl_rms_2D")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(RE_C, color="gray", ls=":", lw=1)
    lo, hi = ax.get_ylim()
    # 左界放到 onset 之前，否则 "onset ~190" 那行字会被裁在轴外（首版就是）
    ax.set_xlim(RE_C - 12, max(re) + 45)
    ax.text(RE_C + 3, lo + 0.06 * (hi - lo), "onset ~190\n(Mode A)", fontsize=6.5,
            color="gray", va="bottom")
    for r in rows:
        ax.annotate(f"{r['dcl_rms_pct']:+.1f}", (r["re"], r["dcl_rms_pct"]),
                    textcoords="offset points", xytext=(3, 4), fontsize=6)
    ax.set_xlabel("Re"); ax.set_ylabel("3D vs 2D  [%]")
    ax.set_title("(c) 3D correction vs Re (Lz=6D, beta=20%)", fontsize=9)
    ax.legend(fontsize=7, loc="center right"); ax.grid(alpha=0.3)

    # (d) 饱和律：log-log 幂律 vs 线性
    ax = axes[1, 1]
    sa = res["saturation"]
    x = np.asarray(sa["x_norm"], float)
    y = np.asarray(sa["E_cl_rms"], float)
    ax.loglog(x, y, "ko", ms=6, label="measured |dCl_rms|")
    pw = sa.get("power_cl_rms")
    if pw:
        xs = np.linspace(x.min() * 0.9, x.max() * 1.05, 50)
        ax.loglog(xs, pw["a"] * xs ** pw["b"], "-", lw=1.2,
                  label=f"power law b={pw['b']:.2f} (R2={pw['r2']:.3f})")
    ln = sa.get("linear_cl_rms")
    if ln:
        xs = np.linspace(x.min() * 0.9, x.max() * 1.05, 50)
        ax.loglog(xs, np.maximum(ln["slope"] * xs + ln["intercept"], 1e-3),
                  "--", lw=1.0,
                  label=f"linear (R2={ln['r2']:.3f})")
    # 两个参考斜率：饱和(b=0) 与 线性(b=1)
    xs = np.asarray([x.min(), x.max()])
    ax.loglog(xs, y[0] * (xs / xs[0]) ** 0.0, ":", color="gray", lw=1,
              label="b=0 (saturated)")
    ax.loglog(xs, y[0] * (xs / xs[0]) ** 1.0, ":", color="tab:red", lw=1,
              label="b=1 (linear)")
    ax.set_xlabel(r"$(Re-190)/110$")
    ax.set_ylabel(r"$|\Delta Cl_{rms}|/Cl_{rms,2D}$  [%]")
    ax.set_title("(d) saturation law of the 3D correction", fontsize=9)
    # log-log 但用普通数字刻度：这里的 x∈[0.5,2]、y∈[20,45] 都是小数量级，
    # 科学计数法（6×10¹ 之类）只会让人多一步换算
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0])
    ax.set_xticks([], minor=True)
    ax.set_yticks([20, 25, 30, 35, 40, 45])
    ax.set_yticks([], minor=True)
    ax.legend(fontsize=6.3); ax.grid(alpha=0.3, which="major")

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "v16_highre.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 第二张图：展向相关（拓扑独立读数，与谱互为对照）
    fig2, ax2 = plt.subplots(figsize=(6.2, 3.8))
    for r in rows:
        cc = r["spectrum"]["stations"][0]["corr"]
        if not cc.get("valid"):
            continue
        R = np.asarray(cc["R"], float)
        rr = np.asarray(cc["r_over_d"], float)
        ax2.plot(rr, R, lw=1.1,
                 label=f"Re={r['re']:.0f} (Lc/D={cc['Lc_over_d']:.2f})")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel(r"spanwise separation  $r/D$")
    ax2.set_ylabel(r"$R_{zz}(r)$")
    ax2.set_title("spanwise two-point correlation, Lz=6D, x/D=3", fontsize=9)
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIGURES / "v16_spanwise_corr.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


# ---------------------------------------------------------------- 冒烟
def smoke():
    """小算例：确认 τ(400)=0.512 在本离散下不炸，相关函数读到正确 r0。"""
    print("smoke: Re=400 nz=16 ny=80, 400 步 ...", flush=True)
    case = Case3D(ny=80, nz=16, re=400.0, **GEO)
    print(f"  tau={case.tau:.4f} (须 > 0.5001)")
    sim = Solver3D(case)
    for _ in range(400):
        sim.step()
    rho, ux, uy, uz = sim.macros()
    rf = rho[:, sim.fluid]
    print(f"  rho_fluid∈[{rf.min():.4f},{rf.max():.4f}] "
          f"|u|max={np.abs(ux).max():.4f} finite={np.isfinite(rf).all()}")
    # 合成纯正弦 m=3（nz=96, d=16 ⇒ λ=2D）：R(0)=1、r0=nz/(4m)=8
    nz = 96
    z = np.arange(nz)
    from lbm.lattice3d import equilibrium3d
    uzz = np.zeros((nz, 80, 320))
    uzz += (1e-3 * np.sin(2 * np.pi * 3 * z / nz))[:, None, None]
    sim.f = equilibrium3d(np.ones_like(uzz), np.full_like(uzz, 0.05),
                          np.zeros_like(uzz), uzz)
    cc = sim.spanwise_correlation(xs=[128])["stations"][0]
    band = mode_band_energy(
        sim.spanwise_spectrum(xs=[128])["stations"][0]["power"], nz, 16.0)
    print(f"  synthetic m=3 -> R(0)={cc['R'][0]:.6f} r0={cc['r0']} "
          f"Lc/D={cc['Lc_over_d']:.3f}")
    print(f"  bands: A={band['bands']['A']['frac_of_p3d']:.3f} "
          f"B={band['bands']['B']['frac_of_p3d']:.3f} "
          f"(m in A {band['bands']['A']['m']}, B {band['bands']['B']['m']})")
    print(f"  representable at Lz=6D: {representable_wavelengths(96, 16.0)[:8]}")


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--cases", default="", help="只跑这些 Re，如 250,300")
    ap.add_argument("--box-extra", action="store_true",
                    help="只跑 Re=250 的 Lz=8D（nz=128），用于盒长/波长分离")
    ap.add_argument("--long250", action="store_true",
                    help="只跑 Re=250/Lz=6D 的长窗对照（40T+10T），检验是否已定常")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    if args.force and CACHE.exists():
        for p in CACHE.glob("*.json"):
            p.unlink()

    if args.long250:
        key = f"v16_3d_re250_nz{NSPAN6}_t{LONG_T_TRANS}"
        r = _load(key)
        if r is None:
            r = run_3d(250.0, NSPAN6, t_trans=LONG_T_TRANS, tag=f"t{LONG_T_TRANS}")
            _save(key, r)
        s0 = r["spectrum"]["stations"][0]
        bb = s0["bands"]["bands"]
        print(f"  [long] Re=250 Lz=6D  {LONG_T_TRANS}T+10T  St={r['st']:.4f} "
              f"Cd={r['cd_mean']:.4f} Cl_rms={r['cl_rms']:.4f} "
              f"uz_rel={r['uz_rel']:.2e} p3d={s0['p3d_rel']:.1e} "
              f"m_peak={s0['m_peak']} A={bb['A']['frac_of_p3d']:.2f} "
              f"B={bb['B']['frac_of_p3d']:.2f} [{r['wall_min']:.0f} min]",
              flush=True)
        return

    if args.box_extra:
        key8 = f"v16_3d_re250_nz{NSPAN8}_t16"
        r8 = _load(key8)
        if r8 is None:
            r8 = run_3d(250.0, NSPAN8)
            _save(key8, r8)
        s0 = r8["spectrum"]["stations"][0]
        print(f"  [box 8D] Re=250 nz={NSPAN8} St={r8['st']:.4f} "
              f"Cd={r8['cd_mean']:.4f} Cl_rms={r8['cl_rms']:.4f} "
              f"uz_rel={r8['uz_rel']:.2e} p3d={s0['p3d_rel']:.1e} "
              f"m_peak={s0['m_peak']} "
              f"Lc/D={s0['corr']['Lc_over_d']:.2f} "
              f"[{r8['wall_min']:.1f} min]", flush=True)
        return

    out_p = RESULTS / "v16_highre.json"
    if args.figures_only:
        res = json.loads(out_p.read_text(encoding="utf-8"))
        make_figures(res)
        make_box_figure(res.get("box"))
        print("仅重画图完成")
        return

    wanted = ([float(v) for v in args.cases.split(",") if v.strip()]
              if args.cases else list(RE_LIST))

    rows = []
    for re in RE_LIST:
        key3 = f"v16_3d_re{int(re)}_nz{NSPAN6}_t16"
        key2 = f"v16_2d_re{int(re)}_ny{NY20}_t16"
        r3, r2 = _load(key3), _load(key2)
        run_here = re in wanted
        if r3 is None and run_here:
            r3 = run_3d(re, NSPAN6)
            _save(key3, r3)
        if r2 is None and run_here:
            r2 = run_2d(re, NY20)
            _save(key2, r2)
        if r3 is None or r2 is None:
            print(f"  [Re={re:.0f}] 跳过（本批未跑且无缓存）", flush=True)
            continue
        s0 = r3["spectrum"]["stations"][0]
        b = s0["bands"]["bands"]
        cen = spectral_centroid(s0, r3["spectrum"]["nz"], r3["spectrum"]["d"])
        rows.append({**r3, "centroid": cen,
                     "st_2d": r2["st"], "cd_2d": r2["cd_mean"],
                     "cl_rms_2d": r2["cl_rms"],
                     "dst_pct": _q(100 * (r3["st"] / r2["st"] - 1)),
                     "dcd_pct": _q(100 * (r3["cd_mean"] / r2["cd_mean"] - 1)),
                     "dcl_rms_pct": _q(100 * (r3["cl_rms"] / r2["cl_rms"] - 1))})
        print(f"  [Re={re:.0f}] 3D St={r3['st']:.4f} Cd={r3['cd_mean']:.4f} "
              f"Cl_rms={r3['cl_rms']:.4f} | 2D St={r2['st']:.4f} "
              f"Cd={r2['cd_mean']:.4f} Cl_rms={r2['cl_rms']:.4f} | "
              f"dSt={rows[-1]['dst_pct']:+.2f}% dCd={rows[-1]['dcd_pct']:+.2f}% "
              f"dCl={rows[-1]['dcl_rms_pct']:+.2f}% | "
              f"uz_rel={r3['uz_rel']:.2e} p3d={s0['p3d_rel']:.1e} "
              f"m_peak={s0['m_peak']} m_c={cen['m_centroid']:.2f} "
              f"Lc/D={s0['corr']['Lc_over_d']:.2f} "
              f"A={b['A']['frac_of_p3d']:.2f} B={b['B']['frac_of_p3d']:.2f} "
              f"tau={r3['tau']:.4f} rho_min={r3['rho_min']:.3f} "
              f"[{r3['wall_min']:.1f} min]", flush=True)

    if len(rows) < len(RE_LIST):
        print(f"部分算例（{len(rows)}/{len(RE_LIST)}）——先落盘缓存，"
              f"其余批次跑完后再合并", flush=True)
        return

    r250 = next((r for r in rows if r["re"] == 250.0), None)
    r8 = _load(f"v16_3d_re250_nz{NSPAN8}_t16")
    box = box_compare(r250, r8) if r250 is not None else None
    rlong = _load(f"v16_3d_re250_nz{NSPAN6}_t{LONG_T_TRANS}")
    stat = None
    if r250 is not None and rlong is not None:
        stat = {
            "re": 250.0, "nz": NSPAN6,
            "t26": {k: r250[k] for k in ("st", "cd_mean", "cl_rms", "uz_rel",
                                         "wall_min")},
            "t40": {k: rlong[k] for k in ("st", "cd_mean", "cl_rms", "uz_rel",
                                          "wall_min")},
            "dst_pct": _q(100 * (rlong["st"] / r250["st"] - 1)),
            "dcd_pct": _q(100 * (rlong["cd_mean"] / r250["cd_mean"] - 1)),
            "dcl_rms_pct": _q(100 * (rlong["cl_rms"] / r250["cl_rms"] - 1)),
            "duz_pct": _q(100 * (rlong["uz_rel"] / r250["uz_rel"] - 1)),
            "run": rlong,
        }
        # 长窗的谱/结构量单列：它才是"定态"那一侧，26T 只作对照
        s0l = rlong["spectrum"]["stations"][0]
        stat["centroid"] = spectral_centroid(s0l, rlong["spectrum"]["nz"],
                                            rlong["spectrum"]["d"])
        stat["m_peak"] = s0l["m_peak"]
        stat["lambda_peak_over_d"] = float(s0l["lambda_over_d"][s0l["m_peak"]])
        stat["frac_A"] = float(s0l["bands"]["bands"]["A"]["frac_of_p3d"])
        stat["frac_B"] = float(s0l["bands"]["bands"]["B"]["frac_of_p3d"])
        stat["p3d_rel"] = float(s0l["p3d_rel"])
        stat["Lc_over_d"] = float(rlong["corr"]["stations"][0]["Lc_over_d"])
        stat["uz_peak"] = float(rlong["uz_rel_peak"])
    res = {"cases": rows, "saturation": saturation_analysis(rows), "box": box,
           "stationarity": stat,
           "protocol": {"geo": GEO, "ny": NY20, "beta": 0.2,
                        "lz_over_d": LZ_D, "nz": NSPAN6,
                        "t_trans": T_TRANS, "t_rec": T_REC,
                        "re_list": list(RE_LIST), "re_c": RE_C,
                        "uz_every": UZ_EVERY,
                        "representable_lambda_over_d":
                            representable_wavelengths(NSPAN6, GEO["d"])[:12].tolist()}}

    # 若长窗对照在场：把 Re=250 那一点换成 50T 的读数重做饱和律，看看结论是否
    # 依赖"26T 是否定常"。这不是"换个数字凑结论"，而是把分母的不确定性显式化。
    if stat is not None:
        sa = res["saturation"]
        e_cl = list(sa["E_cl_rms"])
        e_cd = list(sa["E_cd"])
        e_cl[0] = float(100 * abs(rlong["cl_rms"] / r250["cl_rms_2d"] - 1))
        e_cd[0] = float(100 * abs(rlong["cd_mean"] / r250["cd_2d"] - 1))
        xs = np.asarray(sa["x_norm"], float)
        pc = fit_power(xs, np.asarray(e_cl, float))
        pd = fit_power(xs, np.asarray(e_cd, float))
        stat["sat_with_t40"] = {
            "E_cl_rms": [_q(v) for v in e_cl],
            "E_cd": [_q(v) for v in e_cd],
            "ratio_cl": _q(e_cl[-1] / e_cl[0]),
            "ratio_cd": _q(e_cd[-1] / e_cd[0]),
            "power_cl": pc, "power_cd": pd,
            "note": "Re=250 换成 40T 瞬态 + 10T 记录的同几何读数，其余三点不变",
        }
    out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_q),
                     encoding="utf-8")

    import pandas as pd
    flat = []
    for r in rows:
        d = {k: v for k, v in r.items()
             if k not in ("spectrum", "corr", "uz_series", "rho_min_seq")}
        s0 = r["spectrum"]["stations"][0]
        d.update({"m_centroid": (r.get("centroid") or {}).get("m_centroid"),
                  "lambda_centroid_over_d":
                      (r.get("centroid") or {}).get("lambda_centroid_over_d"),
                  "p3d_rel": s0["p3d_rel"], "m_peak": s0["m_peak"],
                  "lambda_peak_over_d": s0["lambda_peak_over_d"],
                  "frac_A": s0["bands"]["bands"]["A"]["frac_of_p3d"],
                  "frac_B": s0["bands"]["bands"]["B"]["frac_of_p3d"],
                  "Lc_over_d": s0["corr"]["Lc_over_d"],
                  "r0_over_d": s0["corr"]["r0_over_d"]})
        flat.append(d)
    pd.DataFrame(flat).to_csv(RESULTS / "v16_case_rows.csv", index=False)
    make_figures(res)
    make_box_figure(box)
    sa = res["saturation"]
    pw = sa.get("power_cl_rms") or {}
    print(f"饱和律：b={pw.get('b', float('nan')):.2f} "
          f"R2={pw.get('r2', float('nan')):.3f} | "
          f"无模型 E(400)/E(250)={sa.get('ratio_400_250')}")
    print("完成：results/v16_highre.json + figures/v16_highre.png")


if __name__ == "__main__":
    main()
