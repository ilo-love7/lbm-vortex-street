"""N5 v1.10 — Lz=8D 纯盒长分离（Re=250, β=20%, dz=1 lu, 40T+10T 同窗长）。

**要回答的问题**：v1.6 §20.3 的 4D↔6D 差异（St −2.56%）同时改了盒长与
Mode A 的可表示波长（λ_A 4D→3D），无法归因。补上 8D 后：

  4D: λ_A=4D (m=1, 带边)   ← v1.5 短窗有，但**没有长窗**
  6D: λ_A=3D (m=2, 带内)   ← v1.6 有 40T+10T 长窗缓存
  8D: λ_A=4D (m=2, 带内)   ← 本轮新跑

  4D↔8D：λ_A 相同（都是 4D），只差盒长与 m ⇒ **纯盒长判据**
  6D↔8D：盒长与 λ_A 都变 ⇒ 组合
  4D↔6D：盒长与 λ_A 都变 ⇒ 组合（v1.6 无法归因那一对）

**为什么必须同窗长**：短窗（16T+10T）三档 m_peak 全是 1（盒长基模，Mode A
还没建立），λ_peak 就等于盒长本身，分离不了允许波长。6D 长窗（40T）已进
m=2 / λ=3D，所以 4D/8D 也必须跑 40T+10T 才能比。这是 v1.6 报告 §20.3
明确留白的那一格。

**跨盒长可比的读数**（λ 或拓扑定义，与盒长无关）：
  - spectral_centroid 的 lambda_centroid_over_d
  - mode_band_energy 的 band A/B frac_of_p3d
  - spanwise_correlation 的 Lc_over_d
**不可比**的读数：m_peak（λ_z=Lz/m，盒长变则 m 编号变）。

用法：
    python scripts/19_v110_boxlength.py                 # 跑缺档（4D/8D 长窗）+ 汇总
    python scripts/19_v110_boxlength.py --cases 64      # 只跑 4D
    python scripts/19_v110_boxlength.py --cases 128     # 只跑 8D（单独排程，勿抢核）
    python scripts/19_v110_boxlength.py --resummarize   # 不跑，仅从缓存重汇总+出图
    python scripts/19_v110_boxlength.py --smoke         # 自检（0.3T，不用于物理读数）

算例（与 v1.6 逐项一致）：nx=320, d=16, cx_c=80, u_in=0.1, sponge=32,
ny=80, Re=250, t_trans=40, t_rec=10。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# 复用 v1.6 的 run_3d，保证求解器/协议/采样逐位一致
_SPEC = importlib.util.spec_from_file_location(
    "v16_highre", ROOT / "scripts" / "14_v16_highre.py")
v16 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v16)

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = RESULTS / "v110_cache"

RE = 250.0
NZ_LIST = (64, 96, 128)          # Lz = 4D / 6D / 8D
NZ_LABEL = {64: "4D", 96: "6D", 128: "8D"}
T_TRANS, T_REC = 40, 10
D = v16.GEO["d"]                  # 16.0

# 6D 的 40T+10T 缓存是 v1.6 现成产物；直接复用，避免重跑
V16_6D_KEY = "v16_3d_re250_nz96_t40"


def cache_path(nz: int) -> Path:
    return CACHE / f"v110_3d_re250_nz{nz}_t{T_TRANS}.json"


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save(p: Path, payload: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                              default=float), encoding="utf-8")
    tmp.replace(p)


def load_6d_from_v16() -> dict | None:
    """从 v1.6 缓存读 6D 长窗（同协议）；缺档返回 None（不静默造数）。"""
    p = RESULTS / "v16_cache" / f"{V16_6D_KEY}.json"
    return _load(p)


def run_point(nz: int) -> dict:
    """跑一个盒长档（40T+10T）。"""
    r = v16.run_3d(RE, nz=nz, t_trans=T_TRANS, tag=f"t{T_TRANS}")
    r["lz_over_d"] = nz / D
    return r


# ---------------------------------------------------------------- 判据纯函数
def _station0(r: dict) -> dict:
    return r["spectrum"]["stations"][0]


def cross_box_readout(r: dict) -> dict:
    """从一档提取跨盒长可比的读数 + 不可比的峰位（分列，避免混用）。"""
    s0 = _station0(r)
    nz, d = r["spectrum"]["nz"], r["spectrum"]["d"]
    cen = v16.spectral_centroid(s0, nz, d)
    bands = s0.get("bands", {}).get("bands", {})
    corr = s0.get("corr", {})
    return {
        "nz": int(nz), "lz_over_d": float(nz) / float(d),
        "st": float(r["st"]), "cd": float(r["cd_mean"]),
        "cl_rms": float(r["cl_rms"]),
        "uz_rel": float(r["uz_rel"]),
        "p3d_rel": float(s0["p3d_rel"]),
        "wall_min": float(r.get("wall_min", float("nan"))),
        # 跨盒长可比
        "lambda_centroid_over_d": cen["lambda_centroid_over_d"],
        "m_std": cen["m_std"],
        "band_A_frac": (bands.get("A") or {}).get("frac_of_p3d"),
        "band_B_frac": (bands.get("B") or {}).get("frac_of_p3d"),
        "Lc_over_d": corr.get("Lc_over_d"),
        # 不可比（盒长变则编号变）——单列出来提醒
        "m_peak": int(s0["m_peak"]),
        "lambda_peak_over_d": float(s0["lambda_peak_over_d"]),
    }


def box_verdict(points: dict[int, dict]) -> dict:
    """三档盒长分离的联合判据。

    纯盒长对 = 4D↔8D（λ_A 同为 4D）。若它的 St/Cd/Cl_rms 差异显著小于
    含波长变化的 6D↔8D，则差异主因是"允许波长/模式编号"而非盒长本身。
    """
    if not all(k in points for k in NZ_LIST):
        return {"label": "incomplete", "missing":
                [NZ_LABEL[k] for k in NZ_LIST if k not in points]}
    p = {k: cross_box_readout(points[k]) for k in NZ_LIST}
    out = {"pairs": {}, "metrics": {}}
    # 对子：名字 → (nz_i, nz_j, 是否纯盒长)
    pairs = {
        "4D_vs_8D_pure_box": (64, 128, True),
        "6D_vs_8D_box_and_wavelength": (96, 128, False),
        "4D_vs_6D_box_and_wavelength": (64, 96, False),
    }
    for name, (ni, nj, pure) in pairs.items():
        d = {}
        for k in ("st", "cd", "cl_rms", "lambda_centroid_over_d",
                  "Lc_over_d", "band_A_frac"):
            a, b = p[ni].get(k), p[nj].get(k)
            d[k] = (abs(float(a) - float(b))
                    if a is not None and b is not None
                    and np.isfinite(a) and np.isfinite(b) else None)
        d["pure_box"] = pure
        d["abs_dpct_st"] = (100.0 * abs(p[ni]["st"] / p[nj]["st"] - 1.0))
        d["abs_dpct_cl"] = (100.0 * abs(p[ni]["cl_rms"] / p[nj]["cl_rms"] - 1.0))
        d["abs_dpct_cd"] = (100.0 * abs(p[ni]["cd"] / p[nj]["cd"] - 1.0))
        out["pairs"][name] = d
    pure = out["pairs"]["4D_vs_8D_pure_box"]
    mixed = out["pairs"]["6D_vs_8D_box_and_wavelength"]
    # 主判据：力幅 Cl_rms 的纯盒长差 vs 含波长差
    pure_cl, mixed_cl = pure["abs_dpct_cl"], mixed["abs_dpct_cl"]
    if pure_cl < mixed_cl * 0.5:
        label = "wavelength_dominant"
    elif pure_cl > mixed_cl * 2.0:
        label = "box_length_dominant"
    else:
        label = "mixed_or_unresolved"
    out["metrics"] = {"pure_cl_dpct": pure_cl, "mixed_cl_dpct": mixed_cl,
                      "ratio": (pure_cl / mixed_cl if mixed_cl > 0 else None)}
    out["label"] = label
    out["thresholds"] = {"pure_lt_mixed_half": 0.5,
                         "pure_gt_mixed_twice": 2.0,
                         "metric": "Cl_rms 相对差 (%)"}
    return out


def make_figure(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import lbm.plots  # noqa: F401  中文字体副作用

    pts = summary["points"]
    if len(pts) < 2:
        return
    order = [nz for nz in NZ_LIST if nz in pts]
    rd = {nz: cross_box_readout(pts[nz]) for nz in order}
    x = np.arange(len(order))
    fig, ax = plt.subplots(2, 3, figsize=(12.6, 7.2))
    specs = [
        ("st", "St", 0, 0), ("cd", "Cd", 0, 1), ("cl_rms", "Cl_rms", 0, 2),
        ("lambda_centroid_over_d", "lambda_c / D", 1, 0),
        ("band_A_frac", "Mode A band frac", 1, 1),
        ("Lc_over_d", "L_c / D", 1, 2),
    ]
    for key, title, i, j in specs:
        v = np.array([rd[nz].get(key) for nz in order], dtype=float)
        ok = np.isfinite(v)
        a = ax[i, j]
        if ok.any():
            a.plot(x[ok], v[ok], "o-", color="#c0392b")
            for xi, yi in zip(x[ok], v[ok]):
                a.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                           xytext=(0, 7), fontsize=7, ha="center")
        a.set_xticks(x)
        a.set_xticklabels([f"{NZ_LABEL[nz]}\n(m={rd[nz]['m_peak']})"
                           for nz in order], fontsize=8)
        a.set_title(title, fontsize=10)
        a.grid(alpha=0.3)
    fig.suptitle("Box-length separation at Re=250 (40T+10T; 4D vs 8D share lambda_A=4D)",
                 fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "v110_boxlength.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out.relative_to(ROOT)}", flush=True)


def summarize() -> dict:
    points: dict[int, dict] = {}
    for nz in NZ_LIST:
        p = cache_path(nz)
        r = _load(p)
        if r is None:
            print(f"[skip] missing {p.name}", flush=True)
            continue
        points[nz] = r
    # 协议核对：所有档必须同 Re / ny / t_trans / t_rec
    for nz, r in points.items():
        assert abs(float(r["re"]) - RE) < 1e-9, f"nz={nz} Re 不一致"
        assert int(r["ny"]) == v16.NY20, f"nz={nz} ny 不一致"
        assert int(r["nz"]) == nz, f"nz={nz} 缓存 nz 字段不一致"
    verdict = box_verdict(points)
    summary = {
        "protocol": {"re": RE, "ny": v16.NY20, "geo": v16.GEO,
                     "t_trans": T_TRANS, "t_rec": T_REC,
                     "nz_list": list(NZ_LIST),
                     "note": "同窗长三点；6D 复用 v1.6 40T 缓存；4D/8D 本轮新跑"},
        "points": {str(k): cross_box_readout(v) for k, v in points.items()},
        "verdict": verdict,
    }
    out = RESULTS / "v110_boxlength.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                              default=float), encoding="utf-8")
    print(f"summary -> {out.relative_to(ROOT)}", flush=True)
    for nz in sorted(points):
        rd = summary["points"][str(nz)]
        print(f"  [{NZ_LABEL[nz]}] St={rd['st']:.5f} Cd={rd['cd']:.5f} "
              f"Cl={rd['cl_rms']:.5f} lam_c={rd['lambda_centroid_over_d']:.2f}D "
              f"m_peak={rd['m_peak']} A={rd['band_A_frac']:.2f} "
              f"Lc={rd['Lc_over_d']:.2f}D [{rd['wall_min']:.0f} min]", flush=True)
    print("verdict:", json.dumps(verdict, ensure_ascii=False), flush=True)
    make_figure(summary)
    return summary


def smoke() -> None:
    import time as _t
    print("smoke: Re=250 nz=64 ny=80, 0.3T (不用于物理读数) ...", flush=True)
    case = v16.Case3D(ny=v16.NY20, nz=64, re=RE, **v16.GEO)
    T = v16._period(case)
    sim = v16.Solver3D(case)
    n = int(0.3 * T)
    t0 = _t.perf_counter()
    for _ in range(n):
        sim.step()
    s = sim.sample_metrics()
    print(f"  {n} steps in {_t.perf_counter()-t0:.1f}s; uz_rel={s['uz_rel']:.3e} "
          f"finite={bool(np.isfinite(s['uz_rel']))}", flush=True)
    # 判据纯函数自检（合成三点）
    fake = {
        64: {"st": 0.2263, "cd_mean": 1.5776, "cl_rms": 0.5139,
             "uz_rel": 0.05, "spectrum": {"nz": 64, "d": 16.0, "stations": [
                 {"p3d_rel": 1e-3, "m_peak": 1, "lambda_peak_over_d": 4.0,
                  "power": [0, 1, 0, 0], "bands": {"bands": {
                      "A": {"frac_of_p3d": 0.5}, "B": {"frac_of_p3d": 0.2}}},
                  "corr": {"Lc_over_d": 0.4}}]}},
        96: {"st": 0.2205, "cd_mean": 1.5887, "cl_rms": 0.5596,
             "uz_rel": 0.05, "spectrum": {"nz": 96, "d": 16.0, "stations": [
                 {"p3d_rel": 1e-3, "m_peak": 2, "lambda_peak_over_d": 3.0,
                  "power": [0, 0, 1, 0], "bands": {"bands": {
                      "A": {"frac_of_p3d": 0.5}, "B": {"frac_of_p3d": 0.2}}},
                  "corr": {"Lc_over_d": 0.35}}]}},
        128: {"st": 0.2263, "cd_mean": 1.5776, "cl_rms": 0.5145,
              "uz_rel": 0.05, "spectrum": {"nz": 128, "d": 16.0, "stations": [
                  {"p3d_rel": 1e-3, "m_peak": 2, "lambda_peak_over_d": 4.0,
                   "power": [0, 0, 1, 0], "bands": {"bands": {
                       "A": {"frac_of_p3d": 0.5}, "B": {"frac_of_p3d": 0.2}}},
                   "corr": {"Lc_over_d": 0.4}}]}},
    }
    print("  verdict(4D~8D close):", box_verdict(fake)["label"], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cases", default="", help="只跑这些 nz，如 64,128")
    ap.add_argument("--resummarize", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return

    if not args.resummarize:
        todo = ([int(x) for x in args.cases.split(",") if x.strip()]
                if args.cases else [64, 128])
        for nz in todo:
            assert nz in NZ_LIST, f"不支持 nz={nz}"
            p = cache_path(nz)
            if _load(p) is not None:
                print(f"[{NZ_LABEL[nz]}] 已有缓存，跳过", flush=True)
                continue
            if nz == 96:
                print("[6D] 请复用 v1.6 缓存（--resummarize 自动读取）", flush=True)
                continue
            print(f"[{NZ_LABEL[nz]}] 开跑 nz={nz}, {T_TRANS}T+{T_REC}T ...",
                  flush=True)
            r = run_point(nz)
            _save(p, r)
            print(f"[{NZ_LABEL[nz]}] St={r['st']:.5f} Cd={r['cd_mean']:.5f} "
                  f"Cl={r['cl_rms']:.5f} wall={r['wall_min']:.1f} min",
                  flush=True)

    # 6D 从 v1.6 缓存接入本轮目录（同协议，不重跑）
    p6 = cache_path(96)
    if _load(p6) is None:
        r6 = load_6d_from_v16()
        if r6 is not None:
            _save(p6, r6)
            print("[6D] 复用 v1.6 40T 缓存 -> v110_cache", flush=True)
    summarize()


if __name__ == "__main__":
    main()
