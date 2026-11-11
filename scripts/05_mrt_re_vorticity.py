"""05 v1.2 深化：E10 MRT 碰撞 / E11 Re 扫描 / E12 涡量·探针频谱。

  E10 MRT vs BGK：Re ∈ {100, 300, 350}（τ=0.584/0.528/0.524）× {bgk, mrt(magic)}
           —— 力噪声（4 块 std）/慢漂移/低 τ 稳定性是否因 MRT 改善？
  E11 Re 扫描：Re ∈ {75, 125, 175, 250} 新跑 + 复用 v1.1 {50,100,150,200,300}
           —— St(Re)/Cd(Re)/Cl_peak(Re) 全曲线 vs Fey 经验式。
  E12 涡量·探针频谱：Re ∈ {100, 200}，探针 x/D ∈ {1.5, 3, 5}
           —— 升力 St vs uy/ω 探针 St 双测法、互谱对流速度、谐波比。

输出：results/v12_mrt.json / v12_sweep.json / v12_probes.json + 3 张图。

性能：12 个 case 用进程池并行（默认 12 workers，24 核机器墙钟 ~35 min）；
每个 case 结果写 results/v12_cache/<key>.json，重跑自动跳过已完成项。
    python scripts/05_mrt_re_vorticity.py               # 增量（缓存命中即跳过）
    python scripts/05_mrt_re_vorticity.py --workers 8  # 限并行数
    python scripts/05_mrt_re_vorticity.py --force      # 清缓存全部重算
    python scripts/05_mrt_re_vorticity.py --repost     # 仅从探针时程重算后处理（无需重跑 CFD）
    python scripts/05_mrt_re_vorticity.py --figures-only  # 从缓存汇总并重画 3 张图
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 必须早于 numba / numpy 导入：每进程单线程，避免 12 进程 × 24 线程互抢
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm import (CylinderCase, Solver, coherent_cl, force_coefficients,
                 literature_st, run_transient, strouhal, time_stats)
from lbm.config import default_config_path, load_config
from lbm.kernels import CX, CY
from lbm.spectrum import convection_delay, dual_st, harmonic_ratio, probe_st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = RESULTS / "v12_cache"

STATIONS = (1.5, 3.0, 5.0)


# ---------------- 基础工具（纯函数，单测可复用） ----------------
def make_case(cfg, re, collision="bgk"):
    c = cfg["case"]
    return CylinderCase(
        nx=int(c["nx"]), ny=int(c["ny"]), d=float(c["d"]),
        cx_c=float(c["cx_c"]), u_in=float(c["u_in"]), re=float(re),
        rho0=float(c.get("rho0", 1.0)),
        top_bottom=str(c.get("top_bottom", "free_slip")),
        outlet=str(c.get("outlet", "convective")),
        trip=float(c.get("trip", 1e-3)),
        bouzidi=bool(c.get("bouzidi", True)),
        sponge_width=int(c.get("sponge_width", 32)),
        sponge_sigma=float(c.get("sponge_sigma", 0.08)),
        collision=collision,
    )


def block_stats(cd, n_blocks=4):
    blocks = np.array_split(np.asarray(cd, dtype=float), n_blocks)
    means = [float(b.mean()) for b in blocks if b.size]
    half = len(cd) // 2
    return {
        "cd_block_std": float(np.std(means)),
        "cd_drift": float(abs(np.mean(cd[:half]) - np.mean(cd[half:]))),
        "cd_block_means": means,
    }


def run_case(case, n_t, n_r, rec_every=1):
    st_lit = literature_st(case.re)
    T = case.d / (case.u_in * st_lit)
    solver = Solver(case)
    t0 = time.perf_counter()
    run_transient(solver, int(n_t * T), verbose=False)
    cd, cl, t = [], [], []
    for it in range(int(n_r * T)):
        solver.step()
        if it % rec_every == 0:
            fx, fy = solver.force()
            cdi, cli = force_coefficients(fx, fy, case.rho0, case.u_in, case.d)
            cd.append(cdi)
            cl.append(cli)
            t.append(it)
    wall = (time.perf_counter() - t0) / 60.0
    return np.asarray(cd), np.asarray(cl), np.asarray(t), wall, T


def node_probe(f, y, x, ny, nx):
    """节点 uy 与涡量 ω（中心差分），只碰 5 个节点 × 9 槽。"""
    def vel(yy, xx):
        v = f[yy, xx, :]
        rho = float(v.sum())
        if rho <= 0:
            return 0.0, 0.0
        return float((v * CX).sum() / rho), float((v * CY).sum() / rho)

    ux_c, uy_c = vel(y, x)
    _, uy_xp = vel(y, (x + 1) % nx)
    _, uy_xm = vel(y, (x - 1) % nx)
    ux_yp, _ = vel((y + 1) % ny, x)
    ux_ym, _ = vel((y - 1) % ny, x)
    omega = 0.5 * (uy_xp - uy_xm) - 0.5 * (ux_yp - ux_ym)
    return uy_c, omega


# ---------------- 单 case worker（进程池内执行，结果落缓存） ----------------
def _cache_path(key):
    return CACHE / f"{key}.json"


def _save_cache(key, payload):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = _cache_path(key)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                              default=float), encoding="utf-8")
    tmp.replace(p)


def _run_e10(task):
    """MRT vs BGK：8T 瞬态 + 16T 记录，比较力噪声/漂移/低 τ 发散。"""
    cfg, re, coll = task["cfg"], task["re"], task["coll"]
    case = make_case(cfg, re, collision=coll)
    base = {"re": re, "collision": coll, "tau": case.tau, "diverged": False}
    try:
        cd, cl, t, wall, T = run_case(case, 8, 16)
    except RuntimeError as e:
        return {**base, "diverged": True, "error": f"RuntimeError: {e}"}
    if not (np.all(np.isfinite(cd)) and np.all(np.isfinite(cl))) or cd.max() > 1e3:
        return {**base, "diverged": True, "error": "non-finite / blow-up force"}
    stats = time_stats(cd, cl)
    bs = block_stats(cd)
    st = strouhal(cl, case.d, case.u_in)
    coh = coherent_cl(cl, case.d, case.u_in)
    row = {**base, "wall_min": wall,
           "cd_mean": stats["cd_mean"], "cl_rms": stats["cl_rms"],
           "st": st["st"], "cl_peak_amp": coh["cl_peak_amp"],
           "cd_block_std": bs["cd_block_std"], "cd_drift": bs["cd_drift"],
           "cd_block_means": bs["cd_block_means"]}
    np.savez(RESULTS / f"series_v12_mrt_re{int(re)}_{coll}.npz",
             t=t, cd=cd, cl=cl, re=re, tau=case.tau, collision=coll)
    return row


def _run_e11(task):
    """Re 扫描：20T 瞬态 + 30T 记录（与 v1.1 同协议，保证可比）。"""
    cfg, re = task["cfg"], task["re"]
    case = make_case(cfg, re)
    cd, cl, t, wall, T = run_case(case, 20, 30)
    stats = time_stats(cd, cl)
    st = strouhal(cl, case.d, case.u_in)
    coh = coherent_cl(cl, case.d, case.u_in)
    st_lit = literature_st(re)
    row = {"re": re, "tau": case.tau, "cd_mean": stats["cd_mean"],
           "cl_rms": stats["cl_rms"], "cl_peak_amp": coh["cl_peak_amp"],
           "st": st["st"], "st_lit": st_lit,
           "st_rel_err": (st["st"] - st_lit) / st_lit,
           "wall_min": wall, "src": "v1.2"}
    np.savez(RESULTS / f"series_v12_re{int(re)}.npz",
             t=t, cd=cd, cl=cl, re=re, tau=case.tau,
             st=st["st"], st_lit=st_lit, cd_mean=stats["cd_mean"],
             cl_peak_amp=coh["cl_peak_amp"])
    return row


def _run_e12(task):
    """探针频谱：15T 瞬态 + 25T 记录，三站 uy/ω + 升力。"""
    cfg, re = task["cfg"], task["re"]
    case = make_case(cfg, re)
    c = cfg["case"]
    d, u_in = float(c["d"]), float(c["u_in"])
    cx_c = int(c["cx_c"])
    ny, nx = case.ny, case.nx
    y_c = int(round(case.cy))
    xs = [cx_c + int(round(s * d)) for s in STATIONS]
    T = case.d / (case.u_in * literature_st(re))
    n_t, n_r = int(15 * T), int(25 * T)
    solver = Solver(case)
    t0 = time.perf_counter()
    run_transient(solver, n_t, verbose=False)
    uy_p = [[] for _ in xs]
    om_p = [[] for _ in xs]
    cl = []
    for _ in range(n_r):
        solver.step()
        fx, fy = solver.force()
        _, cli = force_coefficients(fx, fy, case.rho0, case.u_in, case.d)
        cl.append(cli)
        for k, xp in enumerate(xs):
            uy, om = node_probe(solver.f, y_c, xp, ny, nx)
            uy_p[k].append(uy)
            om_p[k].append(om)
    wall = (time.perf_counter() - t0) / 60.0
    cl = np.asarray(cl)
    uy_p = [np.asarray(a) for a in uy_p]
    om_p = [np.asarray(a) for a in om_p]

    st_force = strouhal(cl, d, u_in)["st"]
    entry = {"re": re, "st_force": st_force, "wall_min": wall, "stations": {}}
    for s, xp, uy, om in zip(STATIONS, xs, uy_p, om_p):
        du = dual_st(cl, uy, d, u_in)
        st_om = probe_st(om, d, u_in)["st"]
        hr = harmonic_ratio(uy, d, u_in)
        hr_f = harmonic_ratio(cl, d, u_in)
        entry["stations"][f"x{s}D"] = {
            "x": xp, "st_uy": du["st_probe"], "st_omega": st_om,
            "rel_diff_force_vs_uy": du["rel_diff"],
            "harm_ratio_uy": hr["ratio"], "harm_ratio_force": hr_f["ratio"],
        }
    for (s1, s2), k1, k2 in (((1.5, 3.0), 0, 1), ((1.5, 5.0), 0, 2)):
        dx = (s2 - s1) * d
        cdly = convection_delay(uy_p[k1], uy_p[k2], d, u_in, dx=dx)
        entry[f"conv_{s1}D_{s2}D"] = {
            "u_conv_over_u": cdly["u_conv_over_u"],
            "delay_steps": cdly["delay_steps"],
        }
    np.savez(RESULTS / f"probes_v12_re{int(re)}.npz", cl=cl,
             **{f"uy_{k}": a for k, a in enumerate(uy_p)},
             **{f"om_{k}": a for k, a in enumerate(om_p)})
    return entry


def worker(task):
    """进程池入口：任何异常都转成 error 字段，绝不拖垮整批。"""
    key, kind = task["key"], task["kind"]
    t0 = time.perf_counter()
    try:
        fn = {"e10": _run_e10, "e11": _run_e11, "e12": _run_e12}[kind]
        data = fn(task)
        wall = (time.perf_counter() - t0) / 60.0
        payload = {"key": key, "kind": kind, "ok": True,
                   "data": data, "wall_min": wall}
    except Exception as e:  # noqa: BLE001
        payload = {"key": key, "kind": kind, "ok": False, "data": None,
                   "error": f"{type(e).__name__}: {e}"}
    _save_cache(key, payload)
    return payload


def build_tasks(cfg):
    tasks = []
    for re in (100.0, 300.0, 350.0):
        for coll in ("bgk", "mrt"):
            tasks.append({"kind": "e10", "key": f"e10_re{int(re)}_{coll}",
                          "cfg": cfg, "re": re, "coll": coll})
    for re in (75.0, 125.0, 175.0, 250.0):
        tasks.append({"kind": "e11", "key": f"e11_re{int(re)}",
                      "cfg": cfg, "re": re})
    for re in (100.0, 200.0):
        tasks.append({"kind": "e12", "key": f"e12_re{int(re)}",
                      "cfg": cfg, "re": re})
    return tasks


# ---------------- 汇总 / 落盘 ----------------
def collect(results, kind, key_prefix):
    out = []
    for key, p in sorted(results.items()):
        if not key.startswith(key_prefix):
            continue
        if p.get("ok") and p.get("data") is not None:
            out.append(p["data"])
        else:
            print(f"  WARN {key}: {p.get('error', 'no data')}", flush=True)
    return out


def write_mrt(mrt_rows):
    mrt_rows.sort(key=lambda r: (r["re"], r["collision"]))
    (RESULTS / "v12_mrt.json").write_text(
        json.dumps({"rows": mrt_rows}, ensure_ascii=False, indent=1,
                   default=float), encoding="utf-8")


def write_sweep(new_rows):
    merged = []
    for re in (50, 100, 150, 200, 300):
        p = RESULTS / f"baseline_metrics_v11_re{re}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))["rows"][0]
        cl_pk = r.get("cl_peak_amp")
        if cl_pk is None:
            z = np.load(RESULTS / f"series_v11_re{re}.npz")
            cl_pk = coherent_cl(z["cl"], 28.0, 0.1)["cl_peak_amp"]
        merged.append({"re": float(re), "tau": r["tau"],
                       "cd_mean": r["cd_mean"], "cl_rms": r["cl_rms"],
                       "cl_peak_amp": cl_pk, "st": r["st"],
                       "st_lit": r["st_lit"], "st_rel_err": r["st_rel_err"],
                       "src": "v1.1"})
    merged.extend(new_rows)
    merged.sort(key=lambda r: r["re"])
    (RESULTS / "v12_sweep.json").write_text(
        json.dumps({"rows": merged}, ensure_ascii=False, indent=1,
                   default=float), encoding="utf-8")
    pd.DataFrame(merged).to_csv(RESULTS / "v12_sweep_rows.csv", index=False,
                                encoding="utf-8-sig")
    return merged


def write_probes(entries):
    out = {f"re{int(e['re'])}": e for e in sorted(entries, key=lambda e: e["re"])}
    (RESULTS / "v12_probes.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    return out


def repost_harmonics(cfg):
    """谱口径升级后，从 probes_v12_re*.npz 重算谐波比，无需重跑 CFD。

    每个 E12 case 在 worker 内已把 uy/ω/cl 时程落盘，因此后处理口径可以离线刷新；
    这是「长仿真 + 轻量后处理」分离的直接收益。
    """
    c = cfg["case"]
    d, u_in = float(c["d"]), float(c["u_in"])
    changed = []
    for re in (100.0, 200.0):
        npz = RESULTS / f"probes_v12_re{int(re)}.npz"
        key = f"e12_re{int(re)}"
        cp = _cache_path(key)
        if not (npz.exists() and cp.exists()):
            continue
        z = np.load(npz)
        payload = json.loads(cp.read_text(encoding="utf-8"))
        if not payload.get("ok"):
            continue
        hr_f = harmonic_ratio(z["cl"], d, u_in)
        for k, s in enumerate(STATIONS):
            hr = harmonic_ratio(z[f"uy_{k}"], d, u_in)
            st = payload["data"]["stations"][f"x{s}D"]
            st["harm_ratio_uy"] = hr["ratio"]
            st["harm_idx_uy"] = hr["harm_idx"]
            st["harm_ratio_force"] = hr_f["ratio"]
        _save_cache(key, payload)
        changed.append(key)
    entries = []
    for key in ("e12_re100", "e12_re200"):
        cp = _cache_path(key)
        if cp.exists():
            q = json.loads(cp.read_text(encoding="utf-8"))
            if q.get("ok"):
                entries.append(q["data"])
    write_probes(entries)
    return changed


# ---------------- 图 ----------------
def make_figures(mrt_rows, sweep_rows, probe_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lbm import plots as _plots  # noqa: F401  CJK 字体 rcParams 副作用，勿删

    # fig A：E10 块 std / 漂移对比（BGK vs MRT；发散格留空）
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    res = [100, 300, 350]
    xs = np.arange(len(res))
    for k, coll in enumerate(("bgk", "mrt")):
        vals, vals2 = [], []
        for re in res:
            hit = next((r for r in mrt_rows
                        if r["re"] == re and r["collision"] == coll
                        and not r.get("diverged")), None)
            vals.append(hit["cd_block_std"] if hit else np.nan)
            vals2.append(hit["cd_drift"] if hit else np.nan)
        axes[0].bar(xs + (k - 0.5) * 0.36, vals, 0.36, label=coll)
        axes[1].bar(xs + (k - 0.5) * 0.36, vals2, 0.36, label=coll)
    for ax, tt in zip(axes, ["Cd 块间 std（力噪声）", "Cd 前后半漂移"]):
        ax.set_xticks(xs)
        ax.set_xticklabels([f"Re={r}\nτ={8.4/r+0.5:.3f}" for r in res], fontsize=8)
        ax.set_title(tt)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Cd")
    fig.tight_layout()
    fig.savefig(FIGURES / "v12_mrt_noise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig B：Re 扫描 St/Cd/Cl_peak
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    re_s = [r["re"] for r in sweep_rows]
    axes[0].plot(re_s, [r["st"] for r in sweep_rows], "o-", label="LBM")
    axes[0].plot(re_s, [r["st_lit"] for r in sweep_rows], "s--", label="Fey")
    axes[0].set_xlabel("Re"); axes[0].set_ylabel("St")
    axes[0].legend(fontsize=8); axes[0].set_title("St(Re)")
    axes[1].plot(re_s, [r["cd_mean"] for r in sweep_rows], "o-")
    axes[1].set_xlabel("Re"); axes[1].set_ylabel("Cd"); axes[1].set_title("Cd(Re)")
    axes[2].plot(re_s, [r["cl_peak_amp"] for r in sweep_rows], "o-")
    axes[2].set_xlabel("Re"); axes[2].set_ylabel("Cl_peak")
    axes[2].set_title("Cl 相干峰幅(Re)")
    fig.tight_layout()
    fig.savefig(FIGURES / "v12_re_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig C：探针谱双测法（缺失工况跳过）
    present = [k for k in ("re100", "re200") if k in probe_out]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 3.6))
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, present):
            ent = probe_out[key]
            for s, st in ent["stations"].items():
                ax.axvline(st["st_uy"], ls=":", lw=1,
                           label=f"{s} St={st['st_uy']:.3f}")
            ax.axvline(ent["st_force"], color="k", lw=1.5,
                       label=f"force St={ent['st_force']:.3f}")
            ax.set_xlabel("St"); ax.set_title(f"双测法 St 对齐 · {key}")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES / "v12_probe_st.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("图 → v12_mrt_noise.png / v12_re_sweep.png / v12_probe_st.png",
          flush=True)


# ---------------- 主流程 ----------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    workers = 12
    force = False
    if "--force" in argv:
        force = True
    if "--workers" in argv:
        workers = int(argv[argv.index("--workers") + 1])

    cfg = load_config(default_config_path())
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if "--repost" in argv:
        keys = repost_harmonics(cfg)
        print(f"--repost: 已从探针时程重算谐波比并刷新 v12_probes.json，"
              f"更新 {keys}", flush=True)
        return 0
    if "--figures-only" in argv:
        results = {}
        for t in build_tasks(cfg):
            p = _cache_path(t["key"])
            if p.exists():
                q = json.loads(p.read_text(encoding="utf-8"))
                if q.get("ok"):
                    results[t["key"]] = q
        mrt_rows = collect(results, "e10", "e10_")
        sweep_rows = write_sweep(collect(results, "e11", "e11_"))
        probe_out = write_probes(collect(results, "e12", "e12_"))
        make_figures(mrt_rows, sweep_rows, probe_out)
        print("--figures-only: 从缓存汇总并重画 3 张图（无 CFD）", flush=True)
        return 0
    tasks = build_tasks(cfg)
    if force:
        for p in CACHE.glob("*.json"):
            p.unlink()
        print("--force: 缓存已清空", flush=True)

    results, pending = {}, []
    for t in tasks:
        p = _cache_path(t["key"])
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                if payload.get("ok"):
                    results[t["key"]] = payload
                    continue
            except (OSError, ValueError):
                pass
        pending.append(t)
    print(f"任务 {len(tasks)} 个：缓存命中 {len(results)}，待跑 {len(pending)}，"
          f"并行 {min(workers, max(len(pending), 1))} 进程", flush=True)

    if pending:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as ex:
            futs = {ex.submit(worker, t): t["key"] for t in pending}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    payload = fut.result()
                except Exception as e:  # noqa: BLE001
                    payload = {"key": key, "ok": False, "data": None,
                               "error": f"worker crashed: {e}"}
                results[key] = payload
                if payload.get("ok"):
                    print(f"  OK   {key}  [{payload.get('wall_min', 0):.1f} min]",
                          flush=True)
                else:
                    print(f"  FAIL {key}: {payload.get('error')}", flush=True)
        print(f"并行批墙钟 {(time.perf_counter()-t0)/60.0:.1f} min", flush=True)

    mrt_rows = collect(results, "e10", "e10_")
    new_rows = collect(results, "e11", "e11_")
    entries = collect(results, "e12", "e12_")
    write_mrt(mrt_rows)
    sweep_rows = write_sweep(new_rows)
    probe_out = write_probes(entries)
    make_figures(mrt_rows, sweep_rows, probe_out)

    print("\n===== 汇总 =====", flush=True)
    for r in mrt_rows:
        if r.get("diverged"):
            print(f"E10 Re={r['re']:.0f} {r['collision']}: DIVERGED", flush=True)
        else:
            print(f"E10 Re={r['re']:.0f} {r['collision']}: Cd={r['cd_mean']:.3f} "
                  f"blk_std={r['cd_block_std']:.4f} drift={r['cd_drift']:.4f} "
                  f"St={r['st']:.3f}", flush=True)
    for r in sweep_rows:
        if r["src"] == "v1.2":
            print(f"E11 Re={r['re']:.0f}: Cd={r['cd_mean']:.3f} "
                  f"St={r['st']:.4f} ({r['st_rel_err']:+.1%}) "
                  f"Cl_pk={r['cl_peak_amp']:.3f}", flush=True)
    for key, ent in probe_out.items():
        print(f"E12 {key}: force St={ent['st_force']:.4f} "
              + " ".join(f"{s}={v['st_uy']:.4f}"
                         for s, v in ent["stations"].items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
