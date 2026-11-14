"""06 v1.3 深化：E13 阻塞比 / E14 四口径测力 / E15 LES 与入口扰动。

  E13 阻塞比外推（回应 §6.2「偏差主因是 10% 阻塞 + 2D」与 §9 结论 2）
       固定 D=20/U=0.1/nx=560，只改域高 ny ⇒ β=D/H ∈ {20%,14.3%,10%,6.7%,5%}；
       Re=100 全五档、Re=250 三档。对 St/Cd 做 β→0 外推，把「阻塞修正」与
       「2D 约束误差」定量分离：外推残差即不可被加大计算域消除的部分。
  E14 四口径测力互证（§9 最高优先未做项）
       原域 840×280，BGK/MRT × Re=100/300，同一步同时记录：
         MEM 动量交换 / CV-node 围道上碰撞前 Π=Σcc f 面积分（朴素口径）
         / CV-link 链接穿越法 + 逐步扣除 −d/dt∫ρu（严格离散守恒）
         / WAKE 远场尾流动量亏损（严格两截面，只读守恒矩）。
       预研发现：CV-link 扣除累积项后**逐位复现 MEM**（同一恒等式的两种
       写法），CV-node 违反围道无关性，故只有 WAKE 是判定「MRT 把 Cd
       抬高 +1.3~1.9% 是伪影还是真实物理」的独立主判据。另发现经典 wake
       survey 因入口截面落在西侧海绵层内而**低估 53%**，故 WAKE 改用
       「海绵之后两 clean 截面的 (ρu²+p) 直接相减」，并保留经典口径作反例。
  E15 Smagorinsky LES + 持续入口扰动（README 未做项）
       小域 420×140/D=20：LES Re∈{200,400,800} × C_s∈{0,0.16,0.30}；
       入口 Re=200 × ε∈{0,0.05,0.15,0.30}（占 U 的 0–30%，初版 0–6%
       被西侧目标态海绵吸收、四档逐位相同，故扩大到 30% 并量化衰减）。
       判据：2D LES 是否真能「稳定化高 Re」而不只是加数值耗散；持续来流扰动
       是否改变统计吸引子与起振时间。

输出：results/v13_blockage.json / v13_cv_force.json / v13_les_inlet.json
      + 4 张 figures/v13_*.png + results/v13_case_rows.csv。

用法：
    python scripts/06_v13_deepening.py                # 增量（缓存命中即跳过）
    python scripts/06_v13_deepening.py --workers 8
    python scripts/06_v13_deepening.py --force        # 清缓存全部重算
    python scripts/06_v13_deepening.py --figures-only # 只从缓存汇总并重画图
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 必须早于 numba / numpy 导入：每进程单线程，避免多进程互抢 BLAS/_numba 线程
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm import (CylinderCase, Solver, coherent_cl, force_coefficients,
                 literature_st, run_transient, strouhal, time_stats)
from lbm.config import default_config_path, load_config
from lbm.stress import force_coefficients_cv
from lbm.v13 import (blockage_ratio, contour_consistency, fit_extrapolate_zero,
                     method_gap, onset_time, period_average, st_residual)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = RESULTS / "v13_cache"

# ---- 三个实验各自的固定协议（组内唯一变量，跨组不混用绝对数值） ----
BLK = dict(nx=560, d=20.0, cx_c=120.0, u_in=0.1, sponge_width=32)
BLK_NY = (100, 140, 200, 300, 400)
BLK_RE = {100.0: BLK_NY, 250.0: (100, 200, 400)}
CV_RE = (100.0, 300.0)
LES = dict(nx=420, ny=140, d=20.0, cx_c=100.0, u_in=0.1, sponge_width=24)
LES_RE = (200.0, 400.0, 800.0)
LES_CS = (0.0, 0.16, 0.30)
INLET_EPS = (0.0, 0.05, 0.15, 0.30)


def _q(x):
    """把 numpy 标量安全转 float（避免 json.dumps 对 np.float64 报错）。"""
    return float(x)


def base_case(cfg, re, collision="bgk"):
    """E14 原域（与 v1.2 E10 完全同配置，保证 Cd 偏移可比）。"""
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


def blk_case(re, ny):
    return CylinderCase(re=float(re), ny=int(ny), **BLK)


def les_case(re, cs):
    return CylinderCase(re=float(re), les=True, c_smag=float(cs), **LES)


def inlet_case(re, eps):
    return CylinderCase(re=float(re), inlet_eps=float(eps),
                        inlet_period_ratio=2.37, **LES)


def series_stats(case, cd, cl):
    st = strouhal(cl, case.d, case.u_in)
    coh = coherent_cl(cl, case.d, case.u_in)
    ts = time_stats(cd, cl)
    return {"cd_mean": _q(ts["cd_mean"]), "cl_rms": _q(ts["cl_rms"]),
            "st": _q(st["st"]), "st_lit": _q(literature_st(case.re)),
            "cl_peak_amp": _q(coh["cl_peak_amp"]),
            **{k: _q(v) for k, v in st_residual(st["st"],
                                                 literature_st(case.re)).items()}}


def run_series(case, n_t_periods, n_r_periods, with_cv=False, les_sample=0):
    """推进 n_t T 瞬态 + n_r T 记录；with_cv 时同步采四口径测力时程。

    四口径（E14）：
      1) MEM        —— 动量交换（solver.force）
      2) CV-node    —— 围道上碰撞前 Π=Σcc f 的面积分（朴素，方法学反例）
      3) CV-link    —— 链接穿越法 + 逐步扣除 −d/dt∫ρu（严格离散守恒，
                        应逐位复现 MEM；用于验证实现，不作独立判据）
      4) WAKE       —— 远场尾流动量亏损（只读守恒矩，唯一独立主判据）
    """
    st_lit = literature_st(case.re)
    T = case.d / (case.u_in * st_lit)
    q = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    solver = Solver(case)
    t0 = time.perf_counter()
    run_transient(solver, int(n_t_periods * T), verbose=False)
    cd, cl = [], []
    cvs, cl_cv = {}, {}          # h -> 朴素节点口径时程
    link, link_raw = {}, {}      # h -> 严格修正 / 未修正链接口径时程
    wakes = {}                    # x/D -> 严格两截面尾流口径时程
    wakes_classic = {}            # x/D -> 经典 wake survey（方法学反例）
    les_ratios = []
    radii = [h for h, _ in solver._contours]
    for h in radii:
        cvs[h] = cl_cv[h] = []
        link[h], link_raw[h] = [], []
    for s, _ in solver._wake_stations:
        wakes[str(s)] = []
        wakes_classic[str(s)] = []
    for it in range(int(n_r_periods * T)):
        # 逐步累积项必须在 step 前取 P_V(t)，step 后取 P_V(t+1)
        p0 = solver.box_momenta() if with_cv else {}
        solver.step()
        fx, fy = solver.force()
        cdi, cli = force_coefficients(fx, fy, case.rho0, case.u_in, case.d)
        cd.append(cdi)
        cl.append(cli)
        if with_cv:
            p1 = solver.box_momenta()
            for h, cfx, cfy in solver.cv_forces():
                a, b = force_coefficients_cv(cfx, cfy, case.rho0,
                                             case.u_in, case.d)
                cvs[h].append(a)
                cl_cv[h].append(b)
            for h, lx, ly in solver.link_forces():
                # F = links − [P_V(t+1) − P_V(t)]
                cx_corr = lx - (p1[h][0] - p0[h][0])
                link[h].append(cx_corr / q)
                link_raw[h].append(lx / q)
            for s, w in solver.wake_forces():
                wakes[str(s)].append(w["force"] / q)
            for s, w in solver.wake_forces(classic=True):
                wakes_classic[str(s)].append(w["force"] / q)
        if les_sample and it % les_sample == 0:
            diag = solver.les_diagnostics()
            if diag:
                les_ratios.append(diag["nu_t_over_nu0_median"])
    wall = (time.perf_counter() - t0) / 60.0
    return {"cd": np.asarray(cd), "cl": np.asarray(cl),
            "cvs": {h: np.asarray(v) for h, v in cvs.items()},
            "cl_cv": {h: np.asarray(v) for h, v in cl_cv.items()},
            "link": {h: np.asarray(v) for h, v in link.items()},
            "link_raw": {h: np.asarray(v) for h, v in link_raw.items()},
            "wakes": {k: np.asarray(v) for k, v in wakes.items()},
            "wakes_classic": {k: np.asarray(v) for k, v in wakes_classic.items()},
            "les_ratios": np.asarray(les_ratios), "T": T, "wall_min": wall}


# ---------------- 单 case worker ----------------
def _cache_path(key):
    return CACHE / f"{key}.json"


def _save_cache(key, payload):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = _cache_path(key)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                              default=float), encoding="utf-8")
    tmp.replace(p)


def _safe_run(fn, base):
    """把发散/数值异常转成结构化字段，绝不拖垮并行批。"""
    try:
        return fn()
    except RuntimeError as e:
        return {**base, "diverged": True, "error": f"RuntimeError: {e}"}
    except Exception as e:  # noqa: BLE001
        return {**base, "diverged": True,
                "error": f"{type(e).__name__}: {e}"}


def _worker_e13(task):
    re, ny = task["re"], task["ny"]
    case = blk_case(re, ny)
    base = {"re": re, "ny": ny, "beta": blockage_ratio(case.d, ny),
            "tau": _q(case.tau), "diverged": False}

    def go():
        s = run_series(case, 6, 10)
        cd, cl = s["cd"], s["cl"]
        if not (np.all(np.isfinite(cd)) and np.all(np.isfinite(cl))) \
                or np.max(np.abs(cd)) > 1e3:
            return {**base, "diverged": True, "error": "non-finite / blow-up force"}
        row = {**base, **series_stats(case, cd, cl),
               "wall_min": _q(s["wall_min"]), "T": _q(s["T"])}
        np.savez(RESULTS / f"series_v13_blk_re{int(re)}_ny{ny}.npz",
                 t=np.arange(cd.size), cd=cd, cl=cl, re=re, ny=ny,
                 beta=row["beta"], st=row["st"])
        return row

    return _safe_run(go, base)


def _worker_e14(task):
    re, coll = task["re"], task["coll"]
    cfg = task["cfg"]
    case = base_case(cfg, re, collision=coll)
    base = {"re": re, "collision": coll, "tau": _q(case.tau),
            "contour_radii": [h for h, _ in case_contours_for(case)],
            "diverged": False}

    def go():
        s = run_series(case, 6, 10, with_cv=True)
        cd, cl = s["cd"], s["cl"]
        if not (np.all(np.isfinite(cd)) and np.all(np.isfinite(cl))) \
                or np.max(np.abs(cd)) > 1e3:
            return {**base, "diverged": True, "error": "non-finite / blow-up force"}
        pm = period_average(cd, s["T"])
        st_cv, cd_cv, cd_link, cd_link_raw = {}, {}, {}, {}
        cd_wake, cd_wake_classic = {}, {}
        for h, arr in s["cvs"].items():
            p = period_average(arr, s["T"])
            cd_cv[str(h)] = _q(p["mean"])
            cd_link[str(h)] = _q(period_average(s["link"][h], s["T"])["mean"])
            cd_link_raw[str(h)] = _q(period_average(s["link_raw"][h],
                                                    s["T"])["mean"])
            st_cv[str(h)] = _q(strouhal(s["cl_cv"][h], case.d,
                                        case.u_in)["st"])
        for k, arr in s["wakes"].items():
            cd_wake[k] = _q(period_average(arr, s["T"])["mean"])
        for k, arr in s["wakes_classic"].items():
            cd_wake_classic[k] = _q(period_average(arr, s["T"])["mean"])
        classic_ratio = (float(np.mean(list(cd_wake_classic.values())))
                         / pm["mean"]) if cd_wake_classic else float("nan")
        # 恒等式校验：链接口径（含累积修正）应逐步逐位复现 MEM
        link_diff = max(float(np.max(np.abs(s["link"][h] - cd)))
                        for h in s["link"]) if s["link"] else float("nan")
        row = {**base, **series_stats(case, cd, cl),
               "cd_mem_period": _q(pm["mean"]), "cd_mem_full": _q(pm["full_mean"]),
               "phase_residual": _q(pm["phase_residual"]),
               "periods": int(pm["periods"]),
               "cd_cv": cd_cv, "st_cv": st_cv,
               "cd_link": cd_link, "cd_link_raw": cd_link_raw,
               "cd_wake": cd_wake,
               "cd_wake_classic": cd_wake_classic,
               "classic_over_strict": _q(classic_ratio),
               "link_vs_mem_maxdiff": _q(link_diff),
               "contour": contour_consistency(cd_cv),
               "link_contour": contour_consistency(cd_link),
               "wake": contour_consistency(cd_wake),
               "gap": method_gap(pm["mean"],
                                 float(np.mean(list(cd_cv.values())))),
               "gap_wake": method_gap(pm["mean"],
                                      float(np.mean(list(cd_wake.values())))),
               "wall_min": _q(s["wall_min"]), "T": _q(s["T"])}
        np.savez(RESULTS / f"series_v13_cv_re{int(re)}_{coll}.npz",
                 cd_mem=cd, cl_mem=cl, T=s["T"],
                 **{f"cd_cv_{h}": v for h, v in s["cvs"].items()},
                 **{f"cl_cv_{h}": v for h, v in s["cl_cv"].items()},
                 **{f"cd_link_{h}": v for h, v in s["link"].items()},
                 **{f"cd_wake_{k}": v for k, v in s["wakes"].items()},
                 **{f"cd_wake_classic_{k}": v
                    for k, v in s["wakes_classic"].items()})
        return row

    return _safe_run(go, base)


def case_contours_for(case):
    """与 Solver 内一致的围道生成（worker 内需在构造 Solver 前拿到半径列表）。"""
    from lbm.stress import contours_for_case
    return contours_for_case(case)


def _worker_e15_les(task):
    re, cs = task["re"], task["cs"]
    case = les_case(re, cs)
    base = {"re": re, "cs": cs, "tau0": _q(case.tau), "diverged": False}

    def go():
        s = run_series(case, 5, 8, les_sample=400)
        cd, cl = s["cd"], s["cl"]
        if not (np.all(np.isfinite(cd)) and np.all(np.isfinite(cl))) \
                or np.max(np.abs(cd)) > 1e3:
            return {**base, "diverged": True, "error": "non-finite / blow-up force"}
        rr = s["les_ratios"]
        row = {**base, **series_stats(case, cd, cl),
               "nu_ratio_median": _q(np.median(rr)) if rr.size else 0.0,
               "nu_ratio_max": _q(np.max(rr)) if rr.size else 0.0,
               "wall_min": _q(s["wall_min"])}
        tag = f"re{int(re)}_cs{int(round(cs * 100)):03d}"
        np.savez(RESULTS / f"series_v13_les_{tag}.npz",
                 cd=cd, cl=cl, re=re, cs=cs)
        return row

    return _safe_run(go, base)


def _worker_e15_inlet(task):
    re, eps = task["re"], task["eps"]
    case = inlet_case(re, eps)
    base = {"re": re, "eps": eps, "tau": _q(case.tau), "diverged": False}

    def go():
        s = run_series(case, 5, 8)
        cd, cl = s["cd"], s["cl"]
        if not (np.all(np.isfinite(cd)) and np.all(np.isfinite(cl))) \
                or np.max(np.abs(cd)) > 1e3:
            return {**base, "diverged": True, "error": "non-finite / blow-up force"}
        on = onset_time(cl, s["T"])
        row = {**base, **series_stats(case, cd, cl),
               "onset_steps": int(on["onset_steps"]),
               "onset_periods": _q(on["onset_periods"]),
               "sat_rms": _q(on["sat_rms"]),
               "wall_min": _q(s["wall_min"])}
        tag = f"eps{int(round(eps * 1000)):03d}"
        np.savez(RESULTS / f"series_v13_inlet_{tag}.npz",
                 cd=cd, cl=cl, re=re, eps=eps)
        return row

    return _safe_run(go, base)


def worker(task):
    key, kind = task["key"], task["kind"]
    t0 = time.perf_counter()
    try:
        fn = {"e13": _worker_e13, "e14": _worker_e14,
              "e15_les": _worker_e15_les, "e15_inlet": _worker_e15_inlet}[kind]
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
    for re, nys in BLK_RE.items():
        for ny in nys:
            tasks.append({"kind": "e13", "key": f"e13_re{int(re)}_ny{ny}",
                          "re": re, "ny": ny})
    for re in CV_RE:
        for coll in ("bgk", "mrt"):
            tasks.append({"kind": "e14", "key": f"e14_re{int(re)}_{coll}",
                          "re": re, "coll": coll, "cfg": cfg})
    for re in LES_RE:
        for cs in LES_CS:
            tasks.append({"kind": "e15_les",
                          "key": f"e15_les_re{int(re)}_cs{int(round(cs*100)):03d}",
                          "re": re, "cs": cs})
    for eps in INLET_EPS:
        tasks.append({"kind": "e15_inlet",
                      "key": f"e15_inlet_eps{int(round(eps*1000)):03d}",
                      "re": 200.0, "eps": eps})
    return tasks


def collect(results, prefix):
    out = []
    for key, p in sorted(results.items()):
        if not key.startswith(prefix):
            continue
        if p.get("ok") and p.get("data") is not None:
            out.append(p["data"])
        else:
            print(f"  WARN {key}: {p.get('error', 'no data')}", flush=True)
    return out


# ---------------- 汇总 ----------------
def write_blockage(rows):
    rows = sorted(rows, key=lambda r: (r["re"], r["beta"]))
    by_re = {}
    for re in sorted({r["re"] for r in rows}):
        sub = [r for r in rows if r["re"] == re and not r.get("diverged")]
        if len(sub) < 2:
            by_re[str(int(re))] = {"n": len(sub)}
            continue
        b = [r["beta"] for r in sub]
        entry = {"n": len(sub),
                 "beta": b,
                 "st": [r["st"] for r in sub],
                 "cd": [r["cd_mean"] for r in sub],
                 "st_lit": sub[0]["st_lit"],
                 "st_fit_lin": fit_extrapolate_zero(b, [r["st"] for r in sub], 1),
                 "cd_fit_lin": fit_extrapolate_zero(b, [r["cd_mean"] for r in sub], 1)}
        # 二次外推要求 ≥4 点：正好 3 点时 polyfit(2) 是精确插值，
        # 外推到 β=0 会被端点噪声放大，属病态口径（故不给）。
        if len(sub) >= 4:
            entry["st_fit_quad"] = fit_extrapolate_zero(b, [r["st"] for r in sub], 2)
            entry["cd_fit_quad"] = fit_extrapolate_zero(b,
                                                        [r["cd_mean"] for r in sub], 2)
        s0 = entry["st_fit_lin"]["intercept"]
        entry["residual_st_rel"] = _q((s0 - sub[0]["st_lit"]) / sub[0]["st_lit"])
        by_re[str(int(re))] = entry
    out = {"rows": rows, "extrapolation": by_re,
           "protocol": {"d": BLK["d"], "nx": BLK["nx"], "u_in": BLK["u_in"],
                        "transient_periods": 6, "record_periods": 10}}
    (RESULTS / "v13_blockage.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    return out


def write_cv(rows):
    rows = sorted(rows, key=lambda r: (r["re"], r["collision"]))
    groups = {}
    for re in CV_RE:
        sub = {r["collision"]: r for r in rows
               if r["re"] == re and not r.get("diverged")}
        if len(sub) < 2:
            continue
        bgk, mrt = sub["bgk"], sub["mrt"]
        mean_of = lambda r, k: float(np.mean(list(r[k].values())))
        cv_m, lk_m, wk_m = (mean_of(bgk, "cd_cv"),
                            mean_of(bgk, "cd_link"),
                            mean_of(bgk, "cd_wake"))
        cv_M, lk_M, wk_M = (mean_of(mrt, "cd_cv"),
                            mean_of(mrt, "cd_link"),
                            mean_of(mrt, "cd_wake"))
        dmem = mrt["cd_mem_period"] - bgk["cd_mem_period"]
        groups[str(int(re))] = {
            "cd_mem_bgk": bgk["cd_mem_period"],
            "cd_mem_mrt": mrt["cd_mem_period"],
            "delta_mem": _q(dmem),
            "delta_mem_rel": _q(dmem / bgk["cd_mem_period"]),
            "cd_cv_bgk": _q(cv_m), "cd_cv_mrt": _q(cv_M),
            "delta_cv": _q(cv_M - cv_m),
            "delta_cv_rel": _q((cv_M - cv_m) / cv_m),
            "cd_link_bgk": _q(lk_m), "cd_link_mrt": _q(lk_M),
            "delta_link_rel": _q((lk_M - lk_m) / lk_m),
            "cd_wake_bgk": _q(wk_m), "cd_wake_mrt": _q(wk_M),
            "delta_wake": _q(wk_M - wk_m),
            "delta_wake_rel": _q((wk_M - wk_m) / wk_m),
            "gap_bgk": bgk["gap"], "gap_mrt": mrt["gap"],
            "gap_wake_bgk": bgk["gap_wake"], "gap_wake_mrt": mrt["gap_wake"],
            "link_vs_mem_maxdiff": _q(max(bgk["link_vs_mem_maxdiff"],
                                          mrt["link_vs_mem_maxdiff"])),
            "classic_over_strict": (lambda v: _q(float(np.nanmean(v)))
                                    if np.any(np.isfinite(v)) else None)(
                [bgk.get("classic_over_strict", float("nan")),
                 mrt.get("classic_over_strict", float("nan"))]),
            "contour_spread_node": _q(max(bgk["contour"]["rel_spread"],
                                          mrt["contour"]["rel_spread"])),
            "contour_spread_link": _q(max(bgk["link_contour"]["rel_spread"],
                                          mrt["link_contour"]["rel_spread"])),
            "wake_spread": _q(max(bgk["wake"]["rel_spread"],
                                  mrt["wake"]["rel_spread"])),
        }
    out = {"rows": rows, "groups": groups,
           "protocol": {"transient_periods": 6, "record_periods": 10,
                        "contours": "3 正方形围道，h=R+{4,8,14}",
                        "wake_stations_x_over_D": [3.0, 6.0, 10.0],
                        "wake_reference_section": "西侧海绵之后 2 格",
                        "methods": ["MEM", "CV-node(朴素Π)",
                                    "CV-link(严格守恒+累积修正)",
                                    "WAKE(严格两截面/守恒矩)",
                                    "WAKE-classic(方法学反例)"]}}
    (RESULTS / "v13_cv_force.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    return out


def write_les_inlet(les_rows, inlet_rows):
    les_rows = sorted(les_rows, key=lambda r: (r["re"], r["cs"]))
    inlet_rows = sorted(inlet_rows, key=lambda r: r["eps"])
    grid = {}
    for re in LES_RE:
        grid[str(int(re))] = {
            f"cs{r['cs']:.2f}": {"diverged": r.get("diverged", False),
                                 "st": r.get("st"), "cd": r.get("cd_mean"),
                                 "cl_peak": r.get("cl_peak_amp"),
                                 "nu_ratio_median": r.get("nu_ratio_median")}
            for r in les_rows if r["re"] == re}
    out = {"les_rows": les_rows, "inlet_rows": inlet_rows, "grid": grid,
           "protocol": {"domain": LES, "transient_periods": 5,
                        "record_periods": 8,
                        "inlet_forcing_period_ratio": 2.37}}
    (RESULTS / "v13_les_inlet.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    return out


def write_rows_csv(blockage, cv, les):
    rows = []
    for r in blockage["rows"]:
        rows.append({"exp": "E13", "case": f"Re{int(r['re'])}_ny{r['ny']}",
                     "re": r["re"], "var": r["beta"],
                     "st": r.get("st"), "cd": r.get("cd_mean"),
                     "diverged": r.get("diverged", False)})
    for r in cv["rows"]:
        rows.append({"exp": "E14", "case": f"Re{int(r['re'])}_{r['collision']}",
                     "re": r["re"], "var": r["collision"],
                     "st": r.get("st"), "cd": r.get("cd_mem_period"),
                     "diverged": r.get("diverged", False)})
    for r in les["les_rows"]:
        rows.append({"exp": "E15-LES", "case": f"Re{int(r['re'])}_Cs{r['cs']:.2f}",
                     "re": r["re"], "var": r["cs"],
                     "st": r.get("st"), "cd": r.get("cd_mean"),
                     "diverged": r.get("diverged", False)})
    for r in les["inlet_rows"]:
        rows.append({"exp": "E15-INLET", "case": f"eps{r['eps']:.3f}",
                     "re": r["re"], "var": r["eps"],
                     "st": r.get("st"), "cd": r.get("cd_mean"),
                     "diverged": r.get("diverged", False)})
    pd.DataFrame(rows).to_csv(RESULTS / "v13_case_rows.csv", index=False,
                              encoding="utf-8-sig")
    return rows


def make_figures(blockage, cv, les):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lbm import plots as _plots  # noqa: F401  CJK 字体 rcParams 副作用，勿删

    # ---- fig A 阻塞比外推 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for re, ent in blockage["extrapolation"].items():
        if ent.get("n", 0) < 2:
            continue
        b = np.asarray(ent["beta"])
        axes[0].plot(b * 100, ent["st"], "o-", label=f"Re={re}")
        fit = ent["st_fit_lin"]
        bb = np.linspace(0, b.max() * 1.05, 50)
        axes[0].plot(bb * 100, fit["intercept"] + fit["slope"] * bb, "--",
                     alpha=0.5)
        axes[0].plot(0, fit["intercept"], "*", ms=10)
        axes[0].axhline(ent["st_lit"], color="k", ls=":", lw=1)
        axes[1].plot(b * 100, ent["cd"], "s-", label=f"Re={re}")
        axes[2].bar([f"Re={re}"], [ent["residual_st_rel"] * 100])
    axes[0].set_xlabel("阻塞比 β = D/H [%]")
    axes[0].set_ylabel("St")
    axes[0].set_title("St(β) 与 β→0 外推（★）/ 文献（虚线）")
    axes[1].set_xlabel("β [%]")
    axes[1].set_ylabel("Cd")
    axes[1].set_title("Cd(β)")
    axes[2].set_ylabel("β→0 残余 St 偏差 [%]")
    axes[2].set_title("加大计算域消不掉的部分 = 2D 约束误差")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "v13_blockage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- fig B 四口径 ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    groups = cv["groups"]
    keys = sorted(groups, key=int)
    if keys:
        x = np.arange(len(keys))
        w = 0.25
        series = (("delta_mem_rel", "MEM（动量交换）", "tab:blue"),
                  ("delta_cv_rel", "CV-node（朴素 Π）", "tab:orange"),
                  ("delta_wake_rel", "WAKE（独立判据）", "tab:green"))
        for j, (k_, lab, col) in enumerate(series):
            axes[0].bar(x + (j - 1) * w,
                        [groups[k][k_] * 100 for k in keys], w,
                        label=lab, color=col)
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"Re={k}" for k in keys])
        axes[0].set_ylabel("MRT−BGK Cd 变化 [%]")
        axes[0].set_title("三口径下的 MRT 偏移（WAKE 定真伪）")
        axes[0].legend(fontsize=7)
        allrows = [r for r in cv["rows"] if not r.get("diverged")]
        mem = [r["cd_mem_period"] for r in allrows]
        wake = [float(np.mean(list(r["cd_wake"].values()))) for r in allrows]
        axes[1].scatter(mem, wake, c="tab:green", label="WAKE")
        axes[1].scatter(mem,
                        [float(np.mean(list(r["cd_link"].values())))
                         for r in allrows], c="tab:red", marker="x",
                        label="CV-link")
        lim = [min(mem) * 0.9, max(mem) * 1.1]
        axes[1].plot(lim, lim, "k--", lw=1)
        axes[1].set_xlabel("Cd · MEM")
        axes[1].set_ylabel("Cd · 独立口径")
        axes[1].set_title("CV-link 逐位复现 MEM；WAKE 才独立")
        axes[1].legend(fontsize=7)
        spreads = (("contour_spread_node", "CV-node 围道", "tab:orange"),
                   ("contour_spread_link", "CV-link 围道", "tab:red"),
                   ("wake_spread", "WAKE 站位", "tab:green"))
        for j, (k_, lab, col) in enumerate(spreads):
            axes[2].bar(x + (j - 1) * w,
                        [groups[k][k_] * 100 for k in keys], w,
                        label=lab, color=col)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"Re={k}" for k in keys])
        axes[2].set_ylabel("围道/站位间相对分散 [%]")
        axes[2].set_title("守恒律检验：朴素 Π 违反围道无关性")
        axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "v13_cv_force.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- fig C LES ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for re in LES_RE:
        sub = [r for r in les["les_rows"] if r["re"] == re
               and not r.get("diverged")]
        if not sub:
            continue
        cs = [r["cs"] for r in sub]
        axes[0].plot(cs, [r["nu_ratio_median"] for r in sub], "o-",
                     label=f"Re={int(re)}")
        axes[1].plot(cs, [r["cl_peak_amp"] for r in sub], "s-",
                     label=f"Re={int(re)}")
    axes[0].set_xlabel("Smagorinsky C_s")
    axes[0].set_ylabel("median(nu_t/nu0)")
    axes[0].set_title("亚格子耗散占比")
    axes[1].set_xlabel("C_s")
    axes[1].set_ylabel("Cl 相干峰幅")
    axes[1].set_title("LES 对升力的阻尼")
    mat = np.zeros((len(LES_RE), len(LES_CS)))
    for i, re in enumerate(LES_RE):
        for j, cs in enumerate(LES_CS):
            hit = next((r for r in les["les_rows"]
                        if r["re"] == re and r["cs"] == cs), None)
            mat[i, j] = 0 if (hit and not hit.get("diverged")) else (
                1 if hit else -1)
    axes[2].imshow(mat, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    axes[2].set_xticks(range(len(LES_CS)))
    axes[2].set_xticklabels([f"{c:.2f}" for c in LES_CS])
    axes[2].set_yticks(range(len(LES_RE)))
    axes[2].set_yticklabels([f"Re={int(r)}" for r in LES_RE])
    axes[2].set_title("稳定(绿)/发散(红) 矩阵")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "v13_les.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- fig D 入口扰动 ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    rows = [r for r in les["inlet_rows"] if not r.get("diverged")]
    eps = [r["eps"] * 100 for r in rows]
    axes[0].plot(eps, [r["st"] for r in rows], "o-", label="St")
    axes[0].axhline(rows[0]["st_lit"] if rows else 0, color="k", ls=":",
                    label="文献 St")
    axes[0].set_xlabel("入口扰动 ε [%U]")
    axes[0].set_ylabel("St")
    axes[0].set_title("持续来流扰动 → St")
    axes[0].legend(fontsize=8)
    axes[1].plot(eps, [r["onset_periods"] for r in rows], "s-",
                 color="tab:red")
    axes[1].set_xlabel("ε [%U]")
    axes[1].set_ylabel("起振时间 [脱落周期]")
    axes[1].set_title("入口扰动 → 起振")
    fig.tight_layout()
    fig.savefig(FIGURES / "v13_inlet.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("图 → v13_blockage.png / v13_cv_force.png / v13_les.png / "
          "v13_inlet.png", flush=True)


# ---------------- 主流程 ----------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    workers = 12
    if "--workers" in argv:
        workers = int(argv[argv.index("--workers") + 1])
    cfg = load_config(default_config_path())
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if "--force" in argv:
        for p in CACHE.glob("*.json"):
            p.unlink()
        print("--force: 缓存已清空", flush=True)

    tasks = build_tasks(cfg)
    results = {}
    for t in tasks:
        p = _cache_path(t["key"])
        if p.exists():
            try:
                q = json.loads(p.read_text(encoding="utf-8"))
                if q.get("ok"):
                    results[t["key"]] = q
            except (OSError, ValueError):
                pass

    if "--figures-only" not in argv:
        pending = [t for t in tasks if t["key"] not in results]
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
                    tag = "OK  " if payload.get("ok") else "FAIL"
                    print(f"  {tag} {key} "
                          f"[{payload.get('wall_min', 0):.1f} min] "
                          f"{payload.get('error', '')}", flush=True)
            print(f"并行批墙钟 {(time.perf_counter()-t0)/60.0:.1f} min", flush=True)

    blk = write_blockage(collect(results, "e13_"))
    cv = write_cv(collect(results, "e14_"))
    les = write_les_inlet(collect(results, "e15_les_"),
                          collect(results, "e15_inlet_"))
    write_rows_csv(blk, cv, les)
    make_figures(blk, cv, les)

    print("\n===== 汇总 =====", flush=True)
    for re, ent in blk["extrapolation"].items():
        if "st_fit_lin" in ent:
            print(f"E13 Re={re}: St(β→0)={ent['st_fit_lin']['intercept']:.4f} "
                  f"lit={ent['st_lit']:.4f} 残余={ent['residual_st_rel']:+.2%} "
                  f"(R²={ent['st_fit_lin']['r2']:.3f})", flush=True)
    for re, g in cv["groups"].items():
        print(f"E14 Re={re}: ΔCd_MEM={g['delta_mem_rel']:+.2%} "
              f"ΔCd_CV-node={g['delta_cv_rel']:+.2%} "
              f"ΔCd_WAKE={g['delta_wake_rel']:+.2%} "
              f"link≡MEM 最大差={g['link_vs_mem_maxdiff']:.2e} "
              f"node 围道分散={g['contour_spread_node']:.2%} "
              f"经典WAKE/严格={g['classic_over_strict']:.3f}", flush=True)
    for r in les["les_rows"]:
        if r.get("diverged"):
            desc = "DIVERGED"
        else:
            desc = ("St={:.4f} Cd={:.3f} nu_t/nu0={:.3f}".format(
                r["st"], r["cd_mean"], r["nu_ratio_median"]))
        print("E15-LES Re={} Cs={:.2f}: {}".format(
            int(r["re"]), r["cs"], desc), flush=True)
    for r in les["inlet_rows"]:
        print(f"E15-INLET ε={r['eps']:.3f}: St={r.get('st', float('nan')):.4f} "
              f"onset={r.get('onset_periods', float('nan')):.2f}T", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
