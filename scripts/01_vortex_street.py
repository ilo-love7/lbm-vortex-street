"""01 基线实验：Re=50/100/150/200 圆柱绕流 → Cd/Cl/St 与文献经验式对比。

记录长度按预期脱落周期自适应：T ≈ D/(U·St_lit(Re))，
transient 20T + record 30T（频谱分辨率 ΔSt/St ≈ 3%）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm import CylinderCase, Solver, literature_st, run_record, run_transient, strouhal, time_stats
from lbm.config import default_config_path, load_config
from lbm.plots import plot_force_series, plot_spectrum, plot_vorticity

ROOT = Path(__file__).resolve().parents[1]


def run_one(re: float, cfg: dict, out_dir: Path, fig_dir: Path) -> dict:
    c = cfg["case"]
    s_cfg = cfg["sim"]
    case = CylinderCase(
        nx=int(c["nx"]), ny=int(c["ny"]), d=float(c["d"]),
        cx_c=float(c["cx_c"]), u_in=float(c["u_in"]), re=float(re),
        rho0=float(c.get("rho0", 1.0)),
        top_bottom=str(c.get("top_bottom", "free_slip")),
        outlet=str(c.get("outlet", "convective")),
        trip=float(c.get("trip", 1e-3)),
    )
    st_lit = literature_st(re)
    t_period = case.d / (case.u_in * st_lit)
    n_transient = int(s_cfg.get("transient_periods", 20) * t_period)
    n_record = int(s_cfg.get("record_periods", 30) * t_period)
    rec_every = int(s_cfg.get("record_every", 5))
    print(f"\n--- Re={re:.0f}  tau={case.tau:.4f}  T≈{t_period:.0f} steps  "
          f"transient={n_transient}  record={n_record} ---", flush=True)

    solver = Solver(case)
    t0 = time.perf_counter()
    run_transient(solver, n_transient, verbose=True)
    out = run_record(solver, n_record, record_every=rec_every)
    wall = (time.perf_counter() - t0) / 60.0

    stats = time_stats(out["cd"], out["cl"])
    st = strouhal(out["cl"], case.d, case.u_in, dt=float(rec_every))
    row = {
        "re": re, "tau": case.tau, "nu": case.nu,
        "cd_mean": stats["cd_mean"], "cd_std": stats["cd_std"],
        "cl_rms": stats["cl_rms"], "cl_mean": stats["cl_mean"],
        "st": st["st"], "st_lit": st_lit, "st_err": st["st"] - st_lit,
        "st_rel_err": (st["st"] - st_lit) / st_lit,
        "t_period_lu": t_period, "n_transient": n_transient,
        "n_record": n_record, "wall_min": wall,
    }
    print(f"Re={re:.0f}: Cd={row['cd_mean']:.3f}  Cl_rms={row['cl_rms']:.4f}  "
          f"St={row['st']:.4f} (lit {st_lit:.4f}, err {row['st_err']:+.4f})  [{wall:.1f} min]",
          flush=True)

    np.savez(out_dir / f"series_re{int(re)}.npz",
             t=out["t"], cd=out["cd"], cl=out["cl"], **{k: v for k, v in row.items()})
    tag = f"Re={re:.0f}"
    plot_force_series(out["t"], out["cd"], out["cl"], f"Cd/Cl 时程 · {tag}",
                      fig_dir / f"forces_re{int(re)}.png")
    freqs, spec = st["spectrum"]
    plot_spectrum(freqs, spec, st["st"], st_lit, f"升力谱 · {tag}",
                  fig_dir / f"spectrum_re{int(re)}.png", case.d, case.u_in)
    plot_vorticity(solver, f"涡量场 · {tag}", fig_dir / f"vorticity_re{int(re)}.png")
    return row


def main():
    cfg = load_config(default_config_path())
    out_dir = ROOT / cfg["output"]["results_dir"]
    fig_dir = ROOT / cfg["output"]["figures_dir"]
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    re_list = [float(r) for r in cfg["sweep"]["re_list"]]
    if len(sys.argv) > 1 and sys.argv[1] == "--re":
        re_list = [float(sys.argv[2])]

    rows = []
    for re in re_list:
        rows.append(run_one(float(re), cfg, out_dir, fig_dir))

    df = pd.DataFrame(rows)
    suffix = f"_re{int(re_list[0])}" if len(re_list) == 1 else ""
    df.to_csv(out_dir / f"baseline_rows{suffix}.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"baseline_metrics{suffix}.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")

    print("\n===== 汇总：Cd/Cl/St vs 文献 =====")
    for r in rows:
        print(f"Re={r['re']:.0f}: Cd={r['cd_mean']:.3f}  Cl_rms={r['cl_rms']:.4f}  "
              f"St={r['st']:.4f} vs lit {r['st_lit']:.4f}  (rel {r['st_rel_err']:+.1%})")
    print(f"\n共 {len(rows)} 组 → {out_dir}")


if __name__ == "__main__":
    main()
