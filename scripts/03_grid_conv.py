"""03 网格收敛：Re=100、D=21/28/42（几何相似：nx=30D, ny=10D, cx_c=6.07D, U=0.1）。

协议（比基线短，网格收敛看趋势不需要长谱）：transient 12T + record 20T。
输出 results/grid_conv_d{D}.csv + series；三网格 Richardson 外推 φ(h)=φ0+C·h^p 在收口时拟合。

用法：python scripts/03_grid_conv.py --d 21
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm import CylinderCase, Solver, literature_st, run_record, run_transient, strouhal, time_stats
from lbm.postprocess import coherent_cl

ROOT = Path(__file__).resolve().parents[1]


def run_one(d: float) -> dict:
    re = 100.0
    case = CylinderCase(
        nx=int(round(30 * d)), ny=int(round(10 * d)),
        d=float(d), cx_c=6.07 * d, u_in=0.1, re=re,
        bouzidi=True, sponge_width=int(round(32 * d / 28)), sponge_sigma=0.08,
    )
    st_lit = literature_st(re)
    t_period = case.d / (case.u_in * st_lit)
    n_transient = int(12 * t_period)
    n_record = int(20 * t_period)
    print(f"\n--- grid D={d:.0f}  {case.nx}x{case.ny}  tau={case.tau:.4f}  "
          f"T≈{t_period:.0f}  transient={n_transient}  record={n_record} ---", flush=True)

    solver = Solver(case)
    t0 = time.perf_counter()
    run_transient(solver, n_transient, verbose=True)
    out = run_record(solver, n_record, record_every=1)
    wall = (time.perf_counter() - t0) / 60.0

    cd, cl = out["cd"], out["cl"]
    T = t_period
    nblk = int(len(cd) // T)
    blocks = np.array([cd[int(k * T):int((k + 1) * T)].mean() for k in range(nblk)])
    st = strouhal(cl, case.d, case.u_in, dt=1.0)
    coh = coherent_cl(cl, case.d, case.u_in, dt=1.0)
    row = {
        "d": d, "nx": case.nx, "ny": case.ny, "tau": case.tau,
        "cd_mean": float(cd.mean()), "cd_median": float(np.median(cd)),
        "cd_blk": float(blocks.mean()), "cd_blk_std": float(blocks.std()),
        "cl_rms": float(np.sqrt((cl ** 2).mean())),
        "cl_peak_amp": coh["cl_peak_amp"],
        "st": float(st["st"]), "st_lit": st_lit,
        "n_transient": n_transient, "n_record": n_record, "wall_min": wall,
    }
    print(f"D={d:.0f}: Cd_blk={row['cd_blk']:.3f}±{row['cd_blk_std']:.3f}  "
          f"Cl_rms={row['cl_rms']:.4f}  St={row['st']:.4f} (lit {st_lit:.4f})  "
          f"[{wall:.1f} min]", flush=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / f"grid_conv_series_d{int(d)}.npz",
             t=out["t"], cd=cd, cl=cl, **row)
    (out_dir / f"grid_conv_d{int(d)}.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return row


if __name__ == "__main__":
    ds = [21.0, 28.0, 42.0]
    if "--d" in sys.argv:
        ds = [float(sys.argv[sys.argv.index("--d") + 1])]
    for d in ds:
        run_one(d)
