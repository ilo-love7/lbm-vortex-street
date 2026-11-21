"""诊断：固定阻塞比 β 与无量纲域长（nx=20D、cx_c=5D），只加密圆柱直径 D。

背景：v1.3 E13（D=20, nx=560 ⇒ 下游 22D，6T）在 β=10% 给 St=0.17723，
v1.4 生产（D=16, nx=320 ⇒ 下游 15D，16T）给 0.18115，差 2.2%；β=5% 差 4.8%。
两轮同时改了 D、域长与瞬态长度，必须逐项拆开，否则会把"离散口径"错记成
"维度效应"。

本脚本只回答一个问题：**在同一无量纲域长下，D=16 够不够？**
（瞬态长度的影响由 09_v14_transient_sensitivity.py 单独测。）

用法：python scripts/10_v14_d_resolution.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbm.postprocess import literature_st, strouhal   # noqa: E402
from lbm.solver import CylinderCase, Solver           # noqa: E402

T_TRANS, T_REC = 16.0, 10.0
OUT = ROOT / "results" / "v14_d_resolution.json"

# (β, D) —— nx=20D, cx_c=5D, 海绵 32（格子单位）保持不变的无量纲域长
SERIES = [(0.10, 16.0), (0.10, 20.0), (0.10, 24.0),
          (0.05, 16.0), (0.05, 20.0)]


def run(beta: float, d: float) -> dict:
    ny = int(round(d / beta))
    nx = int(round(20 * d))
    case = CylinderCase(nx=nx, ny=ny, d=d, cx_c=5.0 * d, u_in=0.1, re=100.0,
                        sponge_width=32)
    T = case.d / (case.u_in * literature_st(case.re))
    qf = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    sim = Solver(case)
    t0 = time.perf_counter()
    for _ in range(int(T_TRANS * T)):
        sim.step()
    cd, cl = [], []
    for _ in range(int(T_REC * T)):
        sim.step()
        fx, fy = sim.force()
        cd.append(fx / qf)
        cl.append(fy / qf)
    cd, cl = np.asarray(cd), np.asarray(cl)
    st = strouhal(cl, case.d, case.u_in)
    return {"beta": beta, "d": d, "ny": ny, "nx": nx,
            "downstream_over_d": (nx - 5.0 * d) / d,
            "st": float(st["st"]), "cd_mean": float(cd.mean()),
            "cl_rms": float(np.sqrt(np.mean(cl ** 2))),
            "wall_min": round((time.perf_counter() - t0) / 60, 2)}


def main() -> None:
    rows = []
    for beta, d in SERIES:
        r = run(beta, d)
        rows.append(r)
        print(f"  beta={beta:.2f} D={d:4.1f} nx={r['nx']:3d} ny={r['ny']:3d} "
              f"St={r['st']:.5f} Cd={r['cd_mean']:.4f} "
              f"({r['wall_min']:.1f} min)", flush=True)
    OUT.write_text(json.dumps({"rows": rows, "t_trans": T_TRANS,
                               "t_rec": T_REC}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("写出", OUT)


if __name__ == "__main__":
    main()
