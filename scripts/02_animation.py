"""02 涡街动画：Re=100 涡量场 GIF（每 25 步一帧）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm import CylinderCase, Solver
from lbm.config import default_config_path, load_config
from lbm.plots import animate_vorticity

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = load_config(default_config_path())
    c = cfg["case"]
    fig_dir = ROOT / cfg["output"]["figures_dir"]
    fig_dir.mkdir(exist_ok=True)
    case = CylinderCase(
        nx=int(c["nx"]), ny=int(c["ny"]), d=float(c["d"]),
        cx_c=float(c["cx_c"]), u_in=float(c["u_in"]), re=100.0,
        top_bottom=str(c.get("top_bottom", "free_slip")),
        outlet=str(c.get("outlet", "convective")),
        trip=float(c.get("trip", 1e-3)),
    )
    print("生成 Re=100 涡街动画（15000 步，每 25 步一帧）...", flush=True)
    solver = Solver(case)
    n = animate_vorticity(solver, n_steps=15000, every=25,
                          title="卡门涡街 Re=100（涡量）",
                          path=fig_dir / "vortex_street_re100.gif")
    print(f"完成：{n} 帧 → {fig_dir / 'vortex_street_re100.gif'}")


if __name__ == "__main__":
    main()
