"""力系数归一化与时程记录。"""
from __future__ import annotations

import numpy as np


def force_coefficients(fx: float, fy: float, rho0: float, u_in: float, d: float):
    """Cd = Fx / (0.5 ρ U² D)，Cl = Fy / (0.5 ρ U² D)（格子单位、单位深度）。"""
    q = 0.5 * rho0 * u_in * u_in * d
    return fx / q, fy / q


def run_transient(solver, n_transient: int, verbose: bool = False):
    for it in range(n_transient):
        solver.step()
        if it % 500 == 0:
            if verbose:
                print(f"  transient {it}/{n_transient}", flush=True)
            _, ux, _ = solver.macros_masked()
            umax = float(np.nanmax(np.abs(ux)))
            if not np.isfinite(umax) or umax > 1.0:
                raise RuntimeError(f"发散：it={it} |ux|max={umax:.3e}（检查 τ 是否 ≥0.54）")


def run_record(solver, n_steps: int, record_every: int = 1, snapshot_every: int = 0):
    """推进 n_steps，按 record_every 记录 Cd/Cl；snapshot_every>0 时存涡量快照。"""
    c = solver.case
    cd, cl, t = [], [], []
    snaps = []
    for it in range(n_steps):
        solver.step()
        if it % record_every == 0:
            fx, fy = solver.force()
            cd_i, cl_i = force_coefficients(fx, fy, c.rho0, c.u_in, c.d)
            cd.append(cd_i)
            cl.append(cl_i)
            t.append(it)
        if snapshot_every and it % snapshot_every == 0:
            snaps.append(solver.vorticity().copy())
    out = {
        "t": np.asarray(t, dtype=float),
        "cd": np.asarray(cd, dtype=float),
        "cl": np.asarray(cl, dtype=float),
    }
    if snapshot_every:
        out["vorticity_snaps"] = np.asarray(snaps)
    return out
