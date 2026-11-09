"""可视化：涡量云图、Cd/Cl 时程、升力谱、涡街动画 GIF。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from .solver import Solver


def plot_vorticity(solver: Solver, title: str, path: Path, vmax: float = 0.05):
    w = solver.vorticity()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    im = ax.imshow(w, cmap="bwr", vmin=-vmax, vmax=vmax, origin="lower")
    yy, xx = np.mgrid[0 : solver.case.ny, 0 : solver.case.nx]
    ax.contour(xx, yy, solver.cyl.astype(float), levels=[0.5], colors="k", linewidths=1.2)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [lu]")
    ax.set_ylabel("y [lu]")
    fig.colorbar(im, ax=ax, label="vorticity")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_force_series(t, cd, cl, title: str, path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), dpi=120, sharex=True)
    axes[0].plot(t, cd, lw=0.8)
    axes[0].set_ylabel("Cd")
    axes[0].set_title(title)
    axes[1].plot(t, cl, lw=0.8, color="C1")
    axes[1].set_ylabel("Cl")
    axes[1].set_xlabel("t [lu steps]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_spectrum(freqs, spec, st: float, st_lit: float, title: str, path: Path, d: float, u_in: float):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    st_arr = freqs * d / u_in
    ax.plot(st_arr, spec, lw=0.9)
    ax.axvline(st, color="C3", ls="--", label=f"measured St={st:.4f}")
    ax.axvline(st_lit, color="k", ls=":", label=f"literature St={st_lit:.4f}")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("St = fD/U")
    ax.set_ylabel("|Cl| spectrum")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def animate_vorticity(solver: Solver, n_steps: int, every: int, title: str,
                      path: Path, vmax: float = 0.05):
    """逐帧涡量动画（GIF，PillowWriter）。"""
    frames = []
    for it in range(n_steps):
        solver.step()
        if it % every == 0:
            frames.append(solver.vorticity().copy())
    fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
    im = ax.imshow(frames[0], cmap="bwr", vmin=-vmax, vmax=vmax, origin="lower", animated=True)
    yy, xx = np.mgrid[0 : solver.case.ny, 0 : solver.case.nx]
    ax.contour(xx, yy, solver.cyl.astype(float), levels=[0.5], colors="k", linewidths=1)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    def update(k):
        im.set_array(frames[k])
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=40, blit=True)
    ani.save(path, writer=animation.PillowWriter(fps=25))
    plt.close(fig)
    return len(frames)
