"""04 收口分析：v11 基线统计 + v1.0 对比 + 网格收敛 Richardson 外推。

读 results/series_v11_re*.npz 与 results/grid_conv_*.json，输出：
- results/v11_stats.json（块均值/负值占比/Cl 相干幅等）
- stdout 报告表格（可直接贴入 reports/基准评测报告-N5.md §5）

用法：python scripts/04_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbm.postprocess import coherent_cl, literature_st, strouhal

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# v1.0 块均值口径（2026-11-09 收口值，用于对比表）
V10 = {
    50: dict(cd=1.797, st=0.1381, cl_peak=0.035, neg=0.0),
    100: dict(cd=1.638, st=0.1776, cl_peak=0.394, neg=0.35),
    150: dict(cd=1.713, st=0.1958, cl_peak=0.599, neg=0.49),
    200: dict(cd=1.748, st=0.2068, cl_peak=0.868, neg=0.50),
}

LIT_CD = {50: 1.47, 100: 1.35, 150: 1.35, 200: 1.35, 300: 1.33}
LIT_CL = {50: 0.0, 100: 0.33, 150: 0.55, 200: 0.70, 300: 0.80}


def block_stats(cd: np.ndarray, cl: np.ndarray, T: float) -> dict:
    nblk = int(len(cd) // T)
    blocks = np.array([cd[int(k * T):int((k + 1) * T)].mean() for k in range(nblk)])
    q1, q2 = int(len(cd) / 3), int(2 * len(cd) / 3)
    return {
        "cd_mean": float(cd.mean()), "cd_median": float(np.median(cd)),
        "cd_blk": float(blocks.mean()), "cd_blk_std": float(blocks.std()),
        "cd_seg": [float(cd[:q1].mean()), float(cd[q1:q2].mean()), float(cd[q2:].mean())],
        "neg": float((cd < 0).mean()),
        "neg_seg": [float((cd[:q1] < 0).mean()), float((cd[q1:q2] < 0).mean()),
                    float((cd[q2:] < 0).mean())],
        "cl_rms": float(np.sqrt((cl ** 2).mean())),
    }


def main():
    out = {}
    print("===== v1.1 基线（Bouzidi+海绵）=====\n")
    print("| Re | tau | Cd blk | Cd std | neg% | Cl_rms | Cl peak | St | St lit | err |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for re in (50, 100, 150, 200, 300):
        p = RES / f"series_v11_re{re}.npz"
        if not p.exists():
            print(f"| {re} | - | (缺) | | | | | | | |")
            continue
        z = np.load(p)
        cd, cl = z["cd"], z["cl"]
        T = float(z["t_period_lu"])
        bs = block_stats(cd, cl, T)
        st = strouhal(cl, float(z["d"]) if "d" in z.files else 28.0, 0.1, dt=1.0)
        coh = coherent_cl(cl, 28.0, 0.1, dt=1.0)
        st_lit = literature_st(re)
        tau = float(z["tau"])
        row = dict(re=re, tau=tau, **bs,
                   cl_peak=coh["cl_peak_amp"], st=st["st"], st_lit=st_lit,
                   st_rel=(st["st"] - st_lit) / st_lit)
        out[str(re)] = row
        print(f"| {re} | {tau:.3f} | {bs['cd_blk']:.3f} | {bs['cd_blk_std']:.3f} | "
              f"{bs['neg']:.1%} | {bs['cl_rms']:.3f} | {coh['cl_peak_amp']:.3f} | "
              f"{st['st']:.4f} | {st_lit:.4f} | {row['st_rel']:+.1%} |")

    print("\n===== v1.0 → v1.1 对比 =====\n")
    print("| Re | Cd v1.0 | Cd v1.1 | Cd lit | err v1.0 | err v1.1 | St v1.0 | St v1.1 | neg% v1.0→v1.1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for re in (50, 100, 150, 200):
        if str(re) not in out:
            continue
        r = out[str(re)]
        v = V10[re]
        lit = LIT_CD[re]
        print(f"| {re} | {v['cd']:.2f} | {r['cd_blk']:.2f} | {lit:.2f} | "
              f"{(v['cd']-lit)/lit:+.1%} | {(r['cd_blk']-lit)/lit:+.1%} | "
              f"{v['st']:.3f} | {r['st']:.3f} | {v['neg']:.0%} → {r['neg']:.0%} |")

    print("\n===== 网格收敛（Re=100）=====\n")
    gpts = []
    for d in (21, 28, 42):
        p = RES / f"grid_conv_d{d}.json"
        if not p.exists():
            print(f"D={d}: (缺)")
            continue
        g = json.loads(p.read_text(encoding="utf-8"))
        gpts.append(g)
        print(f"D={d}: {g['nx']}x{g['ny']}  Cd_blk={g['cd_blk']:.3f}  "
              f"Cl_peak={g['cl_peak_amp']:.3f}  St={g['st']:.4f} (lit {g['st_lit']:.4f})")
    if len(gpts) >= 3:
        # Richardson: φ(h)=φ0+C·h^p，三网格解 (φ0, C, p)
        hs = np.array([1.0 / g["d"] for g in gpts])
        for key in ("cd_blk", "st", "cl_peak_amp"):
            ph = np.array([g[key] for g in gpts])
            # 网格 1,2,3：p = ln((φ1−φ2)/(φ2−φ3)) / ln(r)，r=h1/h2=h2/h3（几何相似时）
            r = hs[0] / hs[1]
            if abs((hs[1] / hs[2]) - r) > 0.05:
                print(f"  {key}: 网格比非均匀，跳过 Richardson")
                continue
            d12, d23 = ph[0] - ph[1], ph[1] - ph[2]
            if d12 * d23 <= 0:
                print(f"  {key}: 差分非单调（{d12:+.4f}, {d23:+.4f}），外推无意义")
                continue
            p_order = float(np.log(d12 / d23) / np.log(r))
            phi0 = float(ph[2] + d23 / (r ** p_order - 1.0)) if p_order != 0 else float("nan")
            print(f"  {key}: 观测阶 p≈{p_order:.2f}  Richardson 外推 φ0={phi0:.4f}  "
                  f"(细网格值 {ph[2]:.4f}, 相对移动 {(phi0-ph[2])/abs(phi0):+.2%})")

    (RES / "v11_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {RES / 'v11_stats.json'}")


if __name__ == "__main__":
    main()
