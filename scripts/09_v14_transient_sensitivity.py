"""09 v1.4 瞬态敏感性诊断：瞬态到底要多长？

为什么单独做一个实验
--------------------
v1.4 首轮用 6T 瞬态（沿用 v1.3 的 E13/E14 协议）跑 3D-2D 对照，得到
β=20% 差 0.05%、β=5% 差 0.00%，但 **β=10% 差 2.0%**（3D St 0.1759 vs
2D 0.1795），且 3D 的 St(β) 序列非单调。看上去像"三维在低阻塞下才显现"。

真实原因是**瞬态不够，且与 3D 无关**：同一个 2D 求解器把瞬态从 6T 拉到
16T，St/Cd/Cl_rms 也涨 0.9%/3.6%/23%；16/20/24T 完全持平。换成 16T 后
3D 与 2D 的 β=10% 差 0.01%（St）/0.4%（Cd）。

这个实验锁三件事：
1. 饱和阈值：T_TRANS = 16T（6T 低估 Cl_rms 23%，不是小量）；
2. "6T→12T 的跳变"是协议问题，不能归给维度；
3. v1.3 的 E13/E14/E15 都用 6T+10T，其绝对 Cd 系统性偏低 3.6% ——
   凡跨版本比较绝对数值必须记住这一条（报告 §14.5 口径限定）。

用法：
    python scripts/09_v14_transient_sensitivity.py            # 2D 扫描（~7 min）
    python scripts/09_v14_transient_sensitivity.py --with-3d  # 追加 3D 12T（~18 min）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbm.postprocess import literature_st, strouhal   # noqa: E402
from lbm.solver3d import Case3D, Solver3D             # noqa: E402
import numpy as np                                    # noqa: E402

GEO = dict(nx=320, d=16.0, cx_c=80.0, u_in=0.1, re=100.0, sponge_width=32)
NY_2D = 160               # β = 10%，就是首轮出现 2.0% 假差异的那一档
T_LIST = (6, 12, 16, 20, 24)
OUT = ROOT / "results" / "v14_transient_sensitivity.json"


def _load_08():
    spec = importlib.util.spec_from_file_location(
        "v14", ROOT / "scripts" / "08_3d_validation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scan_2d() -> dict:
    m = _load_08()
    out = {}
    for tt in T_LIST:
        m.T_TRANS, m.T_REC = tt, 10
        r = m.run_2d(NY_2D)
        out[str(tt)] = {k: r[k] for k in ("st", "cd_mean", "cl_rms",
                                          "wake_cd_mean", "wall_min")}
        print(f"  2D T_TRANS={tt:2d}T  St={r['st']:.5f} Cd={r['cd_mean']:.4f} "
              f"Cl_rms={r['cl_rms']:.4f} ({r['wall_min']:.1f} min)", flush=True)
    return out


def run_3d_12t() -> dict:
    case = Case3D(ny=NY_2D, nz=16, **GEO)
    T = case.d / (case.u_in * literature_st(case.re))
    qf = 0.5 * case.rho0 * case.u_in ** 2 * case.d
    sim = Solver3D(case)
    for _ in range(int(12 * T)):
        sim.step()
    n_rec = int(10 * T)
    cd, cl = [], []
    for _ in range(n_rec):
        sim.step()
        fx, fy = sim.force()
        cd.append(fx / qf)
        cl.append(fy / qf)
    cd, cl = np.asarray(cd), np.asarray(cl)
    half = n_rec // 2
    out = {}
    for tag, sl in (("full", slice(None)), ("first_half", slice(0, half)),
                    ("second_half", slice(half, None))):
        st = strouhal(cl[sl], case.d, case.u_in)
        out[tag] = {"st": float(st["st"]), "cd_mean": float(cd[sl].mean()),
                    "cl_rms": float(np.sqrt(np.mean(cl[sl] ** 2)))}
        print(f"  3D 12T {tag:12s} St={out[tag]['st']:.5f} "
              f"Cd={out[tag]['cd_mean']:.4f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-3d", action="store_true")
    args = ap.parse_args()

    print("2D ny=160 瞬态扫描 ...", flush=True)
    data = {"probe": {"ny": NY_2D, "beta": 0.1, "geo": GEO,
                      "t_rec": 10, "t_list": list(T_LIST)},
            "2d": scan_2d()}
    s = data["2d"]
    base = s["6"]
    sat = s["16"]
    data["delta_6t_vs_16t"] = {
        "st_pct": 100 * (base["st"] / sat["st"] - 1),
        "cd_pct": 100 * (base["cd_mean"] / sat["cd_mean"] - 1),
        "cl_rms_pct": 100 * (base["cl_rms"] / sat["cl_rms"] - 1)}
    print("  6T 相对 16T: " + "  ".join(
        f"{k}={v:+.2f}%" for k, v in data["delta_6t_vs_16t"].items()), flush=True)

    if args.with_3d:
        print("3D 12T 对照（β=10%，ny=160，nz=16）...", flush=True)
        data["3d_12t"] = run_3d_12t()

    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1,
                              default=float), encoding="utf-8")
    print("写出", OUT)


if __name__ == "__main__":
    main()
