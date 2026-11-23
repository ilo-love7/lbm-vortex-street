"""17 v1.8 低 τ 精度校准：固定离散手段，扫描 τ→0.5 的 ν_eff 偏差。

动机
----
v1.4 E19 的剪切波反测 ν 最低只到 τ=0.55（误差 0.55–1.28%）。
N5 的 Re=400 用 τ=0.512，低于 v1.1 的 0.53 红线；此前只能说
"没炸 ≠ 够准"，却没有独立判据把数值误差量化出来。

做法
----
用 ``shear_wave_decay`` 的解析初场 u=A sin(kx)，对 τ=0.505…0.8 逐档反测
ν_fit，得到 ``ν_fit/ν_input − 1`` 的 τ 依赖。再把该偏差换算成有效 Re 的
相对偏移（Re=UD/ν ⇒ ΔRe/Re = −Δν/ν），给 Re=400 的力/频率读数标一个
**数值误差上限**。这不是修正流场，只是误差预算。

关键：低 τ 档衰减极慢，步数按 7.5 个 e-fold 自适应给出，不能沿用
E19 的固定 4000 步（否则 τ=0.505 只衰减 25%，拟合窗口为空）。

单跑约 18 min（两档 100T 同时在跑时会到 18 min；空闲约 10 min）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from lbm.solver3d import shear_wave_decay  # noqa: E402

RESULTS = ROOT / "results"
TAUS = (0.505, 0.51, 0.515, 0.52, 0.53, 0.55, 0.60, 0.80)
N = 32
AMP = 1e-4


def steps_for(tau: float, n: int = N, e_folds: float = 7.5) -> int:
    """按 ν k² 自适应给步数；至少 4000 步，避免高 τ 档点数不足。"""
    nu = (tau - 0.5) / 3.0
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError(f"tau must be > 0.5, got {tau}")
    rate = nu * (2.0 * np.pi / n) ** 2
    return max(4000, int(np.ceil(e_folds / rate / 20.0)) * 20)


def run(tau: float, n: int = N, amp: float = AMP) -> dict:
    steps = steps_for(tau, n)
    t0 = time.perf_counter()
    r = shear_wave_decay(n=n, nz=4, tau=tau, amp=amp, steps=steps)
    return {"tau": float(tau), "n": n, "steps": steps,
            "nu_input": r["nu_input"], "nu_fit": r["nu_fit"],
            "rel_err_pct": 100.0 * r["rel_err"],
            "n_fit": r["n_fit"], "fit_lo": r["fit_lo"], "fit_hi": r["fit_hi"],
            "wall_min": (time.perf_counter() - t0) / 60.0}


def re_error_budget(rows: list[dict], re_target: float = 400.0,
                    u_in: float = 0.1, d: float = 16.0) -> dict:
    """把 ν 偏差换成有效 Re 偏移；只对最接近目标 Re 的 τ 档给读数。"""
    tau_target = 3.0 * u_in * d / re_target + 0.5
    near = min(rows, key=lambda r: abs(r["tau"] - tau_target))
    err = near["rel_err_pct"] / 100.0
    return {"tau_target": tau_target, "tau_nearest": near["tau"],
            "nu_rel_err_pct": near["rel_err_pct"],
            "re_eff": re_target / (1.0 + err),
            "re_shift_pct": -100.0 * err / (1.0 + err)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default=",".join(str(x) for x in TAUS))
    ap.add_argument("--re-target", type=float, default=400.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rows = []
    for s in args.taus.split(","):
        if not s.strip():
            continue
        tau = float(s)
        p = RESULTS / f"v18_lowtau_tau{tau:.3f}.json"
        if p.exists() and not args.force:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
            print(f"[cached] tau={tau}")
        else:
            r = run(tau)
            p.write_text(json.dumps(r, ensure_ascii=False, indent=1),
                         encoding="utf-8")
            rows.append(r)
            print(f"[done] tau={tau} err={r['rel_err_pct']:.3f}% "
                  f"wall={r['wall_min']:.2f}min", flush=True)
    out = {"rows": rows, "re_budget": re_error_budget(rows, args.re_target),
           "protocol": {"n": N, "amp": AMP, "nz": 4,
                        "e_folds": 7.5, "re_target": args.re_target}}
    (RESULTS / "v18_lowtau.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["re_budget"], indent=1))


if __name__ == "__main__":
    main()
