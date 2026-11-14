"""07 v1.3 汇总：从三份 JSON 生成 results/v13_summary_tables.md（表 A–F）。

报告与 README 的所有数字都引用本文件，避免手抄漂移。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _f(v, n=4, dash="—"):
    if v is None:
        return dash
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return str(v)


def _pc(v, n=2):
    return dash if v is None else f"{float(v) * 100:+.{n}f}%"


dash = "—"


def table_a(blk):
    lines = ["### 表 A · E13 阻塞比逐 case",
             "",
             "| Re | ny | β=D/H | τ | St | St_lit | St 偏差 | Cd | Cl_peak | 墙钟(min) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in blk.get("rows", []):
        if r.get("diverged"):
            lines.append(f"| {int(r['re'])} | {r['ny']} | {_f(r['beta'],3)} | "
                         f"{_f(r.get('tau'),3)} | 发散 | {_f(r.get('st_lit'))} "
                         f"| {dash} | {dash} | {dash} | {_f(r.get('wall_min'),1)} |")
            continue
        lines.append(
            f"| {int(r['re'])} | {r['ny']} | {_f(r['beta'],3)} | {_f(r.get('tau'),3)} "
            f"| {_f(r['st'])} | {_f(r['st_lit'])} | {_pc(r.get('rel_err'))} "
            f"| {_f(r.get('cd_mean'),3)} | {_f(r.get('cl_peak_amp'),3)} "
            f"| {_f(r.get('wall_min'),1)} |")
    return lines


def table_b(blk):
    lines = ["", "### 表 B · E13 β→0 外推（分离阻塞修正与 2D 约束误差）",
             "",
             "| Re | 点数 | St_lit | St(β→0) 线性 | R² | St(β→0) 二次 | 外推残余偏差 | Cd(β→0) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for re, e in sorted(blk.get("extrapolation", {}).items(), key=lambda kv: int(kv[0])):
        if "st_fit_lin" not in e:
            continue
        lin = e["st_fit_lin"]
        quad = e.get("st_fit_quad")
        lines.append(
            f"| {re} | {e['n']} | {_f(e['st_lit'])} | {_f(lin['intercept'])} "
            f"| {_f(lin['r2'],3)} | {_f(quad['intercept']) if quad else dash} "
            f"| {_pc(e['residual_st_rel'])} "
            f"| {_f(e['cd_fit_lin']['intercept'],3)} |")
    return lines


def table_c(cv):
    lines = ["", "### 表 C · E14 逐 case 四口径测力",
             "",
             "| Re | 碰撞 | τ | Cd_MEM | Cd_CV-node | node 围道分散 "
             "| Cd_CV-link | link 围道分散 | Cd_WAKE | 站位分散 | St_MEM |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in cv.get("rows", []):
        if r.get("diverged"):
            lines.append(f"| {int(r['re'])} | {r['collision']} | {_f(r.get('tau'),3)} "
                         f"| 发散 | {dash} | {dash} | {dash} | {dash} | "
                         f"{dash} | {dash} | {dash} |")
            continue
        m = lambda k: (float(sum(r.get(k, {}).values())
                            / len(r[k])) if r.get(k) else None)
        lines.append(
            f"| {int(r['re'])} | {r['collision']} | {_f(r.get('tau'),3)} "
            f"| {_f(r.get('cd_mem_period'),3)} | {_f(m('cd_cv'),3)} "
            f"| {_pc(r.get('contour',{}).get('rel_spread'))} "
            f"| {_f(m('cd_link'),3)} "
            f"| {_pc(r.get('link_contour',{}).get('rel_spread'))} "
            f"| {_f(m('cd_wake'),3)} "
            f"| {_pc(r.get('wake',{}).get('rel_spread'))} "
            f"| {_f(r.get('st'))} |")
    return lines


def table_d(cv):
    lines = ["", "### 表 D · E14 MRT 的 Cd 偏移：四口径判据",
             "",
             "| Re | ΔCd_MEM (MRT−BGK) | ΔCd_CV-node | ΔCd_CV-link "
             "| ΔCd_WAKE（独立主判据） | 判定 |",
             "|---:|---:|---:|---:|---:|---|"]
    for re, g in sorted(cv.get("groups", {}).items(), key=lambda kv: int(kv[0])):
        dm = g.get("delta_mem_rel")
        dw = g.get("delta_wake_rel")
        verdict = dash
        if dm is not None and dw is not None:
            if abs(dw) < 0.3 * abs(dm):
                verdict = "测力伪影（尾流亏损不变）"
            elif abs(dw) > 0.7 * abs(dm):
                verdict = "真实物理改变（尾流亏损同步变）"
            else:
                verdict = "混合（部分真实 + 部分伪影）"
        lines.append(
            f"| {re} | {_pc(dm)} | {_pc(g.get('delta_cv_rel'))} "
            f"| {_pc(g.get('delta_link_rel'))} "
            f"| {_pc(dw)} | {verdict} |")
    return lines


def table_g(cv):
    """E14 方法学：恒等式校验 + 朴素节点口径违反围道无关性。"""
    lines = ["", "### 表 G · E14 方法学校验（严格守恒恒等式与围道无关性）",
             "",
             "| Re | CV-link vs MEM 逐步最大差 | CV-node 围道分散 "
             "| CV-link 围道分散 | WAKE 站位分散 | 经典WAKE/严格WAKE "
             "| Cd_MEM−Cd_WAKE (BGK/MRT) |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for re, g in sorted(cv.get("groups", {}).items(), key=lambda kv: int(kv[0])):
        gap = (f"{_f(g['gap_wake_bgk']['delta'], 4)} / "
               f"{_f(g['gap_wake_mrt']['delta'], 4)}"
               if g.get("gap_wake_bgk") and g.get("gap_wake_mrt") else dash)
        lines.append(
            f"| {re} | {_f(g.get('link_vs_mem_maxdiff'), 8)} "
            f"| {_pc(g.get('contour_spread_node'))} "
            f"| {_pc(g.get('contour_spread_link'))} "
            f"| {_pc(g.get('wake_spread'))} "
            f"| {_f(g.get('classic_over_strict'), 3)} | {gap} |")
    return lines


def table_e(les):
    lines = ["", "### 表 E · E15 Smagorinsky LES 网格",
             "",
             "| Re | τ₀ | C_s | 发散 | median(ν_t/ν₀) | St | Cd | Cl_peak |",
             "|---:|---:|---:|:---:|---:|---:|---:|---:|"]
    for r in les.get("les_rows", []):
        if r.get("diverged"):
            lines.append(f"| {int(r['re'])} | {_f(r.get('tau0'),4)} | {_f(r['cs'],2)} "
                         f"| **是** | {dash} | {dash} | {dash} | {dash} |")
            continue
        lines.append(
            f"| {int(r['re'])} | {_f(r.get('tau0'),4)} | {_f(r['cs'],2)} | 否 "
            f"| {_f(r.get('nu_ratio_median'),4)} | {_f(r.get('st'))} "
            f"| {_f(r.get('cd_mean'),3)} | {_f(r.get('cl_peak_amp'),3)} |")
    return lines


def table_f(les):
    lines = ["", "### 表 F · E15 持续入口扰动",
             "",
             "| ε/U | St | St_lit | Cd | Cl_peak | 起振 [T] |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in les.get("inlet_rows", []):
        if r.get("diverged"):
            lines.append(f"| {_f(r['eps'],3)} | 发散 | {dash} | {dash} | {dash} | {dash} |")
            continue
        lines.append(
            f"| {_f(r['eps'],3)} | {_f(r.get('st'))} | {_f(r.get('st_lit'))} "
            f"| {_f(r.get('cd_mean'),3)} | {_f(r.get('cl_peak_amp'),3)} "
            f"| {_f(r.get('onset_periods'),2)} |")
    return lines


def main():
    blk = _load("v13_blockage.json")
    cv = _load("v13_cv_force.json")
    les = _load("v13_les_inlet.json")
    lines = ["# N5 v1.3 汇总表（自动生成，勿手改）", "",
             "来源：`results/v13_blockage.json` / `v13_cv_force.json` / "
             "`v13_les_inlet.json`；由 `scripts/07_summary_v13.py` 生成。", ""]
    for fn, arg in ((table_a, blk), (table_b, blk), (table_c, cv),
                    (table_d, cv), (table_e, les), (table_f, les),
                    (table_g, cv)):
        lines.extend(fn(arg))
        lines.append("")
    out = RESULTS / "v13_summary_tables.md"
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"→ {out}  ({len(lines)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
