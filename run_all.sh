#!/usr/bin/env bash
# N5 全流程：00 校验 → 01 基线 → 02 动画 → 03 网格收敛 → 04 收口 → 05 v1.2 → 06/07 v1.3
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-D:/项目/venvs/flsoh-py312/Scripts/python.exe}"
for s in 00_verify_env 01_vortex_street 02_animation 03_grid_conv 04_compare 05_mrt_re_vorticity; do
  echo "======== $s ========"
  "$PY" "scripts/${s}.py"
done
echo "======== 06 v1.3 深化（E13 阻塞比 / E14 四口径测力 / E15 LES 与入口扰动） ========"
"$PY" scripts/06_v13_deepening.py --workers 12
echo "======== 07 v1.3 汇总表 ========"
"$PY" scripts/07_summary_v13.py
echo "全部完成。结果见 results/ 与 figures/"
# 增量语义：06 逐 case 断点缓存，重跑只补缺失；--force 清缓存全量重算；
#          --figures-only 只从缓存刷新 JSON/图/CSV，不重跑 CFD。
