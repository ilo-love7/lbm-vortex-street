#!/usr/bin/env bash
# N5 全流程：00 校验 → 01 基线实验（Re 扫描）→ 02 动画 → 03 网格收敛 → 04 收口对比
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-D:/项目/venvs/flsoh-py312/Scripts/python.exe}"
for s in 00_verify_env 01_vortex_street 02_animation 03_grid_conv 04_compare; do
  echo "======== $s ========"
  "$PY" "scripts/${s}.py"
done
echo "全部完成。结果见 results/ 与 figures/"
