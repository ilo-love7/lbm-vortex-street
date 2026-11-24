#!/usr/bin/env bash
# N5 全流程：00 校验 → 01 基线 → 02 动画 → 03 网格收敛 → 04 收口 → 05 v1.2 → 06/07 v1.3 → 08 v1.4 3D
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
echo "======== 08 v1.4 三维校验（E19 解析验证 / E20 3D-2D 对照 / E21 β→0 外推） ========"
"$PY" scripts/08_3d_validation.py
echo "======== 09 v1.4 瞬态敏感性（6/12/16/20/24T，2D）========"
"$PY" scripts/09_v14_transient_sensitivity.py
echo "======== 10 v1.4 口径拆分（D 加密，固定无量纲域长）========"
"$PY" scripts/10_v14_d_resolution.py
echo "======== 11 v1.5 三维失稳区（E22 展向盒长 / E23 失稳 Re 阈值）========"
"$PY" scripts/11_v15_3d_unstable.py
echo "======== 12 v1.5 域长口径（E24 20D 下游 β 系列与截距对账）========"
"$PY" scripts/12_v15_domain20d.py
echo "======== 13 v1.5 缺口审计（E25 v1.3→v1.4 的 6.1pp 逐旋钮替换）========"
"$PY" scripts/13_v15_gap_audit.py
echo "======== 14 v1.6 高 Re 三维（E26 Lz=6D 的 Re=250/300/350/400 与饱和律）========"
"$PY" scripts/14_v16_highre.py
echo "======== 15 v1.7 窗长锁住（四档 50T，每档 5 个 10T 窗）========"
"$PY" scripts/15_v17_window.py
echo "======== 16 v1.8 超长窗（Re=350/400 各 100T，每档 10 个 10T 窗）========"
"$PY" scripts/16_v18_longrun.py
echo "======== 17 v1.8 低 τ 精度校准（τ=0.505-0.8 剪切波反测 ν）========"
"$PY" scripts/17_v18_lowtau.py
echo "======== 18 v1.9 强迫/边界消融（Re=400 100T；trip 幅度/种子/零扰动/壁面）========"
"$PY" scripts/18_v19_forcing.py
echo "======== 19 v1.10 纯盒长分离（Re=250；Lz=4D/8D 各 40T+10T；6D 复用 v1.6 缓存）========"
"$PY" scripts/19_v110_boxlength.py
echo "全部完成。结果见 results/ 与 figures/"
# 增量语义：06/08/11/12/13/14 逐 case 断点缓存，重跑只补缺失；--force 清缓存全量重算；
#          --figures-only 只从缓存刷新 JSON/图/CSV，不重跑 CFD。
# 08 单跑约 30 min（3D ny=320 一档占 26 min，17-18 ms/step）；先跑它可并行做文档。
# 09 约 7 min，10 约 15 min，均为单线程 2D，可与 08 的文档工作并行。
# 11 约 120 min（3D 串行：nz=8→64 四档 + Re=100 长盒对照 + Re=190/220 各一档）。
#    先跑 12（约 12 min）不冲突，2D 单线程只吃一个核。
# 12 约 12 min（2D、nx=400、五点 β 系列）。
# 13 约 35 min（2D 五点 × 四链，nx=424/560；A4 直接复用 v1.4 的 JSON 不重跑）。
# 14 实测约 73/59/72/58 min（四档 3D，两进程各 12 线程并行，总墙钟 2h13m）+ 2D 配对 5 min。
#    可拆批并行：`--cases 250,300` 与 `--cases 350,400` 各起一个进程，
#    并给 NUMBA_NUM_THREADS=12（两进程别抢同一批核）；跑完再不带参数合并汇总。
#    可选的定常性对照：`--long250`（Re=250、40T+10T，约 75-80 min），结果进 §20.5。
#    不要再加 `--box-extra`（8D 档 nz=128 与主批抢内存带宽，实测 2.3 步/秒，本轮取消）。
# 15 四档各 40T+10T（50T），实测单档约 63 min；**串行单进程**跑完约 4h15m，
#    比两进程争核更快（v1.6 实测 26T 并行 58-73 min/档，50T 单独只要 63 min）。
#    `--cases 250` 分批写缓存；四档齐后不带参数一次汇总 JSON/CSV/图。
#    `--smoke` 只跑 0.4T（约 50 s），验采样器不抛错，不用于物理读数。
# 16 Re=350/400 各 40T+60T（100T），10 个 10T 窗；**串行单进程**约 160 min/档。
#    前 40T 四窗必须与 v1.7 逐位相同（硬断言）；40–50T 窗因 v1.7 的 1 步尾窗并入
#    差 1–2 步，只做 1% 容差核对。--cases 分批写缓存；两档齐后汇总出 JSON/CSV/图。
#    --smoke 只跑 0.5T（约 1 min），不用于物理读数。
