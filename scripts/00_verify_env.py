"""00 环境校验：依赖、格子恒等式、JIT 可用性。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


def main():
    import matplotlib
    import numpy
    import scipy

    print(f"numpy {numpy.__version__}  scipy {scipy.__version__}  mpl {matplotlib.__version__}")

    from lbm.kernels import HAVE_NUMBA
    if HAVE_NUMBA:
        import numba
        print(f"numba {numba.__version__}（JIT 内核启用）")
    else:
        print("numba 不可用：回退 numpy 内核（约慢 6 倍）")

    from lbm.lattice import CX, CY, NL, OPP, W, equilibrium
    assert NL == 9
    assert abs(W.sum() - 1.0) < 1e-12
    assert np.all(CX[OPP] == -CX) and np.all(CY[OPP] == -CY)
    rho = np.array([[[1.2, 0.9]]])
    ux = np.array([[[0.05, -0.03]]])
    uy = np.array([[[0.01, 0.02]]])
    feq = equilibrium(rho, ux, uy)
    assert feq.shape == (1, 2, 9)
    m = feq.sum(axis=2)
    assert np.allclose(m, rho, atol=1e-12), m
    print("格子恒等式 OK（权和=1、OPP 对合、feq 质量守恒）")

    from lbm.config import default_config_path, load_config
    cfg = load_config(default_config_path())
    print(f"config OK：re_list={cfg['sweep']['re_list']} nx={cfg['case']['nx']}")
    print("00_verify_env 通过。")


if __name__ == "__main__":
    main()
