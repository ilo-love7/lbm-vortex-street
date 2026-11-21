"""D3Q19 晶格常量与平衡态/宏观量（numpy 向量化，供初始化与后处理共用）。

方向编号（与 ``kernels3d`` 严格一致）：

    i   (cx,cy,cz)   w
    0   ( 0, 0, 0)   1/3
    1   ( 1, 0, 0)   1/18      2  (-1, 0, 0)
    3   ( 0, 1, 0)   1/18      4  ( 0,-1, 0)
    5   ( 0, 0, 1)   1/18      6  ( 0, 0,-1)
    7   ( 1, 1, 0)   1/36      8  ( 1,-1, 0)
    9   (-1, 1, 0)   1/36     10  (-1,-1, 0)
    11  ( 1, 0, 1)   1/36     12  ( 1, 0,-1)
    13  (-1, 0, 1)   1/36     14  (-1, 0,-1)
    15  ( 0, 1, 1)   1/36     16  ( 0, 1,-1)
    17  ( 0,-1, 1)   1/36     18  ( 0,-1,-1)

``OPP`` 为对向表（对合）；``CS2 = 1/3`` 由 ``Σ_i w_i c_iα c_iβ = c_s² δ_αβ`` 给出，
``lattice3d_checks()`` 在导入后按需断言这些性质（不通过则整个模块不可信）。
"""
from __future__ import annotations

import numpy as np

CX3 = np.array([0,
                1, -1, 0, 0, 0, 0,
                1, 1, -1, -1,
                1, 1, -1, -1,
                0, 0, 0, 0], dtype=np.int64)
CY3 = np.array([0,
                0, 0, 1, -1, 0, 0,
                1, -1, 1, -1,
                0, 0, 0, 0,
                1, 1, -1, -1], dtype=np.int64)
CZ3 = np.array([0,
                0, 0, 0, 0, 1, -1,
                0, 0, 0, 0,
                1, -1, 1, -1,
                1, -1, 1, -1], dtype=np.int64)

W3 = np.array([1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12)

OPP3 = np.array([0,
                 2, 1, 4, 3, 6, 5,
                 10, 9, 8, 7,
                 14, 13, 12, 11,
                 18, 17, 16, 15], dtype=np.int64)

NL3 = 19
CS2 = 1.0 / 3.0


def _specular_y_table() -> np.ndarray:
    """y 法向镜面映射：``(cx,cy,cz) → (cx,−cy,cz)`` 的方向编号表。

    三维里 **镜面反射 ≠ 对向**（对向把三个分量全部取反）。2D 之所以能用
    OPP 的写法，是因为 D2Q9 的斜向对向恰好等于镜面；三维若直接套用 OPP，
    自由滑移壁会变成"吸收壁"（v1.4 首版实测均匀流被壁面吃掉 92%）。
    """
    vecs = np.stack([CX3, CY3, CZ3], axis=1)
    table = np.zeros(NL3, dtype=np.int64)
    for i in range(NL3):
        target = np.array([CX3[i], -CY3[i], CZ3[i]])
        hit = np.flatnonzero((vecs == target).all(axis=1))
        if hit.size != 1:
            raise RuntimeError(f"镜面映射 {i} 不在 D3Q19 中")
        table[i] = int(hit[0])
    return table


SPEC_Y3 = _specular_y_table()


def lattice3d_checks() -> dict:
    """返回 D3Q19 的必要性质残差（供单测断言，<1e-14 视为通过）。"""
    vecs = np.stack([CX3, CY3, CZ3], axis=1).astype(np.float64)
    w = W3
    mass = abs(w.sum() - 1.0)
    mom = float(np.max(np.abs(w @ vecs)))
    # Σ w c_a c_b 应为 c_s² δ_ab
    cc = (vecs * w[:, None]).T @ vecs
    iso = float(np.max(np.abs(cc - CS2 * np.eye(3))))
    # 四阶矩各向同性（D3Q19 特有：Σ w c_a² c_b² = c_s⁴ 对 a≠b，= 3c_s⁴ 对 a=b）
    quad = np.einsum("i,ia,ib,ic,id->abcd", w, vecs, vecs, vecs, vecs)
    # 方向唯一性
    uniq = len({tuple(v) for v in vecs.astype(int)})
    opp = float(np.max(np.abs(vecs[OPP3] + vecs)))
    return {"mass": mass, "momentum": mom, "isotropy": iso,
            "quad_moment": float(np.max(np.abs(
                quad[np.arange(3), np.arange(3), np.arange(3), np.arange(3)]
                - 3.0 * CS2 ** 2))),
            "n_unique": uniq, "opposite": opp}


def equilibrium3d(rho, ux, uy, uz, out=None):
    """f_i^eq = w_i ρ [1 + 3(c·u) + 4.5(c·u)² − 1.5|u|²]，最后一维为 19。"""
    rho = np.asarray(rho, dtype=np.float64)
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    uz = np.asarray(uz, dtype=np.float64)
    usq = ux * ux + uy * uy + uz * uz
    shape = rho.shape + (NL3,)
    f = np.empty(shape, dtype=np.float64) if out is None else out
    for i in range(NL3):
        cu = CX3[i] * ux + CY3[i] * uy + CZ3[i] * uz
        f[..., i] = W3[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    return f


def macros3d(f):
    """返回 (rho, ux, uy, uz)；f 形状 (..., 19)。"""
    fa = np.asarray(f, dtype=np.float64)
    rho = fa.sum(axis=-1)
    rs = np.where(rho > 0.0, rho, 1.0)
    ux = (fa @ CX3.astype(np.float64)) / rs
    uy = (fa @ CY3.astype(np.float64)) / rs
    uz = (fa @ CZ3.astype(np.float64)) / rs
    return rho, ux, uy, uz
