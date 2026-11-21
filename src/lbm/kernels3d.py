"""numba JIT 核函数（D3Q19）：碰撞(+Guo 体积力)→海绵→迁移→Bouzidi 反弹→自由滑移。

与 2D ``kernels.py`` 的对应关系与差异：

- 碰撞：BGK，可选 Smagorinsky 隐式 τ（式与 2D 相同，只是 Π^neq 取 19 方向）。
  ``gx`` 非零时按 **Guo 强迫** 加体积力：
  ``S_i = w_i[3(c_i−u)·g + 9(c_i·u)(c_i·g)]``，``f += −(f−f^eq)/τ + (1−1/(2τ))S``，
  宏观速度取 ``u = (Σ c f + g/2)/ρ``；稳态 Poiseuille 因此保留二阶精度。
- 迁移：x/z 恒周期包裹；``wrap_y`` 控制 y 是否周期（通道/圆柱取 False，
  顶底由自由滑移或壁面反弹接管）。
- 反弹：链接表只建在 **一个 y-x 切片** 上（圆柱沿 z 挤出 ⇒ q 与 z 无关），
  核内对 z 并行，``q<0.5`` 的第二节点按 z 周期回绕取。
  力按 z 逐层累积后在 Python 侧求和并除以 nz（单位深度）。
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn

        return wrap

    prange = range  # type: ignore

from .lattice3d import CX3, CY3, CZ3, NL3, OPP3, W3

_CX = CX3.astype(np.float64)
_CY = CY3.astype(np.float64)
_CZ = CZ3.astype(np.float64)


@njit(cache=True, parallel=True)
def collide_sponge3d(f, tau, gx, sigma_x, ftarget,
                     les_on, cs_smag_sq, sponge_on):
    """碰撞（+Guo 力）与海绵层（作用在碰撞后分布上，与 2D 顺序一致）。

    **不使用任何跨节点共享的临时数组**：``prange`` 下把 feq 缓冲区提到循环外
    会造成多线程数据竞争（v1.4 首版即因此全线超差：Poiseuille 幅值 −11%、
    剪切波黏度 10×、Cd 大一个量级）。改为在每节点内联重算 feq。
    """
    nz, ny, nx, _ = f.shape
    inv_tau = 1.0 / tau
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                rho = 0.0
                mx = 0.0
                my = 0.0
                mz = 0.0
                for i in range(NL3):
                    v = f[z, y, x, i]
                    rho += v
                    mx += _CX[i] * v
                    my += _CY[i] * v
                    mz += _CZ[i] * v
                if rho <= 0.0:
                    continue
                # Guo：动量含半步力
                ux = (mx + 0.5 * gx) / rho
                uy = my / rho
                uz = mz / rho
                usq = ux * ux + uy * uy + uz * uz
                if les_on:
                    pxx = 0.0
                    pyy = 0.0
                    pzz = 0.0
                    pxy = 0.0
                    pxz = 0.0
                    pyz = 0.0
                    for i in range(NL3):
                        cu = _CX[i] * ux + _CY[i] * uy + _CZ[i] * uz
                        fe = W3[i] * rho * (1.0 + 3.0 * cu
                                            + 4.5 * cu * cu - 1.5 * usq)
                        fn = f[z, y, x, i] - fe
                        pxx += _CX[i] * _CX[i] * fn
                        pyy += _CY[i] * _CY[i] * fn
                        pzz += _CZ[i] * _CZ[i] * fn
                        pxy += _CX[i] * _CY[i] * fn
                        pxz += _CX[i] * _CZ[i] * fn
                        pyz += _CY[i] * _CZ[i] * fn
                    qn = np.sqrt(2.0 * max(pxx * pxx + pyy * pyy + pzz * pzz
                                           + 2.0 * (pxy * pxy + pxz * pxz
                                                    + pyz * pyz), 0.0))
                    disc = tau * tau + 18.0 * cs_smag_sq * qn / rho
                    omega = 1.0 / (0.5 * (tau + np.sqrt(max(disc, 0.0))))
                else:
                    omega = inv_tau
                c1 = 1.0 - 0.5 * omega
                for i in range(NL3):
                    cu = _CX[i] * ux + _CY[i] * uy + _CZ[i] * uz
                    fe = W3[i] * rho * (1.0 + 3.0 * cu
                                        + 4.5 * cu * cu - 1.5 * usq)
                    d = -(omega * (f[z, y, x, i] - fe))
                    if gx != 0.0:
                        src = (3.0 * (_CX[i] - ux) * gx
                               + 9.0 * cu * _CX[i] * gx)
                        d += c1 * W3[i] * src
                    f[z, y, x, i] += d
    if sponge_on:
        for z in prange(nz):
            for y in range(ny):
                for x in range(nx):
                    sig = sigma_x[x]
                    if sig > 0.0:
                        for i in range(NL3):
                            f[z, y, x, i] += sig * (ftarget[i] - f[z, y, x, i])


@njit(cache=True, parallel=True)
def stream3d(f, fnew, wrap_y):
    """pull 迁移：x/z 恒周期；y 按 wrap_y 或钳位（钳位值随后被边界覆盖）。"""
    nz, ny, nx, _ = f.shape
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                for i in range(NL3):
                    sz = z - CZ3[i]
                    sy = y - CY3[i]
                    sx = x - CX3[i]
                    if sz < 0:
                        sz += nz
                    elif sz >= nz:
                        sz -= nz
                    if sx < 0:
                        sx += nx
                    elif sx >= nx:
                        sx -= nx
                    if sy < 0:
                        sy = sy + ny if wrap_y else 0
                    elif sy >= ny:
                        sy = sy - ny if wrap_y else ny - 1
                    fnew[z, y, x, i] = f[sz, sy, sx, i]


@njit(cache=True, parallel=True)
def bounceback3d(f, fnew, ly, lx, ld, lq, lsy, lsx, lforce, out_f):
    """Bouzidi 反弹写入 fnew，并逐 z 累积圆柱动量交换力到 out_f[:, 3]。"""
    nz = f.shape[0]
    nlink = ly.size
    for z in prange(nz):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        for k in range(nlink):
            y = ly[k]
            x = lx[k]
            i = ld[k]
            q = lq[k]
            fi = f[z, y, x, i]
            j = OPP3[i]
            if q < 0.5:
                sz = z - CZ3[i]
                if sz < 0:
                    sz += nz
                elif sz >= nz:
                    sz -= nz
                bb = 2.0 * q * fi + (1.0 - 2.0 * q) * f[sz, lsy[k], lsx[k], i]
            else:
                inv = 1.0 / (2.0 * q)
                bb = inv * fi + (2.0 * q - 1.0) * inv * f[z, y, x, j]
            fnew[z, y, x, j] = bb
            if lforce[k]:
                fx += _CX[i] * (fi + bb)
                fy += _CY[i] * (fi + bb)
                fz += _CZ[i] * (fi + bb)
        out_f[z, 0] = fx
        out_f[z, 1] = fy
        out_f[z, 2] = fz


def compile_step3d():
    """触发 JIT 编译（单测用，避免首次调用计入计时）。"""
    f = np.ones((2, 4, 4, NL3)) / NL3
    fn = np.empty_like(f)
    from .lattice3d import W3 as _W

    collide_sponge3d(f, 0.8, 0.0, np.zeros(4), _W.copy(), False, 0.0, False)
    stream3d(f, fn, False)
    empty = np.zeros(0, dtype=np.int64)
    bounceback3d(f, fn, empty, empty, empty, np.zeros(0),
                 empty, empty, np.zeros(0, dtype=np.bool_), np.zeros((2, 3)))
