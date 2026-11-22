"""D3Q19 三维 LBM 求解器：圆柱绕流（z 周期、y 自由滑移）、平面通道、周期盒。

为什么要三维
------------
v1.3 报告 §12 把「3D 校验」列为**最高优先未做项**：二维阻塞比外推给出
β→0 时 St 偏差 −4.06%（线性）/ +2.35%（二次），两种拟合差 6.4pp，而
Re=250 的残余 +4.71% 仍随 Re 增长。这些残余究竟是「2D 约束误差」还是
「拟合口径噪声」，只能靠三维计算裁决。本模块提供该裁决所需的求解器。

与 2D 的实现对齐关系（逐条对应，便于交叉验证）
----------------------------------------------
- 入口：3D Zou/He 由「质量守恒 + 非平衡反弹」导出，通式
  ``ρ = (A + A_K) / (1 − s_U + s_K)``；把 D3Q19 换成 D2Q9 时**精确退化为**
  2D 版 ``ρ = (f0+f1+f5+2(f6+f7+f8))/(1−U)``（单测逐位核对）。
- 出口：一阶对流；顶/底：自由滑移镜面；圆柱：Bouzidi 次网格反弹
  （q 由射线与圆在 xy 平面的交点给出，与 z 无关——圆柱沿 z 挤出）。
- 测力：动量交换 ``F = Σ_links e_i[f_i* + f_ī_new]``，逐 z 累积后取
  ``F/nz``（单位深度），与 2D 力可比。
- 尾流：先对 z 做算术平均得到 (ρ, ρu², p) 的展向平均剖面，再套用 v1.3
  严格两截面口径 ``F_D = Σ_ref(ρu_x²+p) − Σ_x(ρu_x²+p)``。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kernels3d import (bounceback3d, collide_sponge3d, stream3d)
from .lattice3d import (CS2, CX3, CY3, CZ3, NL3, OPP3, SPEC_Y3, W3,
                        equilibrium3d, macros3d)
from .postprocess import literature_st
from .wake import wake_ref_x, wake_stations


# ---------------------------------------------------------------- 几何/参数
@dataclass
class Case3D:
    """三维工况（格子单位）。``d=0`` 表示无圆柱（通道/周期盒）。"""

    nx: int = 320
    ny: int = 80
    nz: int = 16
    d: float = 16.0
    cx_c: float = 80.0
    cy_c: float | None = None
    u_in: float = 0.1
    re: float = 100.0
    rho0: float = 1.0
    top_bottom: str = "free_slip"     # free_slip | no_slip | periodic
    inlet: str = "zouhe"              # zouhe | periodic
    outlet: str = "convective"        # convective | extrapolate | periodic
    sponge_width: int = 32
    sponge_sigma: float = 0.08
    trip: float = 1e-3
    bouzidi: bool = True
    body_force: float = 0.0           # 仅 x 向（通道验证用）
    les: bool = False
    c_smag: float = 0.16
    tau_direct: float = 0.0           # >0 时直接指定 τ（d=0 的验证算例用）

    @property
    def radius(self) -> float:
        return self.d / 2.0

    @property
    def cy(self) -> float:
        return self.ny / 2.0 if self.cy_c is None else self.cy_c

    @property
    def nu(self) -> float:
        if self.re <= 0.0:
            raise ValueError("re 必须为正")
        return self.u_in * self.d / self.re

    @property
    def tau(self) -> float:
        if self.tau_direct > 0.0:
            return float(self.tau_direct)
        return 3.0 * self.nu + 0.5

    @property
    def periodic_y(self) -> bool:
        return self.top_bottom == "periodic"

    @property
    def periodic_x(self) -> bool:
        return self.inlet == "periodic"


def _link_q_cyl(y: np.ndarray, x: np.ndarray, direction: int,
                cx_c: float, cy_c: float, radius: float) -> np.ndarray:
    """射线 p + t·(cx,cy) 与圆 |p−C|=R 的首个交点参数 t∈(0,1]。

    三维对角方向 (cx,cy,cz) 的 xy 投影为 (cx,cy)，故二次项系数取
    ``A = cx²+cy²``（面内方向 A=1，与 2D 的 q 完全一致）。
    """
    ex, ey = float(CX3[direction]), float(CY3[direction])
    a2 = ex * ex + ey * ey
    if a2 == 0.0:
        raise ValueError("纯 z 方向不应与 z 不变固体相邻")
    dx = np.asarray(x, dtype=np.float64) - cx_c
    dy = np.asarray(y, dtype=np.float64) - cy_c
    b = dx * ex + dy * ey
    c = dx * dx + dy * dy - radius * radius
    disc = np.maximum(b * b - a2 * c, 0.0)
    t = (-b - np.sqrt(disc)) / a2
    return np.clip(t, 1e-6, 1.0)


def build_link_table3d(fluid: np.ndarray, solid: np.ndarray, cyl: np.ndarray,
                       cx_c: float, cy_c: float, radius: float,
                       bouzidi: bool = True) -> dict:
    """构建 (y,x,dir) 切片链接表；圆柱链接带解析 q，壁面链接恒 q=0.5。

    返回扁平数组 ``ly/lx/ld/lq/lsy/lsx/lforce``；``lforce`` 标记计入圆柱
    动量交换力的链接（壁面不计，避免污染 Cd/Cl）。
    """
    ny, nx = solid.shape
    out = {k: [] for k in ("y", "x", "d", "q", "sy", "sx", "force")}
    for i in range(1, NL3):
        cx, cy = int(CX3[i]), int(CY3[i])
        for y in range(ny):
            ty = y + cy
            if ty < 0 or ty >= ny:
                continue
            nb = np.roll(solid[ty], -cx)          # nb[x] = solid[ty, x+cx]
            xs = np.nonzero(fluid[y] & nb)[0]
            if xs.size == 0:
                continue
            nb_cyl = np.roll(cyl[ty], -cx)[xs]
            q = np.full(xs.size, 0.5)
            if bouzidi and nb_cyl.any():
                q[nb_cyl] = _link_q_cyl(y, xs[nb_cyl], i, cx_c, cy_c, radius)
            sx = (xs - cx) % nx
            sy = y - cy
            if sy < 0 or sy >= ny:
                raise RuntimeError("反弹链接第二节点越界")
            if not fluid[sy, sx].all():
                # 圆柱是凸体：q<0.5 的第二节点必在流体；壁面链接不读第二节点
                if (q < 0.5).any():
                    raise RuntimeError("q<0.5 的第二节点落在固体上")
            out["y"].append(np.full(xs.size, y, dtype=np.int64))
            out["x"].append(xs.astype(np.int64))
            out["d"].append(np.full(xs.size, i, dtype=np.int64))
            out["q"].append(q)
            out["sy"].append(np.full(xs.size, sy, dtype=np.int64))
            out["sx"].append(sx.astype(np.int64))
            out["force"].append(nb_cyl)
    if not out["y"]:
        e = np.zeros(0, dtype=np.int64)
        return {"ly": e, "lx": e, "ld": e, "lq": np.zeros(0),
                "lsy": e, "lsx": e, "lforce": np.zeros(0, dtype=np.bool_)}
    return {"ly": np.concatenate(out["y"]), "lx": np.concatenate(out["x"]),
            "ld": np.concatenate(out["d"]), "lq": np.concatenate(out["q"]),
            "lsy": np.concatenate(out["sy"]), "lsx": np.concatenate(out["sx"]),
            "lforce": np.concatenate(out["force"]).astype(np.bool_)}


# ---------------------------------------------------------------- 求解器
class Solver3D:
    def __init__(self, case: Case3D):
        self.case = c = case
        if not (0.5001 < c.tau < 3.0):
            raise ValueError(f"tau={c.tau:.4f} 超出稳定区间")
        yy, xx = np.mgrid[0:c.ny, 0:c.nx]
        if c.d > 0:
            self.cyl = ((xx - c.cx_c) ** 2 + (yy - c.cy) ** 2) < c.radius ** 2
        else:
            self.cyl = np.zeros((c.ny, c.nx), dtype=bool)
        self.solid = self.cyl.copy()
        if c.top_bottom == "no_slip":
            self.solid[0, :] = True
            self.solid[-1, :] = True
        self.fluid = ~self.solid

        rho = np.full((c.nz, c.ny, c.nx), c.rho0)
        ux = np.full((c.nz, c.ny, c.nx), c.u_in)
        uy = np.zeros((c.nz, c.ny, c.nx))
        uz = np.zeros((c.nz, c.ny, c.nx))
        if c.trip > 0:
            rng = np.random.default_rng(42)
            uy += c.trip * rng.standard_normal(uy.shape)
            uz += c.trip * rng.standard_normal(uz.shape)
        self.f = equilibrium3d(rho, ux, uy, uz)
        self.f[:, self.solid] = 0.0
        self.f_new = np.empty_like(self.f)

        table = build_link_table3d(self.fluid, self.solid, self.cyl,
                                   c.cx_c, c.cy, c.radius, bouzidi=c.bouzidi)
        self._tab = table
        self._out_f = np.zeros((c.nz, 3))

        self._sigma_x = np.zeros(c.nx)
        w = int(c.sponge_width)
        if w > 0 and c.sponge_sigma > 0.0 and not c.periodic_x:
            xi = np.linspace(0.0, 1.0, w)
            ramp = xi * xi * (3.0 - 2.0 * xi)
            self._sigma_x[:w] = c.sponge_sigma * ramp[::-1]
            self._sigma_x[-w:] = c.sponge_sigma * ramp
        cu = CX3 * c.u_in
        self._ftarget = W3 * c.rho0 * (1.0 + 3.0 * cu + 4.5 * cu * cu
                                      - 1.5 * c.u_in ** 2)
        self._sponge_on = bool(self._sigma_x.max() > 0.0)

        # 入口集合（3D Zou/He 一般化的系数）
        self._U_set = [i for i in range(NL3) if CX3[i] > 0]
        self._K_set = [OPP3[i] for i in self._U_set]
        self._known = [i for i in range(NL3) if CX3[i] <= 0]
        U = c.u_in
        self._sU = sum(W3[i] * (1 + 3 * CX3[i] * U + 3 * U * U)
                       for i in self._U_set)
        self._sK = sum(W3[i] * (1 + 3 * CX3[i] * U + 3 * U * U)
                       for i in self._K_set)
        self._den = 1.0 - self._sU + self._sK

        # 尾流站位（仅在非周期 x 时有意义）
        if c.d > 0 and not c.periodic_x:
            self._wake_stations = wake_stations(c.cx_c, c.d, c.nx)
            self._wake_ref_x = wake_ref_x(c.sponge_width)
        else:
            self._wake_stations, self._wake_ref_x = [], None

        self._fx = self._fy = self._fz = 0.0
        self._it = 0

    # ---------- 单步 ----------
    def step(self):
        c = self.case
        t = self._tab
        collide_sponge3d(self.f, c.tau, c.body_force, self._sigma_x,
                         self._ftarget, c.les, c.c_smag ** 2, self._sponge_on)
        stream3d(self.f, self.f_new, c.periodic_y)
        bounceback3d(self.f, self.f_new, t["ly"], t["lx"], t["ld"], t["lq"],
                     t["lsy"], t["lsx"], t["lforce"], self._out_f)
        if t["ly"].size:
            self._fx, self._fy, self._fz = (
                float(self._out_f[:, 0].sum()) / c.nz,
                float(self._out_f[:, 1].sum()) / c.nz,
                float(self._out_f[:, 2].sum()) / c.nz)
        if c.top_bottom == "free_slip":
            self._apply_free_slip()
        if not c.periodic_x:
            self._apply_inlet()
            self._apply_outlet()
        if self.solid.any():
            self.f_new[:, self.solid] = 0.0
        self.f, self.f_new = self.f_new, self.f
        self._it += 1

    def _apply_free_slip(self):
        """y 壁镜面反射：cy>0 槽位 ← 镜面伙伴（cy 反向、cx/cz 保持）。

        必须用 ``SPEC_Y3`` 而非 ``OPP3``：三维对向会把 cx/cz 一并取反，
        那是"反向散射"而不是镜面，会把切向动量带走（实测均匀流被吃掉 92%）。
        """
        f, fn = self.f, self.f_new
        for i in range(NL3):
            if CY3[i] > 0:
                fn[:, 0, :, i] = f[:, 0, :, SPEC_Y3[i]]
            elif CY3[i] < 0:
                fn[:, -1, :, i] = f[:, -1, :, SPEC_Y3[i]]

    def _apply_inlet(self):
        """3D Zou/He：ρ=(A+A_K)/(1−s_U+s_K)，未知向按非平衡反弹补。"""
        f = self.f_new
        A = sum(f[:, :, 0, i] for i in self._known)
        A_K = sum(f[:, :, 0, i] for i in self._K_set)
        rho = (A + A_K) / self._den
        U = self.case.u_in
        for i in self._U_set:
            j = OPP3[i]
            feq_i = W3[i] * rho * (1.0 + 3.0 * CX3[i] * U + 3.0 * U * U)
            feq_j = W3[j] * rho * (1.0 + 3.0 * CX3[j] * U + 3.0 * U * U)
            f[:, :, 0, i] = feq_i + (f[:, :, 0, j] - feq_j)

    def _apply_outlet(self):
        if self.case.outlet == "extrapolate":
            self.f_new[:, :, -1, :] = self.f_new[:, :, -2, :]
            return
        u = self.case.u_in
        f = self.f_new
        f[:, :, -1, :] -= u * (f[:, :, -1, :] - f[:, :, -2, :])

    # ---------- 观测 ----------
    def force(self):
        """动量交换力（单位深度，格子单位）。"""
        return self._fx, self._fy

    def force3(self):
        return self._fx, self._fy, self._fz

    def macros(self):
        rho, ux, uy, uz = macros3d(self.f)
        return rho, ux, uy, uz

    def sample_metrics(self) -> dict:
        """**一次**宏量计算同时给出尾流各站位与展向诊断。

        性能要点：``macros3d`` 对 19 方向全场求和，β=5% 的网格约 1.6M 节点，
        单次约 0.3 s。若按站位各算一次（3 站 + 参考截面 = 4 次），每记录步
        要多付 1.2 s，会把 15 分钟的算例拖成数小时（v1.4 首版实测）。
        """
        rho, ux, uy, uz = macros3d(self.f)
        lo, hi = 1, self.case.ny - 1

        def flux(x):
            r = rho[:, lo:hi, x]
            u = ux[:, lo:hi, x]
            return (float(np.mean(np.sum(r * u * u + CS2 * r, axis=1))),
                    float(np.mean(np.sum(r * u, axis=1))))

        out = {"wake": {}}
        if self._wake_stations:
            ref, _ = flux(self._wake_ref_x)
            for s, x in self._wake_stations:
                fx, mx = flux(x)
                out["wake"][str(s)] = {"force": ref - fx, "flux": fx,
                                       "mass_flux": mx}
        m = self.fluid[lo:hi, :]
        if m.any():
            inplane = np.sqrt(np.mean(ux[:, lo:hi, :][:, m] ** 2
                                      + uy[:, lo:hi, :][:, m] ** 2))
            uz_rms = float(np.sqrt(np.mean(uz[:, lo:hi, :][:, m] ** 2)))
            out["uz_rms"] = uz_rms
            out["uz_rel"] = (uz_rms / inplane) if inplane > 0 else float("nan")
            # 稳定性读数（v1.6）：只统计流体节点，固体里 f≡0 ⇒ ρ≡0 会污染 min。
            # 放在这里而不是每个记录步都算，是因为它要与 uz_rel 共采样。
            rf = rho[:, lo:hi, :][:, m]
            out["rho_min"] = float(rf.min())
            out["rho_max"] = float(rf.max())
        return out

    def section_flux(self, x: int, drop_wall_rows: bool = True) -> dict:
        """截面 x 的展向平均守恒矩：flux=Σ_y(ρu_x²+p)、mass=Σ_y ρu_x。"""
        rho, ux, _, _ = self.macros()
        lo, hi = (1, self.case.ny - 1) if drop_wall_rows else (0, self.case.ny)
        r = rho[:, lo:hi, x]
        u = ux[:, lo:hi, x]
        return {"flux": float(np.mean(np.sum(r * u * u + CS2 * r, axis=1))),
                "mass": float(np.mean(np.sum(r * u, axis=1))),
                "rho": float(np.mean(np.sum(r, axis=1)))}

    def wake_forces(self, include_ref: bool = True):
        """严格两截面尾流口径（展向平均）：返回 [(x/D, dict), ...]。

        内部只做一次宏量计算（见 ``sample_metrics`` 的性能说明）。
        """
        if not self._wake_stations:
            return []
        sm = self.sample_metrics()["wake"]
        return [(s, sm[str(s)]) for s, _ in self._wake_stations]

    def spanwise_diagnostics(self) -> dict:
        """展向非均匀性：u_z 的 rms 相对面内脉动 rms（一次宏量计算）。"""
        sm = self.sample_metrics()
        return {"uz_rms": float(sm.get("uz_rms", 0.0)),
                "rel": float(sm.get("uz_rel", 0.0))}

    def spanwise_spectrum(self, xs=None, lo: int = 1,
                          hi: int | None = None) -> dict:
        """末态展向谱：把 u_z 沿 z 做 rfft，给出「一个盒长内的周期数」m。

        为什么需要它
        ------------
        v1.4 §14.4 的教训：Re=100 下 nz=8/16/24 的 St/Cd 只差 6/150 ppm，
        看着像"展向收敛"，实则**没有三维结构可收敛**——把空判据当证据。
        唯一能把这两种情形分开的读数是展向自由度**自己的**谱：
        ``frac_3d=0``（m=0 绝对主导）⇒ 流动在 z 上均匀，就是二维；
        m≥1 有峰 ⇒ 真有三维结构，且峰位给出 ``λ_z = Lz/m``。

        归一化与口径：``rfft(u_z, axis=0)/nz`` 后对 y 取功率平均。``P[0]`` 是
        展向平均分量的能量（二维分量），``Σ_{m≥1}P`` 是三维能量。

        **不要把 ``frac_3d`` 单独报出来**（v1.5 实测教训）：它是"uz 内部"的
        占比，当整个 uz 场只剩机器零（1e-16）时，p0 也是 1e-33，比值仍可给出
        0.73 —— 首版就因此在"无三维结构"的算例上读到"73% 三维"。真正可比的
        量是 ``p3d_rel = Σ_{m≥1}P / mean(u_x²+u_y²)``：把三维动能直接放在
        面内动能的标尺上，二维流动给 1e-30，真三维给 1e-3–1e-1，相差 20 个
        量级。``is_3d`` 用 ``p3d_rel > 1e-8`` 作门限（远高于 double 噪声、
        远低于任何可测三维结构），但它只表示**三维能量可测**，**不等于失稳**：
        长盒子里最慢的黏性模式衰减很慢（nz=64、Re=250 时 26T 只掉一个量级），
        残余三维能量可以长期停在 1e-7 而不增长。判失稳必须再比对
        ``scripts/11`` 的 ``decay_audit``（实测衰减率 ÷ 黏性衰减率）。

        **限制（必须与结果同报）**：周期盒只允许 ``λ_z = Lz/m``。Lz=4D 时可用
        波长是 4D/2D/1.33D/1D/0.8D…，Mode A（3–4D）落在 m=1、Mode B
        （0.8–1D）落在 m=4/5，都只在**带边**。因此本方法能回答"有没有三维
        结构、哪个 m 占优"，不能替代自由盒长给出自然波长（见报告 §17.3）。
        """
        rho, ux, uy, uz = self.macros()
        nz, ny, nx = uz.shape
        hi = ny - 1 if hi is None else hi
        d = self.case.d if self.case.d > 0 else 1.0
        if xs is None:
            xs = [x for _, x in self._wake_stations] or [nx // 2]
        out = {"nz": nz, "d": d, "stations": []}
        for x in xs:
            u = np.asarray(uz[:, lo:hi, x], dtype=np.float64)
            a = np.fft.rfft(u, axis=0) / nz
            p = np.mean(np.abs(a) ** 2, axis=1)
            m = np.arange(p.size)
            lam = np.where(m > 0, nz / np.maximum(m, 1), np.inf)
            p0 = float(p[0])
            p3 = float(p[1:].sum())
            tot = p0 + p3
            up = np.asarray(ux[:, lo:hi, x], dtype=np.float64)
            vp = np.asarray(uy[:, lo:hi, x], dtype=np.float64)
            inplane = float(np.mean(up * up + vp * vp))
            k = int(np.argmax(p[1:]) + 1) if p.size > 1 else 0
            p3_rel = (p3 / inplane) if inplane > 0 else 0.0
            out["stations"].append({
                "x": int(x),
                "x_over_d": float((x - self.case.cx_c) / d),
                "m": m.tolist(), "power": p.tolist(),
                "lambda_over_d": (lam / d).tolist(),
                "p0": p0, "p3d": p3, "inplane": inplane,
                "p3d_rel": float(p3_rel),
                "p0_rel": float(p0 / inplane) if inplane > 0 else 0.0,
                "is_3d": bool(inplane > 0 and p3_rel > 1e-8),
                "frac_3d": float(p3 / tot) if tot > 0 else 0.0,
                "m_peak": k,
                "lambda_peak_over_d": (float(nz / k / d) if k
                                       else float("inf")),
                "amp_peak": float(np.sqrt(2.0 * p[k])) if p.size > k else 0.0,
            })
        return out


    def spanwise_correlation(self, xs=None, lo: int = 1,
                             hi: int | None = None) -> dict:
        """展向两点相关 ``R(r)`` 与积分尺度 ``L_c``（一个拓扑独立的读数）。

        为什么还要它（谱已经给了 m）
        ---------------------------
        谱的峰位受**盒长**约束（``λ = Lz/m`` 只取离散值），盒子一换，峰位就跳；
        相关函数在物理空间算，是全波长谱的傅里叶变换，换盒子不改形状（只要
        盒子装得下）。两者一起看才能区分"真实空间结构变了"与"盒子只允许别的
        离散波长"。v1.5 的教训是**别只用一把尺子**。

        口径（必须写明，否则数字不可比）
        --------------------------------
        - **循环相关**：``R(r) = mean_z u'(z) u'(z+r)``，``u' = u_z − <u_z>_z``
          （先减展向均值，否则 m=0 的直流项把 R 钉在 1，看不出任何结构）。
        - **积分尺度**：``L_c = Σ_{r=0}^{r0−1} R(r)``（格子单位，矩形法，dz=1），
          ``r0`` 是首个 ``R ≤ 0`` 的位置。不除以 R(0)（已归一化），不做 1/e 拟合
          ——指数拟合在双模式叠加时会给出无意义的长度。
        - 纯正弦 ``u = A sin(2πm z/nz)`` 时解析结果为 ``R(r)=cos(2πm r/nz)``、
          ``r0 = nz/(4m)``：单测按此三条锁死。

        ``valid=False`` 表示该站 ``u_z`` 在 z 上是常数（二维流动），``R`` 无定义。
        """
        rho, ux, uy, uz = self.macros()
        nz, ny, nx = uz.shape
        hi = ny - 1 if hi is None else hi
        d = self.case.d if self.case.d > 0 else 1.0
        if xs is None:
            xs = [x for _, x in self._wake_stations] or [nx // 2]
        nmax = nz // 2
        out = {"nz": nz, "d": d, "stations": []}
        for x in xs:
            u = np.asarray(uz[:, lo:hi, x], dtype=np.float64)
            u = u - u.mean(axis=0, keepdims=True)
            var = float(np.mean(u * u))
            rec = {"x": int(x), "x_over_d": float((x - self.case.cx_c) / d),
                   "valid": bool(var > 0.0)}
            if var <= 0.0:
                rec.update({"R": [0.0] * (nmax + 1), "r0": 0,
                            "Lc_over_d": 0.0, "r0_over_d": 0.0})
                out["stations"].append(rec)
                continue
            R = [float(np.mean(u * np.roll(u, -r, axis=0))) / var
                 for r in range(nmax + 1)]
            r0 = next((i for i, v in enumerate(R) if v <= 0.0), nmax + 1)
            Lc = float(np.sum(R[:r0]))
            rec.update({"R": R, "r_over_d": [r / d for r in range(nmax + 1)],
                        "r0": int(r0), "r0_over_d": float(r0 / d),
                        "Lc_over_d": float(Lc / d)})
            out["stations"].append(rec)
        return out


# 展向模式的物理波长带（文献值，用于把离散 m 归到 A/B 两支）
SPAN_MODE_BANDS = {"A": (3.0, 4.0), "B": (0.8, 1.2)}


def mode_band_energy(power, nz: int, d: float, bands=None) -> dict:
    """把展向谱功率按**物理波长带**汇总（而不是按离散 m 逐个看）。

    周期盒只允许 ``λ = Lz/m``，同一个物理模式在不同盒长下落成不同的 m：
    Lz=4D 时 Mode A(3–4D) 只能落到 m=1、Mode B(0.8–1D) 落到 m=4/5；
    Lz=6D 时前者落 m=2、后者落 m=6/7。**只报峰位 m 无法跨盒长比较**，
    所以这里把「波长落在带内」的 m 全部加总，给出与盒子无关的能量占比。

    返回每个带的 ``m`` 列表、能量、``frac_of_p3d``，以及不属于任何带的
    ``unassigned_m``。``repr_ok=False`` 表示该带在本盒长下**没有可表示的 m**
    （此时"带内能量为 0"是盒子造成的，不能当物理结论——v1.4 空判据的同类陷阱）。
    """
    p = np.asarray(power, dtype=np.float64)
    bands = SPAN_MODE_BANDS if bands is None else bands
    m = np.arange(p.size)
    lam = np.where(m > 0, float(nz) / np.maximum(m, 1) / float(d), np.inf)
    p3 = float(p[1:].sum()) if p.size > 1 else 0.0
    out = {"p3d": p3, "bands": {}}
    used: set[int] = set()
    for name, (lo, hi) in bands.items():
        sel = (m > 0) & (lam >= lo) & (lam <= hi)
        idx = [int(i) for i in np.flatnonzero(sel)]
        e = float(p[sel].sum()) if sel.any() else 0.0
        out["bands"][name] = {
            "lambda_range": [float(lo), float(hi)],
            "m": idx, "repr_ok": bool(idx),
            "energy": e,
            "frac_of_p3d": (e / p3) if p3 > 0 else 0.0}
        used.update(idx)
    out["unassigned_m"] = [i for i in range(1, p.size) if i not in used]
    return out


# ---------------------------------------------------------------- 解析验证
def representable_wavelengths(nz: int, d: float) -> np.ndarray:
    """周期盒允许的展向波长 ``λ_z = Lz/m``（以 D 为单位），按 m=1..nz/2 排列。

    用来判断"某个展向扫描是否有意义"：若所有允许波长都远小于目标物理模式的
    波长，那么"各 nz 结果一致"只能说明盒子把它压掉了，不是分辨率收敛
    （v1.4 §14.4 的空判据）。Mode A 的波长带是 3–4D，故 ``Lz ≥ 3D`` 才可能
    装得下；``Lz = 1D``（nz=16、D=16）在 [3D, 4D] 内没有任何允许波长。
    """
    m = np.arange(1, max(int(nz) // 2, 1) + 1, dtype=float)
    return float(nz) / m / float(d)


def channel_poiseuille(ny: int = 17, nz: int = 4, nx: int = 8, tau: float = 0.7,
                       g: float = 1e-6, steps: int | None = None,
                       rho0: float = 1.0, tail: int = 1000):
    """平面通道 Poiseuille（体积极小 ⇒ Stokes 极限），返回数值/解析剖面。

    壁面在 y=0.5 与 y=ny−1.5（半程反弹），H=ny−2，
    解析 ``u(y) = g/(2ν)·(y−0.5)(H−(y−0.5))``，``u_max = g H²/(8ν)``。

    **步数必须随 H² 缩放（v1.4 修正）**：出口/稳态建立的特征时间是
    ``t_diff = H²/(π²ν)``（注意是 π²，不是 1）。固定 40000 步在 H=63 只折合
    6.6 个特征时间，残余瞬变 ``e^{-6.6} ≈ 1.4e-3`` 与二阶离散误差同量级甚至
    更大，会把二阶收敛**误判成零阶**（首轮实测 ny=65 的 L2 反而回升、
    收敛阶 −0.13）。默认取 ``12 t_diff``（≥4000 步，残余瞬变 ~6e-6），并回传
    末段 ``tail`` 步的相对漂移作为"确已收敛"的证据。
    """
    nu = (tau - 0.5) / 3.0
    H = ny - 2
    t_diff = H * H / (np.pi ** 2 * nu)
    if steps is None:
        steps = int(max(4000, 12.0 * t_diff))
    case = Case3D(nx=nx, ny=ny, nz=nz, d=0.0, u_in=0.0, re=1.0,
                  top_bottom="no_slip", inlet="periodic", outlet="periodic",
                  sponge_width=0, sponge_sigma=0.0, trip=0.0,
                  body_force=g, tau_direct=float(tau))
    sim = Solver3D(case)
    for _ in range(max(0, steps - tail)):
        sim.step()
    _, ux0, _, _ = sim.macros()
    for _ in range(tail):
        sim.step()
    _, ux, _, _ = sim.macros()
    prof = ux.mean(axis=(0, 2))
    prof0 = ux0.mean(axis=(0, 2))
    yc = np.arange(1, ny - 1) - 0.5
    exact = g / (2.0 * nu) * yc * (H - yc)
    umax = float(prof[1:-1].max())
    drift = float(abs(umax - prof0[1:-1].max()) / max(abs(umax), 1e-300))
    return {"y": yc, "u_num": prof[1:-1], "u_exact": exact,
            "u_max_num": umax,
            "u_max_exact": float(g * H * H / (8.0 * nu)),
            "l2_rel": float(np.linalg.norm(prof[1:-1] - exact)
                            / np.linalg.norm(exact)),
            "drift": drift, "steps": int(steps),
            "t_diff": float(t_diff),
            "nu": nu, "tau": tau}


def shear_wave_decay(n: int = 16, nz: int = 4, tau: float = 0.6,
                     amp: float = 1e-4, steps: int = 4000,
                     lo: float = 1e-3, hi: float = 0.6):
    """周期盒剪切波衰减：u_y = A sin(k x) → A(t)=A0 e^{−νk²t}，反测 ν。

    返回拟合 ν 与输入 ν 的相对偏差（验证碰撞算子与黏性关系）。

    **拟合窗口按相对幅值自适应（v1.4 修正）**：旧实现用绝对阈值
    ``amps > 1e-16`` 选点。τ=1.2 时衰减率 ``νk²≈0.036/步``，840 步内
    幅值就从 1e-3 跌到 double 舍入噪声（~1e-19），末段那些"有效点"其实
    全是噪声，``log`` 拟合被拉平，ν 低估 25%。改为只取
    ``A/A0 ∈ [lo, hi]``，舍弃启动瞬变（>hi）与噪声地板（<lo）。
    """
    k = 2.0 * np.pi / n
    case = Case3D(nx=n, ny=nz, nz=nz, d=0.0, u_in=0.0, re=1.0,
                  top_bottom="periodic", inlet="periodic", outlet="periodic",
                  sponge_width=0, sponge_sigma=0.0, trip=0.0,
                  tau_direct=float(tau))
    sim = Solver3D(case)
    x = np.arange(n)
    prof = amp * np.sin(k * x)
    rho = np.ones((nz, nz, n))
    uy = np.broadcast_to(prof[None, None, :], rho.shape).copy()
    sim.f = equilibrium3d(rho, np.zeros_like(rho), uy, np.zeros_like(rho))
    amps = []
    for it in range(steps):
        sim.step()
        if it % 20 == 19:
            _, _, uyy, _ = sim.macros()
            # 投影到 sin(kx) 基上，抗噪
            amps.append(float(2.0 / n * np.sum(uyy[0, 0, :] * np.sin(k * x))))
    amps = np.asarray(amps)
    t = (np.arange(amps.size) + 1) * 20.0
    frac = amps / amps[0]
    m = (frac > lo) & (frac < hi)
    if m.sum() < 4:                       # 退化保护：放宽到噪声地板之上
        m = frac > 1e-8
    slope = np.polyfit(t[m], np.log(amps[m]), 1)[0]
    nu_fit = -slope / (k * k)
    nu_in = (tau - 0.5) / 3.0
    return {"nu_fit": float(nu_fit), "nu_input": float(nu_in),
            "rel_err": float(abs(nu_fit / nu_in - 1.0)),
            "n_fit": int(m.sum()), "fit_lo": float(frac[m].min()),
            "fit_hi": float(frac[m].max()),
            "amps": amps, "t": t, "k": float(k)}


def taylor_green(n: int = 32, tau: float = 0.55, amp: float = 0.05,
                 steps: int = 600, sample: int = 10):
    """3D Taylor-Green 涡（周期盒，**一个周期占满 n 个格子**）：初始衰减率 −6νk_l²。

    初场 ``u=A sin x cos y cos z``、``v=−A cos x sin y cos z``、``w=0``。

    **单位口径（v1.4 的关键澄清）**：LBM 全在格子单位，模式
    ``sin(2π i/n)`` 的格子波数 ``k_l = 2π/n``，离散 Laplacian 本征值
    ``−(2π/n)²``，因此 ``d lnE/dt|₀ = −2ν·3·k_l² = −6ν(2π/n)²``。
    首版误把 k_l 当成 1（物理单位下的"一个周期/2π域"），得到"衰减慢 6.3 倍
    ≈2π"的假异常；按格子波数重算后 n=32/48 的误差为 0.8%/0.3%。
    该初场在三维**不是** Euler 稳态（非线性项驱动 w），但总能衰减率
    ``dE/dt = −2νΩ`` 仍成立，故初始斜率是可解析判据。
    """
    L = 2.0 * np.pi
    x = np.arange(n) * L / n
    # 直接在求解器的 (z, y, x) 轴序上广播构造，避免 meshgrid+transpose 的
    # 轴序歧义（首版用转置，实测散度 4.5e-2、衰减率几乎为 0，即初场不是 TG）
    zz = x[:, None, None]
    yy = x[None, :, None]
    xx = x[None, None, :]
    ux = amp * np.sin(xx) * np.cos(yy) * np.cos(zz)
    uy = -amp * np.cos(xx) * np.sin(yy) * np.cos(zz)
    uz = np.zeros_like(ux)
    ux = np.broadcast_to(ux, (n, n, n)).copy()
    uy = np.broadcast_to(uy, (n, n, n)).copy()
    case = Case3D(nx=n, ny=n, nz=n, d=0.0, u_in=0.0, re=1.0,
                  top_bottom="periodic", inlet="periodic", outlet="periodic",
                  sponge_width=0, sponge_sigma=0.0, trip=0.0,
                  tau_direct=float(tau))
    sim = Solver3D(case)
    rho = np.ones((n, n, n))
    sim.f = equilibrium3d(rho, ux, uy, uz)
    e0 = 0.5 * float(np.mean(ux ** 2 + uy ** 2 + uz ** 2))
    ts, es, divs = [], [], []
    for it in range(steps):
        sim.step()
        if it % sample == sample - 1:
            _, ax, ay, az = sim.macros()
            es.append(0.5 * float(np.mean(ax ** 2 + ay ** 2 + az ** 2)))
            ts.append(float(it + 1))
            div = _divergence_spectral(ax, ay, az)
            divs.append(div)
    ts = np.asarray(ts)
    es = np.asarray(es)
    # 自适应窗口：只取 E/E0 ∈ [2e-4, 0.6] 的采样（避开启动瞬变与噪声底）
    frac = es / es[0]
    m = (frac > 2e-4) & (frac < 0.6)
    if m.sum() < 3:
        m = np.zeros(ts.size, dtype=bool)
        m[1:min(4, ts.size)] = True
    slope0 = float(np.polyfit(ts[m], np.log(es[m]), 1)[0])
    nu = (tau - 0.5) / 3.0
    k_l = 2.0 * np.pi / n          # 格子波数（一个周期占 n 个格子）
    slope_exact = -6.0 * nu * k_l * k_l
    return {"t": ts, "E": es, "E0": e0,
            "slope0": slope0, "slope_exact": float(slope_exact),
            "rel_err": float(abs(slope0 / slope_exact - 1.0)),
            "div_max": float(np.max(divs)) if divs else float("nan"),
            "k_lattice": float(k_l), "fit_window": (float(ts[m][0]),
                                                    float(ts[m][-1])),
            "nu": float(nu)}


def _divergence_spectral(ux, uy, uz) -> float:
    """周期盒散度 ‖∇·u‖_rms / ‖u‖_rms（谱方法）。

    注意数组轴序为 **(z, y, x)**：Kx 沿最后一轴、Ky 中轴、Kz 首轴。
    """
    n = ux.shape[-1]
    k = np.fft.fftfreq(n, d=1.0 / n) * 1j
    KX = k[None, None, :]
    KY = k[None, :, None]
    KZ = k[:, None, None]
    du = (np.fft.ifftn(KX * np.fft.fftn(ux))
          + np.fft.ifftn(KY * np.fft.fftn(uy))
          + np.fft.ifftn(KZ * np.fft.fftn(uz)))
    num = float(np.sqrt(np.mean(np.abs(du) ** 2)))
    scale = float(np.sqrt(np.mean(np.abs(ux) ** 2 + np.abs(uy) ** 2
                                  + np.abs(uz) ** 2)))
    return num / scale if scale > 0 else float("nan")
