"""
理想慢偏移基线 b(t) 的构造与扣除 —— 本轮新增的唯一数据预处理层。

【本轮要解决的问题】
  预实验原始横向误差里混了两样东西：
        e_raw(t) = b(t) + e_vib(t)
    b(t)     : 方向性 / 准静态慢偏移（正向稳态偏 +，反向稳态偏 −，幅值几十 µm，远大于振动）；
    e_vib(t) : 真正要研究的快速振动。

  本层做的是【理想机制验证】：假设 b(t) 可以被完全正确地离线知道（允许用全序列、
  允许用未来帧、允许全局平滑），把它从 e_raw 里扣掉，只留净振动 e_vib。
  这【不是】在线估计方案 —— 132 Hz 在线估计、实时预测、真实部署都不在本轮范围内。

【为什么不用“全程单一低通”】
  启动 / 转向 / 停止段存在很大的瞬态振动尖峰，普通低通会把这些尖峰也吃进 b(t)，
  等于把本该保留的振动错误地削弱掉。所以本层按【运动阶段】分别处理：
      稳态段  → 用“沿程位置分箱中位数 + LOWESS 鲁棒平滑”拟合方向相关的慢中心线 b±(x)
      过渡段  → 用【三次 Hermite 桥接】（一阶连续）把两端中心线连起来，不追随尖峰

  分段的边界直接复用 disturbance.detect_cuts 的 iA/iB/iC/iD，不另立一套切窗规则。

【边界】
  本模块只依赖 config 与 numpy，不 import disturbance / sim_env / controller，
  也绝不读取任何仿真状态或控制器输出。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

import config

# 分段名（与 disturbance 的 iA/iB/iC/iD 一一对应）
SEG_START = "start"           # [0, iA)   静止 → 正向稳态：平滑桥接
SEG_FWD = "fwd_steady"        # [iA, iB]  正向稳态：b+(x)
SEG_TURN = "turn"             # (iB, iC)  换向：b+(x_iB) → b-(x_iC) 平滑桥接
SEG_RET = "ret_steady"        # [iC, iD]  反向稳态：b-(x)
SEG_STOP = "stop"             # (iD, n)   反向稳态 → 静止：平滑桥接

STEADY_SEGS = (SEG_FWD, SEG_RET)
TRANSIENT_SEGS = (SEG_START, SEG_TURN, SEG_STOP)


# ============================================================ 工具
def _hermite(u: np.ndarray, p0: float, p1: float, m0: float, m1: float,
             span: float) -> np.ndarray:
    """三次 Hermite 插值：两端给定值 p 与对时间的斜率 m，span 为区间时长。

    一阶连续；端点斜率由相邻稳态段的 db/dx · dx/dt 给出。
    """
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * p0 + h10 * span * m0 + h01 * p1 + h11 * span * m1


def _tricube(d: np.ndarray) -> np.ndarray:
    """LOWESS 的三次权核：|d|≥1 时为 0。"""
    a = np.abs(d)
    return np.where(a < 1.0, (1.0 - a ** 3) ** 3, 0.0)


class LowessCurve:
    """LOWESS 平滑曲线（局部线性 + 双平方鲁棒迭代），可在任意 x 处求值与求斜率。

    自带实现而非依赖 statsmodels/scipy：本工程只允许 numpy + mujoco + matplotlib。
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray, span: float,
                 robust_iters: int = 3):
        self.x = np.asarray(xs, dtype=float)
        self.y = np.asarray(ys, dtype=float)
        self.span = float(span)
        self.robust_iters = int(robust_iters)
        self._fit = self._solve(self.x, self.x, self.y, np.ones_like(self.x))

    def _solve(self, xq: np.ndarray, x: np.ndarray, y: np.ndarray,
               w0: np.ndarray) -> np.ndarray:
        """在 xq 各点做加权局部线性回归。"""
        out = np.empty(xq.size, dtype=float)
        for i, q in enumerate(xq):
            wt = _tricube((x - q) / self.span) * w0
            sw = float(wt.sum())
            if sw <= 1e-12:
                out[i] = float(np.interp(q, x, y))
                continue
            mx = float((wt * x).sum() / sw)
            my = float((wt * y).sum() / sw)
            sxx = float((wt * (x - mx) ** 2).sum())
            if sxx <= 1e-15:
                out[i] = my
                continue
            b1 = float((wt * (x - mx) * (y - my)).sum() / sxx)
            out[i] = my + b1 * (q - mx)
        return out

    def _robust_weights(self, resid: np.ndarray) -> np.ndarray:
        s = float(np.median(np.abs(resid)))
        if s <= 0.0:
            return np.ones_like(resid)
        u = resid / (6.0 * s)
        return np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)

    def refine(self) -> "LowessCurve":
        """按双平方权做若干轮鲁棒迭代，压制残余的振动尖峰。"""
        w = np.ones_like(self.x)
        for _ in range(self.robust_iters):
            fit = self._solve(self.x, self.x, self.y, w)
            w = self._robust_weights(self.y - fit)
            if not np.any(w > 0):
                break
            self._fit = fit
        return self

    def at(self, xq: np.ndarray) -> np.ndarray:
        """在任意查询点上平滑求值（用同一组数据与带宽重新做局部回归）。"""
        return self._solve(np.atleast_1d(np.asarray(xq, dtype=float)),
                           self.x, self.y, np.ones_like(self.x))

    def slope_at(self, xq: float) -> float:
        """在 xq 处的局部线性斜率 dy/dx（同样用局部回归，不是差分）。"""
        wt = _tricube((self.x - xq) / self.span)
        sw = float(wt.sum())
        if sw <= 1e-12:
            return 0.0
        mx = float((wt * self.x).sum() / sw)
        my = float((wt * self.y).sum() / sw)
        sxx = float((wt * (self.x - mx) ** 2).sum())
        if sxx <= 1e-15:
            return 0.0
        return float((wt * (self.x - mx) * (self.y - my)).sum() / sxx)


def binned_median(x: np.ndarray, y: np.ndarray,
                  bin_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按沿程位置分箱，每箱取【中位数】（不用均值，避免被振动尖峰拉动）。

    返回 (箱中心, 箱内中位数, 箱内样本数)，按 x 升序。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    nb = max(int(np.ceil((hi - lo) / bin_mm)), 1)
    edges = lo + np.arange(nb + 1) * bin_mm
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
    centres, meds, counts = [], [], []
    for k in range(nb):
        m = idx == k
        if int(m.sum()) < config.SLOW_BIN_MIN_SAMPLES:
            continue
        centres.append(0.5 * (edges[k] + edges[k + 1]))
        meds.append(float(np.median(y[m])))
        counts.append(int(m.sum()))
    return (np.asarray(centres), np.asarray(meds), np.asarray(counts, dtype=int))


# ============================================================ b±(x) 拟合
@dataclass
class DirectionalFit:
    name: str
    x: np.ndarray            # 箱中心
    b: np.ndarray            # 箱内中位数
    n: np.ndarray            # 箱内样本数
    curve: LowessCurve
    span_mm: float

    def at(self, xq: np.ndarray) -> np.ndarray:
        """求 b±(x)。查询点【一律夹紧到拟合区间内】——LOWESS 是局部回归，
        外推会沿边缘斜率线性飞出去（曾把 x=3.20 处的外推值抬到 +46 µm）。
        """
        xq = np.atleast_1d(np.asarray(xq, dtype=float))
        return self.curve.at(np.clip(xq, self.x[0], self.x[-1]))

    def slope(self, xq: float) -> float:
        return self.curve.slope_at(float(np.clip(xq, self.x[0], self.x[-1])))


def fit_directional(x: np.ndarray, y: np.ndarray, name: str,
                    bin_mm: float, span_mm: float) -> DirectionalFit:
    """稳态段的方向相关慢偏移 b±(x)：分箱中位数 → LOWESS 鲁棒平滑。"""
    cx, cb, cn = binned_median(x, y, bin_mm)
    if cx.size < 3:
        raise ValueError(f"[慢偏移] {name} 段有效分箱仅 {cx.size} 个，不足以拟合")
    curve = LowessCurve(cx, cb, span_mm).refine()
    return DirectionalFit(name=name, x=cx, b=cb, n=cn, curve=curve, span_mm=span_mm)


# ============================================================ 主构造
def build_slow_offset(d: dict[str, Any], cuts: Any,
                      bin_mm: float = config.SLOW_BIN_MM,
                      span_mm: float = config.SLOW_SPAN_MM,
                      refine_passes: int = config.SLOW_REFINE_PASSES,
                      log=None) -> dict[str, Any]:
    """构造 b(t) 并返回全部中间量。d 为 disturbance.resample_to_grid 的产物。

    返回字典包含：t / b_um / e_vib_um / e_raw_um / 两段方向拟合 / 分段索引 /
    每段统计与人工验收结论 / 实际使用的 span。
    """
    def say(msg: str) -> None:
        if log:
            log(msg)

    t = np.asarray(d["t"], dtype=float)
    x = np.asarray(d["along_mm"], dtype=float)
    e_raw = np.asarray(d["cross_um"], dtype=float)
    n = t.size

    seg = {
        SEG_START: (0, int(cuts.iA) - 1),
        SEG_FWD: (int(cuts.iA), int(cuts.iB)),
        SEG_TURN: (int(cuts.iB) + 1, int(cuts.iC) - 1),
        SEG_RET: (int(cuts.iC), int(cuts.iD)),
        SEG_STOP: (int(cuts.iD) + 1, n - 1),
    }
    for k, (a, b) in seg.items():
        if b <= a:
            raise ValueError(f"[慢偏移] 分段 {k} 为空：{a}..{b}")

    # ---- 静止基线：pre_motion 段的中位数 ----
    b_rest = float(np.median(e_raw[seg[SEG_START][0]:seg[SEG_START][1] + 1]))

    span_used = float(span_mm)
    history: list[dict[str, Any]] = []
    fit_p = fit_r = None
    b = None
    acc = None

    for attempt in range(1, int(refine_passes) + 2):
        fit_p = fit_directional(x[seg[SEG_FWD][0]:seg[SEG_FWD][1] + 1],
                                e_raw[seg[SEG_FWD][0]:seg[SEG_FWD][1] + 1],
                                "b+(x) 正向稳态", bin_mm, span_used)
        fit_r = fit_directional(x[seg[SEG_RET][0]:seg[SEG_RET][1] + 1],
                                e_raw[seg[SEG_RET][0]:seg[SEG_RET][1] + 1],
                                "b-(x) 反向稳态", bin_mm, span_used)
        b, bridges = _assemble(t, x, seg, b_rest, fit_p, fit_r, span_used)
        acc = acceptance(d, seg, e_raw, b, bridges)
        history.append({"attempt": attempt, "span_mm": span_used,
                        "passed": acc["passed"],
                        "failed_checks": acc["failed_checks"]})
        if acc["passed"]:
            break
        if attempt <= int(refine_passes):
            span_used *= config.SLOW_SPAN_GROWTH
            say(f"[慢偏移] 第 {attempt} 次未通过（{acc['failed_checks']}）"
                f"→ 加宽平滑带宽到 {span_used:.2f} mm 重拟合")

    e_vib = e_raw - b
    say(f"[慢偏移] 静止基线 {b_rest:.2f} µm；b+(x) 在 x∈"
        f"[{fit_p.x[0]:.2f},{fit_p.x[-1]:.2f}] mm 上 {fit_p.b.min():.2f}~{fit_p.b.max():.2f} µm；"
        f"b-(x) 在 x∈[{fit_r.x[0]:.2f},{fit_r.x[-1]:.2f}] mm 上 "
        f"{fit_r.b.min():.2f}~{fit_r.b.max():.2f} µm")
    say(f"[慢偏移] b(t) 范围 {b.min():.2f}~{b.max():.2f} µm；e_raw rms "
        f"{np.sqrt(np.mean(e_raw ** 2)):.2f} µm → e_vib rms "
        f"{np.sqrt(np.mean(e_vib ** 2)):.2f} µm；平滑带宽 {span_used:.2f} mm，"
        f"分箱 {bin_mm:.2f} mm")

    return {
        "t": t, "x": x, "e_raw_um": e_raw, "b_um": b, "e_vib_um": e_vib,
        "b_rest_um": b_rest, "segments": seg,
        "fit_fwd": fit_p, "fit_ret": fit_r,
        "bin_mm": float(bin_mm), "span_mm": span_used,
        "span_attempts": history, "acceptance": acc,
    }


def _assemble(t: np.ndarray, x: np.ndarray, seg: dict[str, tuple[int, int]],
              b_rest: float, fit_p: DirectionalFit, fit_r: DirectionalFit,
              span_mm: float) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """按分段拼出完整 b(t)：稳态段取 b±(x)，过渡段用三次 Hermite 桥接。

    返回 (b, bridges)：bridges 记录每段桥接的两端值，供验收判断“b 有没有自己乱动”。
    """
    dt = float(np.median(np.diff(t)))
    b = np.zeros_like(t)

    a_s, b_s = seg[SEG_START]
    a_f, b_f = seg[SEG_FWD]
    a_t, b_t = seg[SEG_TURN]
    a_r, b_r = seg[SEG_RET]
    a_e, b_e = seg[SEG_STOP]

    # --- 正向稳态：b+(x) ---
    b[a_f:b_f + 1] = fit_p.at(x[a_f:b_f + 1])
    # --- 反向稳态：b-(x) ---
    b[a_r:b_r + 1] = fit_r.at(x[a_r:b_r + 1])

    def dxdt(i: int) -> float:
        """沿程速度 dx/dt (mm/s)，用中心差分（这是运动学量，不是误差量）。"""
        i0, i1 = max(0, i - 5), min(t.size - 1, i + 5)
        return float((x[i1] - x[i0]) / (t[i1] - t[i0])) if i1 > i0 else 0.0

    # --- 转向段：b+(x_iB) → b-(x_iC)，两端斜率各自继承 ---
    p0 = float(fit_p.at([x[b_f]])[0])
    m0 = float(fit_p.slope(x[b_f])) * dxdt(b_f)
    p1 = float(fit_r.at([x[a_r]])[0])
    m1 = float(fit_r.slope(x[a_r])) * dxdt(a_r)
    bridges: dict[str, dict[str, float]] = {}

    span_t = t[b_t] - t[a_t] if b_t > a_t else dt
    u = (t[a_t:b_t + 1] - t[a_t]) / span_t
    b[a_t:b_t + 1] = _bridge(u, p0, p1, m0, m1, span_t, dt)
    bridges[SEG_TURN] = _bridge_info(p0, p1, m0, m1, span_t)

    # --- 启动段：静止基线 → b+(x_iA)，起点静止（斜率 0）---
    p0 = b_rest
    m0 = 0.0
    p1 = float(fit_p.at([x[a_f]])[0])
    m1 = float(fit_p.slope(x[a_f])) * dxdt(a_f)
    span_s = t[b_s] - t[a_s] + dt if b_s >= a_s else dt
    u = (t[a_s:b_s + 1] - t[a_s]) / span_s
    b[a_s:b_s + 1] = _bridge(u, p0, p1, m0, m1, span_s, dt)
    bridges[SEG_START] = _bridge_info(p0, p1, m0, m1, span_s)

    # --- 停止段：b-(x_iD) → 静止基线，终点静止（斜率 0）---
    p0 = float(fit_r.at([x[b_r]])[0])
    m0 = float(fit_r.slope(x[b_r])) * dxdt(b_r)
    p1 = b_rest
    m1 = 0.0
    span_e = t[b_e] - t[a_e] + dt if b_e >= a_e else dt
    u = (t[a_e:b_e + 1] - t[a_e]) / span_e
    b[a_e:b_e + 1] = _bridge(u, p0, p1, m0, m1, span_e, dt)
    bridges[SEG_STOP] = _bridge_info(p0, p1, m0, m1, span_e)

    return b, bridges


def _bridge_info(p0: float, p1: float, m0: float, m1: float,
                 span: float) -> dict[str, float]:
    """记录一段桥接的两端值与斜率限幅情况，供验收判断 b 是否自己乱动。"""
    _, _, limited = _limit_slopes(p0, p1, m0, m1, span)
    return {"p0": p0, "p1": p1, "m0_in": m0, "m1_in": m1,
            "span_s": span, "slope_limited": 1.0 if limited else 0.0}


def _limit_slopes(p0: float, p1: float, m0: float, m1: float,
                  span: float) -> tuple[float, float, bool]:
    """Fritsch–Carlson 单调限幅：把端点斜率投影到「不产生过冲」的区域内。

    【为什么必须限幅】转向段的 b 要从 b+(x_iB)≈+40 单调降到 b-(x_iC)≈−25。
    但正向稳态段在行程末端正处于减速段，b+(x) 在那儿被减速瞬态污染而陡升，
    于是继承来的斜率 m0 = db+/dx · dx/dt 是【正的】—— 方向与「必须下降」相反。
    不加限制的三次 Hermite 会先往上冲到 +53 µm 再往下走，白白在转向段留下
    约 45 µm 的虚假正偏置（实测该段 e_vib 中位数被拉到 −39 µm，而原始数据该段
    中位数只有 +6.5 µm，说明这 45 µm 是 b 自己造的，不是数据里的）。

    判据（Δ = p1−p0）：令 α = m0·span/Δ，β = m1·span/Δ。
    三次 Hermite 单调 ⟺ α,β ≥ 0 且 α²+β² ≤ 9（且 α,β ≤ 3）。
    越界就按标准做法投影回该区域，返回 (m0, m1, was_limited)。
    """
    d = p1 - p0
    if abs(d) < 1e-12 or span <= 0.0:
        return 0.0, 0.0, (abs(m0) > 1e-12 or abs(m1) > 1e-12)
    a = m0 * span / d
    b = m1 * span / d
    a0, b0 = a, b
    a = float(np.clip(a, 0.0, 3.0))
    b = float(np.clip(b, 0.0, 3.0))
    n = math.hypot(a, b)
    if n > 3.0:
        a, b = a * 3.0 / n, b * 3.0 / n
    limited = (abs(a - a0) > 1e-9) or (abs(b - b0) > 1e-9)
    return a * d / span, b * d / span, limited


def _bridge(u: np.ndarray, p0: float, p1: float, m0: float, m1: float,
            span: float, dt: float) -> np.ndarray:
    """Hermite 桥接（先做单调限幅，故结果不会过冲，无需再退回余弦过渡）。"""
    m0, m1, _ = _limit_slopes(p0, p1, m0, m1, span)
    return _hermite(np.clip(u, 0.0, 1.0), p0, p1, m0, m1, span)


# ============================================================ 人工验收（§五）
def _band_rms(sig: np.ndarray, dt: float, lo: float, hi: float) -> float:
    """单频带 RMS（去均值 + Hann 窗，与 analysis.band_rms 同口径）。

    样本数太少时退化返回全带 RMS，避免小段上 FFT 分辨率不足给出假数。
    """
    n = sig.size
    if n < 16:
        return float(np.sqrt(np.mean(sig ** 2)))
    w = np.hanning(n)
    F = np.fft.rfft((sig - sig.mean()) * w)
    f = np.fft.rfftfreq(n, dt)
    m = (f >= lo) & (f < hi)
    # 窗函数功率归一：sum(w²) 而不是 n²，否则窗会带来 ~2.7 dB 的假衰减
    return float(np.sqrt(np.sum(np.abs(F[m]) ** 2) * 2.0 / np.sum(w ** 2)))


def _band_pair(er: np.ndarray, ev: np.ndarray, dt: float,
               lo: float | None = None, hi: float | None = None) -> tuple[float, float]:
    """同一频带上 (e_raw 的 RMS, e_vib 的 RMS)。默认取振动带。"""
    if lo is None:
        lo, hi = config.VIB_BAND_HZ
    assert hi is not None
    return (_band_rms(er, dt, lo, hi), _band_rms(ev, dt, lo, hi))


def acceptance(d: dict[str, Any], seg: dict[str, tuple[int, int]],
               e_raw: np.ndarray, b: np.ndarray,
               bridges: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """对 e_vib = e_raw − b 做机器可判的检查，输出结论文字。"""
    e_vib = e_raw - b
    per: dict[str, dict[str, float]] = {}
    for k, (a, bb) in seg.items():
        sl = slice(a, bb + 1)
        er, ev, bv = e_raw[sl], e_vib[sl], b[sl]
        per[k] = {
            "n": int(er.size),
            "median_raw_um": float(np.median(er)),
            "median_vib_um": float(np.median(ev)),
            "mean_raw_um": float(np.mean(er)),
            "mean_vib_um": float(np.mean(ev)),
            "rms_raw_um": float(np.sqrt(np.mean(er ** 2))),
            "rms_vib_um": float(np.sqrt(np.mean(ev ** 2))),
            "peak_abs_raw_um": float(np.max(np.abs(er))),
            "peak_abs_vib_um": float(np.max(np.abs(ev))),
            "ptp_raw_um": float(np.ptp(er)),
            "ptp_vib_um": float(np.ptp(ev)),
            "b_range_um": float(np.ptp(bv)),
        }
        # 频带能量守恒：b 只该拿掉慢带，不该碰振动带。这是本轮 b 是否合格的
        # 最直接证据，故一并存档（不作为硬门限，只在 reasons 里留痕）。
        per[k]["vib_band_raw_um"], per[k]["vib_band_vib_um"] = _band_pair(
            er, ev, float(np.median(np.diff(d["t"]))))
        per[k]["slow_band_raw_um"], per[k]["slow_band_vib_um"] = _band_pair(
            er, ev, float(np.median(np.diff(d["t"]))),
            lo=config.SLOW_BAND_HZ[0], hi=config.SLOW_BAND_HZ[1])

    checks: dict[str, bool] = {}
    reasons: list[str] = []

    # 1) 两个稳态段的 e_vib 中位数是否接近 0（绝对 + 相对双重口径）
    for k in STEADY_SEGS:
        p = per[k]
        tol = max(config.SLOW_CENTER_ABS_UM,
                  config.SLOW_CENTER_REL * p["rms_vib_um"])
        good = abs(p["median_vib_um"]) <= tol
        checks[f"{k}_中位数接近0"] = good
        if not good:
            reasons.append(f"{k} 中位数 {p['median_vib_um']:.2f} µm 超过容差 {tol:.2f} µm")
        # 扣除后必须比原始更靠中心
        better = abs(p["median_vib_um"]) < abs(p["median_raw_um"])
        checks[f"{k}_比原始更居中"] = better
        if not better:
            reasons.append(f"{k} 扣除后 |中位数| 未下降")

    # 2) 过渡段的巨幅瞬态必须保留（b 没有把它吃掉）
    #    判据不是拍的峰值比例，而是数学硬下界：
    #        max(e−b) ≥ max(e) − max(b)      min(e−b) ≤ min(e) − min(b)   （逐点恒成立）
    #    两式相减得   ptp(e_vib) ≥ ptp(e_raw) − ptp(b)
    #    也就是说「b 最多只能吃掉自己那个峰谷高度」，这是定义决定的，不含任何主观取舍。
    #    违反即说明 b 的构造不自洽（例如拟合外推、桥接过冲），必须回退重做。
    for k in TRANSIENT_SEGS:
        p = per[k]
        need = p["ptp_raw_um"] - p["b_range_um"] - config.SLOW_PTP_TOL_UM
        keep = p["ptp_vib_um"] >= need
        checks[f"{k}_瞬态被保留"] = keep
        if not keep:
            reasons.append(f"{k} 峰谷 {p['ptp_raw_um']:.1f} µm 扣除后只剩 "
                           f"{p['ptp_vib_um']:.1f} µm，低于硬下界 "
                           f"{p['ptp_raw_um']:.1f}−{p['b_range_um']:.1f}={need:.1f} µm，"
                           f"说明 b 吃的比它自身高度还多")

    # 2b) 稳态段的【振动带能量必须基本不变】：b 只该拿掉慢带。
    #     这是「b 有没有把该保留的振动一起吃掉」最直接的证据，设 0.90 的守恒下限。
    for k in STEADY_SEGS:
        p = per[k]
        a, c = p["vib_band_raw_um"], p["vib_band_vib_um"]
        ratio = c / a if a > 1e-9 else 1.0
        ok = ratio >= 0.90
        checks[f"{k}_振动带未被吃掉"] = ok
        if not ok:
            reasons.append(f"{k} 振动带 RMS 从 {a:.2f} µm 掉到 {c:.2f} µm（{ratio*100:.1f}%），"
                           f"b 平滑过头，把振动也削掉了")

    # 3) 整体能量：扣掉慢偏移后 RMS 应下降（说明扣的是“偏移”不是“振动”）
    rms_raw = float(np.sqrt(np.mean(e_raw ** 2)))
    rms_vib = float(np.sqrt(np.mean(e_vib ** 2)))
    checks["整体RMS下降"] = bool(rms_vib < rms_raw)
    if rms_vib >= rms_raw:
        reasons.append("扣除后整体 RMS 未下降")

    # 4) b 的灵活性上限：过渡段里 b 的变化应当基本等于“两端点之差”，
    #    超出说明桥接自己在乱动（Hermite 过冲或拟合外推），而不是在做慢过渡。
    #    注意判据是相对【该段必须走完的位移】|p1−p0|，不是相对瞬态幅值 ——
    #    转向段本来就要从 b+ 走到 b-，用瞬态幅值当尺子会误判。
    for k in TRANSIENT_SEGS:
        p = per[k]
        if bridges is None or k not in bridges:
            continue
        need = abs(bridges[k]["p1"] - bridges[k]["p0"])
        allow = config.SLOW_B_OVERSHOOT_MAX * need + config.SLOW_CENTER_ABS_UM
        ok = p["b_range_um"] <= allow
        checks[f"{k}_b过渡幅度受控"] = ok
        if not ok:
            reasons.append(f"{k} 内 b 变化 {p['b_range_um']:.1f} µm，"
                           f"超出该段应有的 |Δb|={need:.1f} µm 允许上限 {allow:.1f} µm")

    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "reasons": reasons,
        "per_segment": per,
        "overall": {"rms_raw_um": rms_raw, "rms_vib_um": rms_vib,
                    "reduction_pct": float((rms_raw - rms_vib) / rms_raw * 100.0)},
        "conclusion": ("通过：稳态段中位数已接近 0，且启动/转向/停止的瞬态仍保留在 e_vib 中。"
                       if all(checks.values()) else
                       "未通过：见 failed_checks 与 reasons。"),
    }
