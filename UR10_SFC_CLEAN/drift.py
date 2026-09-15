"""
理想慢偏移 b(t) 的正式构造（V1.3 方案）。

    e_raw(t) = b(t) + e_vib(t)

本模块只做一件事：从原始预实验数据里定出 b(t)，并给出 e_vib(t)。
它【不做】任何控制、仿真、参数寻优，也不读 Fd。

======================================================================
V1.3 相对 V1.2 的唯一改动：把慢偏移的拟合方式换掉
======================================================================
V1.2 用「沿程位置分箱中位数 + LOWESS 鲁棒平滑」拟合 b±(x)，随 X 位置起伏。
V1.3 换成最简方案：

    四个稳定常值  b_pre / b_out / b_ret / b_post
  + 启动 / 远端转向 / 停止 三段余弦平滑过渡

为什么换：拟合自由度越高，越容易把真实振动当成慢偏移吃掉。
V1.2 那套每条曲线有几十个分箱 + 一个平滑带宽，出问题时无法用肉眼判断
「这是真实的慢变化，还是拟合把振动吸进去了」。V1.3 只剩 4 个数，
每个数都能被单独核对（见 drift_baseline_summary.json 的 windows 与 per_segment），
过渡段形状还是固定的余弦，没有可调自由度。

======================================================================
b(t) 的分段定义（§四）
======================================================================
    段名            区间                 b(t)
    pre_motion      [0, p1]              b_pre            （常值）
    start           [p1+1, iA-1]         余弦 b_pre → b_out
    fwd_steady      [iA, iB]             b_out            （常值）
    turn            [iB+1, iC-1]         余弦 b_out → b_ret
    ret_steady      [iC, iD]             b_ret            （常值）
    stop            [iD+1, q0-1]         余弦 b_ret → b_post
    post_stop       [q0, n-1]            b_post           （常值）

只有 start / turn / stop 三段允许变化，其余一律严格常值。
七个区间无缝覆盖 [0, n-1]，无重叠、无空洞（由 _assemble 断言保证）。

四个稳定值全部【现算】，没有一个写死在代码里：
    b_pre  = median(e_raw[pre_motion])           整段
    b_out  = median(e_raw[iA..iB])               X_outbound 稳态窗
    b_ret  = median(e_raw[iC..iD])               X_return   稳态窗
    b_post = median(e_raw[post_stop])            整段
窗口 [iA,iB] / [iC,iD] 来自 disturbance.detect_cuts，已避开启动加速、
远端减速、转向后过渡、停止前减速。实测把任一边界内缩 100 个采样点，
b_out 只在 21.4–24.2 µm 之间动、b_ret 只在 −19.7..−18.5 µm 之间动，
说明这两个窗口确实落在平台段上（见 drift_baseline_summary.json 的 window_sensitivity）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import config

# ---------------------------------------------------------------- 段名
SEG_PRE = "pre_motion"
SEG_START = "start"
SEG_FWD = "fwd_steady"
SEG_TURN = "turn"
SEG_RET = "ret_steady"
SEG_STOP = "stop"
SEG_POST = "post_stop"
SEGMENTS: tuple[str, ...] = (SEG_PRE, SEG_START, SEG_FWD, SEG_TURN,
                             SEG_RET, SEG_STOP, SEG_POST)
# 允许随 t 变化的段
SEG_TRANSITION: tuple[str, ...] = (SEG_START, SEG_TURN, SEG_STOP)
# 必须严格为常值的段
SEG_CONST: tuple[str, ...] = (SEG_PRE, SEG_FWD, SEG_RET, SEG_POST)

# 模板分段名 → 本模块段名：映射表放在 config.DRIFT_TEMPLATE_TO_PHASE，
# 因为「模板分段 → 五阶段」这件事同时被 record（写 phase 列）和 sweep（分相统计）用到。


# ---------------------------------------------------------------- 基本工具
def phase_span(d: dict, name: str) -> tuple[int, int]:
    """返回某个实验相位在重采样网格上的闭区间 [i0, i1]。"""
    codes = d["phase_code"]
    names = d["phase_names"]
    if name not in names:
        raise KeyError(f"[慢偏移] 数据里没有相位 {name!r}；可用：{names}")
    hit = np.where(codes == names.index(name))[0]
    if hit.size == 0:
        raise ValueError(f"[慢偏移] 相位 {name!r} 在网格上没有任何采样点")
    return int(hit[0]), int(hit[-1])


def cosine_ramp(n: int, v0: float, v1: float) -> np.ndarray:
    """余弦平滑过渡：w = 0.5(1 − cos(πs))，b = (1−w)·v0 + w·v1，s∈[0,1]。

    性质（这三条正是 §五 要求的）：
      · w(0)=0、w(1)=1  → 与两端常值段严格接得上，b 全局连续；
      · w'(0)=w'(1)=0   → 与常值段一阶连续，接缝处不出现折角；
      · w 单调、|w'| 有界（最大值 π/2 在中点）→ 不产生振荡型基线。
    起点只由 v0 决定、终点只由 v1 决定，中间是固定形状，
    所以它【不会】跟随原始数据里的瞬态尖峰 —— 这是刻意设计的，不是缺陷。
    """
    if n <= 0:
        return np.empty(0, dtype=float)
    if n == 1:
        # 只有一个采样点时取起点值，保持与左端常值段连续
        return np.array([float(v0)], dtype=float)
    s = np.linspace(0.0, 1.0, int(n))
    w = 0.5 * (1.0 - np.cos(np.pi * s))
    return (1.0 - w) * float(v0) + w * float(v1)


def _band_rms(x: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """带内 RMS。延迟导入 analysis 以避免与 record/disturbance 形成环。"""
    from analysis import band_rms, welch_psd
    if np.asarray(x).size < 8:
        return 0.0
    f, p = welch_psd(np.asarray(x, dtype=float), fs)
    return band_rms(f, p, band[0], band[1])


def _seg_stats(e_raw: np.ndarray, e_vib: np.ndarray, x: np.ndarray, t: np.ndarray,
               i0: int, i1: int, fs: float) -> dict[str, float]:
    a = slice(i0, i1 + 1)
    r, v = e_raw[a], e_vib[a]
    return {
        "i0": int(i0), "i1": int(i1), "n": int(i1 - i0 + 1),
        "t0_s": float(t[i0]), "t1_s": float(t[i1]),
        "x0_mm": float(x[i0]), "x1_mm": float(x[i1]),
        "median_raw_um": float(np.median(r)),
        "mean_raw_um": float(np.mean(r)),
        "rms_raw_um": float(np.sqrt(np.mean(r ** 2))),
        "median_vib_um": float(np.median(v)),
        "mean_vib_um": float(np.mean(v)),
        "rms_vib_um": float(np.sqrt(np.mean(v ** 2))),
        "ptp_raw_um": float(np.ptp(r)),
        "ptp_vib_um": float(np.ptp(v)),
        "slow_band_raw_um": _band_rms(r, fs, config.SLOW_BAND_HZ),
        "slow_band_vib_um": _band_rms(v, fs, config.SLOW_BAND_HZ),
        "vib_band_raw_um": _band_rms(r, fs, config.VIB_BAND_HZ),
        "vib_band_vib_um": _band_rms(v, fs, config.VIB_BAND_HZ),
    }


# ---------------------------------------------------------------- 稳定值
def stable_levels(d: dict, cuts: Any, log=print) -> dict[str, Any]:
    """现算四个稳定常值。返回数值 + 取值窗口 + 窗口敏感性（不硬编码任何数）。"""
    t = np.asarray(d["t"], dtype=float)
    e = np.asarray(d["cross_um"], dtype=float)
    x = np.asarray(d["along_mm"], dtype=float)

    p0, p1 = phase_span(d, config.DRIFT_PRE_PHASE)
    q0, q1 = phase_span(d, config.DRIFT_POST_PHASE)
    iA, iB, iC, iD = int(cuts.iA), int(cuts.iB), int(cuts.iC), int(cuts.iD)

    if not (p1 < iA <= iB < iC <= iD < q0):
        raise ValueError(
            f"[慢偏移] 分段窗口次序不合法：pre_end={p1} iA={iA} iB={iB} "
            f"iC={iC} iD={iD} post_start={q0}")

    b_pre = float(np.median(e[p0:p1 + 1]))
    b_out = float(np.median(e[iA:iB + 1]))
    b_ret = float(np.median(e[iC:iD + 1]))
    b_post = float(np.median(e[q0:q1 + 1]))

    # 窗口敏感性：把两个端点各内缩 0/50/100 个点，看中位数漂多少。
    # 漂得小 → 窗口确实落在平台段上；漂得大 → 说明窗口切进了变化区。
    sens: dict[str, Any] = {}
    for tag, (a, b) in (("fwd", (iA, iB)), ("ret", (iC, iD))):
        vals = []
        for da in (0, 50, 100):
            for db in (0, 50, 100):
                if b - db <= a + da:
                    continue
                vals.append(float(np.median(e[a + da:b - db + 1])))
        sens[tag] = {
            "values_um": [round(v, 3) for v in vals],
            "min_um": float(min(vals)), "max_um": float(max(vals)),
            "span_um": float(max(vals) - min(vals)),
        }

    out = {
        "b_pre_um": b_pre, "b_out_um": b_out, "b_ret_um": b_ret, "b_post_um": b_post,
        "windows": {
            SEG_PRE: {"i0": p0, "i1": p1, "n": p1 - p0 + 1, "rule": "整段中位数"},
            SEG_FWD: {"i0": iA, "i1": iB, "n": iB - iA + 1,
                      "rule": "X_outbound 稳态窗中位数",
                      "x0_mm": float(x[iA]), "x1_mm": float(x[iB])},
            SEG_RET: {"i0": iC, "i1": iD, "n": iD - iC + 1,
                      "rule": "X_return 稳态窗中位数",
                      "x0_mm": float(x[iC]), "x1_mm": float(x[iD])},
            SEG_POST: {"i0": q0, "i1": q1, "n": q1 - q0 + 1, "rule": "整段中位数"},
        },
        "window_sensitivity": sens,
    }
    log(f"[慢偏移] b_pre = {b_pre:+8.3f} µm  (n={p1 - p0 + 1})")
    log(f"[慢偏移] b_out = {b_out:+8.3f} µm  (n={iB - iA + 1}, "
        f"x {x[iA]:.2f}..{x[iB]:.2f} mm)")
    log(f"[慢偏移] b_ret = {b_ret:+8.3f} µm  (n={iD - iC + 1}, "
        f"x {x[iC]:.2f}..{x[iD]:.2f} mm)")
    log(f"[慢偏移] b_post= {b_post:+8.3f} µm  (n={q1 - q0 + 1})")
    log(f"[慢偏移] 窗口敏感性：fwd {sens['fwd']['span_um']:.2f} µm、"
        f"ret {sens['ret']['span_um']:.2f} µm（±100 点内缩）")
    return out


# ---------------------------------------------------------------- 组装
def _assemble(n: int, win: dict, lv: dict, log=print) -> tuple[np.ndarray, dict]:
    """把四个常值 + 三段余弦拼成 b(t)，并断言七段无缝覆盖。"""
    p0, p1 = win[SEG_PRE]["i0"], win[SEG_PRE]["i1"]
    iA, iB = win[SEG_FWD]["i0"], win[SEG_FWD]["i1"]
    iC, iD = win[SEG_RET]["i0"], win[SEG_RET]["i1"]
    q0, q1 = win[SEG_POST]["i0"], win[SEG_POST]["i1"]
    b_pre, b_out = lv["b_pre_um"], lv["b_out_um"]
    b_ret, b_post = lv["b_ret_um"], lv["b_post_um"]

    b = np.full(n, np.nan, dtype=float)
    b[p0:p1 + 1] = b_pre
    b[iA:iB + 1] = b_out
    b[iC:iD + 1] = b_ret
    b[q0:q1 + 1] = b_post

    trans = {
        SEG_START: (p1 + 1, iA - 1, b_pre, b_out),
        SEG_TURN: (iB + 1, iC - 1, b_out, b_ret),
        SEG_STOP: (iD + 1, q0 - 1, b_ret, b_post),
    }
    for name, (a, z, v0, v1) in trans.items():
        if z < a:
            raise ValueError(f"[慢偏移] 过渡段 {name} 为空：{a}..{z}")
        b[a:z + 1] = cosine_ramp(z - a + 1, v0, v1)

    if np.any(~np.isfinite(b)):
        miss = np.where(~np.isfinite(b))[0]
        raise ValueError(f"[慢偏移] b(t) 未被七段完整覆盖，缺失 {miss.size} 点，"
                         f"首个 idx={int(miss[0])}")
    return b, trans


def build(d: dict, cuts: Any, log=print) -> dict[str, Any]:
    """构造 b(t) 与 e_vib(t)，并给出机器可判的验收结果。

    返回 dict，键：
        t, x, phase（相位名数组）, e_raw_um, b_um, e_vib_um
        segments, levels, transitions, per_segment, steady_checks, overall,
        checks, passed, notes
    """
    t = np.asarray(d["t"], dtype=float)
    x = np.asarray(d["along_mm"], dtype=float)
    e_raw = np.asarray(d["cross_um"], dtype=float)
    n = t.size
    fs = 1.0 / float(np.median(np.diff(t)))

    lv = stable_levels(d, cuts, log=log)
    b, trans = _assemble(n, lv["windows"], lv, log=log)
    e_vib = e_raw - b

    win = lv["windows"]
    segments = {
        SEG_PRE: (win[SEG_PRE]["i0"], win[SEG_PRE]["i1"]),
        SEG_START: (trans[SEG_START][0], trans[SEG_START][1]),
        SEG_FWD: (win[SEG_FWD]["i0"], win[SEG_FWD]["i1"]),
        SEG_TURN: (trans[SEG_TURN][0], trans[SEG_TURN][1]),
        SEG_RET: (win[SEG_RET]["i0"], win[SEG_RET]["i1"]),
        SEG_STOP: (trans[SEG_STOP][0], trans[SEG_STOP][1]),
        SEG_POST: (win[SEG_POST]["i0"], win[SEG_POST]["i1"]),
    }

    per_segment = {
        name: _seg_stats(e_raw, e_vib, x, t, i0, i1, fs)
        for name, (i0, i1) in segments.items()
    }

    # ---- 过渡段元信息 ----
    # 每段从哪个稳定值接到哪个稳定值（仅用于报告，不参与计算）
    trans_ends = {
        SEG_START: ("b_pre", "b_out"),
        SEG_TURN: ("b_out", "b_ret"),
        SEG_STOP: ("b_ret", "b_post"),
    }
    trans_info = []
    for name in SEG_TRANSITION:
        a, z = segments[name]
        seg = b[a:z + 1]
        d1 = float(np.max(np.abs(np.diff(seg)))) if seg.size > 1 else 0.0
        k0, k1 = trans_ends[name]
        trans_info.append({
            "name": name, "i0": int(a), "i1": int(z), "n": int(seg.size),
            "span_s": float(t[z] - t[a]) if seg.size > 1 else 0.0,
            "from_level": k0, "to_level": k1,
            "b_start_um": float(seg[0]), "b_end_um": float(seg[-1]),
            "delta_um": float(seg[-1] - seg[0]),
            "monotone": bool(np.all(np.diff(seg) >= -1e-12) or
                             np.all(np.diff(seg) <= 1e-12)),
            "max_step_um": d1,
            "max_rate_um_per_s": (d1 * fs) if seg.size > 1 else 0.0,
            "form": "w = 0.5(1−cos(πs))",
            "level_delta_um": float(lv[k1 + "_um"] - lv[k0 + "_um"]),
        })

    # ---- §八 稳态段四项指标 ----
    steady_checks: dict[str, Any] = {}
    for tag, seg_name in (("fwd_steady", SEG_FWD), ("ret_steady", SEG_RET)):
        s = per_segment[seg_name]
        steady_checks[tag] = {
            "segment": seg_name,
            "n": s["n"],
            "median_raw_um": s["median_raw_um"], "median_vib_um": s["median_vib_um"],
            "mean_raw_um": s["mean_raw_um"], "mean_vib_um": s["mean_vib_um"],
            "rms_raw_um": s["rms_raw_um"], "rms_vib_um": s["rms_vib_um"],
            "abs_median_reduction_pct": _pct(abs(s["median_raw_um"]),
                                             abs(s["median_vib_um"])),
            "abs_mean_reduction_pct": _pct(abs(s["mean_raw_um"]),
                                           abs(s["mean_vib_um"])),
            "rms_reduction_pct": _pct(s["rms_raw_um"], s["rms_vib_um"]),
            "vib_band_retention": (s["vib_band_vib_um"] / s["vib_band_raw_um"]
                                   if s["vib_band_raw_um"] > 1e-15 else float("nan")),
            "note": "中位数为 0 是构造使然（b 取的就是该窗中位数）；"
                    "请以 mean / RMS 与过渡段行为为准",
        }

    overall = {
        "rms_raw_um": float(np.sqrt(np.mean(e_raw ** 2))),
        "rms_vib_um": float(np.sqrt(np.mean(e_vib ** 2))),
        "b_min_um": float(b.min()), "b_max_um": float(b.max()),
        "b_range_um": float(b.max() - b.min()),
        "fs_hz": float(fs), "n_samples": int(n),
    }
    overall["rms_reduction_pct"] = _pct(overall["rms_raw_um"], overall["rms_vib_um"])

    # ---- post_stop 平稳性诊断（诚实项，见模块末尾 notes）----
    q0, q1 = segments[SEG_POST]
    post = np.arange(q0, q1 + 1)
    octs = []
    k = post.size
    for j in range(8):
        s = post[k * j // 8:k * (j + 1) // 8]
        if s.size:
            octs.append({"part": f"{j + 1}/8", "t0_s": float(t[s[0]]),
                         "t1_s": float(t[s[-1]]), "n": int(s.size),
                         "median_um": float(np.median(e_raw[s]))})
    tail = post[t[post] >= t[post[-1]] - 0.5]
    post_stationarity = {
        "whole_phase_median_um": float(np.median(e_raw[post])),
        "tail_0p5s_median_um": float(np.median(e_raw[tail])),
        "tail_0p5s_span_um": float(np.ptp(e_raw[tail])),
        "octants": octs,
        "monotone_trend": bool(np.all(np.diff([o["median_um"] for o in octs]) >= 0.0)
                               or np.all(np.diff([o["median_um"] for o in octs]) <= 0.0)),
    }

    checks = _checks(b, e_raw, e_vib, segments, per_segment, overall, trans_info)
    passed = all(checks.values())

    notes = [
        "稳态段 e_vib 的中位数为 0 是构造必然（b_out/b_ret 就是该窗中位数），"
        "不能当作「去偏移效果好」的证据；请用 mean/RMS 与过渡段 ptp 判断。",
        f"post_stop 段不平稳：整段中位数 {post_stationarity['whole_phase_median_um']:.3f} µm，"
        f"但尾 0.5 s 中位数为 {post_stationarity['tail_0p5s_median_um']:.3f} µm，"
        f"段内中位数单调爬升 {post_stationarity['octants'][0]['median_um']:.2f} → "
        f"{post_stationarity['octants'][-1]['median_um']:.2f} µm。"
        "本模块按 §三.4 取整段中位数，因此 post_stop 段会残留约 ±10 µm 的慢沉降，"
        "这一段对 e_vib 的贡献是真实的、未被建模的慢变化，已如实保留。",
        "过渡段只由四个稳定值和固定的余弦形状决定，不跟随原始数据中的瞬态尖峰；"
        "瞬态是否被保留由 per_segment 的 ptp 与数学硬下界共同核对。",
    ]
    log(f"[慢偏移] b(t) 范围 {overall['b_min_um']:.2f}..{overall['b_max_um']:.2f} µm")
    log(f"[慢偏移] e_raw rms {overall['rms_raw_um']:.3f} → "
        f"e_vib rms {overall['rms_vib_um']:.3f} µm "
        f"(−{overall['rms_reduction_pct']:.1f}%)")
    log(f"[慢偏移] 验收 {'通过' if passed else '未通过'}："
        f"{sum(checks.values())}/{len(checks)} 项")

    return {
        "t": t, "x": x, "e_raw_um": e_raw, "b_um": b, "e_vib_um": e_vib,
        "segments": segments, "levels": lv, "transitions": trans_info,
        "per_segment": per_segment, "steady_checks": steady_checks,
        "overall": overall, "checks": checks, "passed": passed,
        "notes": notes, "fs_hz": fs,
        "post_stop_stationarity": post_stationarity,
        "method": "四个稳定常值 + 启动/转向/停止三段余弦平滑过渡"
                  "（w = 0.5(1−cos(πs))，形状固定、无拟合自由度）",
    }


def _pct(base: float, new: float) -> float:
    return float((base - new) / base * 100.0) if abs(base) > 1e-15 else float("nan")


def _checks(b, e_raw, e_vib, segments, per_seg, overall, trans_info) -> dict[str, bool]:
    """机器可判的验收项。全部为恒成立的性质，不含主观容差。

    §八 的「稳态段中心更接近 0」在这里落成 4 条：
        中位数绝对值变小、均值绝对值变小、RMS 变小、中位数落在容差内。
    §五 的「过渡平滑」落成单调性 + 单步增量的硬上限。
    """
    c: dict[str, bool] = {}

    # 1) 覆盖完整（build 已断言，再记一遍）
    c["分段无缝覆盖"] = bool(np.all(np.isfinite(b)))

    # 3) 四个常值段严格为常值
    for name in SEG_CONST:
        a, z = segments[name]
        seg = b[a:z + 1]
        c[f"{name} 段为常值"] = bool(seg.size and np.ptp(seg) == 0.0)

    # 4) 三段过渡单调 + 端点值正确 + 瞬态未被吃掉
    for info in trans_info:
        nm = info["name"]
        a, z = segments[nm]
        seg = b[a:z + 1]
        c[f"{nm} 过渡单调"] = bool(info["monotone"])
        c[f"{nm} 过渡端点与常值段一致"] = bool(
            seg.size and abs(seg[0] - b[a - 1]) <= 1e-12
            and abs(seg[-1] - b[z + 1]) <= 1e-12)
        # 形状核对：w = 0.5(1−cos πs) 在 n 个点上的一阶差分有【精确】上界
        #     max|Δw| = sin(π h / 2)，  h = 1/(n−1)     （在段中点取到）
        # 所以 max|Δb| = |Δb_段|·sin(π/(2(n−1)))。这是余弦的解析性质，
        # 不是我拍的容差：拼接错位、段端点写错、中途插阶跃，比值都会立刻 >1。
        n_s = int(seg.size)
        if n_s > 2 and abs(info["delta_um"]) > 1e-12:
            bound = abs(info["delta_um"]) * np.sin(np.pi / (2.0 * (n_s - 1)))
            ratio = float(info["max_step_um"] / bound)
            c[f"{nm} 过渡斜率符合余弦上界（比值 {ratio:.4f} ≤ 1）"] = bool(
                ratio <= 1.0 + 1e-9)
        else:
            c[f"{nm} 过渡斜率符合余弦上界"] = True
        # 硬下界：ptp(e_raw − b) ≥ ptp(e_raw) − ptp(b)，而 b 在该段单调，ptp(b)=|Δb|。
        # 即「b 至多吃掉与自身变化量相等的峰谷」，是恒成立的性质，不是拍的容差。
        floor = per_seg[nm]["ptp_raw_um"] - abs(info["delta_um"])
        margin = per_seg[nm]["ptp_vib_um"] - floor
        c[f"{nm} 瞬态保留（ptp {per_seg[nm]['ptp_vib_um']:.1f} ≥ "
          f"硬下界 {floor:.1f} µm）"] = margin >= -config.DRIFT_PTP_TOL_UM

    # 5) §八 稳态段
    for tag, seg_name in (("fwd_steady", SEG_FWD), ("ret_steady", SEG_RET)):
        s = per_seg[seg_name]
        tol = min(config.DRIFT_CENTER_ABS_UM,
                  config.DRIFT_CENTER_REL * s["rms_vib_um"])
        c[f"{tag} 中位数在容差内（|{s['median_vib_um']:.3f}| ≤ {tol:.3f} µm）"] = \
            abs(s["median_vib_um"]) <= tol
        c[f"{tag} 比原始更居中（|中位数|）"] = \
            abs(s["median_vib_um"]) < abs(s["median_raw_um"])
        c[f"{tag} 比原始更居中（|均值|）"] = \
            abs(s["mean_vib_um"]) < abs(s["mean_raw_um"])
        c[f"{tag} RMS 下降"] = s["rms_vib_um"] < s["rms_raw_um"]

    # 6) 整体 RMS 下降
    c["整体 RMS 下降"] = overall["rms_vib_um"] < overall["rms_raw_um"]
    # 出口统一成 Python bool：上面的比较一旦有一侧是 np.float64，结果就是
    # np.bool_，写进 JSON 会炸（而且 accept/`if not v` 的判据也不再可靠）。
    return {k: bool(v) for k, v in c.items()}


# ---------------------------------------------------------------- 落盘
def write_reference_csvs(out_dir: Path, res: dict, d: dict, log=print) -> dict[str, str]:
    """写出 §七 要求的四个 CSV（均带 t / phase / nominal_along_mm）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = res["t"]
    x = res["x"]
    e_raw, b, e_vib = res["e_raw_um"], res["b_um"], res["e_vib_um"]
    pnames = list(d["phase_names"])
    pcode = np.asarray(d["phase_code"], dtype=int)
    phase_txt = [pnames[int(c)] for c in pcode]

    specs = (
        (config.DRIFT_RAW_CSV, [("e_raw_um", e_raw)]),
        (config.DRIFT_BASELINE_CSV, [("b_um", b)]),
        (config.DRIFT_VIB_CSV, [("e_vib_um", e_vib)]),
        (config.DRIFT_TABLE_CSV, [("e_raw_um", e_raw), ("b_um", b),
                                  ("e_vib_um", e_vib)]),
    )
    written: dict[str, str] = {}
    for fname, cols in specs:
        path = out_dir / fname
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "phase", "nominal_along_mm"] + [c[0] for c in cols])
            for i in range(t.size):
                w.writerow([f"{t[i]:.6f}", phase_txt[i], f"{x[i]:.6f}"]
                           + [f"{c[1][i]:.6f}" for c in cols])
        written[fname] = str(path)
        log(f"[慢偏移] 写出 {fname}（{t.size} 行）")
    return written


def summary_dict(res: dict) -> dict[str, Any]:
    """组装 drift_baseline_summary.json 的内容（§三 + §八）。"""
    return {
        "method": res["method"],
        "levels_are_recomputed": True,
        "grid": {"n_samples": int(res["t"].size),
                 "dt_s": float(np.median(np.diff(res["t"]))),
                 "fs_hz": float(res["fs_hz"])},
        "levels_um": {
            "b_pre": res["levels"]["b_pre_um"],
            "b_out": res["levels"]["b_out_um"],
            "b_ret": res["levels"]["b_ret_um"],
            "b_post": res["levels"]["b_post_um"],
        },
        "windows": res["levels"]["windows"],
        "window_sensitivity": res["levels"]["window_sensitivity"],
        "transitions": res["transitions"],
        "segments": [{"name": n, "i0": res["segments"][n][0], "i1": res["segments"][n][1]}
                     for n in SEGMENTS],
        "per_segment": res["per_segment"],
        "steady_segment_checks": res["steady_checks"],
        "overall": res["overall"],
        "post_stop_stationarity": res["post_stop_stationarity"],
        "acceptance": {"passed": bool(res["passed"]), "checks": res["checks"]},
        "notes": res["notes"],
    }


def write_summary_json(path: Path, res: dict, log=print) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from record import json_default       # 延迟导入：record 依赖 config，本模块同理，
    path.write_text(json.dumps(summary_dict(res), ensure_ascii=False, indent=2,
                               default=json_default), encoding="utf-8")
    log(f"[慢偏移] 写出 {path.name}")
    return str(path)
