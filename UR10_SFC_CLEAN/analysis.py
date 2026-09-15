"""
结果分析 —— 全部为数值统计，本模块【不产生任何图片】。

只回答这些问题：
  ① 五个正式组（A0 / Alinear / ASFC_default / ASFC_medium / ASFC_strong）
     各自的振动水平是多少；ASFC 相对 Alinear 是否存在额外改善；
  ② 测试A：扰动幅值放大/缩小时，ASFC 相对 Alinear 的额外改善是变大还是变小；
  ③ 测试B：慢偏移估计有 ±5 / ±10 µm 偏差时，ASFC 的主要效果是否还成立；
  ④ 测试C：逐个事件看「振出去」的峰值/上升速度 与「回中心」的恢复时间/反向过冲，
     判断 ASFC 是否「压低了峰值但拖长了回正」；
  ⑤ 测试D：分相统计，判断 SFC 的优势主要落在启动/转向/停止这类突变段，
     还是均匀地降低了每一个小波纹。

本模块不做任何参数寻优，也不把某组标成「推荐参数」。
所有 PSD / 频带分析一律用 f_s = 1/median(Δt) 从时间戳反推，见 fs_from_time。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import config
import record

# ------------------------------------------------------------------ 谱工具
def fs_from_time(t: np.ndarray) -> float:
    """从时间戳反推采样率：f_s = 1 / median(Δt)。

    【为什么必须这么做】时序 CSV 是按 TRACE_STRIDE 抽稀后落盘的（1000 Hz → 200 Hz），
    但物理步长是 1 ms。任何对 CSV 做 PSD / 频带积分的分析，如果沿用 1000 Hz 当采样率，
    频率轴会整体错位 5 倍，慢带 / 振动带的带内积分就全错了。所以这里一律以数据自身的
    Δt 为准，不接受外部传入值，也禁止写死 1000。
    """
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        raise ValueError("[分析] 时间序列少于 2 点，无法反推采样率")
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"[分析] 时间戳非单调或 Δt={dt!r}，无法反推采样率")
    return 1.0 / dt


def welch_psd(x: np.ndarray, fs: float, win_s: float = config.WELCH_WIN_S
              ) -> tuple[np.ndarray, np.ndarray]:
    """单边 Welch PSD（Hann 窗、50% 重叠）。返回 (freq_Hz, psd[单位²/Hz])。"""
    x = np.asarray(x, dtype=float)
    nper = int(round(win_s * fs))
    if nper > x.size:
        nper = int(x.size)
    if nper < 8:
        return np.array([0.0]), np.array([0.0])
    step = max(1, nper // 2)
    win = np.hanning(nper)
    scale = 1.0 / (fs * np.sum(win ** 2))

    segs = [x[s:s + nper] for s in range(0, x.size - nper + 1, step)]
    if not segs:
        segs = [x[:nper]]
    psd = np.zeros(nper // 2 + 1)
    for seg in segs:
        X = np.fft.rfft((seg - seg.mean()) * win)
        psd += (np.abs(X) ** 2) * scale
    psd /= len(segs)
    psd[1:-1] *= 2.0
    return np.fft.rfftfreq(nper, d=1.0 / fs), psd


def band_rms(freq: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """带内 PSD 积分开方 → 带内 RMS。"""
    m = (freq >= lo) & (freq < hi)
    if not m.any():
        return 0.0
    return float(np.sqrt(max(float(np.trapezoid(psd[m], freq[m])), 0.0)))


def metrics(e_um: np.ndarray, fs: float) -> dict[str, float]:
    """一段误差的基本统计量。"""
    e = np.asarray(e_um, dtype=float)
    if e.size == 0:
        return {k: float("nan") for k in
                ("e_rms_um", "e_peak_abs_um", "e_ptp_um", "e_mean_um",
                 "e_median_um", "e_slow_rms_um", "e_vib_rms_um")}
    f, p = welch_psd(e, fs)
    return {
        "e_rms_um": float(np.sqrt(np.mean(e ** 2))),
        "e_peak_abs_um": float(np.max(np.abs(e))),
        "e_ptp_um": float(np.ptp(e)),
        "e_mean_um": float(np.mean(e)),
        "e_median_um": float(np.median(e)),
        "e_slow_rms_um": band_rms(f, p, *config.SLOW_BAND_HZ),
        "e_vib_rms_um": band_rms(f, p, *config.VIB_BAND_HZ),
    }


def _ratio(new: float, base: float) -> float:
    """抑振比例 = (base − new)/base。正值表示 new 更小（有抑振）。"""
    return float((base - new) / base) if abs(base) > 1e-15 else float("nan")


# ------------------------------------------------------------------ 标准多组对比
def load_traces(trace_paths: dict[str, Path]) -> tuple[dict[str, dict], dict[str, dict]]:
    """批量读时序，返回 (meta, cols)。缺文件直接报错，不静默跳过。"""
    metas: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    for cid, p in trace_paths.items():
        if p is None or not Path(p).is_file():
            raise FileNotFoundError(f"缺少 {cid} 的结果文件：{p}")
        m, cols = record.load_trace(Path(p))
        metas[cid] = m
        runs[cid] = cols
    return metas, runs


def phase_stats(cols: dict[str, np.ndarray], fs: float
                ) -> dict[str, dict[str, float]]:
    """按 phase 列分相统计（测试D 的核心，summary_main 也复用）。"""
    if "phase" not in cols:
        raise KeyError("[分析] 时序里没有 phase 列，无法分相统计")
    code = np.rint(np.asarray(cols["phase"], dtype=float)).astype(int)
    e = np.asarray(cols["e_um"], dtype=float)
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(config.DRIFT_PHASE_NAMES):
        m = code == i
        if not m.any():
            continue
        st = metrics(e[m], fs)
        st["n"] = int(m.sum())
        st["share_pct"] = float(m.mean() * 100.0)
        out[name] = st
    return out


def compare(trace_paths: dict[str, Path], out_dir: Path,
            ids: tuple[str, ...] | None = None,
            fs: float | None = None) -> dict[str, Any]:
    """标准多组对比（纯数值）。

    fs 默认【从时序 CSV 的时间戳反推】，不信任调用方传入值。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = tuple(config.MODES if ids is None else ids)
    metas, runs = load_traces({c: trace_paths[c] for c in ids})

    t = runs[ids[0]]["t"]
    fs = fs_from_time(t) if fs is None else float(fs)
    print(f"[分析] 由时间戳反推采样率 f_s = {fs:.4f} Hz"
          f"（物理步长 {config.PHYSICS_HZ:g} Hz，抽稀步长 {config.TRACE_STRIDE}）")

    mets = {c: metrics(runs[c]["e_um"], fs) for c in ids}
    phs = {c: phase_stats(runs[c], fs) for c in ids}

    base, lin, strong = config.BASELINE_ID, config.TEST_LINEAR_ID, config.TEST_ASFC_ID
    ratios: dict[str, float] = {}
    if base in ids:
        for c in ids:
            if c == base:
                continue
            for key in ("e_rms_um", "e_vib_rms_um", "e_peak_abs_um"):
                ratios[f"{c}_vs_{base}_{key}"] = _ratio(mets[c][key], mets[base][key])
    if lin in ids and strong in ids:
        for key in ("e_rms_um", "e_vib_rms_um", "e_peak_abs_um"):
            ratios[f"{strong}_vs_{lin}_{key}"] = _ratio(mets[strong][key],
                                                        mets[lin][key])

    # 剪切项是否真的被激活（只看强代表组）
    shear_check: dict[str, Any] = {}
    if strong in runs:
        d = runs[strong]
        lin_f, shr = np.abs(d["F_linear_damping_N"]), np.abs(d["F_shear_N"])
        lr, sr = float(np.sqrt(np.mean(lin_f ** 2))), float(np.sqrt(np.mean(shr ** 2)))
        nz = np.abs(d["F_shear_N"]) > 1e-15
        shear_check = {
            "case_id": strong,
            "v_rms_m_s": float(np.sqrt(np.mean(d["v_m_s"] ** 2))),
            "v_peak_abs_m_s": float(np.max(np.abs(d["v_m_s"]))),
            "F_linear_rms_N": lr, "F_shear_rms_N": sr,
            "F_shear_peak_N": float(np.max(shr)),
            "shear_over_linear_rms": sr / lr if lr > 1e-15 else float("nan"),
            "sign_always_opposes_v": bool(np.all(
                np.sign(d["F_shear_N"][nz]) == -np.sign(d["v_m_s"][nz]))),
            "P_shear_pos_frac": float(np.mean(d["F_shear_N"] * d["v_m_s"] > config.P_TOL_W)),
            "P_linear_pos_frac": float(np.mean(
                d["F_linear_damping_N"] * d["v_m_s"] > config.P_TOL_W)),
        }

    idset = ids
    identical = {
        "same_frozen_Fd": float(np.max(np.abs(runs[idset[0]]["Fd_N"]
                                              - runs[idset[-1]]["Fd_N"]))),
        "same_y_d": float(np.max(np.abs(runs[idset[0]]["y_d"]
                                       - runs[idset[-1]]["y_d"]))),
        "same_along": float(np.max(np.abs(runs[idset[0]]["along_mm"]
                                         - runs[idset[-1]]["along_mm"]))),
        "same_fd_sha": len({metas[c].get("fd_sha256") for c in idset}) == 1,
        "note": "前三行是逐点最大差；全为 0 且 same_fd_sha 为真，才说明各组输入完全一致。",
    }

    summary = {
        "modes": list(ids),
        "n_samples": int(t.size),
        "duration_s": float(t[-1]),
        "trace_stride": config.TRACE_STRIDE,
        "fs_from_timestamps_hz": float(fs),
        "fs_note": "PSD/频带一律用 1/median(Δt)，不是物理步长 1000 Hz",
        "run_params": {c: {k: metas[c].get(k) for k in
                           ("mode", "K_N_per_m", "B0_Ns_per_m", "n", "mu",
                            "r_design", "r_actual", "v_ref_m_s", "fd_sha256",
                            "fd_scale", "bias_um")}
                       for c in idset},
        "metrics": mets,
        "phase_metrics": phs,
        "ratios": ratios,
        "shear_term_check": shear_check,
        "identical_inputs": identical,
    }
    record.save_json(out_dir / "summary.json", summary)
    return summary


def print_report(summary: dict[str, Any]) -> None:
    m, r = summary["metrics"], summary["ratios"]
    ids = summary["modes"]
    print("\n" + "=" * 76)
    print("正式组对照结果（同一冻结 Fd、同一起始状态、同一名义轨迹）")
    print("=" * 76)
    print(f"{'组':<16}{'总RMS(µm)':>12}{'峰值(µm)':>12}{'峰峰(µm)':>12}"
          f"{'慢成分(µm)':>12}{'振动带(µm)':>12}")
    for c in ids:
        d = m[c]
        print(f"{c:<16}{d['e_rms_um']:>12.3f}{d['e_peak_abs_um']:>12.3f}"
              f"{d['e_ptp_um']:>12.3f}{d['e_slow_rms_um']:>12.3f}"
              f"{d['e_vib_rms_um']:>12.3f}")
    print("-" * 76)
    for k, v in r.items():
        tag = k.replace("_e_", "  ").replace("_vs_", " 相对 ")
        print(f"  {tag:<44}{v * 100:+8.2f}%")
    sc = summary.get("shear_term_check") or {}
    if sc:
        print("-" * 76)
        print(f"  剪切项激活（{sc['case_id']}）：|v| rms={sc['v_rms_m_s']:.3e} "
              f"峰值={sc['v_peak_abs_m_s']:.3e} m/s")
        print(f"      F_linear rms={sc['F_linear_rms_N']:.4f} N  "
              f"F_shear rms={sc['F_shear_rms_N']:.4f} N  "
              f"(剪切/线性={sc['shear_over_linear_rms']:.4f})")
        print(f"      剪切项恒与速度反向：{sc['sign_always_opposes_v']}   "
              f"P_shear>0 比例：{sc['P_shear_pos_frac']:.2e}")
    ii = summary.get("identical_inputs", {})
    if ii:
        print(f"  输入一致性：Fd 最大差 {ii['same_frozen_Fd']:.3e}，"
              f"y_d 最大差 {ii['same_y_d']:.3e}，沿程最大差 {ii['same_along']:.3e}，"
              f"Fd SHA 一致 {ii['same_fd_sha']}")
    print("=" * 76 + "\n")


# ============================================================ 测试C：事件级「振出去 / 回中心」
def find_events(e_abs: np.ndarray, fs: float,
                win_s: float = config.EVENT_WIN_S,
                min_sep_s: float = config.EVENT_MIN_SEP_S,
                topk: int = config.EVENT_TOPK) -> np.ndarray:
    """在 |e| 上找彼此至少隔开 min_sep_s 的最大 topk 个局部极大值（返回索引，升序）。

    事件由 A0（无抗振）定义：A0 的响应就是纯激励响应，用它定位「同一批扰动事件」，
    再在固定时间窗里看各个控制组的作为，避免各组各找各的峰、无法配对。
    """
    e_abs = np.asarray(e_abs, dtype=float)
    n = e_abs.size
    half = max(1, int(round(0.05 * fs)))
    if n < 2 * half + 3:
        return np.empty(0, dtype=int)
    cand = np.where((e_abs[1:-1] >= e_abs[:-2]) & (e_abs[1:-1] > e_abs[2:]))[0] + 1
    keep = [int(i) for i in cand
            if e_abs[i] >= e_abs[max(0, i - half):i + half + 1].max() - 1e-12]
    sep = max(1, int(round(min_sep_s * fs)))
    picked: list[int] = []
    for i in sorted(keep, key=lambda k: -e_abs[k]):
        if all(abs(i - j) >= sep for j in picked):
            picked.append(i)
        if len(picked) >= int(topk):
            break
    return np.array(sorted(picked), dtype=int)


def _event_metric(seg: np.ndarray, t_seg: np.ndarray) -> dict[str, float]:
    """单个事件窗内的一组指标。seg 以事件为中心，t_seg 是对应时间。"""
    a = np.abs(seg)
    k = int(np.argmax(a))
    peak = float(a[k])
    t_peak = float(t_seg[k])
    out = {
        "peak_abs_um": peak,
        "t_peak_s": t_peak,
        "t_peak_rel_s": float(t_peak - t_seg[0]),
        "rise_rate_um_per_s": (float((peak - a[0]) / (t_peak - t_seg[0]))
                               if t_peak > t_seg[0] else float("nan")),
        "recover_s": float("nan"),
        "overshoot_um": float("nan"),
    }
    if k + 1 < a.size:
        thr = config.EVENT_RECOVER_FRAC * peak
        hit = np.where(a[k:] <= thr)[0]
        # 只看峰值之后的段；峰值点自身满足时要往后找一段真正的下降
        hit = hit[hit > 0]
        if hit.size:
            out["recover_s"] = float(t_seg[k + hit[0]] - t_peak)
        # 反向过冲：峰值之后朝反方向走的最远幅度
        sign = np.sign(seg[k]) or 1.0
        opp = -sign * seg[k:]
        out["overshoot_um"] = float(max(float(np.max(opp)), 0.0))
    return out


def event_response(trace_paths: dict[str, Path], out_dir: Path,
                   focus: tuple[str, ...] = (config.TEST_LINEAR_ID,
                                             config.TEST_ASFC_ID),
                   event_source: str = config.BASELINE_ID,
                   fs: float | None = None) -> dict[str, Any]:
    """测试C：把每一个突变事件拆成「上升段」与「回落段」分别看。

    关键要回答的是：ASFC 是不是「峰值压低了、但回正反而拖长了」，
    还是「峰值更低、回正也不慢、反向过冲更小」。

    输出：
      summary_event_response.csv  每个组的聚合中位数 + 相对 Alinear 的比值
      event_response_detail.csv   每个事件的逐条明细（可自行复核）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = tuple(dict.fromkeys((event_source,) + tuple(focus)))
    metas, runs = load_traces({c: trace_paths[c] for c in ids})

    t = runs[event_source]["t"]
    fs = fs_from_time(t) if fs is None else float(fs)
    events = find_events(np.abs(runs[event_source]["e_um"]), fs)

    half = int(round(config.EVENT_WIN_S * fs))
    n = t.size
    usable = [int(tp) for tp in events if tp - half >= 0 and tp + half < n]

    detail: list[dict[str, Any]] = []
    for eid, tp in enumerate(usable):
        w = slice(tp - half, tp + half + 1)
        ts = t[w]
        for cid in ids:
            r = _event_metric(np.asarray(runs[cid]["e_um"][w], dtype=float), ts)
            detail.append({"event_id": eid, "case_id": cid,
                           "event_t_s": float(t[tp]), **r})

    def col(cid: str, key: str) -> np.ndarray:
        return np.array([d[key] for d in detail if d["case_id"] == cid], dtype=float)

    agg: dict[str, dict[str, float]] = {}
    for cid in ids:
        agg[cid] = {
            "n_events": int(np.sum([d["case_id"] == cid for d in detail])),
            "peak_abs_median_um": float(np.nanmedian(col(cid, "peak_abs_um"))),
            "peak_abs_mean_um": float(np.nanmean(col(cid, "peak_abs_um"))),
            "rise_rate_median_um_per_s": float(np.nanmedian(
                col(cid, "rise_rate_um_per_s"))),
            "recover_median_s": float(np.nanmedian(col(cid, "recover_s"))),
            "recover_mean_s": float(np.nanmean(col(cid, "recover_s"))),
            "overshoot_median_um": float(np.nanmedian(col(cid, "overshoot_um"))),
        }
        agg[cid]["recover_n_valid"] = int(np.sum(np.isfinite(col(cid, "recover_s"))))

    # 逐事件配对比较（同一 event_id 下 ASFC 与 Alinear 比），比只看中位数可靠
    pair: dict[str, Any] = {}
    if config.TEST_LINEAR_ID in ids and config.TEST_ASFC_ID in ids:
        lin = {d["event_id"]: d for d in detail if d["case_id"] == config.TEST_LINEAR_ID}
        stg = {d["event_id"]: d for d in detail if d["case_id"] == config.TEST_ASFC_ID}
        common = sorted(set(lin) & set(stg))
        if common:
            dp = np.array([stg[k]["peak_abs_um"] - lin[k]["peak_abs_um"] for k in common])
            dr = np.array([stg[k]["recover_s"] - lin[k]["recover_s"] for k in common])
            do = np.array([stg[k]["overshoot_um"] - lin[k]["overshoot_um"]
                           for k in common])
            pair = {
                "n_paired_events": len(common),
                "peak_lower_count": int(np.sum(dp < 0)),
                "peak_equal_count": int(np.sum(dp == 0)),
                "peak_higher_count": int(np.sum(dp > 0)),
                "peak_median_change_um": float(np.nanmedian(dp)),
                "recover_longer_count": int(np.nansum(dr > 0)),
                "recover_shorter_count": int(np.nansum(dr < 0)),
                "recover_median_change_s": float(np.nanmedian(dr)),
                "overshoot_lower_count": int(np.nansum(do < 0)),
                "overshoot_higher_count": int(np.nansum(do > 0)),
                "overshoot_median_change_um": float(np.nanmedian(do)),
            }
            pair["verdict_peak"] = (
                "ASFC 峰值更低" if pair["peak_lower_count"] > pair["peak_higher_count"]
                else "ASFC 峰值并未普遍更低")
            pair["verdict_recover"] = (
                "ASFC 回正更慢（存在阻碍回正）"
                if pair["recover_longer_count"] > pair["recover_shorter_count"]
                else "ASFC 回正不更慢")
            pair["verdict_overshoot"] = (
                "ASFC 反向过冲更小"
                if pair["overshoot_lower_count"] > pair["overshoot_higher_count"]
                else "ASFC 反向过冲并不更小")

    result = {
        "event_source": event_source,
        "fs_from_timestamps_hz": float(fs),
        "event_win_s": config.EVENT_WIN_S,
        "event_min_sep_s": config.EVENT_MIN_SEP_S,
        "recover_threshold_frac": config.EVENT_RECOVER_FRAC,
        "n_events_found": int(events.size),
        "n_events_used": len(usable),
        "aggregate": agg,
        "paired_ASFC_vs_Alinear": pair,
    }

    # ---- summary_event_response.csv ----
    cols = ["case_id", "n_events", "peak_abs_median_um", "peak_abs_mean_um",
            "rise_rate_median_um_per_s", "recover_median_s", "recover_mean_s",
            "recover_n_valid", "overshoot_median_um"]
    with open(out_dir / "summary_event_response.csv", "w", newline="",
              encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for cid in ids:
            fh.write(",".join([cid] + [f"{agg[cid].get(c, float('nan')):.6g}"
                                       for c in cols[1:]]) + "\n")
    # 相对 Alinear 的比值单列成文件后半段不合适（列数不同），改为在 JSON 里给结论

    with open(out_dir / "event_response_detail.csv", "w", newline="",
              encoding="utf-8") as fh:
        dcols = ["event_id", "case_id", "event_t_s", "t_peak_s", "t_peak_rel_s",
                 "peak_abs_um", "rise_rate_um_per_s", "recover_s", "overshoot_um"]
        fh.write(",".join(dcols) + "\n")
        for d in detail:
            fh.write(",".join([str(d["event_id"]), d["case_id"]]
                              + [f"{d[c]:.6g}" for c in dcols[2:]]) + "\n")
    return result


# ============================================================ 测试D：分相统计
def phasewise(trace_paths: dict[str, Path], out_dir: Path,
              ids: tuple[str, ...] | None = None,
              fs: float | None = None) -> dict[str, Any]:
    """测试D：按 5 个运动阶段分别统计，看 SFC 的优势落在哪一段。

    关键问题：SFC 是「均匀降低了每一个小波纹」，还是「只在启动/转向/停止
    这类突变段起大作用」。判据是各相的 ASFC 相对 Alinear 改善率。

    输出 summary_phasewise.csv。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = tuple(config.MODES if ids is None else ids)
    metas, runs = load_traces({c: trace_paths[c] for c in ids})

    fs = fs_from_time(runs[ids[0]]["t"]) if fs is None else float(fs)
    per_phase = {c: phase_stats(runs[c], fs) for c in ids}

    rows: list[dict[str, Any]] = []
    for phase in config.DRIFT_PHASE_NAMES:
        for cid in ids:
            st = per_phase[cid].get(phase)
            if st is None:
                continue
            rows.append({
                "phase": phase, "case_id": cid, "n": st["n"],
                "share_pct": st["share_pct"],
                "rms_um": st["e_rms_um"], "peak_abs_um": st["e_peak_abs_um"],
                "ptp_um": st["e_ptp_um"], "mean_um": st["e_mean_um"],
                "median_um": st["e_median_um"],
                "slow_rms_um": st["e_slow_rms_um"], "vib_rms_um": st["e_vib_rms_um"],
            })

    cols = ["phase", "case_id", "n", "share_pct", "rms_um", "peak_abs_um", "ptp_um",
            "mean_um", "median_um", "slow_rms_um", "vib_rms_um"]
    with open(out_dir / "summary_phasewise.csv", "w", newline="",
              encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join([r["phase"], r["case_id"]]
                              + [f"{r[c]:.6g}" for c in cols[2:]]) + "\n")

    # 各相 ASFC_strong 相对 Alinear 的改善率（负值 = ASFC 更差）
    improvement: dict[str, dict[str, float]] = {}
    lin, stg, base = config.TEST_LINEAR_ID, config.TEST_ASFC_ID, config.BASELINE_ID
    for phase in config.DRIFT_PHASE_NAMES:
        a, b = per_phase.get(lin, {}).get(phase), per_phase.get(stg, {}).get(phase)
        c0 = per_phase.get(base, {}).get(phase)
        if not a or not b:
            continue
        improvement[phase] = {
            "rms_um_Alinear": a["e_rms_um"], "rms_um_ASFC": b["e_rms_um"],
            "asfc_vs_linear_rms_pct": _ratio(b["e_rms_um"], a["e_rms_um"]) * 100.0,
            "asfc_vs_linear_peak_pct": _ratio(b["e_peak_abs_um"],
                                              a["e_peak_abs_um"]) * 100.0,
            "asfc_vs_linear_ptp_pct": _ratio(b["e_ptp_um"], a["e_ptp_um"]) * 100.0,
        }
        if c0:
            improvement[phase]["linear_vs_A0_rms_pct"] = \
                _ratio(a["e_rms_um"], c0["e_rms_um"]) * 100.0
            improvement[phase]["asfc_vs_A0_rms_pct"] = \
                _ratio(b["e_rms_um"], c0["e_rms_um"]) * 100.0

    trans = [p for p in ("start", "turn", "stop") if p in improvement]
    steady = [p for p in ("fwd_steady", "ret_steady") if p in improvement]
    m_trans = float(np.mean([improvement[p]["asfc_vs_linear_rms_pct"] for p in trans])) \
        if trans else float("nan")
    m_steady = float(np.mean([improvement[p]["asfc_vs_linear_rms_pct"] for p in steady])) \
        if steady else float("nan")
    verdict = {
        "transient_phases": trans, "steady_phases": steady,
        "mean_asfc_vs_linear_rms_pct_transient": m_trans,
        "mean_asfc_vs_linear_rms_pct_steady": m_steady,
        "conclusion": (
            "SFC 的优势集中在突变段（启动/转向/停止），稳态段收益明显更小"
            if (np.isfinite(m_trans) and np.isfinite(m_steady) and m_trans > m_steady)
            else "SFC 的相对收益在稳态段与突变段相当（未表现出只对突变段有效）"),
    }
    return {"fs_from_timestamps_hz": float(fs), "per_phase": per_phase,
            "improvement": improvement, "verdict": verdict}
