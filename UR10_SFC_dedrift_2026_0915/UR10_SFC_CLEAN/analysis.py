"""
结果分析 —— 标准三组对比 + 批量扫描的可视化与客观排序。

只回答这些问题：
  A0 / Alinear / ASFC 各自的振动水平是多少；
  ASFC 相对 Alinear 是否存在额外改善；
  n 与 r 改变时非线性强度与抑振效果如何变化；
  是否存在“抑振更好但峰值/力矩代价更大”的情况。

本模块不做任何参数寻优，也不把某组标成“推荐参数”。
图只用于本地自查；交付 ZIP 不含图片。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import config
import record

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 图里要写中文，matplotlib 默认的 DejaVu Sans 没有 CJK 字形，会全部画成方框。
for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
           "WenQuanYi Zen Hei", "PingFang SC", "Heiti SC"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

MODE_COLOR = {"A0": "#B03A2E", "Alinear": "#1F6FB2", "ASFC": "#1E8449"}


# ------------------------------------------------------------------ 谱工具
def fs_from_time(t: np.ndarray) -> float:
    """从时间戳反推采样率：f_s = 1 / median(Δt)。

    【为什么必须这么做】时序 CSV 是按 TRACE_STRIDE 抽稀后落盘的（1000 Hz → 200 Hz），
    但物理步长是 1 ms。任何对 CSV 做 PSD / 频带积分的分析，如果沿用 1000 Hz 当采样率，
    频率轴会整体错位 5 倍，慢带 / 振动带的带内积分就全错了（曾出现把 200 Hz 数据
    按 1000 Hz 算 PSD 的问题）。所以这里一律以数据自身的 Δt 为准，不接受外部传入值。
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
    e = np.asarray(e_um, dtype=float)
    f, p = welch_psd(e, fs)
    return {
        "e_rms_um": float(np.sqrt(np.mean(e ** 2))),
        "e_peak_abs_um": float(np.max(np.abs(e))),
        "e_ptp_um": float(np.ptp(e)),
        "e_slow_rms_um": band_rms(f, p, *config.SLOW_BAND_HZ),
        "e_vib_rms_um": band_rms(f, p, *config.VIB_BAND_HZ),
    }


def _ratio(new: float, base: float) -> float:
    """抑振比例 = (base − new)/base。正值表示 new 更小（有抑振）。"""
    return float((base - new) / base) if abs(base) > 1e-15 else float("nan")


# ------------------------------------------------------------------ 标准三组对比
def compare(trace_paths: dict[str, Path], out_dir: Path,
            fs: float | None = None) -> dict[str, Any]:
    """标准三组对比。fs 默认【从时序 CSV 的时间戳反推】，不信任调用方传入值。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas: dict[str, dict] = {}
    runs: dict[str, dict[str, np.ndarray]] = {}
    for mode in config.MODES:
        p = trace_paths.get(mode)
        if p is None or not Path(p).is_file():
            raise FileNotFoundError(f"缺少 {mode} 的结果文件：{p}")
        m, cols = record.load_trace(Path(p))
        metas[mode] = m
        runs[mode] = cols

    t = runs["A0"]["t"]
    fs = fs_from_time(t) if fs is None else float(fs)
    print(f"[分析] 由时间戳反推采样率 f_s = {fs:.4f} Hz"
          f"（物理步长 {config.PHYSICS_HZ:g} Hz，抽稀步长 {config.TRACE_STRIDE}）")
    mets = {m: metrics(runs[m]["e_um"], fs) for m in config.MODES}

    ratios = {
        "Alinear_vs_A0": _ratio(mets["Alinear"]["e_rms_um"], mets["A0"]["e_rms_um"]),
        "ASFC_vs_A0": _ratio(mets["ASFC"]["e_rms_um"], mets["A0"]["e_rms_um"]),
        "ASFC_vs_Alinear": _ratio(mets["ASFC"]["e_rms_um"], mets["Alinear"]["e_rms_um"]),
        "Alinear_vs_A0_vib": _ratio(mets["Alinear"]["e_vib_rms_um"],
                                    mets["A0"]["e_vib_rms_um"]),
        "ASFC_vs_A0_vib": _ratio(mets["ASFC"]["e_vib_rms_um"], mets["A0"]["e_vib_rms_um"]),
        "ASFC_vs_Alinear_vib": _ratio(mets["ASFC"]["e_vib_rms_um"],
                                      mets["Alinear"]["e_vib_rms_um"]),
    }

    # ---- 图 1：Y 向振动时域 ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for m in config.MODES:
        axes[0].plot(runs[m]["t"], runs[m]["e_um"], lw=0.7, color=MODE_COLOR[m],
                     label=f"{m}  rms={mets[m]['e_rms_um']:.2f} µm")
    axes[0].set_ylabel("e = y_sim − y_d  (µm)")
    axes[0].set_xlabel("t (s)")
    axes[0].set_title("末端 Y 向偏差：A0 / Alinear / ASFC（同一冻结 Fd）")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    k = int(np.argmax(np.abs(runs["A0"]["e_um"])))
    a, b = max(0, k - int(2.0 * fs)), min(t.size, k + int(2.0 * fs))
    for m in config.MODES:
        axes[1].plot(t[a:b], runs[m]["e_um"][a:b], lw=1.0, color=MODE_COLOR[m], label=m)
    axes[1].set_ylabel("e (µm)")
    axes[1].set_xlabel("t (s)")
    axes[1].set_title(f"放大：t = {t[a]:.2f}–{t[b]:.2f} s（A0 峰值附近）")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_time_domain.png", dpi=130)
    plt.close(fig)

    # ---- 图 2：PSD ----
    fig, ax = plt.subplots(figsize=(11, 5))
    for m in config.MODES:
        f, p = welch_psd(runs[m]["e_um"], fs)
        ax.semilogy(f, p, lw=1.0, color=MODE_COLOR[m], label=m)
    ax.axvspan(*config.SLOW_BAND_HZ, color="#8E44AD", alpha=0.10, label="慢成分带")
    ax.axvspan(*config.VIB_BAND_HZ, color="#27AE60", alpha=0.08, label="振动带")
    ax.set_xlim(0, config.VIB_BAND_HZ[1])
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("PSD (µm²/Hz)")
    ax.set_title("末端 Y 向偏差功率谱（Welch，2 s Hann，50% 重叠）")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_psd.png", dpi=130)
    plt.close(fig)

    # ---- 图 3：RMS 柱状 ----
    groups = ["e_rms_um", "e_slow_rms_um", "e_vib_rms_um"]
    labels = ["总 RMS", f"慢成分 {config.SLOW_BAND_HZ[0]}–{config.SLOW_BAND_HZ[1]} Hz",
              f"振动带 {config.VIB_BAND_HZ[0]}–{config.VIB_BAND_HZ[1]} Hz"]
    x = np.arange(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, m in enumerate(config.MODES):
        vals = [mets[m][g] for g in groups]
        ax.bar(x + (i - 1) * w, vals, w, color=MODE_COLOR[m], label=m)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMS (µm)")
    ax.set_title("三组抑振效果对比（同一冻结 Fd、同一起始状态、同一名义轨迹）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_rms_bar.png", dpi=130)
    plt.close(fig)

    shear_check = {}
    if "ASFC" in runs:
        d = runs["ASFC"]
        lin, shr = np.abs(d["F_linear_damping_N"]), np.abs(d["F_shear_N"])
        lr, sr = float(np.sqrt(np.mean(lin ** 2))), float(np.sqrt(np.mean(shr ** 2)))
        nz = np.abs(d["F_shear_N"]) > 1e-15
        shear_check = {
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

    summary = {
        "modes": list(config.MODES),
        "n_samples": int(t.size),
        "duration_s": float(t[-1]),
        "trace_stride": config.TRACE_STRIDE,
        "fs_from_timestamps_hz": float(fs),
        "fs_note": "PSD/频带一律用 1/median(Δt)，不是物理步长 1000 Hz",
        "run_params": {m: {k: metas[m].get(k) for k in
                           ("K_N_per_m", "B0_Ns_per_m", "n", "mu", "r_design",
                            "r_actual", "v_ref_m_s", "fd_sha256")}
                       for m in config.MODES},
        "metrics": mets,
        "ratios": ratios,
        "shear_term_check": shear_check,
        "identical_inputs": {
            "same_frozen_Fd": float(np.max(np.abs(runs["A0"]["Fd_N"] - runs["ASFC"]["Fd_N"]))),
            "same_y_d": float(np.max(np.abs(runs["A0"]["y_d"] - runs["ASFC"]["y_d"]))),
            "same_along": float(np.max(np.abs(runs["A0"]["along_mm"] - runs["ASFC"]["along_mm"]))),
            "same_fd_sha": len({metas[m].get("fd_sha256") for m in config.MODES}) == 1,
            "note": "前三行是逐点最大差；全为 0 且 same_fd_sha 为真，才说明三组输入完全一致。",
        },
    }
    record.save_json(out_dir / "summary.json", summary)
    return summary


def print_report(summary: dict[str, Any]) -> None:
    m, r = summary["metrics"], summary["ratios"]
    print("\n" + "=" * 68)
    print("标准三组对照结果")
    print("=" * 68)
    print(f"{'模式':<10}{'总RMS(µm)':>12}{'峰峰(µm)':>12}{'慢成分(µm)':>12}{'振动带(µm)':>12}")
    for mode in summary["modes"]:
        d = m[mode]
        print(f"{mode:<10}{d['e_rms_um']:>12.3f}{d['e_ptp_um']:>12.3f}"
              f"{d['e_slow_rms_um']:>12.3f}{d['e_vib_rms_um']:>12.3f}")
    print("-" * 68)
    print(f"  Alinear 相对 A0       抑振比例：{r['Alinear_vs_A0'] * 100:+.2f}%"
          f"   （振动带 {r['Alinear_vs_A0_vib'] * 100:+.2f}%）")
    print(f"  ASFC    相对 A0       抑振比例：{r['ASFC_vs_A0'] * 100:+.2f}%"
          f"   （振动带 {r['ASFC_vs_A0_vib'] * 100:+.2f}%）")
    print(f"  ASFC    相对 Alinear  额外改善：{r['ASFC_vs_Alinear'] * 100:+.2f}%"
          f"   （振动带 {r['ASFC_vs_Alinear_vib'] * 100:+.2f}%）")
    sc = summary.get("shear_term_check") or {}
    if sc:
        print("-" * 68)
        print(f"  剪切项激活检查：|v| rms={sc['v_rms_m_s']:.3e} 峰值={sc['v_peak_abs_m_s']:.3e} m/s")
        print(f"                  F_linear rms={sc['F_linear_rms_N']:.4f} N  "
              f"F_shear rms={sc['F_shear_rms_N']:.4f} N  "
              f"(剪切/线性={sc['shear_over_linear_rms']:.4f})")
        print(f"                  剪切项恒与速度反向：{sc['sign_always_opposes_v']}   "
              f"P_shear>0 比例：{sc['P_shear_pos_frac']:.2e}")
    ii = summary.get("identical_inputs", {})
    if ii:
        print(f"  输入一致性：Fd 最大差 {ii['same_frozen_Fd']:.3e}，"
              f"y_d 最大差 {ii['same_y_d']:.3e}，沿程最大差 {ii['same_along']:.3e}，"
              f"Fd SHA 一致 {ii['same_fd_sha']}")
    print("=" * 68 + "\n")


# ------------------------------------------------------------------ 批量扫描可视化
def sweep_plots(rows: list[dict], out_dir: Path) -> None:
    """少量本地自查图（不入交付 ZIP）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows if r.get("status") == "OK"
          and "round1" in str(r.get("group", ""))
          and np.isfinite(r.get("e_rms_um", float("nan")))]
    if not ok:
        return

    ns = sorted({r["n"] for r in ok})
    rs = sorted({r["r_design"] for r in ok})

    def grid(key: str):
        Z = np.full((len(rs), len(ns)), np.nan)
        for r in ok:
            if r["n"] in ns and r["r_design"] in rs:
                Z[rs.index(r["r_design"]), ns.index(r["n"])] = r[key]
        return Z

    # 图 A：相对 A0 的抑制率 heatmap
    fig, ax = plt.subplots(figsize=(7.5, 5))
    Z = grid("suppress_vs_A0_pct")
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(ns)), [f"{n:g}" for n in ns])
    ax.set_yticks(range(len(rs)), [f"{r:g}" for r in rs])
    ax.set_xlabel("n")
    ax.set_ylabel("r（v_ref 处剪切/线性强度）")
    ax.set_title("ASFC 相对 A0 的 RMS 抑制率 (%)")
    for i in range(len(rs)):
        for j in range(len(ns)):
            if np.isfinite(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center",
                        color="w", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "figA_suppress_heatmap.png", dpi=130)
    plt.close(fig)

    # 图 B：实际剪切/线性力比
    fig, ax = plt.subplots(figsize=(7.5, 5))
    Z = grid("F_shear_over_linear")
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="magma")
    ax.set_xticks(range(len(ns)), [f"{n:g}" for n in ns])
    ax.set_yticks(range(len(rs)), [f"{r:g}" for r in rs])
    ax.set_xlabel("n")
    ax.set_ylabel("r")
    ax.set_title("实际 F_shear_rms / F_linear_rms")
    for i in range(len(rs)):
        for j in range(len(ns)):
            if np.isfinite(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center",
                        color="w", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "figB_shearratio_heatmap.png", dpi=130)
    plt.close(fig)

    # 图 C：额外改善 vs 剪切强度，按 n 分组
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in ns:
        sub = sorted([r for r in ok if r["n"] == n], key=lambda r: r["r_design"])
        ax.plot([r["F_shear_over_linear"] for r in sub],
                [r["extra_vs_Alinear_pct"] for r in sub], "o-", label=f"n={n:g}")
    ax.set_xscale("log")
    ax.set_xlabel("F_shear_rms / F_linear_rms")
    ax.set_ylabel("ASFC 相对 Alinear 的额外改善 (%)")
    ax.set_title("非线性强度 vs 额外抑振收益")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "figC_extra_vs_shear.png", dpi=130)
    plt.close(fig)


# ============================================================ §四 慢偏移中间过程图
def _csv_col(path: Path, col: int = 1) -> np.ndarray:
    """读一个带 `#` 注释表头的 CSV，取第 col 列。"""
    return np.genfromtxt(str(path), delimiter=",", comments="#")[:, col].astype(float)


def slow_offset_plots(packet_dir: Path, out_dir: Path) -> None:
    """§四 要求的中间过程图：只供人工检查，【不入任何交付 ZIP】。

    依次输出：
      图S1 原始 e_raw 与慢偏移基线 b(t) 同轴
      图S2 去慢偏移后的净振动 e_vib(t)
      图S3/S4 正向/反向稳态 b±(x)：分箱中位数 + LOWESS 曲线
      图S5 启动 / 转向 / 停止三段局部放大（确认 b 平滑过渡、没追随尖峰）
    """
    packet_dir, out_dir = Path(packet_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    need = ("b_slow_offset.csv", "e_raw_um.csv", "e_vib_um.csv", "schedule.csv")
    miss = [f for f in need if not (packet_dir / f).is_file()]
    if miss:
        raise FileNotFoundError(f"慢偏移中间图缺文件：{miss}（需先用 --source vib 重建）")

    t = _csv_col(packet_dir / "b_slow_offset.csv", 0)
    b = _csv_col(packet_dir / "b_slow_offset.csv", 1)
    e_raw = _csv_col(packet_dir / "e_raw_um.csv", 1)
    e_vib = _csv_col(packet_dir / "e_vib_um.csv", 1)
    x = _csv_col(packet_dir / "schedule.csv", 1)

    meta: dict = {}
    mp = packet_dir / config.FD_META_FILENAME
    if mp.is_file():
        meta = json.loads(mp.read_text(encoding="utf-8")).get("slow_offset", {}) or {}

    # ---- 图S1：e_raw vs b(t) ----
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(t, e_raw, lw=0.6, color="#8899AA", label="原始横向误差 $e_{raw}$")
    ax.plot(t, b, lw=2.0, color="#B03A2E", label="理想慢偏移基线 $b(t)$")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("横向误差 (µm)")
    ax.set_title("原始误差与慢偏移基线（b(t) 只跟随慢中心线，不追随振动）")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figS1_raw_vs_b.png", dpi=130)
    plt.close(fig)

    # ---- 图S2：e_vib ----
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.plot(t, e_vib, lw=0.6, color="#1E8449",
            label=f"净振动 $e_{{vib}}=e_{{raw}}-b$  rms={np.sqrt(np.mean(e_vib ** 2)):.2f} µm")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("$e_{vib}$ (µm)")
    ax.set_title("去慢偏移后的净振动（本轮扰动重建与 SFC 验证的唯一输入）")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figS2_e_vib.png", dpi=130)
    plt.close(fig)

    # ---- 图S3/S4：b±(x) ----
    for fname, tag, colour in (("b_plus_x.csv", "b_+(x) 正向稳态", "#1F6FB2"),
                               ("b_minus_x.csv", "b_-(x) 反向稳态", "#B9770E")):
        fp = packet_dir / fname
        if not fp.is_file():
            continue
        z = np.genfromtxt(str(fp), delimiter=",", comments="#", ndmin=2)
        bx, bmed, bn, bsmo = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.plot(bx, bmed, "o", ms=4, color="#95A5A6",
                label=f"分箱中位数（箱宽 {meta.get('bin_mm', float('nan')):.2f} mm）")
        ax.plot(bx, bsmo, "-", lw=2.0, color=colour,
                label=f"LOWESS 平滑（带宽 {meta.get('span_mm', float('nan')):.2f} mm）")
        ax.set_xlabel("沿程位置 x (mm)")
        ax.set_ylabel("慢偏移 (µm)")
        ax.set_title(f"{tag}：分箱中位数 + 鲁棒平滑 → 慢中心线")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"figS3_{fname.replace('.csv', '')}.png", dpi=130)
        plt.close(fig)

    # ---- 图S5：三段过渡局部放大 ----
    seg: dict = meta.get("segments") or {}
    windows = [("start", "启动段"), ("turn", "转向段"), ("stop", "停止段")]
    have = [(k, lbl) for k, lbl in windows if k in seg]
    if have:
        fig, axes = plt.subplots(len(have), 1, figsize=(12, 3.2 * len(have)),
                                 squeeze=False)
        for ax, (key, lbl) in zip(axes[:, 0], have):
            a, bb = int(seg[key][0]), int(seg[key][1])
            pad = max(1, int(0.15 * (bb - a + 1)))
            a, bb = max(0, a - pad), min(t.size - 1, bb + pad)
            sl = slice(a, bb + 1)
            ax.plot(t[sl], e_raw[sl], lw=0.8, color="#8899AA", label="$e_{raw}$")
            ax.plot(t[sl], b[sl], lw=2.2, color="#B03A2E", label="$b(t)$")
            ax.plot(t[sl], e_vib[sl], lw=0.8, color="#1E8449", alpha=0.75,
                    label="$e_{vib}$")
            ax.axhline(0.0, color="k", lw=0.7, alpha=0.4)
            ax.set_title(f"{lbl}（{key}）：b 应是平滑过渡，尖峰留在 $e_{{vib}}$ 里")
            ax.set_ylabel("µm")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper right")
        axes[-1, 0].set_xlabel("t (s)")
        fig.tight_layout()
        fig.savefig(out_dir / "figS5_transients.png", dpi=130)
        plt.close(fig)

    print(f"[分析] 慢偏移中间过程图 → {out_dir}（figS1..figS5）")


# ============================================================ §9.1 XY 抖动图
def compare_xy_plots(trace_paths: dict[str, Path], out_dir: Path,
                     packet_dir: Path | None = None) -> dict[str, Any]:
    """XY 抖动图（本轮口径）：横轴沿程位置，纵轴横向偏差。

    主图只画【处理后的振动】：预实验 e_vib 与三组仿真残差 e_um。
    原始 e_raw 只作为辅图留一行做对照，不再作为主结论依据（§9.1）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict[str, np.ndarray]] = {}
    for mode in config.MODES:
        p = trace_paths.get(mode)
        if p is not None and Path(p).is_file():
            runs[mode] = record.load_trace(Path(p))[1]

    # 参考序列必须用【模板网格】版本 ref_*.csv：它们与仿真时序同为 62853 点、
    # 同一条时间轴；记录网格的 e_vib_um.csv 只有 7756 点，直接画会维度不匹配。
    e_vib = e_raw = xs = None
    if packet_dir is not None and (Path(packet_dir) / "ref_e_vib_um.csv").is_file():
        pd_ = Path(packet_dir)
        e_vib = _csv_col(pd_ / "ref_e_vib_um.csv", 1)
        e_raw = _csv_col(pd_ / "ref_e_raw_um.csv", 1)
        xs = _csv_col(pd_ / "schedule.csv", 1)

    if not runs:
        raise FileNotFoundError("XY 抖动图：没有任何可用的时序文件")

    x_sim = runs["A0"]["along_mm"]
    zoom = 20.0   # §四 要求至少给出 0–20 mm 局部区域

    # ---- 图X1：主图 = 处理后的振动（预实验 e_vib + 三组仿真残差） ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax = axes[0]
    if e_vib is not None:
        ax.plot(xs, e_vib, lw=0.5, color="#7F8C8D", alpha=0.9,
                label="预实验净振动 $e_{vib}$（重建目标）")
    for m in config.MODES:
        if m in runs:
            ax.plot(x_sim, runs[m]["e_um"], lw=0.7, color=MODE_COLOR[m], label=m)
    ax.axhline(0.0, color="k", lw=0.9)
    ax.set_xlabel("沿程位置 (mm)")
    ax.set_ylabel("横向偏差 (µm)")
    ax.set_title("【主图·处理后】去慢偏移后的横向振动：预实验目标 vs 三组仿真残差")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if e_raw is not None:
        ax.plot(xs, e_raw, lw=0.5, color="#BBBBBB", alpha=0.9,
                label="原始 $e_{raw}$（含慢偏移）")
    for m in config.MODES:
        if m in runs:
            ax.plot(x_sim, runs[m]["e_um"], lw=0.7, color=MODE_COLOR[m], label=m)
    ax.axhline(0.0, color="k", lw=0.9)
    ax.set_xlabel("沿程位置 (mm)")
    ax.set_ylabel("横向偏差 (µm)")
    ax.set_title("【辅图·仅对照】原始 e_raw 与三组仿真残差（不作主结论依据）")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figX1_xy_main_vs_raw.png", dpi=130)
    plt.close(fig)

    # ---- 图X2：0–20 mm 局部放大（处理后） ----
    fig, ax = plt.subplots(figsize=(12, 4.8))
    mz = x_sim <= zoom
    if e_vib is not None:
        ax.plot(xs[xs <= zoom], e_vib[xs <= zoom], lw=0.6, color="#7F8C8D",
                alpha=0.9, label="预实验 $e_{vib}$")
    for m in config.MODES:
        if m in runs:
            ax.plot(x_sim[mz], runs[m]["e_um"][mz], lw=0.8, color=MODE_COLOR[m],
                    label=m)
    ax.axhline(0.0, color="k", lw=0.9)
    ax.set_xlabel("沿程位置 (mm)")
    ax.set_ylabel("横向偏差 (µm)")
    ax.set_title(f"【处理后 XY 局部放大】沿程 0–{zoom:.0f} mm")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figX2_xy_zoom0_20.png", dpi=130)
    plt.close(fig)

    print(f"[分析] XY 抖动图 → {out_dir}（figX1 主/辅、figX2 局部放大）")
    return {"zoom_max_mm": zoom, "n_sim_points": int(x_sim.size)}


# ============================================================ §9.2 回正阻碍局部复查
def return_local_analysis(trace_paths: dict[str, Path], out_dir: Path,
                          strong_id: str, along_mm: float, half_mm: float
                          ) -> dict[str, Any]:
    """在指定沿程窗口内逐项检查力与速度方向，判断「ASFC 妨碍回正」是否仍成立。

    判据（全部基于【去慢偏移后】的 e_vib 与仿真量）：
      * e 的符号   → 当前偏离中心线在哪一侧
      * v 的符号   → 正在往哪一侧走（回正时应指向 0）
      * F_spring = −K·e 恒指回中心线，是回正主力
      * F_linear_damping = −B0·v 恒与速度反向：回正时它在【助推】，
        远离中心线时它在刹车 —— 所以「阻尼拖慢回正」与「阻尼抑制过冲」
        是同一件事的两面，结论必须区分，不能只看回正变慢就判过阻尼。
      * F_shear 与 F_linear_damping 同号（都含 −v），只是幅值随 |v|^(n−1) 放大。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict[str, np.ndarray]] = {}
    for mode in config.MODES:
        p = trace_paths.get(mode)
        if p is not None and Path(p).is_file():
            runs[mode] = record.load_trace(Path(p))[1]
    if "A0" not in runs or "ASFC" not in runs:
        raise FileNotFoundError("回正复查需要 A0 与 ASFC 的完整时序")

    x = runs["A0"]["along_mm"]
    win = np.abs(x - along_mm) <= half_mm
    if int(win.sum()) < 10:
        raise ValueError(f"沿程 {along_mm}±{half_mm} mm 窗口内样本仅 {int(win.sum())} 个")

    e_vib = runs["A0"].get("e_vib_um", np.zeros_like(x))
    stats: dict[str, Any] = {}
    for m in config.MODES:
        if m not in runs:
            continue
        d = runs[m]
        v, e = d["v_m_s"][win], d["e_um"][win]
        fs_, fl_, sh_, sfc_ = (d["F_spring_N"][win], d["F_linear_damping_N"][win],
                               d["F_shear_N"][win], d["F_SFC_N"][win])
        # 「回正相」= 速度与位移反向（正朝中心线走）的样本
        ret = (np.sign(v) == -np.sign(e)) & (np.abs(e) > 1e-9)
        ret_frac = float(np.mean(ret)) if ret.size else float("nan")
        hinder = (float(np.mean(np.sign(fl_[ret]) == -np.sign(v[ret])))
                  if ret.any() else float("nan"))
        stats[m] = {
            "n_samples": int(win.sum()),
            "e_median_um": float(np.median(e)),
            "e_mean_um": float(np.mean(e)),
            "v_median_m_s": float(np.median(v)),
            "F_spring_median_N": float(np.median(fs_)),
            "F_linear_damping_median_N": float(np.median(fl_)),
            "F_shear_median_N": float(np.median(sh_)),
            "F_SFC_median_N": float(np.median(sfc_)),
            "回正相占比": ret_frac,
            "回正相内阻尼与速度反向占比": hinder,
            "RMS_um": float(np.sqrt(np.mean(e ** 2))),
        }

    # ---- 图R1：e / v / 四个力分量同轴 ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].axhline(0.0, color="k", lw=0.8, alpha=0.5)
    axes[0].plot(x[win], e_vib[win], lw=1.2, color="#7F8C8D", label="预实验 $e_{vib}$")
    for m in config.MODES:
        if m in runs:
            axes[0].plot(x[win], runs[m]["e_um"][win], lw=1.0,
                         color=MODE_COLOR[m], label=m)
    axes[0].set_ylabel("$e$ (µm)")
    axes[0].set_title(f"回正段局部复查（沿程 {along_mm:.1f}±{half_mm:.1f} mm）")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.0, color="k", lw=0.8, alpha=0.5)
    for m in config.MODES:
        if m in runs:
            axes[1].plot(x[win], runs[m]["v_m_s"][win] * 1e3, lw=1.0,
                         color=MODE_COLOR[m], label=m)
    axes[1].set_ylabel("$v$ (mm/s)")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)

    ax = axes[2]
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    d = runs["ASFC"]
    for key, colour, lbl in (("F_spring_N", "#1F6FB2", "$-K e$"),
                             ("F_linear_damping_N", "#B9770E", "$-B_0 v$"),
                             ("F_shear_N", "#8E44AD", "$-\\mu|v|^{n-1}v$"),
                             ("F_SFC_N", "#1E8449", "$F_{SFC}$")):
        ax.plot(x[win], d[key][win], lw=1.0, color=colour, label=lbl)
    ax.set_xlabel("沿程位置 (mm)")
    ax.set_ylabel("力 (N)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"figR1_return_local_{strong_id}.png", dpi=130)
    plt.close(fig)

    a0r, asr, alr = (stats[m]["RMS_um"] for m in ("A0", "ASFC", "Alinear"))
    extra = (alr - asr) / alr * 100.0 if alr > 0 else float("nan")
    verdict = (
        f"回正段局部复查：A0 {a0r:.2f} µm，Alinear {alr:.2f} µm，ASFC {asr:.2f} µm；"
        f"ASFC 相对 Alinear {extra:+.2f}%。"
        + ("本窗口内 ASFC 仍不优于 Alinear；" if extra < 0
           else "本窗口内 ASFC 已优于 Alinear；")
        + "阻尼项在回正相内确实与速度反向（对回正方向不利），"
          "但它同时抑制了反向过冲——是否算「过阻尼」要看该窗口 RMS 是否真的更大。"
    )
    out = {"window_along_mm": along_mm, "window_half_mm": half_mm,
           "strong_case": strong_id, "per_mode": stats,
           "ASFC_vs_Alinear_pct_in_window": float(extra),
           "verdict": verdict,
           "figure": f"figR1_return_local_{strong_id}.png"}
    record.save_json(out_dir / "return_local_analysis.json", out)
    print("[分析] " + verdict)
    return out
