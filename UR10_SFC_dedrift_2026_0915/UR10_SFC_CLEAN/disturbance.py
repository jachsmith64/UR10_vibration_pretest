"""
扰动重建与冻结 —— 真实预实验振动 → 等效扰动力 Fd(t)。

【本模块的两半，界线必须清楚】

  A. 离线重建（只跑一次，产物冻结）
        真实实验横向振动 e_exp(t)
          → 1 ms 网格重采样 → 按 P05R01 阈值切窗
          → “单程拉长”模板 e_des(t) 与沿程 schedule(t)
          → 标定 Plant 频响 G(f)：在 INIT_Q 构型测“Y 向外力 → Y 向位移”
          → 频域带限反演   Fd0 = irfft( E_des / G · taper )
          → 关闭 SFC 回放 → e_sim
          → 残差迭代修正   Fd ← Fd + lr · invert(e_des − e_sim)
          → 达到复现要求后冻结（写 CSV + SHA-256）

  B. 在线加载（正式三组 A0/Alinear/ASFC 共用）
        只读冻结文件；严格校验；force(t) 是纯插值查表。

【术语纪律】Fd 是“等效扰动力”，不是真实机械臂实际受到的物理外力。
【冻结纪律】正式运行中 Fd 绝不根据当前 e_sim 重算。本模块不读任何控制器输出。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

import config
import slowoffset
from sim_env import SimEnv, build_nominal_joint_trajectory

# ============================================================ 通用小工具


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _write_csv(path: Path, t: np.ndarray, v: np.ndarray, header: str) -> None:
    np.savetxt(str(path), np.column_stack([t, v]), delimiter=",",
               fmt="%.6f", header=header, comments="#")


# ============================================================ 1. 读实验数据
_COLS = ("time_s", "nominal_along_mm", "cross_track_um", "vision_speed_mm_s", "phase")


def read_experiment(path: Path) -> dict[str, np.ndarray]:
    """读预实验 CSV。表头名精确匹配；编码自动尝试 utf-8-sig / utf-8 / gbk。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"预实验数据不存在：{path}")
    rows = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, encoding=enc, newline="") as fh:
                rows = list(csv.reader(fh))
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if not rows:
        raise ValueError(f"无法解码 {path}")

    hdr = [h.replace("﻿", "").strip() for h in rows[0]]
    idx: dict[str, int] = {}
    for want in _COLS:
        hit = [i for i, h in enumerate(hdr) if h == want]
        if not hit:
            raise ValueError(f"{path} 缺列 {want}（表头前 8 列：{hdr[:8]}）")
        idx[want] = hit[0]

    body = [r for r in rows[1:] if len(r) > max(idx.values()) and r[idx[_COLS[0]]]]
    if not body:
        raise ValueError(f"{path} 无有效数据行")

    def col(name: str) -> np.ndarray:
        return np.array([float(r[idx[name]]) for r in body])

    raw = {
        "t": col("time_s"),
        "along_mm": col("nominal_along_mm"),
        "cross_um": col("cross_track_um"),
        "speed_mm_s": col("vision_speed_mm_s"),
        "phase": np.array([r[idx["phase"]].strip() for r in body], dtype=object),
    }
    dt = np.diff(raw["t"])
    med = float(np.median(dt)) if dt.size else float("nan")
    raw["sampling"] = {
        "n": int(raw["t"].size),
        "dt_median_s": med,
        "dt_max_s": float(np.max(dt)) if dt.size else float("nan"),
        "gap_pct": float(100.0 * np.mean(dt > 1.5 * med)) if med > 0 else 0.0,
        "note": "相机原始帧 ~132 Hz，含丢帧；未重采样",
    }
    return raw


def resample_to_grid(raw: dict[str, Any], dt: float = config.GRID_DT_S) -> dict[str, np.ndarray]:
    """线性内插到 dt 均匀网格，供切窗/拼接/FFT 用。

    注意：内插不会创造真实带宽；高频段本质是插值噪声，这也是反演带限的理由。
    """
    t = raw["t"]
    tg = np.arange(float(t[0]), float(t[-1]) + dt / 2, dt)
    phase_codes = {p: i for i, p in enumerate(dict.fromkeys(raw["phase"]))}
    codes = np.array([phase_codes[p] for p in raw["phase"]])
    names = list(phase_codes)
    return {
        "t": tg,
        "along_mm": np.interp(tg, t, raw["along_mm"]),
        "cross_um": np.interp(tg, t, raw["cross_um"]),
        "speed_mm_s": np.interp(tg, t, raw["speed_mm_s"]),
        "phase_names": names,
        "phase_code": codes[np.clip(np.searchsorted(t, tg, side="right") - 1, 0, t.size - 1)],
    }


# ============================================================ 2. 切窗
@dataclass
class Cuts:
    iA: int   # 启动结束，进前向稳态
    iB: int   # 前向稳态结束，开始刹车
    iP: int   # 远端换向顶点
    iC: int   # 返程瞬态结束，回稳态
    iD: int   # 返程稳态结束，开始减速停


def _phase_span(d: dict[str, Any], name: str) -> tuple[int, int]:
    if name not in d["phase_names"]:
        raise ValueError(f"数据缺少阶段 {name}（现有 {d['phase_names']}）")
    hit = np.where(d["phase_code"] == d["phase_names"].index(name))[0]
    if hit.size == 0:
        raise ValueError(f"阶段 {name} 无样本")
    return int(hit[0]), int(hit[-1])


def detect_cuts(d: dict[str, Any]) -> Cuts:
    """按 P05R01 显式阈值切窗；任一条件不命中即报错（不静默兜底）。"""
    al, cr, sp = d["along_mm"], d["cross_um"], d["speed_mm_s"]
    o0, o1 = _phase_span(d, "X_outbound")
    r0, r1 = _phase_span(d, "X_return")
    ob = al[o0:o1 + 1]

    def first_at_least(cond: np.ndarray, what: str) -> int:
        hit = np.where(cond)[0]
        if hit.size == 0:
            raise ValueError(f"[切窗] {what} 未命中，数据不满足 P05R01 规则")
        return int(hit[0])

    iA = o0 + first_at_least(ob >= config.CUT_IA_ALONG_MM,
                             f"X_outbound 内沿程 ≥ {config.CUT_IA_ALONG_MM} mm")
    iB = o0 + first_at_least(ob >= config.CUT_IB_ALONG_MM,
                             f"X_outbound 内沿程 ≥ {config.CUT_IB_ALONG_MM} mm")
    iP = o0 + int(np.where(ob == ob.max())[0][-1])

    seg = cr[r0:r1 + 1]
    lim = min(seg.size, int(config.CUT_IC_SEARCH_S / config.GRID_DT_S) + 1)
    kmin = int(np.argmin(seg[:lim]))
    up = np.where(seg[kmin:] >= config.CUT_IC_CROSS_UM)[0]
    if up.size == 0:
        raise ValueError(f"[切窗] 返程未回升越过 {config.CUT_IC_CROSS_UM} µm")
    iC = r0 + kmin + int(up[0])

    mv = np.where(sp[r0:r1 + 1] < config.CUT_ID_SPEED_MM_S)[0]
    if mv.size == 0:
        raise ValueError(f"[切窗] 返程无 < {config.CUT_ID_SPEED_MM_S} mm/s 的全速段")
    iD = r0 + int(mv[-1])

    if not (iA < iB < iP < iC < iD):
        raise ValueError(f"[切窗] 顺序异常 iA..iD = {iA},{iB},{iP},{iC},{iD}")
    return Cuts(iA=iA, iB=iB, iP=iP, iC=iC, iD=iD)


# ============================================================ 3. 单程拉长模板
@dataclass
class Template:
    t: np.ndarray
    along_mm: np.ndarray
    e_um: np.ndarray
    segments: list[tuple[str, int, int]] = field(default_factory=list)
    seam_jumps_um: list[float] = field(default_factory=list)
    n_fwd_repeat: int = 1
    n_ret_repeat: int = 0


def _smooth_seams(x: np.ndarray, seams: list[int], half: int) -> np.ndarray:
    """在每条拼接缝处，以 ±half 窗把信号按 sin² 权重混向两端点连线。"""
    out = x.copy()
    n = len(out)
    for s in seams:
        a, b = max(0, s - half), min(n - 1, s + half)
        if b - a < 4:
            continue
        line = np.linspace(out[a], out[b], b - a + 1)
        w = np.sin(np.pi * np.linspace(0.0, 1.0, b - a + 1)) ** 2
        out[a:b + 1] = (1.0 - w) * out[a:b + 1] + w * line
    return out


def build_template(d: dict[str, Any], duration_s: float = config.TEMPLATE_DURATION_S,
                   n_stroke: int | None = None,
                   e_source: np.ndarray | None = None) -> Template:
    """把真实实验的稳态片段确定性复制拉长成模板（不含新正弦/随机相位/随机噪声）。

    拼接顺序： start → mid_fwd×N → turn_fwd → turn_ret → mid_ret×M → mid_ret_last → stop
    假设：中段稳态振动统计平稳，故可直接重复真实片段，不需造信号。

    e_source：本层要用的误差序列。默认取原始 cross_um（上一轮口径）；
              本轮传 slowoffset 扣掉慢偏移后的 e_vib，模板与沿程 schedule 完全不变，
              只把「被重建的那条曲线」换掉。
    """
    t, al = d["t"], d["along_mm"]
    cr = np.asarray(d["cross_um"], dtype=float) if e_source is None \
        else np.asarray(e_source, dtype=float)
    if cr.shape != al.shape:
        raise ValueError(f"e_source 形状 {cr.shape} 与沿程 {al.shape} 不一致")
    c = detect_cuts(d)
    iE = len(t) - 1
    half = int(round(config.SEAM_SMOOTH_S / config.GRID_DT_S))

    def unit(sl: slice) -> tuple[np.ndarray, np.ndarray]:
        return (al[sl] - al[sl.start]).copy(), cr[sl].copy()

    d_start, e_start = unit(slice(0, c.iA + 1))
    d_fwd, e_fwd = unit(slice(c.iA, c.iB + 1))
    d_tf, e_tf = unit(slice(c.iB, c.iP + 1))
    d_tr, e_tr = unit(slice(c.iP, c.iC + 1))
    d_ret, e_ret = unit(slice(c.iC, c.iD + 1))
    d_stop, e_stop = unit(slice(c.iD, iE + 1))

    launch = float(d_start[-1]); duf = float(d_fwd[-1])
    fwd_turn = float(d_tf[-1]); turn_ret = float(-d_tr[-1])
    ur_drop = float(-d_ret[-1]); stop_drop = float(-d_stop[-1])

    def assemble(n_fwd: int):
        Lf = launch + duf * n_fwd + fwd_turn
        need_ret = max((Lf - turn_ret) - stop_drop, 0.0)
        m_full = int(need_ret // ur_drop)
        rem = need_ret - m_full * ur_drop

        along_parts: list[np.ndarray] = []
        e_parts: list[np.ndarray] = []
        segments: list[tuple[str, int, int]] = []
        seams: list[int] = []
        base = 0.0
        pos = 0

        def place(name: str, dloc: np.ndarray, e: np.ndarray, seam: bool) -> None:
            nonlocal base, pos
            if seam and along_parts:
                seams.append(pos)
            along_parts.append(base + dloc)
            e_parts.append(e)
            segments.append((name, pos, pos + len(dloc)))
            base = float(base + dloc[-1])
            pos += len(dloc)

        place("start", d_start, e_start, False)
        for k in range(n_fwd):
            place("mid_fwd", d_fwd, e_fwd, seam=(k >= 1))
        place("turn_fwd", d_tf, e_tf, False)
        place("turn_ret", d_tr, e_tr, False)
        for k in range(m_full):
            place("mid_ret", d_ret, e_ret, seam=(k >= 1))
        if rem > 1e-6 and len(d_ret) > 10:
            take = int(np.clip(int(np.searchsorted(np.abs(d_ret), rem)) + 1, 2, len(d_ret)))
            place("mid_ret_last", d_ret[:take], e_ret[:take], True)
        place("stop", d_stop, e_stop, False)

        along = np.concatenate(along_parts)
        e = _smooth_seams(np.concatenate(e_parts), seams, half)
        return along, e, segments, seams, m_full, along.size * config.GRID_DT_S

    if n_stroke is None:
        n_stroke = 1
        while assemble(n_stroke)[-1] < duration_s - 1e-6:
            n_stroke += 1
    along, e, segments, seams, m_ret, dur = assemble(n_stroke)

    return Template(
        t=np.arange(e.size, dtype=float) * config.GRID_DT_S,
        along_mm=along, e_um=e, segments=segments,
        seam_jumps_um=[float(abs(e[i] - e[i - 1])) for i in seams if 0 < i < len(e)],
        n_fwd_repeat=n_stroke, n_ret_repeat=m_ret,
    )


# ============================================================ 4. Plant 频响标定
@dataclass
class Plant:
    freq_hz: np.ndarray
    gain: np.ndarray          # 复增益 [µm/N]
    dc_um_per_n: float

    def at(self, freqs: np.ndarray) -> np.ndarray:
        f = np.asarray(freqs, dtype=float)
        fs = self.freq_hz
        real = np.interp(f, fs, self.gain.real)
        imag = np.interp(f, fs, self.gain.imag)
        dc = f <= 0.0
        real[dc] = self.dc_um_per_n
        imag[dc] = 0.0
        return real + 1j * imag


def _schroeder_phase(k: int, n: int) -> float:
    """Schroeder 相位：把多正弦的峰值因子压到最低，避免探针本身饱和执行器。"""
    return 0.0 if k == 0 else math.pi * k * (k - 1) / n


def measure_plant(env: SimEnv | None = None, log: Callable[[str], None] = print) -> Plant:
    """sum-of-sines 扫频：在 INIT_Q 构型测 “末端 Y 外力 → 末端 Y 位移” 的复频响。

    先跑稳定段丢弃初始伺服瞬态，再逐点施加 [0, f(t), 0] 并记录 TCP Y 位移。
    """
    dt = config.physics_dt()
    n = int(round(config.PROBE_DURATION_S / dt))
    kmax = int(round(config.PROBE_FMAX_HZ * config.PROBE_DURATION_S))
    freq_bins = np.arange(1, kmax + 1) / config.PROBE_DURATION_S
    ph = np.array([_schroeder_phase(k, kmax + 1) for k in range(1, kmax + 1)])

    tt = np.arange(n, dtype=float) * dt
    force = np.zeros(n)
    for s in range(0, freq_bins.size, 500):
        fb = freq_bins[s:s + 500, None]
        pb = ph[s:s + 500, None]
        force += np.sum(config.PROBE_AMP_N * np.sin(2 * np.pi * fb * tt[None, :] + pb), axis=0)

    env = env or SimEnv()
    env.reset(config.INIT_Q)
    for _ in range(config.PROBE_SETTLE_STEPS):
        env.step()
    y0 = env.tcp_y()

    y = np.zeros(n)
    for i in range(n):
        env.apply_disturbance_force_y(float(force[i]))
        env.step()
        y[i] = (env.tcp_y() - y0) * 1e6

    Y = np.fft.rfft(y)
    F = np.fft.rfft(force)
    freqs = np.fft.rfftfreq(n, d=dt)
    g = np.full(freqs.shape, np.nan, dtype=complex)
    kidx = (freqs * config.PROBE_DURATION_S + 0.5).astype(int)
    valid = (kidx >= 1) & (kidx <= kmax)
    g[valid] = Y[valid] / F[valid]

    dc = measure_dc_compliance()
    keep = ~np.isnan(g)
    plant = Plant(freq_hz=freqs[keep], gain=g[keep], dc_um_per_n=dc)
    log(f"[plant] {plant.freq_hz.size} 个 bin（{plant.freq_hz[0]:.3f}–{plant.freq_hz[-1]:.1f} Hz），"
        f"DC 静柔度 = {dc:.2f} µm/N")
    return plant


def measure_dc_compliance(env: SimEnv | None = None) -> float:
    """静柔度：1 N 恒定 Y 力稳定后 TCP Y 位移 (µm/N)。"""
    dt = config.physics_dt()
    env = env or SimEnv()
    env.reset(config.INIT_Q)
    for _ in range(config.PROBE_DC_SETTLE_STEPS):
        env.step()
    y0 = env.tcp_y()
    for _ in range(int(round(1.0 / dt))):
        env.apply_disturbance_force_y(1.0)
        env.step()
    return float((env.tcp_y() - y0) * 1e6)


# ============================================================ 5. 带限反演
def cos_taper(freq_hz: np.ndarray, f_pass: float, f_stop: float) -> np.ndarray:
    """1（f ≤ f_pass）→ 余弦渐减（f_pass..f_stop）→ 0（f ≥ f_stop）。"""
    taper = np.ones_like(freq_hz)
    up = (freq_hz > f_pass) & (freq_hz < f_stop)
    taper[up] = 0.5 * (1.0 + np.cos(np.pi * (freq_hz[up] - f_pass) / (f_stop - f_pass)))
    taper[freq_hz >= f_stop] = 0.0
    return taper


def invert_series(e_um: np.ndarray, plant: Plant,
                  f_pass: float = config.INV_F_PASS_HZ,
                  f_stop: float = config.INV_F_STOP_HZ) -> np.ndarray:
    """频域复数除法 + 带限： Fd = irfft( rfft(e) / G · taper )。"""
    x = np.asarray(e_um, dtype=float)
    n = int(x.size)
    freq = np.fft.rfftfreq(n, d=config.GRID_DT_S)
    G = plant.at(freq)
    taper = cos_taper(freq, f_pass, f_stop)
    W = np.zeros_like(np.fft.rfft(x))
    ok = (taper > 0.0) & (np.abs(G) > 1e-6)
    W[ok] = (np.fft.rfft(x)[ok] / G[ok]) * taper[ok]
    return np.fft.irfft(W, n=n)


# ============================================================ 6. 基线回放（重建专用）
def run_baseline(e_um_target: np.ndarray, q_traj: np.ndarray, fd_N: np.ndarray,
                 y_d: float = config.Y_NOMINAL,
                 log: Callable[[str], None] | None = None) -> np.ndarray:
    """关闭 SFC，施加 Fd 并按名义轨迹运动，返回 e_sim(µm)。

    记录对齐：第 i 步先读当前 TCP Y（对应 t_i）再施加该步的 ctrl 与 Fd，然后推进。
    这是重建阶段的专用回放，不涉及任何控制器。
    """
    dt = config.GRID_DT_S
    n = int(fd_N.size)
    env = SimEnv()
    env.reset(q_traj[0])
    e_sim = np.zeros(n)
    for i in range(n):
        e_sim[i] = (env.tcp_y() - y_d) * 1e6
        env.set_ctrl(q_traj[i])
        env.apply_disturbance_force_y(float(fd_N[i]))
        env.step()
    if log and n:
        log(f"[baseline] {n} 步 / {n * dt:.1f} s，e_sim rms = "
            f"{float(np.sqrt(np.mean(e_sim ** 2))):.2f} µm")
    return e_sim


# ============================================================ 7. 重建主流程 + 冻结
def reconstruct_fd(out_dir: Path, duration_s: float = config.TEMPLATE_DURATION_S,
                   refine_iters: int = config.REFINE_ITERS,
                   lr: float = config.REFINE_LR,
                   source: str = config.RECON_SOURCE_DEFAULT,
                   log: Callable[[str], None] = print) -> dict[str, Any]:
    """执行 A 段全流程并冻结 Fd。返回 meta。

    source = "vib"：先构造并扣除理想慢偏移 b(t)，只对净振动 e_vib 做反演与迭代（本轮正式口径）；
    source = "raw"：直接对原始 e_raw 做重建（上一轮口径，仅用于对照/画辅图）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if source not in ("vib", "raw"):
        raise ValueError(f"未知重建来源 source={source!r}（只允许 'vib' / 'raw'）")

    raw = read_experiment(config.DATA_CSV)
    log(f"[重建] 读入实验数据 {raw['sampling']['n']} 帧，"
        f"dt 中位 {raw['sampling']['dt_median_s'] * 1e3:.2f} ms，"
        f"丢帧 {raw['sampling']['gap_pct']:.2f}%")
    d = resample_to_grid(raw)
    cuts = detect_cuts(d)
    log(f"[重建] 切窗 iA/iB/iP/iC/iD = {cuts.iA}/{cuts.iB}/{cuts.iP}/{cuts.iC}/{cuts.iD}")

    # ---- 理想慢偏移：构造 b(t) 并扣除 → e_vib ----
    slow: dict[str, Any] | None = None
    e_source = None
    if source == "vib":
        slow = slowoffset.build_slow_offset(d, cuts, log=log)
        e_source = slow["e_vib_um"]
        _write_csv(out_dir / "b_slow_offset.csv", slow["t"], slow["b_um"], "t_s,b_um")
        _write_csv(out_dir / "e_raw_um.csv", slow["t"], slow["e_raw_um"], "t_s,e_raw_um")
        _write_csv(out_dir / "e_vib_um.csv", slow["t"], slow["e_vib_um"], "t_s,e_vib_um")
        for fit, fname in ((slow["fit_fwd"], "b_plus_x.csv"),
                           (slow["fit_ret"], "b_minus_x.csv")):
            np.savetxt(str(out_dir / fname),
                       np.column_stack([fit.x, fit.b, fit.n, fit.curve.at(fit.x)]),
                       delimiter=",", fmt="%.6f",
                       header="x_mm,b_bin_median_um,n_samples,b_lowess_um", comments="#")
        acc = slow["acceptance"]
        log(f"[慢偏移] 人工验收：{acc['conclusion']}")
        if not acc["passed"]:
            log(f"[慢偏移] 未通过项 {acc['failed_checks']}；原因：{acc['reasons']}")
    else:
        log("[重建] 来源 = 原始 e_raw（不做慢偏移扣除，仅供对照）")

    T = build_template(d, duration_s=duration_s, e_source=e_source)
    # 慢偏移三件套的【模板网格】版本。build_template 的分段只由 d 决定（与 e_source
    # 无关），所以用同一个 d / duration_s 再建两次，得到的就是同一套时间轴上的
    # e_raw(t) 与 b(t)。时序 CSV 里要带的参考列必须是模板网格（62.85 s）而不是
    # 原始记录网格（7.755 s），否则与仿真量对不齐。
    ref_tpl: dict[str, np.ndarray] = {}
    if slow is not None:
        ref_tpl = {
            "e_raw_um": build_template(d, duration_s=duration_s,
                                       e_source=slow["e_raw_um"]).e_um,
            "b_um": build_template(d, duration_s=duration_s,
                                   e_source=slow["b_um"]).e_um,
            "e_vib_um": T.e_um,
        }
        if any(v.shape != T.e_um.shape for v in ref_tpl.values()):
            raise ValueError("[重建] 慢偏移模板网格长度不一致，无法作为时序参考列")
    log(f"[重建] 模板 {T.t[-1]:.2f} s（{T.e_um.size} 点），远点 {T.along_mm.max():.1f} mm，"
        f"前向复制 ×{T.n_fwd_repeat} 返向 ×{T.n_ret_repeat}，"
        f"缝跳跃 均值 {np.mean(T.seam_jumps_um):.3f} / 最大 {np.max(T.seam_jumps_um):.3f} µm")

    log("[重建] 标定 Plant 频响（sum-of-sines 探针，约 60 s 仿真）…")
    plant = measure_plant(log=log)

    log("[重建] 名义关节轨迹 IK…")
    q_traj = build_nominal_joint_trajectory(T.along_mm, config.Y_NOMINAL, config.Z_HOLD)

    fd = invert_series(T.e_um, plant)
    log(f"[重建] 反演初值 Fd：rms {np.sqrt(np.mean(fd ** 2)):.3f} N，"
        f"峰值 {np.max(np.abs(fd)):.3f} N，带限 ≤ {config.INV_F_PASS_HZ:.0f} Hz"
        f"（{config.INV_F_STOP_HZ:.0f} Hz 渐减到 0）")

    history: list[dict[str, float]] = []
    target_rms = float(np.sqrt(np.mean(T.e_um ** 2)))
    stop_rms = config.REFINE_STOP_RESIDUAL_FRAC * target_rms
    for it in range(1, int(refine_iters) + 1):
        e_sim = run_baseline(T.e_um, q_traj, fd, log=log)
        res = T.e_um - e_sim
        rr = float(np.sqrt(np.mean(res ** 2)))
        cc = float(np.corrcoef(T.e_um, e_sim)[0, 1])
        history.append({
            "iter": it,
            "e_sim_rms_um": float(np.sqrt(np.mean(e_sim ** 2))),
            "corr": cc, "residual_rms_um": rr,
            "residual_over_target": rr / target_rms if target_rms > 0 else float("nan"),
            "gate_pass": bool(cc >= config.GATE_CORR_MIN
                              and rr <= config.GATE_RESIDUAL_FRAC_MAX * target_rms),
        })
        log(f"[重建] 迭代 {it}：e_sim rms {history[-1]['e_sim_rms_um']:.2f} µm，"
            f"corr {cc:.4f}，残差 rms {rr:.2f} µm"
            f"（残差/目标 {rr / target_rms * 100:.1f}%）"
            f"{'  ← 已达验收门槛' if history[-1]['gate_pass'] else ''}")
        if rr <= stop_rms:
            log(f"[重建] 残差 rms ≤ {stop_rms:.2f} µm（目标的 "
                f"{config.REFINE_STOP_RESIDUAL_FRAC * 100:.0f}%），提前收敛")
            break
        if it < int(refine_iters):
            fd = fd + lr * invert_series(res, plant)
        else:
            # 最后一次迭代不再更新：否则被冻结的 Fd 会比最后一次实测的 e_sim 多走一步，
            # 冻结产物就成了“没有验收过的版本”。宁可少修正一次，也不要冻结未测量的结果。
            log("[重建] 已达设定迭代上限，冻结当前 Fd（不再多做一次未测量的修正）")

    # ---- 施力点数值自检（修正后的强制检查）----
    _chk = SimEnv()
    _chk.reset(config.INIT_Q)
    tcp_check = _chk.check_tcp_force_generalized(1.0)
    log(f"[重建] TCP 施力点检查：max|q_applyFT − J_tcp^T·F| = "
        f"{tcp_check['max_abs_err']:.3e}（参考量级 {tcp_check['ref_max_abs']:.3e}），"
        f"{'通过' if tcp_check['passed'] else '不通过'}")
    if not tcp_check["passed"]:
        raise RuntimeError("TCP 施力点折算出的广义力与 J_tcp^T·F 不一致，Fd 作用点未真正落在 TCP 上")

    # ---- 冻结 ----
    fd_path = out_dir / config.FD_FILENAME
    _write_csv(fd_path, T.t, fd, "t_s,Fd_N")
    _write_csv(out_dir / "schedule.csv", T.t, T.along_mm, "t_s,along_mm")
    _write_csv(out_dir / "e_des_um.csv", T.t, T.e_um, "t_s,e_des_um")
    for _k, _v in ref_tpl.items():          # 模板网格上的 e_raw / b / e_vib
        _write_csv(out_dir / f"ref_{_k}.csv", T.t, _v, f"t_s,{_k}")

    meta = {
        "kind": "equivalent_disturbance_force",
        "disclaimer": "Fd 是等效扰动力，不是真实机械臂实际受到的物理外力。",
        "recon_source": source,
        "source_note": ("先扣除理想慢偏移 b(t)，只重建净振动 e_vib" if source == "vib"
                        else "直接重建原始 e_raw（上一轮口径，仅对照）"),
        "data_id": "P05R01",
        "source_csv": str(config.DATA_CSV),
        "source_csv_sha256": sha256_file(config.DATA_CSV),
        "source_sampling": raw["sampling"],
        "grid_dt_s": config.GRID_DT_S,
        "slow_offset": None if slow is None else {
            "method": "稳态段=沿程分箱中位数+LOWESS 鲁棒平滑 → b±(x)；"
                      "启动/转向/停止段=三次 Hermite 一阶连续桥接",
            "bin_mm": slow["bin_mm"], "span_mm": slow["span_mm"],
            "b_rest_um": slow["b_rest_um"],
            "span_attempts": slow["span_attempts"],
            "segments": {k: [int(v[0]), int(v[1])] for k, v in slow["segments"].items()},
            "b_range_um": [float(slow["b_um"].min()), float(slow["b_um"].max())],
            "fit_fwd": {"x_range_mm": [float(slow["fit_fwd"].x[0]),
                                       float(slow["fit_fwd"].x[-1])],
                        "b_range_um": [float(slow["fit_fwd"].b.min()),
                                       float(slow["fit_fwd"].b.max())],
                        "n_bins": int(slow["fit_fwd"].x.size)},
            "fit_ret": {"x_range_mm": [float(slow["fit_ret"].x[0]),
                                       float(slow["fit_ret"].x[-1])],
                        "b_range_um": [float(slow["fit_ret"].b.min()),
                                       float(slow["fit_ret"].b.max())],
                        "n_bins": int(slow["fit_ret"].x.size)},
            "acceptance": slow["acceptance"],
            "files": {"b": "b_slow_offset.csv", "e_raw": "e_raw_um.csv",
                      "e_vib": "e_vib_um.csv", "b_plus_x": "b_plus_x.csv",
                      "b_minus_x": "b_minus_x.csv"},
            "files_template_grid": {"e_raw": "ref_e_raw_um.csv", "b": "ref_b_um.csv",
                                    "e_vib": "ref_e_vib_um.csv"},
            "note_template_grid": "记录网格（7.755 s）与模板网格（62.85 s）两套："
                                  "上面的 *_um.csv 在记录网格上，供中间过程图与人工检查；"
                                  "ref_*.csv 在模板网格上，与 Fd 等长，供时序 CSV 的参考列。",
        },
        "template": {
            "dur_s": float(T.t[-1]), "n_samples": int(T.e_um.size),
            "max_along_mm": float(T.along_mm.max()),
            "n_segments": len(T.segments),
            "seam_jump_mean_um": float(np.mean(T.seam_jumps_um)),
            "seam_jump_max_um": float(np.max(T.seam_jumps_um)),
            "n_fwd_repeat": int(T.n_fwd_repeat), "n_ret_repeat": int(T.n_ret_repeat),
            "e_rms_um": float(np.sqrt(np.mean(T.e_um ** 2))),
            "e_ptp_um": float(np.ptp(T.e_um)),
            "e_median_um": float(np.median(T.e_um)),
            "e_mean_um": float(np.mean(T.e_um)),
        },
        "cuts": {"iA": cuts.iA, "iB": cuts.iB, "iP": cuts.iP,
                 "iC": cuts.iC, "iD": cuts.iD},
        "force_application": {
            "method": "mujoco.mj_applyFT(force=[0,Fd,0], torque=0, point=TCP site, "
                      "body=" + config.FORCE_BODY_NAME + ")",
            "point": "TCP site '" + config.TCP_SITE_NAME + "' 的世界坐标",
            "note": "Fd 作用在 TCP 空间点，而非 wrist_3_link 刚体质心；"
                    "广义力与 J_tcp^T·[0,F,0] 数值一致（见 fd_meta 的 tcp_force_check）。",
            "tcp_force_check": tcp_check,
        },
        "plant": {"dc_um_per_n": plant.dc_um_per_n,
                  "n_bins": int(plant.freq_hz.size),
                  "probe_dur_s": config.PROBE_DURATION_S,
                  "probe_fmax_hz": config.PROBE_FMAX_HZ,
                  "probe_amp_n": config.PROBE_AMP_N,
                  "config": "INIT_Q",
                  "excitation_point": "TCP site（与 Fd 作用点一致）"},
        "inversion": {"f_pass_hz": config.INV_F_PASS_HZ, "f_stop_hz": config.INV_F_STOP_HZ,
                      "method": "Fd = irfft( rfft(e_des)/G · cos_taper )"},
        "refine": {"iters_done": len(history), "lr": lr, "history": history},
        "frozen": {"file": config.FD_FILENAME, "sha256": sha256_file(fd_path),
                   "n_samples": int(fd.size),
                   "Fd_rms_N": float(np.sqrt(np.mean(fd ** 2))),
                   "Fd_ptp_N": float(np.ptp(fd)),
                   "Fd_peak_abs_N": float(np.max(np.abs(fd)))},
    }
    (out_dir / config.FD_META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[重建] 已冻结 {fd_path.name}  sha256={meta['frozen']['sha256'][:12]}…")
    return meta


# ============================================================ 8. 在线加载（正式运行唯一入口）
class FrozenDisturbance:
    """冻结 Fd(t) 的只读句柄。force(t) 是纯插值，不看任何仿真状态。"""

    def __init__(self, path: Path, t: np.ndarray, f: np.ndarray, meta: dict[str, Any]):
        self.path = Path(path)
        self.t = t
        self.f = f
        self.meta = meta
        self.sha256 = meta.get("frozen", {}).get("sha256", "")

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    @property
    def n_samples(self) -> int:
        return int(self.t.size)

    def force(self, tt: float) -> float:
        """t 时刻的等效扰动力 (N)。窗外夹紧到端点。"""
        return float(np.interp(tt, self.t, self.f))

    def describe(self) -> str:
        return (f"冻结 Fd：{self.path.name}（sha12={self.sha256[:12]}，"
                f"{self.n_samples} 点，t∈[{self.t[0]:.3f}, {self.t[-1]:.3f}] s）")


def load_frozen(packet_dir: Path, required_t_end: float | None = None) -> FrozenDisturbance:
    """加载冻结 Fd 并严格校验；覆盖不足直接报错，禁止循环/补零/替换。"""
    packet_dir = Path(packet_dir)
    fd_path = packet_dir / config.FD_FILENAME
    meta_path = packet_dir / config.FD_META_FILENAME
    if not fd_path.is_file():
        raise FileNotFoundError(f"冻结 Fd 不存在：{fd_path}（请先运行 main.py --reconstruct）")

    data = np.genfromtxt(str(fd_path), delimiter=",", comments="#")
    data = data[~np.isnan(data).any(axis=1)]
    if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 2:
        raise ValueError(f"冻结 Fd 至少需两列且 ≥2 行：{fd_path}")
    t, f = data[:, 0].astype(float), data[:, 1].astype(float)

    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(f))):
        raise ValueError(f"冻结 Fd 必须全为有限数：{fd_path}")
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"冻结 Fd 时间列必须严格递增：{fd_path}")
    if required_t_end is not None and float(t[-1]) < float(required_t_end) - 1e-9:
        raise ValueError(
            f"冻结 Fd 覆盖不足：文件到 t={t[-1]:.3f}s，本次需要到 {required_t_end:.3f}s。"
            "禁止循环/补零/替换。")

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    got = sha256_file(fd_path)
    if meta.get("frozen", {}).get("sha256") and meta["frozen"]["sha256"] != got:
        raise ValueError(f"冻结 Fd 哈希不符（可能被改动过）：{fd_path}")
    return FrozenDisturbance(fd_path, t, f, meta)
