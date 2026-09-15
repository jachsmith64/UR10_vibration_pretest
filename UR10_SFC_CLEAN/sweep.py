"""
批处理驱动 —— 5 个正式组 + 三组机制验证测试（V1.3）。

本模块【不做参数扫描、不做自动寻优】。跑什么全部由 config 写死：
    config.ROUND_GROUPS  5 个正式组里的 3 个 ASFC 组（n 与 r 由用户指定）
    config.SCALE_FACTORS 测试A 的扰动幅值缩放档
    config.BIAS_UM       测试B 的慢偏移估计偏差档

设计纪律：
  * 架构不变。本模块只调用已有的 扰动支路 / 控制支路 / 执行支路，不新增控制概念。
  * A0 与 Alinear 各只跑一次（s=1.0 档），作为公共基准，不每组重复跑。
  * 所有组共用同一份冻结 Fd（同一 FrozenDisturbance 句柄，逐字节相同，逐组记 SHA-256）。
    测试A 只把同一个 Fd 整体乘一个常数 s，测试B 用同一份 Plant 与同一份名义轨迹重建。
  * 任何一组出现 NaN / 发散 / 跑飞 → 立即中止该组、标记 INVALID、继续下一组，
    绝不让单组失败带崩整批。
  * 本模块【不产生任何图片】。

μ 的标定（见 config.py 与 sfc.mu_from_r）：
        v_ref = percentile(|v_A0|, 75%)            （去掉首尾各 3 s 瞬态）
        μ     = r · B0 / v_ref^(n−1)
  这样 r=1 表示“在 v_ref 处剪切阻尼与线性阻尼等强”，不同 n 之间才可比。
"""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

import analysis
import config
import disturbance
import record
import sfc
from controller import SFCController
from sim_env import SimEnv, build_nominal_joint_trajectory

PACKET_DIR = config.FD_PACKET_DIR
SWEEP_DIR = config.OUTPUT_ROOT / "sweep"
TRACE_DIR = SWEEP_DIR / "traces"
COMPARE_DIR = config.OUTPUT_ROOT / "compare"
BIAS_ROOT = config.OUTPUT_ROOT / "bias_packets"


# ============================================================ 慢偏移参考序列
def load_reference(packet_dir: Path = PACKET_DIR) -> dict[str, np.ndarray]:
    """读回本轮的慢偏移分解（e_raw / b / e_vib），作为时序里的参考列。

    读的是【模板网格】版本 `ref_*.csv`（62.85 s，与 Fd 同长度），由
    disturbance.reconstruct_fd 写出。V1.3 只有 e_vib 一个口径，因此这里
    读不到就直接报错 —— 不再有「回退到原始 e_raw」这种会让两套口径混用的分支。
    """
    out: dict[str, np.ndarray] = {}
    for col in record.REFERENCE_COLS:
        q = Path(packet_dir) / f"ref_{col}.csv"
        if not q.is_file():
            raise FileNotFoundError(
                f"缺参考序列 {q}。V1.3 的正式口径只有 e_vib 一条路，"
                f"请先运行 `--reconstruct` 重建冻结包。")
        out[col] = np.genfromtxt(str(q), delimiter=",", comments="#")[:, 1].astype(float)
    return out


def load_phase_codes(packet_dir: Path = PACKET_DIR) -> np.ndarray:
    """从 fd_meta.json 的模板分段还原 5 个运动阶段的编码（时序里的 phase 列）。"""
    mp = Path(packet_dir) / config.FD_META_FILENAME
    if not mp.is_file():
        raise FileNotFoundError(f"缺少 {mp}，无法还原 phase 列")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    segs = meta.get("template", {}).get("segments")
    if not segs:
        raise KeyError(f"{mp} 里没有 template.segments，无法还原 phase 列")
    return record.phase_codes_from_segments(
        [(str(s[0]), int(s[1]), int(s[2])) for s in segs])


# ============================================================ 公共输入
def get_q_traj(along_mm: np.ndarray) -> np.ndarray:
    """名义关节轨迹（IK），缓存到 outputs 以免每组重复求解。"""
    cache = config.OUTPUT_ROOT / "q_traj.npz"
    if cache.is_file():
        z = np.load(str(cache))
        if z["along"].shape == along_mm.shape and np.allclose(z["along"], along_mm):
            return z["q"]
    q = build_nominal_joint_trajectory(along_mm, config.Y_NOMINAL, config.Z_HOLD)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache), along=along_mm, q=q)
    return q


def load_schedule(packet_dir: Path) -> np.ndarray:
    p = Path(packet_dir) / "schedule.csv"
    if not p.is_file():
        raise FileNotFoundError(f"缺少名义运动 schedule：{p}（请先运行 --reconstruct）")
    return np.genfromtxt(str(p), delimiter=",", comments="#")[:, 1].astype(float)


# ============================================================ 单组运行
def run_case(case_id: str, mode: str, K: float, B0: float, n: float, mu: float,
             r_design: float, v_ref: float, fd, along: np.ndarray,
             q_traj: np.ndarray, phase_code: np.ndarray | None = None,
             fd_scale: float = 1.0, bias_um: float = 0.0,
             keep_trace: bool = False, trace_name: str | None = None,
             ref: dict[str, np.ndarray] | None = None,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    """跑一组并返回统计行。异常/发散不会抛出，而是以 status 标记返回。

    fd_scale  : 测试A 用。施加的力按 Fd_scale = fd_scale · Fd(t)，其余完全相同。
    bias_um   : 测试B 用。只用于记录该组用的是哪个 Δb 重建出来的 Fd，不参与计算。
    ref       : 慢偏移参考序列（e_raw / b / e_vib，长度与 along 相同）。它只随模板
                时间轴变化，与 mode / K / B0 / n 无关；落进每组时序是为了让任一组都能
                直接做「原始 vs 去慢偏移 vs 仿真残差」的同轴对照。
    """
    dt = config.GRID_DT_S
    N = fd.n_samples

    row: dict[str, Any] = {
        "case_id": case_id, "group": "", "mode": mode, "status": "OK", "note": "",
        "K_N_per_m": float(K), "B0_Ns_per_m": float(B0), "n": float(n),
        "mu": float(mu), "r_design": float(r_design), "v_ref_m_s": float(v_ref),
        "fd_scale": float(fd_scale), "bias_um": float(bias_um),
        "clip_applied": False,
        "fd_sha256_12": fd.sha256[:12],
    }

    if ref is None:
        ref = {}
    ref_cols: dict[str, np.ndarray] = {}
    for c in record.REFERENCE_COLS:
        src = ref.get(c)
        ref_cols[c] = (np.zeros(N) if src is None
                       else np.asarray(src, dtype=float)[:N])

    buf = {c: np.zeros(N) for c in
           ("t", "phase", "along_mm", "y_d", "y_sim", "e_um", "v_m_s", "Fd_N",
            "F_spring_N", "F_linear_damping_N", "F_shear_N", "F_SFC_N")}
    buf.update(ref_cols)
    if phase_code is not None:
        pc = np.asarray(phase_code, dtype=float)
        if pc.size != N:
            raise ValueError(f"phase_code 长度 {pc.size} 与 Fd {N} 不一致")
        buf["phase"] = pc
    tau_buf = np.zeros((N, 6))
    t_all = np.arange(N, dtype=float) * dt

    done = 0
    try:
        env = SimEnv()
        env.reset(q_traj[0])
        ctrl = SFCController(env, {"K": K, "B0": B0, "mu": mu, "n": n}, mode)

        for i in range(N):
            y_d = config.Y_NOMINAL
            y = env.tcp_y()

            out = ctrl.step(y_d)                      # 控制支路：不接触 Fd
            f = fd.force(t_all[i]) * float(fd_scale)  # 扰动支路：纯查表 × 常数
            if not np.isfinite(f):
                row["status"], row["note"] = "FAILED", f"t={t_all[i]:.3f}s 处 Fd 非有限"
                break

            buf["t"][i] = t_all[i]
            buf["along_mm"][i] = along[i]
            buf["y_d"][i] = y_d
            buf["y_sim"][i] = y
            buf["e_um"][i] = out["e"] * 1e6
            buf["v_m_s"][i] = out["v"]
            buf["Fd_N"][i] = f
            buf["F_spring_N"][i] = out["F_K"]
            buf["F_linear_damping_N"][i] = out["F_B"]
            buf["F_shear_N"][i] = out["F_shear"]
            buf["F_SFC_N"][i] = out["F_sfc"]
            tau_buf[i] = np.asarray(out["tau"], dtype=float).ravel()[:6]

            env.apply_disturbance_force_y(f)
            env.set_ctrl(q_traj[i])
            env.step()
            done = i + 1

            qv = env.data.qvel
            if not np.isfinite(qv).all() or not np.isfinite(env.data.qpos).all():
                row["status"], row["note"] = "INVALID", f"t={t_all[i]:.3f}s 出现 NaN/Inf"
                break
            qvmax = float(np.max(np.abs(qv)))
            if qvmax > config.ABORT_QVEL_MAX:
                row["status"] = "INVALID"
                row["note"] = (f"t={t_all[i]:.3f}s 关节速度 {qvmax:.1f} rad/s "
                               f"> {config.ABORT_QVEL_MAX}，判定发散")
                break
            if abs(y - config.Y_NOMINAL) > config.ABORT_E_ABS_MAX:
                row["status"] = "INVALID"
                row["note"] = (f"t={t_all[i]:.3f}s 末端偏差 "
                               f"{abs(y - config.Y_NOMINAL) * 1e6:.0f} µm 超限，判定跑飞")
                break
    except Exception as exc:                                            # noqa: BLE001
        row["status"], row["note"] = "FAILED", f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
        log(f"    ! {case_id} 异常：{exc}")

    row["n_steps_done"] = done
    row["n_steps_total"] = N
    if done < 2:
        row["status"] = row["status"] if row["status"] != "OK" else "FAILED"
        row["note"] = row["note"] or "有效步数不足"
        for c in ("e_rms_um", "e_peak_abs_um", "e_ptp_um", "e_slow_rms_um",
                  "e_vib_rms_um", "F_shear_rms_N", "F_linear_rms_N",
                  "F_shear_over_linear", "F_SFC_peak_N", "tau_max_norm_Nm",
                  "frac_absF_gt_10N", "P_shear_pos_frac", "P_linear_pos_frac",
                  "E_shear_J", "E_linear_J", "psd_peak_hz", "psd_peak_amp_um2_per_hz"):
            row[c] = float("nan")
        return row

    # ---------------- 统计（只用有效前缀） ----------------
    sl = slice(0, done)
    e_um = buf["e_um"][sl]
    v = buf["v_m_s"][sl]
    fshr = buf["F_shear_N"][sl]
    flin = buf["F_linear_damping_N"][sl]
    fsfc = buf["F_SFC_N"][sl]
    tau = tau_buf[sl]

    # 统计口径与落盘口径对齐：metrics 一律在【抽稀后】的序列上算，
    # 采样率从抽稀时间戳反推（TRACE_STRIDE 抽稀后是 200 Hz，不是物理 1000 Hz）。
    # 这样 sweep_summary.csv 与 compare/summary.json 的数字逐位一致。
    stride = max(1, config.TRACE_STRIDE)
    t_dec = buf["t"][:done:stride]
    e_dec = e_um[::stride]
    fs = analysis.fs_from_time(t_dec)
    met = analysis.metrics(e_dec, fs)
    row.update(met)
    row["metrics_fs_hz"] = float(fs)
    row["metrics_n_samples"] = int(e_dec.size)

    row["F_shear_rms_N"] = float(np.sqrt(np.mean(fshr ** 2)))
    row["F_linear_rms_N"] = float(np.sqrt(np.mean(flin ** 2)))
    row["F_shear_over_linear"] = float(
        row["F_shear_rms_N"] / row["F_linear_rms_N"]) if row["F_linear_rms_N"] > 1e-15 \
        else float("nan")
    row["F_SFC_peak_N"] = float(np.max(np.abs(fsfc)))
    row["tau_max_norm_Nm"] = float(np.max(np.linalg.norm(tau, axis=1)))
    row["frac_absF_gt_10N"] = float(np.mean(np.abs(fsfc) > config.F_DIAG_REF_N))

    # ---------------- 耗散正确性 ----------------
    P_lin = flin * v
    P_shr = fshr * v
    row["P_linear_pos_frac"] = float(np.mean(P_lin > config.P_TOL_W))
    row["P_shear_pos_frac"] = float(np.mean(P_shr > config.P_TOL_W))
    row["E_linear_J"] = float(np.sum(P_lin) * dt)
    row["E_shear_J"] = float(np.sum(P_shr) * dt)

    # ---------------- 频域（同一口径：抽稀后序列 + 反推采样率） ----------------
    f_ax, psd = analysis.welch_psd(e_dec, fs)
    k = int(np.argmax(psd[1:])) + 1 if psd.size > 1 else 0
    row["psd_peak_hz"] = float(f_ax[k])
    row["psd_peak_amp_um2_per_hz"] = float(psd[k])

    # 实际 r（由实际 μ、B0 反推，用于检查标定是否自洽）
    row["r_actual"] = float(mu * (v_ref ** (n - 1.0)) / B0) if v_ref > 0 and B0 > 0 \
        else float("nan")

    if keep_trace:
        # τ 单独存在 (N,6) 缓冲里，落盘前并回列字典
        cols_out = dict(buf)
        for _j, _c in enumerate(record.TAU_COLS):
            cols_out[_c] = tau_buf[:, _j]
        p = record.write_trace(
            TRACE_DIR / f"{trace_name or case_id}.csv", cols_out, {
                "case_id": case_id, "mode": mode,
                "K_N_per_m": float(K), "B0_Ns_per_m": float(B0),
                "n": float(n), "mu": float(mu), "r_design": float(r_design),
                "r_actual": row["r_actual"], "v_ref_m_s": float(v_ref),
                "fd_scale": float(fd_scale), "bias_um": float(bias_um),
                "fd_sha256": fd.sha256, "status": row["status"],
                "note": row["note"], "n_steps_done": done,
            })
        row["trace_file"] = str(p.relative_to(SWEEP_DIR)).replace("\\", "/")

    return row


# ============================================================ v_ref 标定
def compute_v_ref(a0_cols: dict[str, np.ndarray]) -> dict[str, Any]:
    """从 A0 的速度分布定出 v_ref：去掉首尾瞬态后取 |v| 的 75 百分位。"""
    t, v = a0_cols["t"], a0_cols["v_m_s"]
    lo, hi = config.V_REF_TRIM_S, float(t[-1]) - config.V_REF_TRIM_S
    m = (t >= lo) & (t <= hi)
    if m.sum() < 100:
        m = np.ones_like(t, dtype=bool)
    av = np.abs(v[m])
    p75 = float(np.percentile(av, config.V_REF_PERCENTILE))
    rms = float(np.sqrt(np.mean(av ** 2)))
    fallback = None
    if p75 < config.V_REF_MIN_M_S:
        fallback = (f"p75(|v|)={p75:.3e} m/s 低于 V_REF_MIN_M_S="
                    f"{config.V_REF_MIN_M_S:g}，异常接近 0，改用有效运动区间 RMS 速度")
        p75 = rms
    return {"v_ref_m_s": p75, "p75_m_s": p75, "rms_m_s": rms,
            "peak_m_s": float(av.max()), "trim_s": config.V_REF_TRIM_S,
            "percentile": config.V_REF_PERCENTILE, "n_used": int(m.sum()),
            "fallback_reason": fallback}


# ============================================================ 主流程
def run_batch(log: Callable[[str], None] = print) -> dict[str, Any]:
    for d in (SWEEP_DIR, TRACE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    rows: list[dict[str, Any]] = []

    def note_err(msg: str) -> None:
        errors.append(msg)
        log("    ! " + msg)

    # ---------------- 0. 载入冻结扰动与公共输入 ----------------
    fd = disturbance.load_frozen(PACKET_DIR)
    along = load_schedule(PACKET_DIR)
    if along.size != fd.n_samples:
        raise ValueError(f"schedule 与 Fd 长度不一致：{along.size} vs {fd.n_samples}")
    q_traj = get_q_traj(along)
    phase_code = load_phase_codes(PACKET_DIR)
    ref = load_reference(PACKET_DIR)
    if ref["e_vib_um"].size != fd.n_samples:
        raise ValueError(f"参考序列长度 {ref['e_vib_um'].size} 与 Fd {fd.n_samples} 不一致")
    log(f"[批处理] {fd.describe()}")
    log(f"[批处理] 名义轨迹 {len(along)} 点，远点 {along.max():.1f} mm")
    _u, _c = np.unique(phase_code, return_counts=True)
    log("[批处理] phase 列已还原：" + "  ".join(
        f"{config.DRIFT_PHASE_NAMES[int(k)]}={int(v)}" for k, v in zip(_u, _c))
        + f"（编码 {config.DRIFT_PHASE_NAMES}）")
    log(f"[批处理] 参考列 e_raw/b/e_vib 各 {ref['e_vib_um'].size} 点"
        f"（e_vib rms={np.sqrt(np.mean(ref['e_vib_um'] ** 2)):.3f} µm）")

    K0, B0_ = config.SFC_K, config.SFC_B0

    def run_and_store(case_id, group, mode, K, B0, n, mu, r_d, v_ref,
                      keep_trace=False, fd_obj=None, along_arr=None,
                      phase_arr=None, fd_scale=1.0, bias_um=0.0,
                      ref_obj=None, q_obj=None) -> dict:
        log(f"  → {case_id:28s} mode={mode:16s} K={K:8.1f} B0={B0:7.1f} "
            f"n={n:4.2f} μ={mu:.4e} r={r_d:5.2f}"
            + (f"  [s={fd_scale:g}]" if fd_scale != 1.0 else "")
            + (f"  [Δb={bias_um:+.0f} µm]" if bias_um else ""))
        r = run_case(case_id, mode, K, B0, n, mu, r_d, v_ref,
                     fd_obj if fd_obj is not None else fd,
                     along_arr if along_arr is not None else along,
                     q_obj if q_obj is not None else q_traj,
                     phase_code=phase_arr if phase_arr is not None else phase_code,
                     fd_scale=fd_scale, bias_um=bias_um, keep_trace=keep_trace,
                     ref=ref_obj if ref_obj is not None else ref, log=log)
        r["group"] = group
        rows.append(r)
        if r["status"] != "OK":
            note_err(f"{case_id}: {r['status']} —— {r['note']}")
        else:
            log(f"      e_rms={r['e_rms_um']:.3f} µm  "
                f"峰值={r['e_peak_abs_um']:.3f} µm  "
                f"F_shr/F_lin={r['F_shear_over_linear']:.4f}")
        return r

    # ---------------- 1. A0（基准，只跑一次） ----------------
    log("\n[1] A0（无控制基准，只跑一次）")
    a0 = run_and_store(config.BASELINE_ID, "baseline", "A0", K0, B0_,
                       config.SFC_N_DEFAULT, 0.0, 0.0, float("nan"),
                       keep_trace=True)

    # ---------------- 2. v_ref 标定 ----------------
    log("\n[2] 由 A0 速度分布定 v_ref")
    _, a0_cols = record.load_trace(TRACE_DIR / f"{config.BASELINE_ID}.csv")
    vref = compute_v_ref(a0_cols)
    log(f"  v_ref = {vref['v_ref_m_s']:.4e} m/s（|v| 的 {vref['percentile']:.0f} 百分位，"
        f"已去首尾各 {vref['trim_s']:.0f} s；rms={vref['rms_m_s']:.4e}，"
        f"峰值={vref['peak_m_s']:.4e}）")
    if vref["fallback_reason"]:
        note_err("v_ref 回退：" + vref["fallback_reason"])
    v_ref = vref["v_ref_m_s"]

    # ---------------- 3. Alinear（只跑一次） ----------------
    log("\n[3] Alinear（线性阻抗基准，只跑一次）")
    alin = run_and_store(config.TEST_LINEAR_ID, "baseline", "Alinear",
                         K0, B0_, 1.0, 0.0, 0.0, v_ref, keep_trace=True)

    # ---------------- 4. 3 个 ASFC 正式组 ----------------
    log("\n[4] ASFC 正式组（参数由 config.ROUND_GROUPS 指定，不做任何寻优）")
    asfc_rows: dict[str, dict] = {}
    for cid, n_i, r_i in config.ROUND_GROUPS:
        mu_i = sfc.mu_from_r(r_i, B0_, n_i, v_ref)
        asfc_rows[cid] = run_and_store(cid, "main", "ASFC", K0, B0_, n_i, mu_i,
                                       r_i, v_ref, keep_trace=True)

    # ---------------- 5. 5 个正式组的对比与主汇总 ----------------
    log("\n[5] 正式组对比（纯数值，无图）")
    main_ids = (config.BASELINE_ID, config.TEST_LINEAR_ID, *asfc_rows.keys())
    cmp_sum = analysis.compare({c: TRACE_DIR / f"{c}.csv" for c in main_ids},
                              COMPARE_DIR)
    analysis.print_report(cmp_sum)
    _write_summary_main(cmp_sum)
    log(f"  已写出 {config.SUMMARY_MAIN}")

    strong = asfc_rows.get(config.TEST_ASFC_ID, {})

    # ---------------- 6. 测试A：扰动幅值缩放 ----------------
    log("\n[6] 测试A：Fd_test = s·Fd，只跑 Alinear 与 ASFC_strong")
    scale_rows: list[dict] = []
    n_s, r_s = 0.0, 0.0
    for cid, n_i, r_i in config.ROUND_GROUPS:
        if cid == config.TEST_ASFC_ID:
            n_s, r_s = n_i, r_i
    mu_s = sfc.mu_from_r(r_s, B0_, n_s, v_ref) if r_s else float("nan")
    for s in config.SCALE_FACTORS:
        if abs(s - 1.0) < 1e-12:
            # s=1.0 就是本轮正式组：直接复用，不重复运行
            log(f"  = s=1.00 复用正式组 {config.TEST_LINEAR_ID} / {config.TEST_ASFC_ID}")
            if alin.get("status") == "OK":
                scale_rows.append(dict(alin, scale=1.0))
            if strong.get("status") == "OK":
                scale_rows.append(dict(strong, scale=1.0))
            continue
        tag = f"s{s:.2f}"
        for cid, mode, K_, B_, n_, mu_, r_ in (
                (f"{tag}_{config.TEST_LINEAR_ID}", "Alinear", K0, B0_, 1.0, 0.0, 0.0),
                (f"{tag}_{config.TEST_ASFC_ID}", "ASFC", K0, B0_, n_s, mu_s, r_s)):
            r = run_and_store(cid, "testA_scale", mode, K_, B_, n_, mu_, r_,
                              v_ref, keep_trace=True, fd_scale=s)
            scale_rows.append(dict(r, scale=s))
    _write_summary_scale(scale_rows, a0_rms=a0["e_rms_um"], log=log)
    log(f"  已写出 {config.SUMMARY_SCALE}")

    # ---------------- 7. 测试B：慢偏移估计偏差 ----------------
    log("\n[7] 测试B：b̂ = b + Δb，重新重建 Fd 并只跑 ASFC_strong")
    bias_rows: list[dict] = []
    if strong.get("status") == "OK":
        try:
            plant = disturbance.load_plant(PACKET_DIR / disturbance.PLANT_FILENAME)
            if plant is None:
                note_err("冻结包内没有 plant_frf.npz，测试B 将重新标定 Plant（更慢）")
                plant = disturbance.measure_plant(log=log)
            prepared = disturbance.prepare(log=log)
            for db in config.BIAS_UM:
                tag = f"bias{'+' if db > 0 else '-'}{abs(db):.0f}"
                pdir = BIAS_ROOT / tag
                log(f"\n  --- Δb = {db:+.1f} µm → 重建到 {pdir.relative_to(config.OUTPUT_ROOT)} ---")
                disturbance.reconstruct_fd(pdir, bias_um=db, prepared=prepared,
                                           plant=plant, q_traj=q_traj,
                                           write_references=False, log=log)
                fdb = disturbance.load_frozen(pdir)
                r = run_and_store(f"{config.TEST_ASFC_ID}_{tag}", "testB_bias",
                                  "ASFC", K0, B0_, n_s, mu_s, r_s, v_ref,
                                  keep_trace=True, fd_obj=fdb,
                                  along_arr=load_schedule(pdir),
                                  phase_arr=load_phase_codes(pdir),
                                  bias_um=db)
                bias_rows.append(r)
        except Exception as exc:                                        # noqa: BLE001
            note_err(f"测试B 执行失败：{type(exc).__name__}: {exc}")
            log(traceback.format_exc())
    else:
        note_err(f"{config.TEST_ASFC_ID} 未成功，测试B 跳过")
    _write_summary_bias(bias_rows, strong, log=log)
    log(f"  已写出 {config.SUMMARY_BIAS}")

    # ---------------- 8. 测试C：事件级 振出去 / 回中心 ----------------
    log("\n[8] 测试C：逐事件比较「峰值 / 上升速度 / 恢复时间 / 反向过冲」（纯离线）")
    ev = None
    try:
        traces_c = {c: TRACE_DIR / f"{c}.csv" for c in
                    (config.BASELINE_ID, config.TEST_LINEAR_ID, config.TEST_ASFC_ID)}
        ev = analysis.event_response(traces_c, SWEEP_DIR)
        p = ev["paired_ASFC_vs_Alinear"]
        log(f"  事件数 {ev['n_events_used']}（由 {ev['event_source']} 定位），"
            f"配对 {p['n_paired_events']} 个")
        log(f"  峰值：{p['verdict_peak']}"
            f"（更低 {p['peak_lower_count']} / 持平 {p['peak_equal_count']} / "
            f"更高 {p['peak_higher_count']}，中位变化 {p['peak_median_change_um']:+.3f} µm）")
        log(f"  回正：{p['verdict_recover']}"
            f"（更慢 {p['recover_longer_count']} / 更快 {p['recover_shorter_count']}，"
            f"中位变化 {p['recover_median_change_s']:+.4f} s）")
        log(f"  过冲：{p['verdict_overshoot']}"
            f"（更小 {p['overshoot_lower_count']} / 更大 {p['overshoot_higher_count']}，"
            f"中位变化 {p['overshoot_median_change_um']:+.3f} µm）")
        log(f"  已写出 {config.SUMMARY_EVENT}")
    except Exception as exc:                                            # noqa: BLE001
        note_err(f"测试C 失败：{type(exc).__name__}: {exc}")
        log(traceback.format_exc())

    # ---------------- 9. 测试D：分相统计 ----------------
    log("\n[9] 测试D：分相统计（稳态段 vs 突变段）")
    ph = None
    try:
        ph = analysis.phasewise({c: TRACE_DIR / f"{c}.csv" for c in main_ids},
                                SWEEP_DIR)
        v = ph["verdict"]
        for name, d in ph["improvement"].items():
            log(f"  {name:12s} ASFC 相对 Alinear RMS 改善 "
                f"{d['asfc_vs_linear_rms_pct']:+7.2f}%   "
                f"峰值改善 {d['asfc_vs_linear_peak_pct']:+7.2f}%   "
                f"（线性相对 A0 {d.get('linear_vs_A0_rms_pct', float('nan')):+6.2f}%）")
        log(f"  突变段平均改善 {v['mean_asfc_vs_linear_rms_pct_transient']:+.2f}%  vs  "
            f"稳态段 {v['mean_asfc_vs_linear_rms_pct_steady']:+.2f}%")
        log(f"  结论：{v['conclusion']}")
        log(f"  已写出 {config.SUMMARY_PHASE}")
    except Exception as exc:                                            # noqa: BLE001
        note_err(f"测试D 失败：{type(exc).__name__}: {exc}")
        log(traceback.format_exc())

    # ---------------- 10. 汇总落盘 ----------------
    log("\n[10] 汇总落盘")
    _finalize(rows, a0, alin, strong, vref, fd, errors,
              scale_rows, bias_rows, ev, ph)

    return {"rows": rows, "a0": a0, "alinear": alin, "strong": strong,
            "asfc_rows": asfc_rows, "v_ref": vref, "v_ref_value": v_ref,
            "scale": scale_rows, "bias": bias_rows,
            "event_response": ev, "phasewise": ph, "errors": errors}


# ============================================================ 五张汇总表
def _write_csv_rows(path: Path, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            out = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, bool):
                    out.append(str(v))
                elif isinstance(v, (int, np.integer)):
                    out.append(str(int(v)))
                elif isinstance(v, float):
                    out.append("nan" if math.isnan(v) else f"{v:.6g}")
                else:
                    out.append(str(v))
            fh.write(",".join(out) + "\n")


def _fmt(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _write_summary_main(cmp_sum: dict[str, Any]) -> None:
    """正式 5 组主汇总。正向/反向单程 RMS 直接取分相统计的稳态段。"""
    mets = cmp_sum["metrics"]
    phs = cmp_sum["phase_metrics"]
    params = cmp_sum["run_params"]
    ids = cmp_sum["modes"]
    a0_r = mets[config.BASELINE_ID]["e_rms_um"]
    al_r = mets[config.TEST_LINEAR_ID]["e_rms_um"]
    a0_v = mets[config.BASELINE_ID]["e_vib_rms_um"]
    al_v = mets[config.TEST_LINEAR_ID]["e_vib_rms_um"]
    rows = []
    for c in ids:
        d = mets[c]
        p = params.get(c, {})
        ph = phs.get(c, {})
        rows.append({
            "case_id": c, "mode": p.get("mode") or ("A0" if c == "A0" else ""),
            "n": p.get("n"), "r_design": p.get("r_design"), "mu": p.get("mu"),
            "v_ref_m_s": p.get("v_ref_m_s"),
            "total_rms_um": d["e_rms_um"], "peak_abs_um": d["e_peak_abs_um"],
            "ptp_um": d["e_ptp_um"],
            "slow_rms_um": d["e_slow_rms_um"], "vib_rms_um": d["e_vib_rms_um"],
            "fwd_single_stroke_rms_um": ph.get("fwd_steady", {}).get("e_rms_um"),
            "ret_single_stroke_rms_um": ph.get("ret_steady", {}).get("e_rms_um"),
            "start_peak_um": ph.get("start", {}).get("e_peak_abs_um"),
            "turn_ptp_um": ph.get("turn", {}).get("e_ptp_um"),
            "stop_peak_um": ph.get("stop", {}).get("e_peak_abs_um"),
            "suppress_vs_A0_pct": _pct(a0_r, d["e_rms_um"]),
            "extra_vs_Alinear_pct": _pct(al_r, d["e_rms_um"]),
            "suppress_vib_vs_A0_pct": _pct(a0_v, d["e_vib_rms_um"]),
            "extra_vib_vs_Alinear_pct": _pct(al_v, d["e_vib_rms_um"]),
        })
    cols = ["case_id", "mode", "n", "r_design", "mu", "v_ref_m_s",
            "total_rms_um", "peak_abs_um", "ptp_um", "slow_rms_um", "vib_rms_um",
            "fwd_single_stroke_rms_um", "ret_single_stroke_rms_um",
            "start_peak_um", "turn_ptp_um", "stop_peak_um",
            "suppress_vs_A0_pct", "extra_vs_Alinear_pct",
            "suppress_vib_vs_A0_pct", "extra_vib_vs_Alinear_pct"]
    _write_csv_rows(SWEEP_DIR / config.SUMMARY_MAIN, cols, rows)


def _write_summary_scale(scale_rows: list[dict], a0_rms: float, log=print) -> None:
    """测试A 汇总：每一档的绝对量 + ASFC 相对【该档线性组】的额外改善。

    「相对 A0」一列只能以 s=1.0 的那次 A0 为分母 —— 每档按要求只跑了
    Alinear 与 ASFC_strong 两组，没有各档自己的 A0。列名与 note 都写明这一点，
    避免被误读成「该档的 A0」。
    """
    by_s: dict[float, dict[str, dict]] = {}
    for r in scale_rows:
        by_s.setdefault(round(float(r.get("scale", 1.0)), 6), {})[r["mode"]] = r
    rows = []
    for s in sorted(by_s):
        grp = by_s[s]
        lin = grp.get("Alinear")
        stg = grp.get("ASFC")
        for mode, r in grp.items():
            if r.get("status") != "OK":
                continue
            rows.append({
                "scale": s, "case_id": r["case_id"], "mode": mode,
                "e_rms_um": r["e_rms_um"], "peak_abs_um": r["e_peak_abs_um"],
                "ptp_um": r["e_ptp_um"], "vib_rms_um": r["e_vib_rms_um"],
                "asfc_vs_this_level_linear_rms_pct":
                    (_pct(lin["e_rms_um"], r["e_rms_um"])
                     if (mode == "ASFC" and lin) else float("nan")),
                "asfc_vs_this_level_linear_peak_pct":
                    (_pct(lin["e_peak_abs_um"], r["e_peak_abs_um"])
                     if (mode == "ASFC" and lin) else float("nan")),
                "vs_A0_at_s1_rms_pct": float("nan"),   # 下一轮填
                "note": "",
            })
    for r in rows:
        if np.isfinite(a0_rms):
            r["vs_A0_at_s1_rms_pct"] = _pct(a0_rms, r["e_rms_um"])
            r["note"] = "相对 A0 一列以 s=1.0 的 A0 为分母（每档只跑了 Alinear/ASFC）"
    cols = ["scale", "case_id", "mode", "e_rms_um", "peak_abs_um", "ptp_um",
            "vib_rms_um", "asfc_vs_this_level_linear_rms_pct",
            "asfc_vs_this_level_linear_peak_pct", "vs_A0_at_s1_rms_pct", "note"]
    _write_csv_rows(SWEEP_DIR / config.SUMMARY_SCALE, cols, rows)
    # 结论：ASFC 的额外改善是否随扰动幅值增大
    xs, ys = [], []
    for s in sorted(by_s):
        grp = by_s[s]
        if grp.get("Alinear", {}).get("status") == "OK" and \
                grp.get("ASFC", {}).get("status") == "OK":
            xs.append(s)
            ys.append(_pct(grp["Alinear"]["e_rms_um"], grp["ASFC"]["e_rms_um"]))
    if len(xs) >= 2:
        log(f"  额外改善随幅值变化："
            + "  ".join(f"s={x:g} → {y:+.2f}%" for x, y in zip(xs, ys)))
        log(f"  {'额外改善随扰动幅值增大而增大' if ys[-1] > ys[0] else '额外改善未随扰动幅值增大（甚至变小）'}")


def _write_summary_bias(bias_rows: list[dict], strong: dict, log=print) -> None:
    """测试B 汇总：Δb ≠ 0 时 ASFC_strong 相对【零偏差的 ASFC_strong】退化多少。"""
    base_r = _fmt(strong.get("e_rms_um", float("nan")))
    base_p = _fmt(strong.get("e_peak_abs_um", float("nan")))
    rows = []
    for r in bias_rows:
        rr, pp = _fmt(r.get("e_rms_um")), _fmt(r.get("e_peak_abs_um"))
        rms_chg = (rr - base_r) / base_r * 100.0 if base_r > 0 else float("nan")
        pk_chg = (pp - base_p) / base_p * 100.0 if base_p > 0 else float("nan")
        rows.append({
            "bias_um": r.get("bias_um"), "case_id": r["case_id"],
            "status": r.get("status"),
            "e_rms_um": rr, "peak_abs_um": pp, "ptp_um": r.get("e_ptp_um"),
            "vib_rms_um": r.get("e_vib_rms_um"),
            "rms_change_vs_nobias_pct": rms_chg,
            "peak_change_vs_nobias_pct": pk_chg,
            "note": "正数 = 比零偏差更差",
        })
    if not rows:
        rows.append({"bias_um": float("nan"), "case_id": "未执行", "status": "SKIP",
                     "note": "ASFC_strong 未成功或无 Δb 设定"})
    cols = ["bias_um", "case_id", "status", "e_rms_um", "peak_abs_um", "ptp_um",
            "vib_rms_um", "rms_change_vs_nobias_pct", "peak_change_vs_nobias_pct",
            "note"]
    _write_csv_rows(SWEEP_DIR / config.SUMMARY_BIAS, cols, rows)
    if base_r == base_r:                                                # 非 NaN
        log(f"  零偏差 ASFC_strong：rms {base_r:.3f} µm，峰值 {base_p:.3f} µm")
        for r in rows:
            if r["status"] == "OK":
                log(f"  Δb={r['bias_um']:+5.1f} µm → rms {r['e_rms_um']:.3f} µm "
                    f"({r['rms_change_vs_nobias_pct']:+.2f}%)，"
                    f"峰值 {r['peak_abs_um']:.3f} µm "
                    f"({r['peak_change_vs_nobias_pct']:+.2f}%)")


# ============================================================ 收尾落盘
def _finalize(rows, a0, alin, strong, vref, fd, errors,
              scale_rows, bias_rows, ev, ph) -> None:
    a0_r, al_r = a0["e_rms_um"], alin["e_rms_um"]
    a0_v, al_v = a0["e_vib_rms_um"], alin["e_vib_rms_um"]
    for r in rows:
        er = r.get("e_rms_um", float("nan"))
        ev_ = r.get("e_vib_rms_um", float("nan"))
        r["suppress_vs_A0_pct"] = _pct(a0_r, er)
        r["extra_vs_Alinear_pct"] = _pct(al_r, er)
        r["suppress_vib_vs_A0_pct"] = _pct(a0_v, ev_)
        r["extra_vib_vs_Alinear_pct"] = _pct(al_v, ev_)
        r.setdefault("trace_file", "")

    cols = ["case_id", "group", "mode", "status", "note",
            "fd_scale", "bias_um",
            "K_N_per_m", "B0_Ns_per_m", "n", "mu", "r_design", "r_actual",
            "v_ref_m_s",
            "e_rms_um", "e_peak_abs_um", "e_ptp_um", "e_slow_rms_um", "e_vib_rms_um",
            "suppress_vs_A0_pct", "extra_vs_Alinear_pct",
            "suppress_vib_vs_A0_pct", "extra_vib_vs_Alinear_pct",
            "F_shear_rms_N", "F_linear_rms_N", "F_shear_over_linear",
            "F_SFC_peak_N", "tau_max_norm_Nm",
            "clip_applied", "frac_absF_gt_10N",
            "P_shear_pos_frac", "P_linear_pos_frac", "E_shear_J", "E_linear_J",
            "psd_peak_hz", "psd_peak_amp_um2_per_hz",
            "n_steps_done", "n_steps_total", "fd_sha256_12", "trace_file"]
    _write_csv_rows(SWEEP_DIR / config.SWEEP_CSV, cols, rows)

    record.save_json(SWEEP_DIR / "baseline_summary.json", {
        "v_ref": vref,
        "A0": a0, "Alinear": alin, "ASFC_strong": strong,
        "suppression": {
            "Alinear_vs_A0_pct": _pct(a0_r, al_r),
            "ASFC_strong_vs_A0_pct": _pct(a0_r, strong.get("e_rms_um", float("nan"))),
            "ASFC_strong_vs_Alinear_pct": _pct(al_r, strong.get("e_rms_um",
                                                              float("nan"))),
        },
    })

    mid = json.loads((PACKET_DIR / config.FD_META_FILENAME).read_text(encoding="utf-8"))
    last = (mid.get("refine", {}).get("history") or [{}])[-1]
    record.save_json(SWEEP_DIR / "disturbance_fit_summary.json", {
        "recon_source": "vib",
        "source_note": "只对去慢偏移后的净振动 e_vib 重建（§六 唯一口径）",
        "drift_baseline_method": mid.get("drift_baseline", {}).get("method"),
        "drift_levels_um": mid.get("drift_baseline", {}).get("levels_um"),
        "drift_acceptance_passed": (mid.get("drift_baseline", {})
                                    .get("acceptance", {}).get("passed")),
        "frozen_file": config.FD_FILENAME,
        "sha256": mid["frozen"]["sha256"],
        "target_e_rms_um": mid["template"]["e_rms_um"],
        "sim_e_rms_um": last.get("e_sim_rms_um"),
        "residual_rms_um": last.get("residual_rms_um"),
        "residual_over_target": last.get("residual_over_target"),
        "corr": last.get("corr"),
        "e_ptp_target_um": mid["template"]["e_ptp_um"],
        "gate": {"corr_min": config.GATE_CORR_MIN,
                 "residual_frac_max": config.GATE_RESIDUAL_FRAC_MAX,
                 "passed": bool(last.get("gate_pass", False))},
        "refine_iters_used": mid.get("refine", {}).get("iters_done"),
        "force_application": mid.get("force_application"),
        "plant_dc_um_per_n": mid["plant"]["dc_um_per_n"],
        "frozen_Fd_rms_N": mid["frozen"]["Fd_rms_N"],
        "frozen_Fd_peak_abs_N": mid["frozen"]["Fd_peak_abs_N"],
    })

    record.save_json(SWEEP_DIR / "run_manifest.json", {
        "scope": {
            "main_groups": list(config.MODES),
            "testA_scale_factors": list(config.SCALE_FACTORS),
            "testA_new_runs": [r["case_id"] for r in scale_rows
                               if abs(float(r.get("scale", 1.0)) - 1.0) > 1e-12],
            "testB_bias_um": list(config.BIAS_UM),
            "testB_new_runs": [r["case_id"] for r in bias_rows],
            "testC": "离线（event_response_detail.csv + summary_event_response.csv）",
            "testD": "离线（summary_phasewise.csv）",
        },
        "n_runs": len(rows),
        "n_ok": sum(1 for r in rows if r["status"] == "OK"),
        "n_invalid": sum(1 for r in rows if r["status"] == "INVALID"),
        "n_failed": sum(1 for r in rows if r["status"] == "FAILED"),
        "fd_sha256": fd.sha256,
        "all_runs_same_fd_sha": len({r["fd_sha256_12"] for r in rows}) == 1,
        "run_ids": [r["case_id"] for r in rows],
    })

    record.save_json(SWEEP_DIR / "config_snapshot.json", config.snapshot())

    record.save_json(SWEEP_DIR / "test_results.json", {
        "testA_disturbance_scale": [
            {"scale": r.get("scale"), "case_id": r.get("case_id"),
             "e_rms_um": r.get("e_rms_um"), "peak_abs_um": r.get("e_peak_abs_um"),
             "ptp_um": r.get("e_ptp_um")} for r in scale_rows],
        "testB_bias_robustness": [
            {"bias_um": r.get("bias_um"), "case_id": r.get("case_id"),
             "e_rms_um": r.get("e_rms_um"), "peak_abs_um": r.get("e_peak_abs_um"),
             "status": r.get("status")} for r in bias_rows],
        "testC_event_response": ev,
        "testD_phasewise_verdict": (ph or {}).get("verdict"),
    })

    with open(SWEEP_DIR / "errors.txt", "w", encoding="utf-8") as fh:
        fh.write("本次批处理的问题记录（成功组不在此列）\n")
        fh.write("=" * 60 + "\n")
        if not errors:
            fh.write("无：全部组与全部测试均正常完成。\n")
        for e in errors:
            fh.write(e + "\n")
        fh.write("\n" + "=" * 60 + "\n")
        fh.write("INVALID / FAILED 组的完整回溯：\n")
        for r in rows:
            if r["status"] != "OK" and r.get("traceback"):
                fh.write(f"\n[{r['case_id']}] {r['status']}\n{r['traceback']}\n")


def _pct(base: float, new: float) -> float:
    """(base − new)/base × 100。正值 = new 更小 = 有抑振。"""
    if not (np.isfinite(base) and np.isfinite(new)) or abs(base) < 1e-15:
        return float("nan")
    return float((base - new) / base * 100.0)
