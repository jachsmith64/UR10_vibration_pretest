"""
批量实验驱动 —— 标准三组 + 第一轮主扫描 + 条件极端诊断 + 第二轮局部扩展。

设计纪律：
  * 架构不变。本模块只调用已有的 扰动支路 / 控制支路 / 执行支路，不新增任何控制概念。
  * A0 与 Alinear 各只跑一次，作为所有 ASFC 组的公共基准（不是每组重复跑）。
  * 所有组共用同一份冻结 Fd（同一 FrozenDisturbance 句柄，逐字节相同，逐组记录 SHA-256）。
  * 不做“自动寻优”，不把任何组写回 config 当作推荐参数；只输出数据与客观排序。
  * 任何一组出现 NaN / 发散 / 跑飞 → 立即中止该组、标记 INVALID、继续下一组，
    绝不让单组失败带崩整批。

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
RUNS_DIR = config.OUTPUT_ROOT / "runs"
SWEEP_DIR = config.OUTPUT_ROOT / "sweep"
TRACE_DIR = SWEEP_DIR / "traces"


# ============================================================ 慢偏移参考序列
def load_reference(packet_dir: Path = PACKET_DIR) -> dict[str, np.ndarray] | None:
    """读回本轮的慢偏移分解（e_raw / b / e_vib），作为时序里的参考列。

    读的是【模板网格】版本 `ref_*.csv`（62.85 s，与 Fd 同长度），由
    disturbance.reconstruct_fd 在 source="vib" 时写出。读不到就返回 None ——
    那说明当前冻结包是上一轮的 raw 口径，此时时序里这三列写 0 并在日志里明确
    说明，绝不静默混用两套口径。
    """
    if not (Path(packet_dir) / "ref_e_vib_um.csv").is_file():
        return None
    out: dict[str, np.ndarray] = {}
    for col in ("e_raw_um", "b_um", "e_vib_um"):
        q = Path(packet_dir) / f"ref_{col}.csv"
        if not q.is_file():
            raise FileNotFoundError(f"缺慢偏移参考序列：{q}")
        out[col] = np.genfromtxt(str(q), delimiter=",", comments="#")[:, 1].astype(float)
    return out


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
             q_traj: np.ndarray, keep_trace: bool = False,
             trace_name: str | None = None, ref: dict[str, np.ndarray] | None = None,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    """跑一组并返回统计行。异常/发散不会抛出，而是以 status 标记返回。

    ref 为慢偏移参考序列（e_raw / b / e_vib，长度与 along 相同）。它只随模板时间轴
    变化，与 mode / K / B0 / n 无关；落进每组时序是为了让任一组都能直接做
    「原始 vs 去慢偏移 vs 仿真残差」的同轴对照。
    """
    dt = config.GRID_DT_S
    N = fd.n_samples

    row: dict[str, Any] = {
        "case_id": case_id, "group": "", "mode": mode, "status": "OK", "note": "",
        "K_N_per_m": float(K), "B0_Ns_per_m": float(B0), "n": float(n),
        "mu": float(mu), "r_design": float(r_design), "v_ref_m_s": float(v_ref),
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
           ("t", "along_mm", "y_d", "y_sim", "e_um", "v_m_s", "Fd_N",
            "F_spring_N", "F_linear_damping_N", "F_shear_N", "F_SFC_N")}
    buf.update(ref_cols)
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
            f = fd.force(t_all[i])                    # 扰动支路：纯查表
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

    # ---------------- 频域（与上面同一口径：抽稀后序列 + 反推采样率） ----------------
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
def run_batch(do_sweep: bool = True,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """do_sweep=False 时只跑标准三组并出对比（不做 n×r 扫描与第二轮）。"""
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    rows: list[dict[str, Any]] = []

    def note_err(msg: str) -> None:
        errors.append(msg)
        log("    ! " + msg)

    # ---------------- 0. 载入冻结扰动 ----------------
    fd = disturbance.load_frozen(PACKET_DIR)
    along = load_schedule(PACKET_DIR)
    if along.size != fd.n_samples:
        raise ValueError(f"schedule 与 Fd 长度不一致：{along.size} vs {fd.n_samples}")
    q_traj = get_q_traj(along)
    log(f"[批处理] {fd.describe()}")
    log(f"[批处理] 名义轨迹 {len(along)} 点，远点 {along.max():.1f} mm")

    ref = load_reference(PACKET_DIR)
    if ref is None:
        note_err("冻结包内没有慢偏移分解（e_raw/b/e_vib）——当前是 raw 口径；"
                 "时序里这三列写 0。本轮正式口径应为 --reconstruct --source vib。")
        log("[批处理] 参考列（e_raw/b/e_vib）：不可用，写 0")
    else:
        n_ref = len(ref["e_vib_um"])
        log(f"[批处理] 参考列已载入：e_raw/b/e_vib 各 {n_ref} 点"
            f"（e_vib rms={np.sqrt(np.mean(ref['e_vib_um'] ** 2)):.3f} µm）")
        if n_ref != fd.n_samples:
            raise ValueError(f"慢偏移序列长度 {n_ref} 与 Fd {fd.n_samples} 不一致")

    K0, B0_ = config.SFC_K, config.SFC_B0

    def run_and_store(case_id, group, mode, K, B0, n, mu, r_d, v_ref,
                      keep_trace=False) -> dict:
        log(f"  → {case_id:28s} mode={mode:7s} K={K:8.1f} B0={B0:7.1f} "
            f"n={n:4.2f} μ={mu:.4e} r={r_d:5.2f}")
        r = run_case(case_id, mode, K, B0, n, mu, r_d, v_ref, fd, along,
                     q_traj, keep_trace=keep_trace, ref=ref, log=log)
        r["group"] = group
        rows.append(r)
        if r["status"] != "OK":
            note_err(f"{case_id}: {r['status']} —— {r['note']}")
        else:
            log(f"      e_rms={r['e_rms_um']:.3f} µm  "
                f"F_shr/F_lin={r['F_shear_over_linear']:.4f}  "
                f"E_shr={r['E_shear_J']:.3e} J")
        return r

    # ---------------- 1. A0（基准，只跑一次） ----------------
    log("\n[1] A0（无控制基准，只跑一次）")
    a0 = run_and_store("A0", "baseline", "A0", K0, B0_, config.SFC_N_DEFAULT, 0.0,
                       0.0, float("nan"), keep_trace=True)

    # ---------------- 2. v_ref 标定 ----------------
    log("\n[2] 由 A0 速度分布定 v_ref")
    _, a0_cols = record.load_trace(TRACE_DIR / "A0.csv")
    vref = compute_v_ref(a0_cols)
    log(f"  v_ref = {vref['v_ref_m_s']:.4e} m/s（|v| 的 {vref['percentile']:.0f} 百分位，"
        f"已去首尾各 {vref['trim_s']:.0f} s；rms={vref['rms_m_s']:.4e}，"
        f"峰值={vref['peak_m_s']:.4e}）")
    if vref["fallback_reason"]:
        note_err("v_ref 回退：" + vref["fallback_reason"])
    v_ref = vref["v_ref_m_s"]

    # ---------------- 3. Alinear（只跑一次） ----------------
    log("\n[3] Alinear（线性阻抗基准，只跑一次）")
    alin = run_and_store("Alinear", "baseline", "Alinear", K0, B0_, 1.0, 0.0,
                         0.0, v_ref, keep_trace=True)

    # ---------------- 4. 默认 ASFC ----------------
    n_d, r_d = config.SFC_N_DEFAULT, config.SFC_R_DEFAULT
    mu_d = sfc.mu_from_r(r_d, B0_, n_d, v_ref)
    log(f"\n[4] 默认 ASFC（n={n_d}, r={r_d} → μ={mu_d:.4e}）")
    asfc = run_and_store(f"ASFC_n{n_d:.1f}_r{r_d:.2f}", "default", "ASFC",
                         K0, B0_, n_d, mu_d, r_d, v_ref, keep_trace=True)

    # ---------------- 5. 标准三组验收 ----------------
    log("\n[5] 标准三组验收")
    std_checks = _standard_checks(a0, alin, asfc, fd)
    for k, v in std_checks.items():
        log(f"  {k}: {v}")

    a0_rms = a0["e_rms_um"]
    alin_rms = alin["e_rms_um"]
    log(f"  A0 rms {a0_rms:.3f} µm  |  Alinear rms {alin_rms:.3f} µm "
        f"（相对 A0 {(a0_rms - alin_rms) / a0_rms * 100:+.2f}%）  |  "
        f"ASFC rms {asfc['e_rms_um']:.3f} µm "
        f"（相对 Alinear {(alin_rms - asfc['e_rms_um']) / alin_rms * 100:+.2f}%）")

    # 标准三组对比图（对照用；图不入交付 ZIP）
    try:
        cmp_sum = analysis.compare(
            {"A0": TRACE_DIR / "A0.csv", "Alinear": TRACE_DIR / "Alinear.csv",
             "ASFC": TRACE_DIR / f"{asfc['case_id']}.csv"},
            config.OUTPUT_ROOT / "compare")
        analysis.print_report(cmp_sum)
    except Exception as exc:                                            # noqa: BLE001
        note_err(f"标准三组对比图生成失败：{exc}")

    if not do_sweep:
        log("\n[仅标准三组] 未执行代表组对照")
        _finalize(rows, a0, alin, asfc, vref, fd, std_checks, [], False,
                  [], errors, None)
        return {"rows": rows, "a0": a0, "alinear": alin, "asfc": asfc,
                "v_ref": vref, "v_ref_value": v_ref, "representatives": [],
                "round": [], "return_local": None, "errors": errors}

    # ---------------- 6. 本轮正式对照：少量代表组（不再做几十组全扫描） ----------------
    log("\n[6] 代表组对照（去慢偏移口径）")
    round_rows: list[dict] = []
    strong: dict[str, Any] | None = None
    for (n_i, r_i) in config.ROUND_ASFC:
        if (n_i, r_i) == (n_d, r_d):
            # 该点已在 [4] 作为默认 ASFC 跑过：复用同一行，不复制出第二条同 case_id。
            asfc["group"] = "default+round"
            log(f"  = ASFC_n{n_i:.1f}_r{r_i:.2f}（复用默认 ASFC 结果，不重复运行）")
            round_rows.append(asfc)
            continue
        mu_i = sfc.mu_from_r(r_i, B0_, n_i, v_ref)
        row = run_and_store(f"ASFC_n{n_i:.1f}_r{r_i:.2f}", "round", "ASFC",
                            K0, B0_, n_i, mu_i, r_i, v_ref, keep_trace=True)
        round_rows.append(row)
        # 强代表组：r 最大、n 最接近 1 的那组（本轮取 config 里最后一项）
        strong = row

    ok_r = [r for r in round_rows if r["status"] == "OK" and np.isfinite(r["e_rms_um"])]
    log(f"  代表组完成：{len(ok_r)}/{len(round_rows)} 组有效")
    reps = [r for r in round_rows if r["status"] == "OK"]

    # ---------------- 7. §9.2「回正阻碍」局部复查 ----------------
    ret_local = None
    if strong is not None:
        try:
            ret_local = analysis.return_local_analysis(
                {"A0": TRACE_DIR / "A0.csv", "Alinear": TRACE_DIR / "Alinear.csv",
                 "ASFC": TRACE_DIR / f"{strong['case_id']}.csv"},
                config.PLOT_DIR, strong["case_id"],
                along_mm=config.RETURN_LOCAL_ALONG_MM,
                half_mm=config.RETURN_LOCAL_HALF_MM)
            log("  " + ret_local["verdict"])
        except Exception as exc:                                        # noqa: BLE001
            note_err(f"回正阻碍局部复查失败：{exc}")

    # ---------------- 8. §四 中间过程图（只供人工检查，不入交付 ZIP） ----------------
    try:
        analysis.slow_offset_plots(PACKET_DIR, config.PLOT_DIR)
        analysis.compare_xy_plots(
            {"A0": TRACE_DIR / "A0.csv", "Alinear": TRACE_DIR / "Alinear.csv",
             "ASFC": TRACE_DIR / f"{asfc['case_id']}.csv"},
            config.PLOT_DIR, PACKET_DIR)
        log(f"  中间过程图已写入 {config.PLOT_DIR}（不入任何 ZIP）")
    except Exception as exc:                                            # noqa: BLE001
        note_err(f"中间过程图生成失败：{exc}")

    # ---------------- 9. 汇总落盘 ----------------
    log("\n[9] 汇总落盘")
    _finalize(rows, a0, alin, asfc, vref, fd, std_checks, reps, False,
              [], errors, ret_local)

    return {"rows": rows, "a0": a0, "alinear": alin, "asfc": asfc,
            "v_ref": vref, "v_ref_value": v_ref,
            "representatives": [r["case_id"] for r in reps],
            "round": round_rows, "return_local": ret_local, "errors": errors}


# ============================================================ 组内工具
def _standard_checks(a0, alin, asfc, fd) -> dict[str, Any]:
    """§四要求的标准三组重新验证。"""
    shas = {r["case_id"]: r["fd_sha256_12"] for r in (a0, alin, asfc)}
    same_sha = len(set(shas.values())) == 1
    diff = abs(a0["e_rms_um"] - alin["e_rms_um"])
    return {
        "三组Fd_SHA256一致": bool(same_sha),
        "Fd_sha12": fd.sha256[:12],
        "A0与Alinear的RMS差_um": round(float(diff), 6),
        "A0与Alinear有合理差异": bool(diff > 1e-6),
        "ASFC剪切项恒与速度反向": "见 --check 与 sweep_summary 的 P_shear_pos_frac==0",
        "tau_SFC_=_J_tcp^T·F_SFC": "由 controller.JacobianTorqueExecution 保证；"
                                  "TCP 施力点一致性见 fd_meta 的 tcp_force_check",
        "Fd运行期不读控制器状态": "结构保证：FrozenDisturbance.force(t) 只做查表",
    }


def _finalize(rows, a0, alin, asfc, vref, fd, std_checks, reps, inactive,
              extreme_rows, errors, ret_local=None) -> None:
    # --- sweep_summary.csv：包含全部成功、失败与 INVALID 组 ---
    cols = ["case_id", "group", "mode", "status", "note",
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

    a0_r, al_r = a0["e_rms_um"], alin["e_rms_um"]
    a0_v, al_v = a0["e_vib_rms_um"], alin["e_vib_rms_um"]
    for r in rows:
        er = r.get("e_rms_um", float("nan"))
        ev = r.get("e_vib_rms_um", float("nan"))
        r["suppress_vs_A0_pct"] = _pct(a0_r, er)
        r["extra_vs_Alinear_pct"] = _pct(al_r, er)
        r["suppress_vib_vs_A0_pct"] = _pct(a0_v, ev)
        r["extra_vib_vs_Alinear_pct"] = _pct(al_v, ev)
        r.setdefault("trace_file", "")

    with open(SWEEP_DIR / config.SWEEP_CSV, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            out = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    out.append("nan" if math.isnan(v) else f"{v:.6g}")
                elif isinstance(v, bool):
                    out.append(str(v))
                else:
                    out.append(str(v))
            fh.write(",".join(out) + "\n")

    record.save_json(SWEEP_DIR / "baseline_summary.json", {
        "v_ref": vref,
        "A0": a0, "Alinear": alin, "default_ASFC": asfc,
        "standard_checks": std_checks,
        "suppression": {
            "Alinear_vs_A0_pct": _pct(a0_r, al_r),
            "default_ASFC_vs_A0_pct": _pct(a0_r, asfc["e_rms_um"]),
            "default_ASFC_vs_Alinear_pct": _pct(al_r, asfc["e_rms_um"]),
        },
        "representative_cases": [r["case_id"] for r in reps],
        "nonlinear_inactive_after_round1": bool(inactive),
    })

    mid = json.loads((PACKET_DIR / config.FD_META_FILENAME).read_text(encoding="utf-8"))
    last = (mid.get("refine", {}).get("history") or [{}])[-1]
    tgt = mid["template"]["e_rms_um"]
    record.save_json(SWEEP_DIR / "disturbance_fit_summary.json", {
        "frozen_file": config.FD_FILENAME,
        "sha256": mid["frozen"]["sha256"],
        "target_e_rms_um": tgt,
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
        "n_runs": len(rows),
        "n_ok": sum(1 for r in rows if r["status"] == "OK"),
        "n_invalid": sum(1 for r in rows if r["status"] == "INVALID"),
        "n_failed": sum(1 for r in rows if r["status"] == "FAILED"),
        "fd_sha256": fd.sha256,
        "all_runs_same_fd_sha": len({r["fd_sha256_12"] for r in rows}) == 1,
        "runs": rows,
    })

    record.save_json(SWEEP_DIR / "config_snapshot.json", config.snapshot())

    # 记录本轮的口径（去慢偏移 vs 原始）与 §9.2 的局部复查结论
    record.save_json(SWEEP_DIR / "recon_source.json", {
        "recon_source": config.RECON_SOURCE_DEFAULT,
        "fd_packet_dir": str(PACKET_DIR),
        "return_local_verdict": (ret_local or {}).get("verdict", "未执行"),
        "return_local_along_mm": config.RETURN_LOCAL_ALONG_MM,
        "return_local_half_mm": config.RETURN_LOCAL_HALF_MM,
    })

    # 少量本地自查图（不入交付 ZIP）
    try:
        analysis.sweep_plots(rows, SWEEP_DIR / "figs")
    except Exception as exc:                                            # noqa: BLE001
        errors.append(f"扫描自查图生成失败（不影响数据）：{exc}")

    with open(SWEEP_DIR / "errors.txt", "w", encoding="utf-8") as fh:
        fh.write("本次批处理的问题记录（成功组不在此列）\n")
        fh.write("=" * 60 + "\n")
        if not errors:
            fh.write("无：全部组均正常完成。\n")
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
