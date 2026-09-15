"""
UR10 末端 TCP 的 Y 向抗振仿真 —— 入口。

用法：
    python main.py --reconstruct --source vib  ① 构造慢偏移 b(t) → 净振动 e_vib → 重建冻结 Fd
    python main.py --sweep                     ② 标准三组 + 本轮代表组对照（A0/Alinear/3 个 ASFC）
    python main.py --standard                  ③ 只跑标准三组并出对比图
    python main.py --check                     ④ 验收自检（旧方案残留 / TCP 施力点 / 符号与耗散）

本轮（V1.2）唯一核心变化在数据入口：
    e_raw(t) = b(t) + e_vib(t)
  先用「沿程分箱中位数 + LOWESS 鲁棒平滑 + 三次 Hermite 桥接」构造理想慢偏移 b(t)
  （稳态段按运动方向分别拟合 b±(x)，启动/转向/停止段平滑桥接，不追随瞬态尖峰），
  再扣掉它得到净振动 e_vib；扰动重建、v_ref 标定、三组对照全部只针对 e_vib。
  --source raw 保留上一轮口径，仅作对照，不是本轮正式结论。

三条支路严格独立：
    扰动支路   Fd(t) ──mj_applyFT(作用点=TCP site)──▶ UR10
    控制支路   UR10 状态 → e,v → F_SFC → J_tcp^T → τ_SFC ──▶ UR10
    名义运动   schedule(t) → IK → q_d(t) ──data.ctrl──▶ 模型内置关节位置伺服
SFC 绝不读取 Fd；Fd 绝不根据 e_sim 重算；正式运行只读冻结文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Windows 控制台默认 GBK，日志里的 µ/µm 会直接抛 UnicodeEncodeError。统一强制 UTF-8。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

import analysis
import config
import disturbance
import record
import sweep

PACKET_DIR = config.FD_PACKET_DIR
SWEEP_DIR = sweep.SWEEP_DIR
TRACE_DIR = sweep.TRACE_DIR

# 旧方案特征词（验收第 1 步的静态扫描用）。
# 刻意用拼装方式写：否则这份“禁用词清单”本身会被自己扫出来，永远误报 main.py。
LEGACY_TOKENS = tuple("".join(p) for p in (
    ("Virtual", "Cart"), ("virtual", "_cart"), ("AP", "F"), ("ap", "f"),
    ("delta", "_y"), ("delta", "_q"), ("F_", "vir"), ("y_sfc", "_offset"),
    ("closed", "_loop"),
))


# ============================================================ 验收自检
def cmd_check(packet_dir: Path, log=print) -> int:
    ok = True
    log("=" * 68)
    log("验收自检")
    log("=" * 68)

    # --- ① 旧方案残留扫描 ---
    log("\n[1] 旧控制链残留扫描（本工程全部 .py）")
    hits = []
    for py in sorted(config.PROJECT_DIR.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for tok in LEGACY_TOKENS:
            if tok in text:
                hits.append((py.name, tok))
    if hits:
        for name, tok in hits:
            log(f"    ! {name} 含特征词 {tok!r}（请确认只是说明性文字）")
    else:
        log(f"    OK 已扫描 {len(LEGACY_TOKENS)} 个旧控制链特征词（虚拟小车 / Δy 一类），"
            "全部未命中")

    # --- ② 冻结 Fd + TCP 施力点 ---
    log("\n[2] 冻结 Fd 覆盖、哈希与 TCP 施力点")
    try:
        fd = disturbance.load_frozen(packet_dir)
        log(f"    OK {fd.describe()}")
        meta = fd.meta
        hist = meta.get("refine", {}).get("history", [])
        if hist:
            last = hist[-1]
            log(f"    重建最终：e_sim rms {last['e_sim_rms_um']:.2f} µm，"
                f"corr {last['corr']:.4f}，残差 rms {last['residual_rms_um']:.2f} µm，"
                f"残差/目标 {last['residual_over_target'] * 100:.1f}%")
        fa = meta.get("force_application", {})
        if fa:
            chk = fa.get("tcp_force_check", {})
            log(f"    作用点：{fa.get('point')}")
            log(f"    TCP 施力点检查：max|q_applyFT − J_tcp^T·F| = "
                f"{chk.get('max_abs_err', float('nan')):.3e} → "
                f"{'通过' if chk.get('passed') else '不通过'}")
            ok &= bool(chk.get("passed", False))
    except Exception as exc:                                   # noqa: BLE001
        log(f"    X {exc}")
        ok = False

    # --- ③ 标准三组的符号与耗散检查 ---
    log("\n[3] 控制律符号与耗散正确性（基于已跑结果）")
    base = SWEEP_DIR / "baseline_summary.json"
    if not base.is_file():
        log("    （尚无运行结果，跳过；先跑 python main.py --standard）")
    else:
        bs = json.loads(base.read_text(encoding="utf-8"))
        a0_id = "A0"
        al_id = "Alinear"
        as_id = bs["default_ASFC"]["case_id"]
        for cid, label in ((a0_id, "A0"), (al_id, "Alinear"), (as_id, "ASFC(默认)")):
            p = TRACE_DIR / f"{cid}.csv"
            if not p.is_file():
                log(f"    （缺 {label} 轨迹，跳过）")
                continue
            _, d = record.load_trace(p)
            v = d["v_m_s"]
            if label == "A0":
                mx = float(np.max(np.abs(d["F_SFC_N"])))
                log(f"    [{label:<12}] F_SFC 最大绝对值 {mx:.3e} N（应为 0）")
                ok &= mx < 1e-12
                continue
            e = d["e_um"] * 1e-6
            m_e, m_v = np.abs(e) > 1e-7, np.abs(v) > 1e-7
            sgn_e = bool(np.all(np.sign(-d["F_spring_N"][m_e]) == np.sign(e[m_e])))
            sgn_v = bool(np.all(np.sign(-d["F_linear_damping_N"][m_v]) == np.sign(v[m_v])))
            P_lin = d["F_linear_damping_N"] * v
            P_shr = d["F_shear_N"] * v
            log(f"    [{label:<12}] 正 e → 负 F_spring：{sgn_e}   "
                f"正 v → 负 F_linear_damping：{sgn_v}")
            log(f"    [{label:<12}] P_linear>tol 比例 {np.mean(P_lin > config.P_TOL_W):.2e}   "
                f"P_shear>tol 比例 {np.mean(P_shr > config.P_TOL_W):.2e}")
            ok &= bool(sgn_e and sgn_v)
            ok &= bool(np.mean(P_lin > config.P_TOL_W) == 0.0)
            ok &= bool(np.mean(P_shr > config.P_TOL_W) == 0.0)
            if label.startswith("ASFC"):
                lr = float(np.sqrt(np.mean(d["F_linear_damping_N"] ** 2)))
                sr = float(np.sqrt(np.mean(d["F_shear_N"] ** 2)))
                log(f"    [{label:<12}] F_shear rms {sr:.4f} N vs F_linear rms {lr:.4f} N "
                    f"(比值 {sr / lr if lr > 0 else float('nan'):.4f})")
                log(f"    [{label:<12}] 恒与速度反向：{bool(np.all(np.sign(d['F_shear_N'][m_v]) == -np.sign(v[m_v])))}")

    log("\n" + "=" * 68)
    log("自检结论：" + ("全部通过" if ok else "存在未通过项，见上"))
    log("=" * 68)
    return 0 if ok else 1


# ============================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="UR10 末端 TCP 的 Y 向抗振仿真（冻结等效扰动 + SFC 批量实验）")
    ap.add_argument("--reconstruct", action="store_true",
                    help="重建并冻结等效扰动力 Fd（离线，只跑一次）")
    ap.add_argument("--sweep", action="store_true",
                    help="标准三组 + 本轮代表组对照（A0/Alinear/默认+中等+强 ASFC）")
    ap.add_argument("--standard", action="store_true",
                    help="只跑标准三组并出对比图（不做代表组对照）")
    ap.add_argument("--source", default=config.RECON_SOURCE_DEFAULT,
                    choices=("vib", "raw"),
                    help="扰动重建的输入口径：vib=去慢偏移后的净振动（本轮正式）；"
                         "raw=原始 e_raw（上一轮口径，仅对照）")
    ap.add_argument("--check", action="store_true", help="验收自检")
    ap.add_argument("--packet", default=str(PACKET_DIR), help="冻结数据包目录")
    args = ap.parse_args(argv)

    if args.reconstruct:
        meta = disturbance.reconstruct_fd(Path(args.packet), source=args.source)
        print(f"\n[完成] 冻结数据包 → {Path(args.packet)}"
              f"（来源口径 source={args.source}）")
        sl = meta.get("slow_offset") or {}
        if sl:
            acc = sl.get("acceptance") or {}
            print(f"        慢偏移：分箱 {sl.get('bin_mm')} mm，带宽 {sl.get('span_mm')} mm，"
                  f"b(t) 范围 {sl.get('b_range_um', [float('nan')] * 2)[0]:.2f}"
                  f"~{sl.get('b_range_um', [float('nan')] * 2)[1]:.2f} µm")
            print(f"        人工验收：{'通过' if acc.get('passed') else '未通过'}"
                  f" —— {acc.get('conclusion', '')}")
            if not acc.get("passed"):
                print(f"        未通过项：{acc.get('failed_checks')}")
                for rs in acc.get("reasons", []):
                    print(f"          · {rs}")
        f = meta["frozen"]
        print(f"        Fd rms {f['Fd_rms_N']:.3f} N，峰值 {f['Fd_peak_abs_N']:.3f} N，"
              f"sha256 {f['sha256'][:12]}…")
        h = meta["refine"]["history"][-1]
        print(f"        复现：e_sim rms {h['e_sim_rms_um']:.2f} µm vs 目标 "
              f"{meta['template']['e_rms_um']:.2f} µm，corr {h['corr']:.4f}，"
              f"残差 {h['residual_rms_um']:.2f} µm"
              f"（{h['residual_over_target'] * 100:.1f}%）")
        return 0

    if args.check:
        return cmd_check(Path(args.packet))

    if args.sweep:
        sweep.run_batch(do_sweep=True)
        return 0

    # 默认 / --standard：只跑标准三组
    sweep.run_batch(do_sweep=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
