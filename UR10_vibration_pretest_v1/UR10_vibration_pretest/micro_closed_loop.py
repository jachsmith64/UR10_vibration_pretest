"""
一键 XY 多目标视觉闭环逼近测试（micro_closed_loop）。

当前主动流程使用每轴 0→100→250→460→360→210→0 μm 的目标序列，
观察误差进入个位数/十几微米后何时稳定、停滞或进入极限振荡。

要回答的不是"最终误差是多少"，而是三件事：
1. 完整迭代序列长什么样（例如 0 → +13 → −2 → +12 → −1 → +14 …）；
2. 每个真实命令 ↔ 真实视觉响应的对应关系（据此可反推 2/3/5/8/10 μm 命令实际走多少）；
3. 达不到目标时，卡住的是"控制器忽略了命令"还是"机械柔性吸收掉了"。

本模块与既有实验的关系：
- 复用 robot.py 的 MoveL 微动链路（URRobot / Waypoint / validate_trajectory），
  不切换到 servoj / RTDE 高速伺服，也不改动 robot.py 本身；
- 复用 camera.py 已经过预实验验证的棋盘格亚像素链路（CheckerboardTracker.process），
  不重新设计识别算法；
- **不复用 batch_camera_worker**：那套把编解码器写死为 BATCH_VIDEO_CODEC。

录制格式：**未压缩 RAW**。原因是 OpenCV 的 VideoWriter_fourcc('FFV1') 写 Mono8 时
会经 swscale 转成 YUV420P（色度抽样 + 有限范围 16–235），并非位精确；而本实验要的
分辨率是 3 μm ≈ 0.044 px，这种系统性量化是实打实的损失，抽检帧比对也查不出来。
未压缩 RAW 位精确、不依赖编解码器行为。编码在这里本来也不受实时约束——采集先把帧
缓冲进内存，落盘发生在片段结束之后——所以 RAW 的唯一代价（体积）由"用后即删"吸收。

为什么采集要先缓冲进内存：实测全幅 1936×1464 实时写盘每帧均值 2.27 ms、峰值 7.38 ms，
而相机周期是 7.56 ms，会丢掉 6.11% 的帧。memcpy 进内存只需约 0.3 ms，余量 25 倍。
识别与录制**完全串行**：片段关闭之后，父进程才逐帧跑棋盘格，识别再慢也挤不掉采集。

**本模块有意绕过手电筒门禁**：batch_camera_worker 的 _FlashGate 会硬抛
"手电筒门禁未完成"。微动闭环不需要那套门禁（它依赖操作者打手电筒，而本实验的
时间对齐由全机共享的 perf_counter_ns 保证），所以新相机进程不构造 _FlashGate。
这是有意的设计决定，不是安全回归——若将来有人要给本模式加回门禁，请连同这段说明一起改。
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Iterator, Sequence

import numpy as np

import config


# =============================================================================
# 0. 常量
# =============================================================================

AXIS_INDEX: dict[str, int] = {"X": 0, "Y": 1}
AXES: tuple[str, ...] = ("X", "Y")
AMPLITUDES_UM: tuple[int, ...] = (5, 20, 50)
DIRECTIONS: tuple[int, ...] = (1, -1)

# 迭代终止状态。前面 8 个是"已结束"，STILL_SHRINKING 表示"还没结束，继续迭代"。
TERMINAL_CONVERGED = "CONVERGED"
TERMINAL_CONVERGED_NOISE_LIMITED = "CONVERGED_NOISE_LIMITED"
# "误差落在容差里，但机器人根本没被观察到朝目标走过"——
# 这是 5 μm 档最可能出现的**假成功**，必须与 CONVERGED 分开报。
# 它既不是成功也不是失败，而是"这个幅值下视觉分不出来"这一条真实结论。
TERMINAL_MEASUREMENT_NOISE_LIMITED = "MEASUREMENT_NOISE_LIMITED"
TERMINAL_LIMIT_CYCLE = "LIMIT_CYCLE"
TERMINAL_STALLED = "STALLED_MIN_EFFECTIVE_MOTION"
TERMINAL_DIVERGING = "DIVERGING"
TERMINAL_MAX_ITER = "MAX_ITER"
TERMINAL_MEASUREMENT_LOST = "MEASUREMENT_LOST"
TERMINAL_ABORTED = "ABORTED"
TERMINAL_STILL_SHRINKING = "STILL_SHRINKING"

# 单次迭代的执行结果（与终止状态是两回事：一个目标可以有很多次 NO_MOTION 迭代）。
STEP_VALID = "VALID"
STEP_NO_MOTION = "NO_MOTION_BELOW_RESOLUTION"
STEP_MEASUREMENT_LOST = "MEASUREMENT_LOST"
STEP_INVALID_JUMP = "INVALID_JUMP"
STEP_SMALL = "INVALID_SMALL"
STEP_LARGE = "INVALID_LARGE"
STEP_SIGN = "INVALID_SIGN"
STEP_TIMEOUT = "INVALID_TIMEOUT"

ITERATION_COLUMNS: tuple[str, ...] = (
    "iteration_id",
    "axis",
    "direction",
    "target_amplitude_um",
    "target_um",
    "measured_position_um",
    "error_before_um",
    "command_um",
    "achieved_robot_um",
    "actual_visual_displacement_um",
    "error_after_um",
    "robot_tcp_before",
    "robot_tcp_after",
    "transverse_um",
    "g_obs",
    "n_valid_frames",
    "n_tail_frames",
    "sigma_tail_um",
    "timestamp",
    "step_status",
    "termination_state",
    "note",
)

FRAME_COLUMNS: tuple[str, ...] = (
    "iteration_id",
    "frame_index",
    "frame_id",
    "host_ns",
    "t_rel_s",
    "camera_timestamp_raw",
    "checker_dx_mm",
    "checker_dy_mm",
    "checker_is_valid",
    "accepted",
    "checker_quality",
    "checker_residual_px",
    "checker_angle_deg",
    "checker_scale",
    "checker_found_by",
    "reject_reason",
    "in_tail_window",
)

STATIC_FRAME_COLUMNS: tuple[str, ...] = (
    "frame_index",
    "frame_id",
    "host_ns",
    "t_rel_s",
    "camera_timestamp_raw",
    "checker_dx_mm",
    "checker_dy_mm",
    "checker_is_valid",
    "accepted",
    "checker_quality",
    "checker_residual_px",
    "checker_found_by",
    "reject_reason",
)

# 相机进程自己的逐帧时间戳旁车（临时文件，随 RAW 一起删除）。
RAW_FRAME_COLUMNS: tuple[str, ...] = (
    "segment_frame_index",
    "frame_id",
    "host_ns",
    "camera_timestamp_raw",
)

MASTER_COLUMNS: tuple[str, ...] = (
    "axis",
    "target_amplitude_um",
    "target_direction",
    "iteration_count",
    "termination_state",
    "final_mean_error_um",
    # 这是**相对目标的残差**（target_um − final_measured_um）。
    # 旧名 final_abs_error_um 装的是 abs(最终位置)，是位置不是误差，
    # 目标在 ±50 μm 时会稳定地报出 50 μm 的"误差"。
    "closed_loop_final_residual_um",
    # 最终位置本身也留着：残差说"离目标多远"，位置说"人站在哪儿"，
    # 复盘回差与零点漂移时要的是后者。
    "final_position_um",
    "final_error_peak_to_peak_um",
    "max_overshoot_um",
    "smallest_command_issued_um",
    "corresponding_actual_motion_um",
    "actual_frames",
    "expected_frames",
    "frame_ratio",
    "valid_chessboard_frames",
    "chessboard_detection_rate",
    "initial_error_um",
    "residual_snr",
    "est_sigma_um",
    "limit_cycle_amplitude_um",
    "plateau_um",
    "min_effective_motion_limit_um",
    "drift_um_per_min",
    "ref_cross_group_delta_um",
    "converged_verified",
    "axis_reference_um",
    "temp_bytes_peak",
    "trash_bytes",
    "note",
)

TEMP_DIR_NAME = "_micro_temp"
TRASH_DIR_NAME = "_micro_trash"

# 证据帧用途标签（每个目标最多 MICRO_LOOP_EVIDENCE_MAX 张）。
EVIDENCE_START = "01_start_stable"
EVIDENCE_PEAK = "02_max_offset_or_overshoot"
EVIDENCE_FINAL = "03_final_converged_or_limit_cycle"

# 当前多目标视觉闭环的终止状态。旧状态名保留给历史结果读取，当前主流程只写下面四种。
TARGET_STABLE_REACHED = "STABLE_REACHED"
TARGET_LIMIT_CYCLE = "LIMIT_CYCLE"
TARGET_SMALL_COMMAND_STALL = "SMALL_COMMAND_STALL"
TARGET_MAX_ITER_REACHED = "MAX_ITER_REACHED"

TARGET_ITERATION_COLUMNS: tuple[str, ...] = (
    "axis",
    "target_index",
    "target_nominal_um",
    "target_absolute_um",
    "iteration",
    "position_before_um",
    "error_before_um",
    "command_um",
    "vision_delta_um",
    "position_after_um",
    "error_after_um",
    "orthogonal_axis_before_um",
    "orthogonal_axis_after_um",
    "orthogonal_drift_um",
    "command_and_motion_same_direction",
    "rtde_before",
    "rtde_after",
    "rtde_delta_um",
    "baseline_noise_reference_um",
    "stop_reason",
    "timestamp",
    # 额外保留测量质量与时间对齐证据，不改变上面规定字段的含义。
    "before_valid_frames",
    "after_valid_frames",
    "before_sigma_um",
    "after_sigma_um",
    "command_start_ns",
    "command_end_ns",
    "note",
)


# =============================================================================
# 1. 纯计算层（不 import 相机 SDK / ur_rtde，可离线单测）
# =============================================================================

def build_targets(
    *,
    axes: Sequence[str] = AXES,
    amplitudes_um: Sequence[int] = AMPLITUDES_UM,
    directions: Sequence[int] = DIRECTIONS,
) -> list[dict[str, Any]]:
    """
    生成 12 个核心目标：X:+5,−5,+20,−20,+50,−50 然后 Y 同样。

    输入：轴、幅值、方向（默认都取本模块常量）。
    输出：目标字典列表，按执行顺序排列。

    实验作用：±A 不是"一条命令"，而是"一个视觉目标位置 + 允许反复闭环纠偏"。
    同一个幅值的 +A 与 −A 共用同一个局部参考，不重新定义参考——
    这样 +A→−A 的切换才真正检验反向时的死区/回差。
    """

    targets: list[dict[str, Any]] = []
    for axis in axes:
        for amplitude_um in amplitudes_um:
            for direction in directions:
                targets.append(
                    {
                        "target_id": f"{axis}_{int(amplitude_um):03d}um_{'pos' if direction > 0 else 'neg'}",
                        "axis": str(axis),
                        "amplitude_um": int(amplitude_um),
                        "direction": int(direction),
                        "target_um": float(direction) * float(amplitude_um),
                    }
                )
    return targets


def build_multi_target_sequence(
    *,
    axes: Sequence[str] = AXES,
    relative_steps_um: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """生成当前实验的 12 个轴向目标，不包含斜向目标。"""

    steps = tuple(
        int(value)
        for value in (
            config.MICRO_TARGET_RELATIVE_STEPS_UM
            if relative_steps_um is None
            else relative_steps_um
        )
    )
    targets: list[dict[str, Any]] = []
    for axis in axes:
        absolute = 0
        for index, step in enumerate(steps, start=1):
            absolute += step
            targets.append(
                {
                    "axis": str(axis),
                    "target_index": index,
                    "target_nominal_um": step,
                    "target_absolute_um": absolute,
                }
            )
    return targets


def _sign_changes(values: Sequence[float]) -> int:
    signs = [1 if float(value) > 0.0 else -1 for value in values if float(value) != 0.0]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def classify_multi_target_stop(
    errors_um: Sequence[float],
    commands_um: Sequence[float],
    motions_um: Sequence[float],
    *,
    baseline_noise_um: float,
    max_iterations: int | None = None,
) -> str | None:
    """
    判断一个多目标闭环是否结束；返回 None 表示继续。

    稳定带固定为 ±3 μm，不随噪声放宽。极限振荡至少在第 6 次以后判断；
    小命令停滞的运动阈值随本次 5 s 静止基线变化。
    """

    errors = [float(value) for value in errors_um if math.isfinite(float(value))]
    commands = [float(value) for value in commands_um if math.isfinite(float(value))]
    motions = [float(value) for value in motions_um if math.isfinite(float(value))]
    max_iter = int(config.MICRO_TARGET_MAX_ITER if max_iterations is None else max_iterations)
    stable_count = int(config.MICRO_TARGET_STABLE_COUNT)
    stable_tol = float(config.MICRO_TARGET_STABLE_TOL_UM)
    if len(errors) >= stable_count and all(
        abs(value) <= stable_tol for value in errors[-stable_count:]
    ):
        return TARGET_STABLE_REACHED

    window = int(config.MICRO_TARGET_LIMIT_WINDOW)
    if len(errors) >= 6 and len(errors) >= window:
        recent = errors[-window:]
        prior = errors[max(0, len(errors) - 2 * window) : -window]
        noise = (
            abs(float(baseline_noise_um))
            if math.isfinite(float(baseline_noise_um))
            else 0.0
        )
        small_range = max(20.0, 6.0 * noise)
        if (
            _sign_changes(recent)
            >= int(config.MICRO_TARGET_LIMIT_MIN_SIGN_CHANGES)
            and prior
            and float(np.median(np.abs(recent)))
            >= (1.0 - float(config.MICRO_TARGET_LIMIT_IMPROVEMENT_RATIO))
            * float(np.median(np.abs(prior)))
            and max(recent) - min(recent) <= small_range
        ):
            return TARGET_LIMIT_CYCLE

    # 三次小命令均只产生接近静止噪声的位移，而且误差没有获得超过噪声的改善。
    patience = 3
    if len(errors) >= patience + 1 and len(commands) >= patience and len(motions) >= patience:
        noise = (
            abs(float(baseline_noise_um))
            if math.isfinite(float(baseline_noise_um))
            else 0.0
        )
        motion_gate = max(1.0, 2.0 * noise)
        improvement_gate = max(1.0, noise)
        recent_commands = commands[-patience:]
        recent_motions = motions[-patience:]
        improvement = abs(errors[-(patience + 1)]) - abs(errors[-1])
        if (
            all(abs(value) <= float(config.MICRO_TARGET_SMALL_COMMAND_UM) for value in recent_commands)
            and all(abs(value) <= motion_gate for value in recent_motions)
            and improvement <= improvement_gate
        ):
            return TARGET_SMALL_COMMAND_STALL

    if len(errors) >= max_iter:
        return TARGET_MAX_ITER_REACHED
    return None


def build_groups(
    *,
    axes: Sequence[str] = AXES,
    amplitudes_um: Sequence[int] = AMPLITUDES_UM,
) -> list[dict[str, Any]]:
    """
    按"一个幅值一组"分组，组内先 +A 后 −A。

    输入：轴、幅值。
    输出：组列表，每组含 axis / amplitude_um / 两个目标。

    实验作用：参考只在组的开头建立一次。组内 +A 与 −A 共用它，
    幅度切换时才重新建立新参考——这正是操作者要求的流程。
    """

    groups: list[dict[str, Any]] = []
    for axis in axes:
        for amplitude_um in amplitudes_um:
            groups.append(
                {
                    "axis": str(axis),
                    "amplitude_um": int(amplitude_um),
                    "targets": build_targets(
                        axes=(axis,), amplitudes_um=(amplitude_um,), directions=DIRECTIONS
                    ),
                }
            )
    return groups


# =============================================================================
# 1b. 开环微动模式（seq / alt）
# =============================================================================
#
# 开环与闭环测的是两件不同的事，指标必须严格分开，不能互相替代：
#   开环 = "给一个固定命令，实际走了多少" → 这才是最小可靠微动的来源；
#   闭环 = "能不能用反复纠偏逼近一个目标" → 这是逼近能力，最终残差不是最小微动。
# 上一版把闭环残差当最小微动报出去，是概念错误，本轮用字段名彻底分开。

OPEN_MODE_SEQ = "seq"
OPEN_MODE_ALT = "alt"

# 开环单步的执行结果。
OPEN_STEP_VALID = "VALID"
# "完全没动"与"动了但只走了一半"是两种不同的物理现象，分开计数：
# 前者是控制器忽略了命令/死区，后者是摩擦、柔性或增益不足。
OPEN_STEP_ZERO = "ZERO_RESPONSE"
OPEN_STEP_PARTIAL = "PARTIAL_RESPONSE"
OPEN_STEP_DIRECTION_ERROR = "DIRECTION_ERROR"
OPEN_STEP_ABNORMAL_JUMP = "ABNORMAL_JUMP"
OPEN_STEP_MEASUREMENT_LOST = "MEASUREMENT_LOST"
OPEN_STEP_NOT_SETTLED = "NOT_SETTLED"

# 开环档位的整体判读。刻意不用 "min resolution" 这种词——
# 本轮只测三个档位，只能说"这一档稳不稳"，推不出精确最小分辨率。
OPEN_VERDICT_STABLE = "STABLE"
OPEN_VERDICT_PARTIAL = "PARTIAL"
OPEN_VERDICT_NO_RESPONSE = "NO_RESPONSE"
OPEN_VERDICT_UNRELIABLE = "UNRELIABLE"

OPEN_STEP_COLUMNS: tuple[str, ...] = (
    "block_id",
    "axis",
    "mode",
    "target_level_um",
    "step_index",
    "direction",
    "commanded_increment_um",
    "vision_axis",
    "vision_before_um",
    "vision_after_um",
    "measured_increment_um",
    "robot_tcp_before",
    "robot_tcp_after",
    "robot_achieved_um",
    "command_timestamp_ns",
    "measurement_timestamp_ns",
    "settling_time_s",
    "settle_ok",
    "n_valid_frames",
    "n_tail_frames",
    "sigma_tail_um",
    "accept_rate",
    "reject_reasons",
    "status",
    "note",
)


def open_loop_steps(
    *,
    axis: str,
    mode: str,
    level_um: int,
    seq_positive: int | None = None,
    seq_negative: int | None = None,
    alt_steps: int | None = None,
) -> list[dict[str, Any]]:
    """
    生成一个开环实验块的固定命令序列。

    输入：轴、模式（seq / alt）、档位（μm）、各模式的步数（默认取配置）。
    输出：step 字典列表，字段与 OPEN_STEP_COLUMNS 对应。

    seq（连续同向）：+Δ×3 然后 −Δ×3 —— 共 6 步，净位移 ≈0。
      看的是：微小命令是否真的产生运动、连续同向能否累积、正负是否对称，
      以及有没有"前两步不动、第三步突然跳很大"的死区/积累现象。
    alt（频繁换向）：+Δ −Δ 交替 6 步 —— 净位移也 ≈0。
      模拟未来 SFC 可能出现的高频正负修正，重点看换向死区、回差、摩擦与方向延迟。

    两种模式都以"净位移 ≈0"结束，所以块与块之间不需要额外的回位动作，
    全局漂移也不会随着块数累积。
    """

    mode = str(mode)
    if mode == OPEN_MODE_SEQ:
        positive = (
            int(config.MICRO_LOOP_OPEN_SEQ_POSITIVE_STEPS)
            if seq_positive is None
            else int(seq_positive)
        )
        negative = (
            int(config.MICRO_LOOP_OPEN_SEQ_NEGATIVE_STEPS)
            if seq_negative is None
            else int(seq_negative)
        )
        directions = [1] * positive + [-1] * negative
    elif mode == OPEN_MODE_ALT:
        count = int(config.MICRO_LOOP_OPEN_ALT_STEPS) if alt_steps is None else int(alt_steps)
        directions = [1 if index % 2 == 0 else -1 for index in range(count)]
    else:
        raise ValueError(f"未知开环模式 {mode!r}，只能是 'seq' 或 'alt'。")

    level = float(level_um)
    block_id = f"{axis}_open_{mode}_{int(level_um):03d}um"
    steps: list[dict[str, Any]] = []
    for index, direction in enumerate(directions, start=1):
        steps.append(
            {
                "block_id": block_id,
                "axis": str(axis),
                "mode": mode,
                "target_level_um": float(level_um),
                "step_index": index,
                "direction": int(direction),
                "commanded_increment_um": float(direction) * level,
            }
        )
    return steps


def classify_open_step(
    commanded_um: float,
    measured_um: float | None,
    *,
    min_ratio: float | None = None,
    max_ratio: float | None = None,
    zero_um: float | None = None,
    jump_um: float | None = None,
) -> tuple[str, str]:
    """
    给开环单步定一个状态码。

    输入：命令增量、视觉实测增量（μm，已归一到机器人轴正向）、各判据阈值。
    输出：(状态码, 说明)。

    判据与闭环的 achieved 断言是两套，刻意不复用：闭环那套看的是编码器侧，
    这一套看的是**视觉侧**——两者并排才能把"控制器忽略了命令"与
    "机械柔性吸收掉了"分开。
    """

    if measured_um is None or not math.isfinite(float(measured_um)):
        return OPEN_STEP_MEASUREMENT_LOST, "本步没有可用的视觉测量"

    lo = float(config.MICRO_LOOP_OPEN_MIN_RATIO if min_ratio is None else min_ratio)
    hi = float(config.MICRO_LOOP_OPEN_MAX_RATIO if max_ratio is None else max_ratio)
    zero = float(config.MICRO_LOOP_OPEN_ZERO_UM if zero_um is None else zero_um)
    jump = float(config.MICRO_LOOP_OPEN_JUMP_UM if jump_um is None else jump_um)

    commanded = float(commanded_um)
    measured = float(measured_um)
    if commanded == 0.0:
        return OPEN_STEP_ZERO, "命令为零"

    # 异常跳变先判：它比"比例不对"更严重，是检测层出问题的典型症状
    # （棋盘格索引错位一格会给出一个残差≈0、质量≈1.0 的假平移）。
    if abs(measured) > jump:
        return (
            OPEN_STEP_ABNORMAL_JUMP,
            f"实测增量 {measured:+.1f} μm 超过异常跳变上限 {jump:g} μm",
        )

    if abs(measured) <= zero:
        return (
            OPEN_STEP_ZERO,
            f"命令 {commanded:+.1f} μm，实测增量只有 {measured:+.2f} μm，"
            f"低于零响应阈值 {zero:g} μm",
        )

    if measured * commanded < 0.0:
        return (
            OPEN_STEP_DIRECTION_ERROR,
            f"命令方向 {math.copysign(1.0, commanded):+.0f}，"
            f"实测方向 {math.copysign(1.0, measured):+.0f}，方向相反",
        )

    ratio = abs(measured) / abs(commanded)
    if ratio < lo:
        return (
            OPEN_STEP_PARTIAL,
            f"实测 {abs(measured):.2f} μm 只有命令 {abs(commanded):.1f} μm 的 "
            f"{ratio:.0%}，低于 {lo:.0%}",
        )
    if ratio > hi:
        return (
            OPEN_STEP_ABNORMAL_JUMP,
            f"实测 {abs(measured):.2f} μm 是命令 {abs(commanded):.1f} μm 的 "
            f"{ratio:.0%}，超过 {hi:.0%}",
        )
    return OPEN_STEP_VALID, f"实测 {measured:+.2f} μm（命令 {commanded:+.1f} μm，{ratio:.0%}）"


def summarize_open_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    把一个开环实验块的逐步记录压成一档的响应统计。

    输入：该块的逐步行（字段见 OPEN_STEP_COLUMNS）。
    输出：含成功率、实测增量的均值/标准差、方向错误数、零响应数、异常跳变数与判读的字典。

    这些数就是"最小可靠微动"的唯一来源；闭环那边的最终残差不参与这里的任何一项。
    """

    if not rows:
        return {
            "steps": 0,
            "measured_steps": 0,
            "valid_steps": 0,
            "zero_response_steps": 0,
            "partial_response_steps": 0,
            "direction_error_steps": 0,
            "abnormal_jump_steps": 0,
            "lost_steps": 0,
            "success_ratio": None,
            "mean_increment_um": None,
            "std_increment_um": None,
            "mean_abs_increment_um": None,
            "response_ratio": None,
            "net_displacement_um": None,
            "verdict": OPEN_VERDICT_UNRELIABLE,
        }

    measured = [
        float(row["measured_increment_um"])
        for row in rows
        if row.get("measured_increment_um") not in (None, "")
        and math.isfinite(float(row["measured_increment_um"]))
    ]
    commanded = [
        abs(float(row.get("commanded_increment_um", 0.0)))
        for row in rows
        if row.get("commanded_increment_um") not in (None, "")
    ]
    statuses = [str(row.get("status", "")) for row in rows]
    valid = statuses.count(OPEN_STEP_VALID)
    zero = statuses.count(OPEN_STEP_ZERO)
    partial = statuses.count(OPEN_STEP_PARTIAL)
    direction = statuses.count(OPEN_STEP_DIRECTION_ERROR)
    jump = statuses.count(OPEN_STEP_ABNORMAL_JUMP)
    lost = statuses.count(OPEN_STEP_MEASUREMENT_LOST) + statuses.count(OPEN_STEP_NOT_SETTLED)

    array = np.asarray(measured, dtype=float) if measured else np.asarray([], dtype=float)
    mean_increment = float(array.mean()) if array.size else None
    std_increment = float(array.std(ddof=1)) if array.size >= 2 else None
    mean_abs = float(np.abs(array).mean()) if array.size else None
    mean_commanded = float(np.mean(commanded)) if commanded else None
    response_ratio = (
        (mean_abs / mean_commanded)
        if mean_abs is not None and mean_commanded not in (None, 0.0)
        else None
    )
    net = float(np.sum(array)) if array.size else None

    # 单步利用率：有多少步真正"按命令走够了"。
    # NO_RESPONSE 特指"一步都没走够"，部分响应是另一档，两者不能混。
    if not measured or (valid == 0 and partial == 0):
        verdict = OPEN_VERDICT_NO_RESPONSE
    elif valid == 0 or valid * 2 < len(rows):
        verdict = OPEN_VERDICT_UNRELIABLE
    elif valid < len(rows):
        verdict = OPEN_VERDICT_PARTIAL
    else:
        verdict = OPEN_VERDICT_STABLE

    return {
        "steps": len(rows),
        "measured_steps": len(measured),
        "valid_steps": valid,
        "zero_response_steps": zero,
        "partial_response_steps": partial,
        "direction_error_steps": direction,
        "abnormal_jump_steps": jump,
        "lost_steps": lost,
        "success_ratio": (valid / len(rows)) if rows else None,
        "mean_increment_um": mean_increment,
        "std_increment_um": std_increment,
        "mean_abs_increment_um": mean_abs,
        "response_ratio": response_ratio,
        "net_displacement_um": net,
        "verdict": verdict,
    }


def min_reliable_level(
    block_stats: Sequence[dict[str, Any]], *, mode: str, axis: str
) -> float | None:
    """
    在测过的档位里挑出"最小且稳定有效"的那一档。

    输入：同一 (轴, 模式) 下各档位的统计、模式、轴。
    输出：最小 STABLE 档位的幅值（μm）；没有 STABLE 档时返回 None。

    只在这三个档位里挑，所以它只能说"5 或 20 或 50 μm 是已测档位里最小且稳定的"，
    推不出精确的最小分辨率——那需要扫更多档位，正是本轮刻意不做的。
    """

    candidates = sorted(
        (
            float(block["level_um"])
            for block in block_stats
            if str(block.get("axis")) == str(axis)
            and str(block.get("mode")) == str(mode)
            and str(block.get("verdict")) == OPEN_VERDICT_STABLE
        )
    )
    return candidates[0] if candidates else None


def sign_flip_detected(
    *,
    command_um: float,
    achieved_robot_um: float,
    measured_delta_um: float | None,
    g_obs: float,
    floor_um: float | None = None,
    min_command_um_for_g_obs: float = 5.0,
) -> bool:
    """
    这一轮是否呈现"指令与实测方向相反"。

    输入：本轮指令（机器人轴正向，μm）、机器人侧实测位移、视觉侧实测位移、
          observed gain、判据地板。
    输出：True = 疑似符号写反，调用方据此累加 sign_flip_streak。

    ★ 三个入参**必须全部已经在机器人测试轴坐标系里**：
      - command_um 由 command_for_error 从已归一的 error 算出（它不接 sign）；
      - achieved_robot_um 是机器人自己的 TCP 读数，本来就是机器人坐标；
      - measured_delta_um 来自 axis_measured_um(本次) − axis_measured_um(上次)，
        两端都归过符号。
    任何一个再乘一次 sign，都会让**正常的、sign=−1 的相机安装**被判成方向异常：
    机器人老实按指令走，位移与指令却被读成反号，连续两次就触发
    SIGN_INVERTED 中止整个目标。这正是本函数被单独抽出来的原因——
    把判据放在一个能直接喂数测试的地方，而不是埋在 3000 行主流程里。

    ★ 但要如实说明这条联锁**管不了**什么：如果符号是在控制律里用错了
    （command 与 command_for_error 各乘一次 sign），指令与实测位移会恒同号，
    本函数恒返回 False——那种缺陷不会在这里暴露，表现为 DIVERGING。
    换句话说本函数只能抓"测量侧符号不一致"，抓不了"控制侧符号用两次"；
    后者靠的是 command_for_error 干脆不接收 sign。

    两条判据（与上一版一致，只是搬了位置）：
      1. 机器人侧确实动了（超过地板值）却和指令反向；
      2. 视觉侧算出的 g_obs < 0，且指令足够大（小指令的 g_obs 是噪声）。
    """

    floor = (
        float(config.MICRO_LOOP_ACHIEVED_FLOOR_UM) if floor_um is None else float(floor_um)
    )

    if measured_delta_um is None:
        # 视觉侧这轮没测到，只剩机器人侧可判——不足以单独定罪，
        # 与上一版一致：不做任何假设，返回 False。
        return False

    if (
        math.isfinite(float(achieved_robot_um))
        and abs(float(achieved_robot_um)) >= floor
        and (float(achieved_robot_um) > 0.0) != (float(command_um) > 0.0)
    ):
        return True
    if (
        abs(float(command_um)) >= float(min_command_um_for_g_obs)
        and math.isfinite(float(g_obs))
        and float(g_obs) < 0.0
    ):
        return True
    return False


def measured_delta_and_gain(
    *,
    previous_measured_um: float | None,
    measured_um: float | None,
    command_um: float,
) -> tuple[float | None, float]:
    """用尚未被覆盖的上一位置计算本轮视觉位移与观测增益。"""

    if previous_measured_um is None or measured_um is None:
        return None, float("nan")
    delta_um = float(measured_um) - float(previous_measured_um)
    gain = delta_um / float(command_um) if float(command_um) != 0.0 else float("nan")
    return delta_um, gain


def closed_loop_final_residual_um(
    target_um: float, final_measured_um: float | None
) -> float | None:
    """返回目标位置减最终实测位置；没有最终测量时返回 None。"""

    if final_measured_um is None:
        return None
    return float(target_um) - float(final_measured_um)


def command_for_error(
    error_um: float,
    *,
    kp: float | None = None,
    max_correction_um: float | None = None,
    min_command_um: float | None = None,
) -> float:
    """
    由剩余误差算出这一轮要发的笛卡尔修正量（μm）。

    输入：误差（μm，**机器人测试轴正向**坐标系）、Kp、限幅、死区。
    输出：有符号修正量（μm）。死区内或非法输入返回 0.0。

    控制律就是操作者指定的 command = Kp · error，这里不做 PID、不做滤波、
    不做死区补偿、不做自适应。

    ★ 本函数**不接收 sign，也不得接收**。理由不是风格，是这里曾经有一个
    把 sign 用了两次的真实缺陷：

        视觉位置 --(×sign)--> axis_measured_um --(target − measured)--> error
                 --(×sign again)--> command          ← 第二次，错

    轴向符号（视觉正方向是否等于机器人正方向）必须在**唯一一处**视觉测量
    入口（axis_measured_um / axis_positions_um）里用掉，之后 error、
    measured_delta、observed gain、以及本函数算出的 command 全部已经是
    机器人测试轴坐标系。到这里再乘一次 sign，只会在 sign = −1 的相机安装
    （例如机器人 +Y 在图像上是 −y）下把方向重新翻反。

    这个错误的**实际后果（按代码逐轮推演并跑过模拟，不是推测）**：以
    Kp=1、sign=−1、目标 +50 μm 为例，误差按 `err_{k+1} = 2·err_k` 翻倍——
    50 → 100 → 200 → 300 → 400 → 500 μm，机器人 6 轮内跑到 −550 μm 之外，
    结束状态是 **DIVERGING**（并伴随轴向漂移告警，但走位量级还够不到
    1500 μm 的中止线）。

    ★ 它**不会**被判成 SIGN_INVERTED。方向判据比较的是"指令"与"实测位移"，
    而在双符号下这两者**恒同号**：指令被翻反，视觉读数也被同一个翻反的
    sign 归一回来，两边自洽，g_obs 恒为 +1。也就是说这个缺陷**不会被方向
    联锁抓住**，只会以"闭环发散、残差巨大"的形式出现在报告里——看起来像是
    "这台 UR10 连 50 μm 都逼近不了"，而不是像"代码写错了"。

    sign = +1 时它更是**完全不可见**（乘 1 是恒等），所以只有在装反了相机
    的那台机器上才会暴露，而暴露出来的症状还会被误读成机械性能问题。
    把 sign 从签名里删掉，是为了让"再乘一次"在类型层面就写不出来。

    死区是必需的，不是调参：robot.validate_trajectory 拒绝 length <= 1e-6 m
    （"两点位置重合"）。而收敛时 |error| 可能只有 1 μm，Kp=1 时命令 1 μm，
    正好撞在这条硬线上——那会在一个目标即将成功的时刻把整轮弄死。
    """

    kp = float(config.MICRO_LOOP_KP) if kp is None else float(kp)
    max_correction_um = (
        float(config.MICRO_LOOP_MAX_CORRECTION_UM)
        if max_correction_um is None
        else float(max_correction_um)
    )
    min_command_um = (
        float(config.MICRO_LOOP_MIN_COMMAND_UM)
        if min_command_um is None
        else float(min_command_um)
    )

    if not math.isfinite(float(error_um)):
        return 0.0

    raw = kp * float(error_um)
    clamped = max(-max_correction_um, min(max_correction_um, raw))
    if abs(clamped) < min_command_um:
        return 0.0
    return clamped


def directional_motion_confirmed(
    *,
    direction: int,
    amplitude_um: float,
    final_measured_um: float | None,
    sigma_um: float,
    min_fraction: float | None = None,
) -> tuple[bool, str]:
    """
    确认机器人**确实朝目标方向累计走过**了幅值的一个可观比例。

    输入：目标方向（+1/−1）、目标幅值、最终实测位置（机器人轴正向，μm）、
          本组噪声、要求的最小比例（默认取配置）。
    输出：(是否确认, 说明文字)。

    为什么必须有这道门：5 μm 档最危险的错误不是"没收敛"，而是**完全没动却判成功**。
    当有效容差已经和幅值同量级（tol_eff ≥ A）时，"误差落在容差内"几乎不构成
    证据——机器人一步不走，误差恰好就是 A，也满足 |error| ≤ tol_eff。
    这时若直接报 CONVERGED，等于把"视觉分辨不出这么小的位移"包装成"逼近成功"。

    方向必须用 `direction` 归一后再比：目标在 −50 μm 时机器人合法地向 −X 走，
    未归一化的位移是负的，会被误判成"没有朝目标方向运动"。
    这也是 sign=−1 时最容易被写错的一处。

    用**净位移**而不是每一步位移的绝对值之和：绝对值之和不区分方向，
    朝左走 10 μm 再朝右走 10 μm 会被它算成"走了 20 μm"并据此确认收敛。
    净位移对零均值噪声免疫，它问的正是"最后到底过去了没有"。
    """

    if final_measured_um is None or not math.isfinite(float(final_measured_um)):
        return False, "没有可用的最终位置"

    fraction = (
        float(config.MICRO_LOOP_MIN_DIRECTIONAL_FRACTION)
        if min_fraction is None
        else float(min_fraction)
    )
    # 已朝目标方向走过的净位移（μm）。两个量都在机器人轴帧里，
    # 所以这里的乘法就是"投影到目标方向"。
    net_um = float(direction) * float(final_measured_um)
    # 门槛还要再抬到噪声之上：比一个 σ 还小的"位移"和没动在统计上不可分，
    # 拿它当方向性证据就是自欺。
    required_um = max(0.0, fraction) * abs(float(amplitude_um))
    noise_um = float(sigma_um)
    if math.isfinite(noise_um) and noise_um > 0.0:
        required_um = max(required_um, noise_um)
    if net_um >= required_um:
        return True, f"朝目标方向净位移 {net_um:.2f} μm（要求 ≥ {required_um:.2f} μm）"
    return (
        False,
        f"朝目标方向净位移仅 {net_um:.2f} μm，低于门槛 {required_um:.2f} μm"
        f"（幅值 {abs(float(amplitude_um)):g} μm 的 {fraction:.0%} 与噪声底取大者）",
    )


def effective_tolerance_um(tol_um: float, noise_um: float) -> float:
    """
    把标称容差放宽到噪声底之上，返回实际使用的收敛容差。

    输入：配置容差、本组实测测量噪声（μm）。
    输出：有效容差（μm）。

    实验作用：如果一次闭环测量的不确定度本身就有 4 μm，那么"误差 ≤ 3 μm"
    这个判据只是在筛噪声，不是判收敛。取 mode 决定是绝对容差还是随噪声放宽。
    """

    tol_um = float(tol_um)
    noise_um = float(noise_um)
    if not math.isfinite(noise_um) or noise_um <= 0.0:
        return tol_um
    if str(config.MICRO_LOOP_POSITION_TOL_MODE) == "absolute":
        return tol_um
    return max(tol_um, float(config.MICRO_LOOP_NOISE_SIGMA_MULT) * noise_um)


def median_of_tail(values: Sequence[float], times_ns: Sequence[int], window_s: float) -> float | None:
    """
    取"最后 window_s 时间内有效值"的中位数。

    输入：值与对应时间戳（同长）、尾部窗口秒数。
    输出：中位数；窗口内没有值则 None。

    实验作用：操作者明确要求用"最后一段有效数据的中位数"而不是单帧。
    单帧既受噪声影响，又可能正好落在一次偶发误检上；中位数对两者都稳。
    按时间戳而不是"最后 N 帧"取窗口，是因为相机实测会零星丢帧，
    "最后 N 帧"在丢帧时对应的真实时长会变。
    """

    if not values or not times_ns or len(values) != len(times_ns):
        return None
    cutoff_ns = int(times_ns[-1]) - int(float(window_s) * 1_000_000_000)
    tail = [float(v) for v, t in zip(values, times_ns) if int(t) >= cutoff_ns]
    if not tail:
        return None
    return float(np.median(tail))


def expected_frames(record_start_ns: int, record_stop_ns: int, fps: float) -> int:
    """
    按本段真实的录制起止时间和实际帧率推算应有多少帧。

    输入：录制起、止的 host_ns，相机实际帧率。
    输出：预期帧数。

    实验作用：不假设所有片段一样长。每段都记自己的起止时间，
    "实际帧数 / 预期帧数"才是有意义的成对指标。
    """

    if fps <= 0.0:
        return 0
    span_s = (int(record_stop_ns) - int(record_start_ns)) / 1_000_000_000.0
    if span_s <= 0.0:
        return 0
    return int(round(span_s * float(fps)))


def estimate_window_bytes(width: int, height: int, fps: float, seconds: float) -> int:
    """
    估算一段全幅 Mono8 未压缩 RAW 的字节数。

    输入：宽、高、帧率、秒数。
    输出：字节数（1 字节/像素）。

    实验作用：磁盘保护按峰值而不是稳态核算——同时可能存在的
    有一段正在录、有上一段还没删掉。
    """

    return int(round(float(width) * float(height) * float(fps) * float(seconds)))


def disk_free_bytes(path: Path | str) -> int:
    """
    返回给定路径所在卷的可用字节数。

    输入：任意路径，**不要求它已经存在**。
    输出：字节数。

    磁盘检查必须发生在创建任何目录之前，而 OUTPUT_ROOT 可能还没被建出来，
    所以这里向上找到最近一个真实存在的祖先再问文件系统。
    """

    probe = Path(path).resolve()
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return int(shutil.disk_usage(str(probe)).free)


def free_gb(path: Path | str) -> float:
    """返回给定路径所在卷的可用空间（GiB）。"""

    return disk_free_bytes(path) / (1024.0 ** 3)


def format_gib(value: int | float) -> str:
    """把字节数格式化成 GiB 字符串。"""

    return f"{float(value) / (1024.0 ** 3):.2f} GiB"


def available_ram_gb() -> float:
    """
    返回当前可用物理内存（GiB）。

    输入：无。
    输出：可用物理内存。取不到时返回 nan——**取不到不等于没有风险**，
    调用方必须把 nan 当作"未知"处理并如实记进日志，而不是当成"充足"。
    """

    try:
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return float("nan")
        return float(status.ullAvailPhys) / (1024.0 ** 3)
    except Exception:
        return float("nan")


# -----------------------------------------------------------------------------
# 运行预算：时长与体积
# -----------------------------------------------------------------------------

# 单帧棋盘格识别的实测耗时（秒）。本机实测 902 帧全幅 1936×1464：
# mean 163 ms / median 163 ms / max 191 ms。
MEASURE_FRAME_SECONDS = 0.165
# 单帧的全幅 Mono8 字节数（与分辨率无关的默认值，仅在相机未就绪时使用）。
FULL_FRAME_BYTES = int(config.MICRO_LOOP_FULL_FRAME_WIDTH) * int(
    config.MICRO_LOOP_FULL_FRAME_HEIGHT
)
# 落盘带宽（字节/秒）。实测全幅顺序写约 1.25 GB/s，这里保守取 600 MB/s，
# 估算偏保守不会把"放不下"算成"放得下"。
RAW_WRITE_BYTES_PER_SECOND = 600.0 * 1024.0 * 1024.0


def open_block_id(axis: str, mode: str, level_um: int) -> str:
    """
    开环实验块的目录名。

    输入：轴、模式、档位。
    输出：如 "X_open_seq_005um"。

    命名规则要能一眼分辨"哪个轴 / 哪种模式 / 哪个档位"，因为落地之后
    是几十个目录并列，靠序号认不出来谁是谁。
    """

    if str(mode) not in (OPEN_MODE_SEQ, OPEN_MODE_ALT):
        raise ValueError(f"未知开环模式 {mode!r}。")
    return f"{str(axis)}_open_{str(mode)}_{int(level_um):03d}um"


def closed_block_id(axis: str, level_um: int) -> str:
    """闭环实验块的目录名，如 "X_closed_005um"。"""

    return f"{str(axis)}_closed_{int(level_um):03d}um"


def block_dir(root: Path | str, block_id: str) -> Path:
    """
    取得（并在不存在时创建）一个实验块的落地目录。

    输入：根目录、块名。
    输出：块目录 Path。

    绝不覆盖已有内容：同名块目录若已存在就复用它，里面的文件按名字区分，
    不会静默删掉上一次的东西。
    """

    path = Path(root) / str(block_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def spanning_every_n(frame_count: int, target: int) -> int:
    """
    算一个抽帧步长，使整段里大约取到 target 个采样点，且跨满整段。

    输入：片段总帧数、想要的采样点数。
    输出：>= 1 的抽帧步长。

    为什么需要它：静止基线要的是"整段 5 s 里画面抖了多少"，
    如果只分析尾部 40 帧，等于把 4.7 s 的记录扔掉、只看最后 0.3 s，
    噪声底会被系统性低估。抽帧则能在同样的分析预算下跨满整段。
    被抽掉的帧仍在保留的 RAW 里，事后可以逐帧复查。
    """

    return max(1, int(round(int(frame_count) / max(1, int(target)))))


def estimate_run_plan(
    *,
    frame_bytes: int | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    """
    按当前配置算出整轮实验的窗口数、时长与数据量。

    输入：单帧字节数、帧率（默认取配置里的全幅尺寸与 EXPECTED_VISION_FPS）。
    输出：含各阶段窗口数、每窗口耗时分解、正常/最坏总时长与总字节数的字典。

    为什么必须在启动前算：本轮每一步都要落一段未压缩 RAW，
    总时长几乎完全由"分析多少帧 × 0.165 s"决定（机械等待只占约 4%）。
    估不出来就没法在空间或时间不够时**在动手之前**拒绝启动——
    而"实验跑一半才发现放不下"只能靠事后删数据收场，那正是被明令禁止的。

    正常与最坏的差别只有两处：闭环是否跑满迭代预算、以及稳定等待是否超时。
    这两处都按**最坏**算第二组数，所以"最坏"是真正的上界。
    """

    frame_bytes = int(frame_bytes or FULL_FRAME_BYTES)
    fps = float(fps or config.EXPECTED_VISION_FPS or 132.0)

    axes = list(config.MICRO_LOOP_AXES)
    levels = list(config.MICRO_LOOP_AMPLITUDES_UM)
    modes = list(config.MICRO_LOOP_OPEN_MODES)

    # 每个开环块的步数按模式分别算，不写死 6——
    # 步数是配置项，估算跟着配置走才不会在改配置后失真。
    steps_per_open_block = {
        mode: len(
            open_loop_steps(axis=axes[0], mode=mode, level_um=levels[0])
        )
        for mode in modes
    }
    open_blocks = len(axes) * len(modes) * len(levels)
    open_steps = sum(
        steps_per_open_block[mode] * len(levels) * len(axes) for mode in modes
    )

    # 闭环目标数**从 build_targets() 取**，不自己再算一遍。
    # 上一版这里写的是 len(axes) * len(levels) = 6，漏了**方向**这一维：
    # 正式闭环是 2 轴 × 3 档 × 2 方向 = 12 个目标（X:+5,−5,+20,−20,+50,−50
    # 然后 Y 同样）。把 12 当成 6，时长、RAW 数据量、磁盘检查、UI 提示
    # 就全部只按一半的量在算——估算与实际状态机循环数对不上。
    # 直接复用 build_targets() 而不是写 len(axes)*len(levels)*len(DIRECTIONS)，
    # 是为了让"实际会跑多少个目标"只有一个定义处，以后加方向或改过滤条件
    # 时估算跟着自动走。
    closed_targets = len(build_targets(axes=axes, amplitudes_um=levels))
    closed_worst = closed_targets * int(config.MICRO_LOOP_MAX_ITER)
    # 收敛后的零命令验证：每个目标跑 **1 次**。VERIFY_MAX_ATTEMPTS 是"验证没通过
    # 才重开闭环"的重试上限，把它乘进窗口数等于假设每个目标都验证失败并重试到底——
    # 那是"闭环根本没收敛"的场景，此时该看的是收尾逻辑而不是时长预算。
    # 按 1 次算，最坏估算是可达上界；剩下的重试余量由时间预算兜住。
    closed_verify_worst = closed_targets
    # 正常情况按"每个目标 4 轮命中收敛/停滞"估：以本轮 3 个档位的定位，
    # 5 μm 走不动会早退，20/50 μm 通常 2–3 轮到位。
    closed_normal = closed_targets * 4
    closed_verify_normal = closed_targets

    static_windows = 1
    probe_motion_windows = 2 * len(axes)  # 每轴正反各一段
    probe_remeasure_windows = (
        2 * len(axes) * int(config.MICRO_LOOP_PROBE_REMEASURE_ATTEMPTS)
    )

    pre_s = float(config.MICRO_LOOP_PRE_SECONDS)
    post_s = float(config.MICRO_LOOP_POST_SECONDS)

    def window_seconds(*, worst: bool, motion_s: float | None = None) -> float:
        """一段窗口的真实时长：前段 + 运动与稳定 + 后段。"""

        if motion_s is None:
            if worst:
                motion_s = (
                    float(config.MICRO_LOOP_STEP_TIMEOUT_S)
                    + float(config.MICRO_LOOP_POST_STABLE_TIMEOUT_S)
                    + float(config.MICRO_LOOP_MOTION_SETTLE_SECONDS)
                )
            else:
                # 实测：motion_in_progress 的 200 ms 宽限 + 稳定保持 0.3 s。
                motion_s = 0.2 + float(config.MICRO_LOOP_STABLE_HOLD_SECONDS)
        return pre_s + float(motion_s) + post_s

    # 探针单独算：它走 MICRO_LOOP_PROBE_UM = 1000 μm，是微动步的 20~200 倍，
    # 用微动的超时会必然超时（见 config 里 MICRO_LOOP_PROBE_TIMEOUT_S 的说明）。
    probe_motion_normal = (
        float(config.MICRO_LOOP_PROBE_UM) / 1000.0 / float(config.MICRO_LOOP_SPEED_MM_S)
        + 0.4
    )
    probe_motion_worst = float(config.MICRO_LOOP_PROBE_TIMEOUT_S) + float(
        config.MICRO_LOOP_POST_STABLE_TIMEOUT_S
    ) + float(config.MICRO_LOOP_MOTION_SETTLE_SECONDS)

    seconds_normal = window_seconds(worst=False)
    seconds_worst = window_seconds(worst=True)
    probe_normal = window_seconds(worst=False, motion_s=probe_motion_normal)
    probe_worst = window_seconds(worst=False, motion_s=probe_motion_worst)

    def part(count: int, per_window_seconds: float, analyze_frames: int) -> dict[str, float]:
        record_s = count * per_window_seconds
        bytes_ = int(round(count * per_window_seconds * fps)) * frame_bytes
        write_s = bytes_ / RAW_WRITE_BYTES_PER_SECOND
        analyze_s = count * analyze_frames * MEASURE_FRAME_SECONDS
        return {
            "windows": count,
            "record_s": record_s,
            "write_s": write_s,
            "analyze_s": analyze_s,
            "total_s": record_s + write_s + analyze_s,
            "bytes": bytes_,
        }

    measure_frames = int(config.MICRO_LOOP_MEASURE_FRAMES)
    reference_frames = int(config.MICRO_LOOP_REFERENCE_FRAMES)
    probe_measure_frames = int(config.MICRO_LOOP_PROBE_MEASURE_FRAMES)
    probe_reference_frames = int(config.MICRO_LOOP_PROBE_REFERENCE_FRAMES)
    # 建零点要**扫两遍**：第一遍挑 medoid 帧、第二遍让正式 tracker 锁存参考并定零点。
    prime_frames = 2 * reference_frames
    static_frames = int(config.MICRO_LOOP_STATIC_ANALYZE_FRAMES)
    # 静止基线要跨满整段 5 s，所以它的分析量是"扫一遍抽帧 + 一次测量"。
    static_analyze = static_frames + measure_frames
    reference_analyze = prime_frames + measure_frames
    # 闭环每个 (轴, 幅值) 组都要新建一次视觉参考（组内 ±A 共用同一个零点，
    # 这样反向时的死区/回差才真正被测到，而不是被"重新定义零点"抹掉）。
    group_reference_windows = len(axes) * len(levels)

    probe_motion_part = part(
        probe_motion_windows, probe_normal, probe_measure_frames
    )
    # 探针参考帧在 prime 时扫两遍：先选 medoid，再以它建立正式零点。
    probe_reference_part = part(
        1,
        float(config.MICRO_LOOP_PROBE_REFERENCE_SECONDS),
        2 * probe_reference_frames,
    )
    probe_part = {
        key: probe_motion_part[key] + probe_reference_part[key]
        for key in probe_motion_part
    }

    parts_normal = {
        "static": part(
            static_windows, float(config.MICRO_LOOP_STATIC_SECONDS), static_analyze
        ),
        "probe": probe_part,
        "group_reference": part(
            group_reference_windows,
            float(config.MICRO_LOOP_REFERENCE_SECONDS),
            reference_analyze,
        ),
        "open_loop": part(open_steps, seconds_normal, measure_frames),
        "closed_loop": part(closed_normal, seconds_normal, measure_frames),
        "closed_verify": part(closed_verify_normal, seconds_normal, measure_frames),
    }
    parts_worst = {
        "static": dict(parts_normal["static"]),
        "probe": dict(parts_normal["probe"]),
        "group_reference": dict(parts_normal["group_reference"]),
        "open_loop": part(open_steps, seconds_worst, measure_frames),
        "closed_loop": part(closed_worst, seconds_worst, measure_frames),
        "closed_verify": part(closed_verify_worst, seconds_worst, measure_frames),
    }
    # 最坏情形允许每个正/反探针都用满 5 次原地复测后才合格。
    # 每次复测前的机器人稳定等待发生在相机窗口外，因此单独计入时间；
    # 复测窗口本身只录 MOTION_SETTLE_SECONDS，不发送任何运动命令。
    probe_remeasure = part(
        probe_remeasure_windows,
        float(config.MICRO_LOOP_MOTION_SETTLE_SECONDS),
        probe_measure_frames,
    )
    probe_remeasure_stabilize_s = probe_remeasure_windows * float(
        config.MICRO_LOOP_STABLE_TIMEOUT_S
    )
    probe_remeasure["stabilize_s"] = probe_remeasure_stabilize_s
    probe_remeasure["total_s"] += probe_remeasure_stabilize_s
    parts_worst["probe_remeasure"] = probe_remeasure
    # 参考/静止片段本身是静止录制，时长不随稳定超时变化，所以两档共用同一份。
    for name in ("static", "group_reference"):
        parts_worst[name]["bytes"] = parts_normal[name]["bytes"]

    # 固定开销：机器人连接与安全自检、相机开启与预热、参考建立、回位、写汇总。
    # 这些与窗口数无关，用实测量级的常数而不是拍脑袋的百分比。
    setup_s = (
        5.0                                  # 机器人连接 + 安全状态检查
        + 3.0 + float(config.MICRO_LOOP_CAMERA_WARMUP_SECONDS)  # MVS 节点 + 预热
        + 3.0                                # 探针参考建立（一次 prime）
        + 6.0                                # 返回初始位姿 + 写汇总
    )

    def total(parts: dict[str, dict[str, float]]) -> float:
        return setup_s + sum(item["total_s"] for item in parts.values())

    bytes_normal = sum(item["bytes"] for item in parts_normal.values())
    bytes_worst = sum(item["bytes"] for item in parts_worst.values())

    # RAW 在每组结果落盘后立即删除，因此启动门禁按“最大同时驻盘组”而不是
    # 整轮累计生成量计算。闭环一组包含参考、正负两个目标的迭代与验证；
    # 开环一组是一个固定步序列；探针组还包含最坏情况下的全部原地复测。
    closed_groups = max(1, len(axes) * len(levels))
    peak_group_bytes_worst = max(
        int(parts_worst["static"]["bytes"]),
        int(parts_worst["probe"]["bytes"] + parts_worst["probe_remeasure"]["bytes"]),
        int(parts_worst["open_loop"]["bytes"] / max(1, open_blocks)),
        int(
            (
                parts_worst["group_reference"]["bytes"]
                + parts_worst["closed_loop"]["bytes"]
                + parts_worst["closed_verify"]["bytes"]
            )
            / closed_groups
        ),
    )

    # ---- 时间与数据量都是**信息**，不是截止条件 ----
    # 正常与最坏这两个数都是"设计算术值"：窗口数由实验设计固定，每窗口又要
    # 串行地"录像 → 分析 → 落盘"（闭环必须等测量结果才能决定下一条命令，
    # 这一步无法并行），所以整轮必然是十几分钟量级。
    #
    # 上一版在这里还算了一个"硬上限"（时间预算 + 一个最坏窗口 + 收尾），
    # 运行时靠它强制收尾。那个策略已按操作者要求删除：机器人、相机、磁盘
    # 都正常时，不能因为"到 10 分钟了"就跳过剩余目标。所以这里只报
    # 正常预计与设计最坏两个数，供启动前知情，**不产生任何截断**。
    wrap_up_s = 6.0            # 安全停臂 + 返回初始位姿 + 写汇总

    return {
        "kind": "MICRO_CLOSED_LOOP_BUDGET",
        "fps": fps,
        "frame_bytes": frame_bytes,
        "measure_frames": measure_frames,
        "open_blocks": open_blocks,
        "open_steps": open_steps,
        "steps_per_open_block": steps_per_open_block,
        # 闭环目标数 = 2 轴 × 3 档 × 2 方向 = 12。由 build_targets() 得出，
        # 见上面关于"上一版漏了方向这一维"的说明。
        "closed_targets": closed_targets,
        "closed_steps_max": closed_worst,
        "windows_normal": int(sum(item["windows"] for item in parts_normal.values())),
        "windows_worst": int(sum(item["windows"] for item in parts_worst.values())),
        "window_seconds_normal": seconds_normal,
        "window_seconds_worst": seconds_worst,
        "setup_s": setup_s,
        "parts_normal": parts_normal,
        "parts_worst": parts_worst,
        "seconds_normal": total(parts_normal),
        "seconds_worst": total(parts_worst),
        "bytes_normal": bytes_normal,
        "bytes_worst": bytes_worst,
        "peak_group_bytes_worst": peak_group_bytes_worst,
        "wrap_up_s": wrap_up_s,
        # 时长提示线：运行中超过它只打一条日志，不中止实验。
        "time_notice_s": float(config.MICRO_LOOP_TIME_NOTICE_S),
    }


def estimate_multi_target_run_plan(
    *, frame_bytes: int | None = None, fps: float | None = None
) -> dict[str, Any]:
    """估算当前 12 个多目标闭环的时长和逐窗口 RAW 峰值。"""

    frame_bytes = int(frame_bytes or FULL_FRAME_BYTES)
    fps = float(fps or config.EXPECTED_VISION_FPS or 132.0)
    targets = len(build_multi_target_sequence())
    normal_iterations = targets * 5
    worst_iterations = targets * int(config.MICRO_TARGET_MAX_ITER)
    analyze_per_iteration_s = (
        2 * int(config.MICRO_LOOP_MEASURE_FRAMES) * MEASURE_FRAME_SECONDS
    )
    pre_analysis_s = int(config.MICRO_LOOP_MEASURE_FRAMES) * MEASURE_FRAME_SECONDS
    record_normal_s = (
        float(config.MICRO_LOOP_PRE_SECONDS)
        + pre_analysis_s
        + 0.2
        + float(config.MICRO_LOOP_STABLE_HOLD_SECONDS)
        + float(config.MICRO_LOOP_POST_SECONDS)
    )
    record_worst_s = (
        float(config.MICRO_LOOP_PRE_SECONDS)
        + pre_analysis_s
        + float(config.MICRO_LOOP_PRE_STABLE_TIMEOUT_S)
        + float(config.MICRO_LOOP_MOTION_SETTLE_SECONDS)
        + float(config.MICRO_LOOP_STEP_TIMEOUT_S)
        + float(config.MICRO_LOOP_POST_STABLE_TIMEOUT_S)
        + float(config.MICRO_LOOP_MOTION_SETTLE_SECONDS)
        + float(config.MICRO_LOOP_POST_SECONDS)
    )

    def iteration_total(count: int, record_s: float) -> tuple[float, int]:
        bytes_ = int(round(count * record_s * fps)) * frame_bytes
        write_s = bytes_ / RAW_WRITE_BYTES_PER_SECOND
        analyze_s = count * analyze_per_iteration_s
        return count * record_s + write_s + analyze_s, bytes_

    iteration_normal_s, iteration_normal_bytes = iteration_total(
        normal_iterations, record_normal_s
    )
    iteration_worst_s, iteration_worst_bytes = iteration_total(
        worst_iterations, record_worst_s
    )
    # 探针、静止基线、实验原点及两次回位的保守固定预算。
    setup_normal_s = 5.0 + 4 * 5.0 + 12.0 + 8.0
    setup_worst_s = 8.0 + 24 * 5.0 + 15.0 + 12.0
    setup_bytes = int(round((5.0 + 5.0 + 1.0) * fps)) * frame_bytes
    probe_worst_bytes = int(round(25.0 * fps)) * frame_bytes
    snapshot_bytes = int(config.MICRO_LOOP_MEASURE_FRAMES) * frame_bytes
    peak_iteration_bytes = int(round(record_worst_s * fps)) * frame_bytes + snapshot_bytes
    return {
        "kind": "MICRO_MULTI_TARGET_BUDGET",
        "fps": fps,
        "frame_bytes": frame_bytes,
        "measure_frames": int(config.MICRO_LOOP_MEASURE_FRAMES),
        "closed_targets": targets,
        "closed_steps_max": worst_iterations,
        "windows_normal": normal_iterations + 7,
        "windows_worst": worst_iterations + 27,
        "window_seconds_normal": record_normal_s,
        "window_seconds_worst": record_worst_s,
        "setup_s": setup_normal_s,
        "seconds_normal": setup_normal_s + iteration_normal_s,
        "seconds_worst": setup_worst_s + iteration_worst_s,
        "bytes_normal": setup_bytes + iteration_normal_bytes,
        "bytes_worst": probe_worst_bytes + setup_bytes + iteration_worst_bytes,
        "peak_group_bytes_worst": max(peak_iteration_bytes, probe_worst_bytes),
        "time_notice_s": float(config.MICRO_LOOP_TIME_NOTICE_S),
    }


def format_multi_target_budget_lines(plan: dict[str, Any]) -> list[str]:
    """把多目标闭环预算格式化为启动日志/UI 文本。"""

    return [
        f"实验规模：X 6 个目标 + Y 6 个目标，共 {plan['closed_targets']} 个目标；"
        f"每目标最多 {int(config.MICRO_TARGET_MAX_ITER)} 次迭代",
        "目标序列（每轴绝对位置）：0 → 100 → 250 → 460 → 360 → 210 → 0 μm；"
        "无斜向目标",
        f"每次迭代同一窗口采集 before/command/after；before 与 after 各分析 "
        f"{plan['measure_frames']} 个有效帧",
        f"预计时长：正常 {plan['seconds_normal'] / 60.0:.1f} 分钟，"
        f"设计最坏 {plan['seconds_worst'] / 60.0:.1f} 分钟；时长只提示，"
        "不中止、不跳过目标",
        f"预计累计临时 RAW：正常 {format_gib(int(plan['bytes_normal']))}，"
        f"最坏 {format_gib(int(plan['bytes_worst']))}；每次迭代结果落盘后立即删除",
        f"预计最大同时驻盘 {format_gib(int(plan['peak_group_bytes_worst']))}，"
        "正常结束、停止和异常退出均有兜底清理",
    ]


def format_budget_lines(plan: dict[str, Any]) -> list[str]:
    """
    把预算字典整理成几行可直接打印的中文说明。

    输入：estimate_run_plan() 的结果。
    输出：字符串列表（每行一条），供启动日志与 UI 直接显示。
    """

    def minutes(seconds: float) -> str:
        return f"{seconds / 60.0:.1f} 分钟"

    def gib(bytes_: int) -> str:
        return format_gib(bytes_)

    per_mode = "、".join(
        f"{mode} {count} 步"
        for mode, count in sorted(plan["steps_per_open_block"].items())
    )
    max_iter = plan["closed_steps_max"] // max(1, plan["closed_targets"])
    analyze_s = plan["measure_frames"] * MEASURE_FRAME_SECONDS

    lines = [
        f"实验规模：开环 {plan['open_blocks']} 块（{per_mode}）共 "
        f"{plan['open_steps']} 步；闭环 {plan['closed_targets']} 个目标"
        f"（2 轴 × 3 档 × 正负 2 方向）× 最多 {max_iter} 轮，"
        f"加收敛验证最多 {plan['closed_targets']} 段",
        f"共 {plan['windows_normal']} 段（最坏 {plan['windows_worst']} 段）；"
        f"单段窗口约 {plan['window_seconds_normal']:.2f} s"
        f"（超时情形 {plan['window_seconds_worst']:.2f} s），"
        f"每段在线分析 {plan['measure_frames']} 帧 × "
        f"{MEASURE_FRAME_SECONDS * 1000:.0f} ms = {analyze_s:.1f} s",
        f"预计时长：正常 {minutes(plan['seconds_normal'])}"
        f"（含固定开销 {plan['setup_s']:.0f} s），"
        f"设计最坏 {minutes(plan['seconds_worst'])}",
        # 这几行是**信息**，不是承诺也不是截止线。上一版这里写的是"硬上限…
        # 超时即安全收尾"，已被操作者明确取消：正常跑完全部实验优先于压时长。
        f"时长提示线 {minutes(plan['time_notice_s'])}：只是提示，"
        "**到点不中止实验、不跳过任何目标**；"
        "只要机器人/相机/磁盘正常，12 个闭环目标与 12 个开环块全部跑完",
        f"预计原始数据：正常 {gib(plan['bytes_normal'])}，"
        f"最坏 {gib(plan['bytes_worst'])}（未压缩 Mono8，不裁剪不压缩）",
        f"视频按组落盘后立即删除；预计最大单组驻盘 "
        f"{gib(plan['peak_group_bytes_worst'])}（任务结束/停止另有兜底清理）",
    ]
    return lines


def decide_raw_retention(
    plan: dict[str, Any], raw_root: Path | str
) -> dict[str, Any]:
    """
    决定这次能不能全量保留原始实验块。

    输入：预算字典、原始数据根目录。
    输出：含 keep_all / reason / free_gb / needed_bytes 的字典。

    规则（全部在创建任何目录之前算完）：
    - 只要最坏估算 + 余量放得下，就全量保留；
    - 放不下时**降级为"静止基线 + 每块代表步"**，并在日志里明写原因，
      绝不在实验结束后静默删除；
    - 连降级后的量都放不下，由调用方拒绝启动。
    """

    reserve = float(config.MICRO_LOOP_KEEP_ALL_RESERVE_GB) * (1024.0 ** 3)
    needed = int(plan["bytes_worst"])
    free = disk_free_bytes(raw_root)
    keep_all = bool(config.MICRO_LOOP_KEEP_ALL_RAW) and (needed + reserve) <= free

    if keep_all:
        reason = "空间充足，全量保留每个实验块的原始 RAW"
    elif not bool(config.MICRO_LOOP_KEEP_ALL_RAW):
        reason = "配置为 MICRO_LOOP_KEEP_ALL_RAW=False，只保留静止基线与每块代表步"
    else:
        reason = (
            f"空间不足：最坏需 {format_gib(needed)} + 余量 "
            f"{format_gib(reserve)}，而该卷只剩 {format_gib(free)}；"
            "降级为“静止基线 + 每块代表步”"
        )

    # 降级后的量：静止基线整段 + 每个实验块 1 步代表窗口。
    blocks = int(plan["open_blocks"]) + int(plan["closed_targets"])
    representative_bytes = int(plan["parts_normal"]["static"]["bytes"]) + int(
        round(
            blocks
            * plan["window_seconds_normal"]
            * float(plan["fps"])
            * int(plan["frame_bytes"])
        )
    )
    return {
        "kind": "MICRO_CLOSED_LOOP_RAW_RETENTION",
        "keep_all": keep_all,
        "reason": reason,
        "free_bytes": free,
        "free_gb": free / (1024.0 ** 3),
        "needed_bytes": needed,
        "needed_gb": needed / (1024.0 ** 3),
        "representative_bytes": representative_bytes,
        "representative_gb": representative_bytes / (1024.0 ** 3),
        "reserve_gb": float(config.MICRO_LOOP_KEEP_ALL_RESERVE_GB),
        "raw_root": str(raw_root),
    }


def allocate_closed_loop_run(
    output_root: Path, now: datetime | None = None
) -> tuple[str, Path]:
    """
    创建本次运行的输出目录，绝不覆盖已有目录。

    输入：输出根目录、可选时间戳。
    输出：(运行 id, 运行目录)。目录已存在时依次追加 _02、_03。
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    run_id = f"micro_motion_{stamp}"
    candidate = output_root / run_id
    counter = 2
    while candidate.exists():
        run_id = f"micro_motion_{stamp}_{counter:02d}"
        candidate = output_root / run_id
        counter += 1
    candidate.mkdir(parents=True)
    return run_id, candidate


def target_dir(run_dir: Path, axis: str, amplitude_um: int) -> Path:
    """目标组目录，例如 <run>/X/005um。"""

    return Path(run_dir) / str(axis) / f"{int(amplitude_um):03d}um"


def direction_slug(direction: int) -> str:
    """方向对应的文件名前缀：+1 → positive，−1 → negative。"""

    return "positive" if int(direction) > 0 else "negative"


def write_rows_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    *,
    overwrite: bool = True,
) -> None:
    """写一个通用 CSV；实验中途反复重写，以便崩溃后仍留有已完成的数据。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8-sig", newline="", buffering=65_536) as file:
        writer = csv.DictWriter(file, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


def read_rows_csv(path: Path) -> list[dict[str, str]]:
    """读取一个通用 CSV，返回字符串行。"""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_json(path: Path, payload: Any) -> None:
    """写一个 JSON（先落盘再返回，供崩溃后仍可读）。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True, default=str)
    with path.open("w", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())


def append_log(path: Path, text: str) -> None:
    """向实验日志追加一行（带时间戳，落盘）。"""

    stamp = datetime.now().strftime("%H:%M:%S")
    with Path(path).open("a", encoding="utf-8", buffering=1) as file:
        file.write(f"[{stamp}] {text}\n")


def _format_pose(pose: Any) -> str:
    """把一个 6 维 TCP 位姿写成 CSV 里的单个字段（分号分隔，9 位小数）。"""

    if pose is None:
        return ""
    try:
        return ";".join(f"{float(value):.9f}" for value in pose)
    except (TypeError, ValueError):
        return ""


def mad_sigma(values: Sequence[float]) -> float:
    """用 MAD 估计标准差（1.4826 × 中位绝对偏差），对离群点比 std 稳。"""

    data = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if data.size == 0:
        return float("nan")
    return float(np.median(np.abs(data - np.median(data))) * 1.4826)


def estimate_measurement_sigma_um(
    values: Sequence[float], tail_frames: int, *, every_n: int = 1
) -> float:
    """
    用"闭环自己的估计量"实测一次测量的不确定度。

    输入：参考片段的合格帧位移序列（μm）、尾部窗口应有帧数、抽帧步长。
    输出：一次闭环测量的标准差估计（μm）。

    做法：对整条序列做块自助——把每一个可能位置的"尾部窗口中位数"都算出来，
    取这些中位数的标准差。这正是闭环每一步真正在做的事，所以这个数才是
    闭环的测量不确定度，而不是单帧噪声。用实测而不是假设，
    是因为它直接决定 POSITION_TOL 能不能被判据满足。
    """

    data = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    window = max(1, int(tail_frames) // max(1, int(every_n)))
    if data.size < window + 1:
        return float("nan")
    windows = [float(np.median(data[i : i + window])) for i in range(data.size - window + 1)]
    return float(np.std(windows))


def no_progress_plateau(
    values: Sequence[float],
    *,
    tol_eff_um: float,
    flat_gate_um: float,
) -> bool:
    """
    判断一段误差序列是不是"卡住了、并且看不出还在改善"。

    输入：一段（按时间排序的）误差值、有效容差、平坦门限。
    输出：True 表示这是一个平台。

    为什么不能只看峰峰值相对中位幅值：CYCLE_WINDOW 被 MICRO_LOOP_MAX_ITER=6
    逼到只有 3 以后，窗内只剩 2 个差分，"每步降 2 μm、误差 20 μm"这种
    平滑下降的 ptp/|中位| 只有 0.2，和真正冻结的平台无法区分——会把
    "闭环正常工作、误差正在收敛"误报成 STALLED，正好把结论反过来。

    真正的区别在**趋势**而不是**幅度**：平台是"每一步变化都在门限以内、
    整段的净变化也在门限以内"；平滑下降则至少有一个差分超出门限。
    门限取 max(噪声, STALL_MIN_IMPROVE_UM)，即"改善必须大到能盖过噪声才算改善"，
    这正是用户说的"连续 2~3 次误差没有明显改善"。
    """

    if len(values) < 2:
        return False

    diffs = [b - a for a, b in zip(values, values[1:])]
    net = values[0] - values[-1]
    median_abs = float(np.median([abs(value) for value in values]))

    return (
        median_abs > tol_eff_um
        and abs(net) < flat_gate_um
        and all(abs(diff) < flat_gate_um for diff in diffs)
    )


def evaluate_termination(
    errors_um: Sequence[float],
    valid: Sequence[bool],
    *,
    tol_um: float,
    noise_um: float,
) -> str:
    """
    判断闭环该不该停，以及为什么停。

    输入：误差序列（第 0 项是迭代前的初始误差，之后每项是上一次迭代后的残差）、
          与它等长的有效标志、有效容差、本组测量噪声。
    输出：终止状态常量；返回 STILL_SHRINKING 表示继续迭代。

    判据不只是"换号 + 峰峰值不缩"。5 μm 档最可能出现的其实不是极限环，而是
    单边平台：控制器忽略了小于阈值的命令，误差于是冻结在一个恒定值上，
    **一次号都不换**——原始判据永远不会触发，最后报 MAX_ITER，而 MAX_ITER
    完全掩盖了"这里有一个最小有效运动尺度"这个结论。所以单边平台独立成桶。

    两道防早期误判的门（这才是判据能不能用的关键）：
    1. 先过噪声底：最近的误差已经低到噪声量级时，换号是噪声，不是极限环。
    2. 先过"无进展"：必须至少走到路程一半，才允许判极限环。否则
       "误差还很大但在单调收缩"会被早期误判成极限环。
    """

    cycle_window = int(config.MICRO_LOOP_CYCLE_WINDOW)
    min_sign_changes = int(config.MICRO_LOOP_CYCLE_MIN_SIGN_CHANGES)
    ptp_keep = float(config.MICRO_LOOP_CYCLE_PTP_KEEP)
    abs_keep = float(config.MICRO_LOOP_CYCLE_ABS_KEEP)
    converge_count = int(config.MICRO_LOOP_CONVERGE_COUNT)

    tol_eff = effective_tolerance_um(tol_um, noise_um)
    noise_floor = tol_eff

    pairs = [(index, float(value)) for index, value in enumerate(errors_um) if valid[index]]
    if not pairs:
        return TERMINAL_STILL_SHRINKING

    # 收敛判据：连续 converge_count 次落在有效容差内。
    tail = pairs[-converge_count:]
    if len(tail) == converge_count and all(abs(value) <= tol_eff for _, value in tail):
        return TERMINAL_CONVERGED

    # 早退：连续 stall_patience 次"误差几乎没有变小"就判停滞。
    # 这条必须排在 CYCLE_WINDOW 的长度门之前：本轮迭代上限只有 6 次，
    # 极限环/平台判据要攒够 2×CYCLE_WINDOW 个样本才启动（第 5 轮），
    # 那时预算已经快用完了——8 次以内想给出结论，必须有一条更早的门。
    # 门限 1 μm 是刻意留宽的：Kp=1 时正常闭环 1–2 步到位；即使有效增益只有 0.3，
    # 50 μm 目标的误差序列也是 50 → 35 → 24.5 → 17，每步改善远大于 1 μm，
    # 所以这条不会误伤"收敛得慢但确实在收敛"的目标。
    stall_patience = int(config.MICRO_LOOP_STALL_PATIENCE)
    # 平坦门限同时要盖过测量噪声和"明显改善"的门槛：改善小于噪声时，
    # 无论看到什么都无法断言它在动，"卡住"是唯一诚实的标签。
    flat_gate = max(float(noise_um), float(config.MICRO_LOOP_STALL_MIN_IMPROVE_UM))
    if len(pairs) >= stall_patience + 1 and no_progress_plateau(
        [value for _, value in pairs[-(stall_patience + 1) :]],
        tol_eff_um=tol_eff,
        flat_gate_um=flat_gate,
    ):
        return TERMINAL_STALLED

    if len(pairs) < 2 * cycle_window:
        return TERMINAL_STILL_SHRINKING

    recent_pairs = pairs[-cycle_window:]
    prior_pairs = pairs[-2 * cycle_window : -cycle_window]

    # 不允许跨测量缺口比较：中间有太多无效迭代时，前后两组根本不是同一段过程。
    span = recent_pairs[-1][0] - recent_pairs[0][0]
    if span > cycle_window + 4:
        return TERMINAL_STILL_SHRINKING

    recent = [value for _, value in recent_pairs]
    prior = [value for _, value in prior_pairs]
    first_error = pairs[0][1]

    # 门 1（对下面所有分支生效）：已经低到噪声量级时，换号就是噪声，不是现象。
    if min(abs(value) for value in recent) <= noise_floor:
        return TERMINAL_STILL_SHRINKING

    sign_changes = sum(1 for a, b in zip(recent, recent[1:]) if a * b < 0.0)
    ptp_recent = max(recent) - min(recent)
    ptp_prior = max(prior) - min(prior)
    abs_recent = float(np.median([abs(value) for value in recent]))
    abs_prior = float(np.median([abs(value) for value in prior]))

    # 发散的判据是"幅值在长大"，与"有没有走够路程"无关，所以必须排在门 2 之前。
    # 排在后面的话：发散序列的 min|recent| 恰好总是大于首误差的一半，
    # 会被门 2 当成 STILL_SHRINKING 吞掉，而这个状态是最该被报出来的。
    if abs_recent > 1.2 * abs_prior:
        return TERMINAL_DIVERGING

    if sign_changes >= min_sign_changes:
        # 门 2（只对极限环生效）：必须至少走到路程一半，才允许下极限环结论。
        # 这一道专门杀掉"误差还很大但在单调收缩、只是因为噪声换了一两次号"的早期误判。
        if abs(first_error) > 0.0 and min(abs(value) for value in recent) >= 0.5 * abs(first_error):
            return TERMINAL_STILL_SHRINKING
        # 真极限环：反复换号，且峰峰值与中位幅值都不再收缩。
        # 这两条同时把"增益略大于 1 但仍在收敛"的振荡排除掉：
        # 那种情况下 6 步包络会缩到 |1−Kp·g|^6（|1−Kp·g| = 0.85 时是 0.38），
        # 远低于 0.5 与 0.7，只有约等于单位环路增益的真极限环能存活。
        if ptp_recent >= ptp_keep * ptp_prior and abs_recent >= abs_keep * abs_prior:
            return TERMINAL_LIMIT_CYCLE
        return TERMINAL_STILL_SHRINKING

    # 单边平台：一次号都不换、窗口很窄、但幅值仍在容差之上——
    # 控制器忽略了小于某个阈值的命令，误差冻结在一个恒定值上。
    # **这一条不能受门 2 管辖**：停在 32 μm 而目标是 50 μm 时，
    # min|recent| 恰好大于首误差的一半，门 2 会把 5 μm 档最可能出现的这个结果
    # 一直压到 MAX_ITER，而 MAX_ITER 完全掩盖了"存在最小有效运动尺度"这个结论。
    if no_progress_plateau(recent, tol_eff_um=tol_eff, flat_gate_um=flat_gate):
        return TERMINAL_STALLED

    return TERMINAL_STILL_SHRINKING


def convergence_quality(
    errors_um: Sequence[float],
    valid: Sequence[bool],
    *,
    noise_um: float,
) -> tuple[float, float]:
    """
    判断"收敛"是真不动点还是运气好落进噪声里。

    输入：误差序列、有效标志、本组测量噪声。
    输出：(有符号残差 μm, 残差信噪比)。

    实验作用：残差信噪比低于 2 时，残差与零在统计上不可分，
    只能诚实地记作 CONVERGED_NOISE_LIMITED；信噪比高且残差大于容差，
    说明存在一个可重复的物理下限——那才是本实验最有价值的单个数。
    """

    converge_count = int(config.MICRO_LOOP_CONVERGE_COUNT)
    recent = [float(v) for i, v in enumerate(errors_um) if valid[i]][-converge_count:]
    if not recent:
        return float("nan"), float("nan")
    residual_um = float(np.median(recent))
    sigma = float(noise_um) / math.sqrt(len(recent)) if noise_um and math.isfinite(noise_um) else float("nan")
    if not math.isfinite(sigma) or sigma <= 0.0:
        snr = math.inf if abs(residual_um) > 0.0 else 0.0
    else:
        snr = abs(residual_um) / sigma
    return residual_um, snr


@dataclass
class ProbeGains:
    """启动探针测出的视觉↔机器人映射。"""

    gain: dict[str, dict[str, float]] = field(default_factory=dict)
    sign: dict[str, float] = field(default_factory=dict)
    vision_axis: dict[str, str] = field(default_factory=dict)
    hysteresis: dict[str, float] = field(default_factory=dict)
    cross_coupling: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        matrix = probe_gain_matrix(self) if self.gain else np.empty((0, 0))
        return {
            "gain_matrix": self.gain,
            "gain_matrix_visual_rows_robot_columns": matrix.tolist(),
            "gain_matrix_condition": (
                float(np.linalg.cond(matrix)) if matrix.shape == (2, 2) else None
            ),
            "sign": self.sign,
            "vision_axis_for_robot_axis": self.vision_axis,
            "hysteresis_ratio": self.hysteresis,
            "cross_coupling_ratio": self.cross_coupling,
            "notes": list(self.notes),
        }


def probe_gain_matrix(gains: ProbeGains) -> np.ndarray:
    """
    返回 G：行是视觉 x/y，列是机器人 X/Y。

    ProbeGains.gain 的外层键是机器人轴，因此这里必须显式转置成数学定义的布局。
    """

    try:
        matrix = np.asarray(
            [
                [float(gains.gain["X"]["x"]), float(gains.gain["Y"]["x"])],
                [float(gains.gain["X"]["y"]), float(gains.gain["Y"]["y"])],
            ],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("轴向探针没有形成完整的 2×2 增益矩阵。") from exc
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("轴向探针的 2×2 增益矩阵包含非法值。")
    return matrix


def inverse_probe_matrix(gains: ProbeGains) -> np.ndarray:
    """检查探针矩阵质量并返回 G⁻¹。"""

    matrix = probe_gain_matrix(gains)
    determinant = float(np.linalg.det(matrix))
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-6:
        raise RuntimeError(
            f"轴向探针矩阵近似奇异（det={determinant:.3g}），无法安全反演。"
        )
    if not math.isfinite(condition) or condition > float(
        config.MICRO_TARGET_MATRIX_MAX_CONDITION
    ):
        raise RuntimeError(
            f"轴向探针矩阵条件数 {condition:.2f} 超过 "
            f"{float(config.MICRO_TARGET_MATRIX_MAX_CONDITION):g}，"
            "视觉→机器人坐标换算会放大噪声，正式微动拒绝开始。"
        )
    return np.linalg.inv(matrix)


def visual_to_robot_xy(
    position_us: dict[str, float], gains: ProbeGains
) -> dict[str, float]:
    """使用完整 G⁻¹ 把视觉 x/y 位移换算成机器人 X/Y 位移。"""

    vector = np.asarray(
        [float(position_us["x"]), float(position_us["y"])], dtype=float
    )
    robot = inverse_probe_matrix(gains) @ vector
    return {"X": float(robot[0]), "Y": float(robot[1])}


def evaluate_probe_gains(
    baseline: dict[str, float],
    forward: dict[str, dict[str, float]],
    backward: dict[str, dict[str, float]],
) -> ProbeGains:
    """
    由探针片段位移算出增益矩阵并逐项过门禁。

    输入：
    - baseline：探针参考片段的 (x_um, y_um)；
    - forward：每个机器人轴正向移动后的 (x_um, y_um)；
    - backward：每个机器人轴反向回到起点后的 (x_um, y_um)。
    输出：ProbeGains；任一门禁不通过时抛 RuntimeError（具名中止，不继续）。

    增益矩阵 G[视觉轴][机器人轴] = 视觉位移(μm) / 机器人位移(μm)。
    这里同时定三件事：轴向对应、符号、以及增益量级是否落在闭环能收敛的范围内。
    |g| 明显大于 1 意味着过冲振荡风险；明显小于 1 意味着在 MICRO_LOOP_MAX_ITER
    次迭代内走不完最大目标幅值（本轮上限只有 6 次）。
    """

    probe_um = float(config.MICRO_LOOP_PROBE_UM)
    gain_min = float(config.MICRO_LOOP_GAIN_MIN)
    gain_max = float(config.MICRO_LOOP_GAIN_MAX)
    hysteresis_ratio = float(config.MICRO_LOOP_PROBE_HYSTERESIS_RATIO)

    def delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
        return {
            "x": float(after.get("x", 0.0)) - float(before.get("x", 0.0)),
            "y": float(after.get("y", 0.0)) - float(before.get("y", 0.0)),
        }

    gains = ProbeGains()
    forward_delta: dict[str, dict[str, float]] = {}
    backward_delta: dict[str, dict[str, float]] = {}
    position = {"x": float(baseline.get("x", 0.0)), "y": float(baseline.get("y", 0.0))}

    for robot_axis in ("X", "Y"):
        moved = delta(forward.get(robot_axis, position), position)
        back = delta(backward.get(robot_axis, position), forward.get(robot_axis, position))
        forward_delta[robot_axis] = moved
        backward_delta[robot_axis] = back
        gains.gain[robot_axis] = {
            "x": moved["x"] / probe_um,
            "y": moved["y"] / probe_um,
        }

        # 选主响应轴。注意下面全部门禁比的都是**增益**（视觉位移 ÷ 机器人位移，无量纲），
        # 不是位移本身——探针位移是 1000 μm 量级，拿它去比 [0.3, 2.0] 的增益带，
        # 无论机器多好都会判定"超出范围"而在启动时中止整轮实验。
        main_fwd = max(((abs(moved[vision]), vision) for vision in ("x", "y")), default=(0.0, "x"))
        main_bwd = max(((abs(back[vision]), vision) for vision in ("x", "y")), default=(0.0, "x"))
        gains.vision_axis[robot_axis] = main_fwd[1]

        signed_gain = moved[main_fwd[1]] / probe_um
        fwd_gain = abs(signed_gain)
        bwd_gain = main_bwd[0] / probe_um
        gains.sign[robot_axis] = 1.0 if signed_gain >= 0.0 else -1.0

        if fwd_gain <= 0.0:
            raise RuntimeError(
                f"轴向探针：机器人 {robot_axis} 轴走 {probe_um:g} μm，视觉上没有任何位移。"
                "可能相机视野外、镜头被遮挡，或该轴与像平面近乎平行。"
            )

        if not gain_min <= fwd_gain <= gain_max:
            raise RuntimeError(
                f"轴向探针：机器人 {robot_axis} 轴 → 视觉 {main_fwd[1]} 的增益为 "
                f"{fwd_gain:+.3f}（机器人走 {probe_um:g} μm，视觉测到 "
                f"{main_fwd[0]:+.1f} μm），超出允许范围 [{gain_min:g}, {gain_max:g}]。"
                f"增益过大意味着闭环振荡风险，过小意味着在 {int(config.MICRO_LOOP_MAX_ITER)} 次迭代内走不完目标幅值。"
            )

        if bwd_gain > 0.0:
            hysteresis = abs(fwd_gain - bwd_gain) / fwd_gain
            gains.hysteresis[robot_axis] = float(hysteresis)
            if hysteresis > hysteresis_ratio:
                raise RuntimeError(
                    f"轴向探针：机器人 {robot_axis} 轴正向增益 {fwd_gain:.3f}、"
                    f"反向增益 {bwd_gain:.3f}，相差 {hysteresis:.1%}，"
                    f"超过允许的 {hysteresis_ratio:.0%}。回差大到这个程度时闭环不是良定的。"
                )
        else:
            gains.hysteresis[robot_axis] = float("nan")

        other = [vision for vision in ("x", "y") if vision != main_fwd[1]]
        if other:
            cross = abs(moved[other[0]]) / main_fwd[0]
            gains.cross_coupling[robot_axis] = float(cross)
            if cross >= 0.3:
                gains.notes.append(
                    f"机器人 {robot_axis} 轴对视觉 {other[0]} 的交叉耦合为 {cross:.2f}，"
                    "已记录但未补偿（本实验只按被测轴纠偏）。"
                )

    if gains.vision_axis.get("X") == gains.vision_axis.get("Y"):
        raise RuntimeError(
            f"轴向探针：机器人 X 与 Y 都主要映射到视觉 {gains.vision_axis.get('X')} 轴，"
            "说明像平面与机器人 XY 平面接近侧视，两个自由度不可分辨，闭环无法工作。"
        )
    # 当前多目标实验不再丢弃交叉项；这里在任何正式微动前确认完整矩阵可逆且
    # 不会异常放大视觉噪声。inverse_probe_matrix 的异常信息会直接说明失败原因。
    inverse_probe_matrix(gains)
    return gains


# =============================================================================
# 2. 视觉测量层（复用已验证的棋盘格亚像素链路）
# =============================================================================

def _accept_frame(
    result: dict[str, Any],
    *,
    ref_found_by: str | None,
) -> tuple[bool, str]:
    """
    判断一帧的棋盘格结果能不能进入测量。

    输入：CheckerboardTracker.process() 返回的结果字典、参考帧的检测方法。
    输出：(是否接受, 拒绝原因)。

    为什么每一帧都要过这一套：棋盘格**索引错位一格**时，所有角点位移同一个向量
    （约一个方格 3 mm），estimateAffinePartial2D 会拟合成一个残差≈0、内点率 1.0、
    质量分≈1.0、转角≈0、尺度≈1 的**纯平移**——它和"真的移动了 3 mm"
    在返回字典里完全无法区分。3000 μm 的假误差喂进限幅 100 μm 的闭环，
    就是 20 步朝任意方向走 2 mm。

    其中 found_by 相等这一条最廉价也最关键：findChessboardCornersSB 与传统兜底
    findChessboardCorners 不保证角点排序一致，参考帧若来自 sb、某帧回落到 legacy，
    配对就可能被置换。
    """

    if not result.get("checker_is_valid"):
        return False, "识别失败"

    found_by = result.get("checker_found_by")
    if ref_found_by is not None and found_by != ref_found_by:
        return False, f"检测方法从 {ref_found_by} 变为 {found_by}，角点排序可能不同"

    quality = result.get("checker_quality")
    if quality is None or not math.isfinite(float(quality)) or float(quality) < float(
        config.MICRO_LOOP_QUALITY_MIN
    ):
        return False, f"质量分 {quality} 低于 {config.MICRO_LOOP_QUALITY_MIN}"

    angle = result.get("checker_angle_deg")
    if angle is None or not math.isfinite(float(angle)) or abs(float(angle)) > float(
        config.MICRO_LOOP_ANGLE_MAX_DEG
    ):
        return False, f"转角 {angle}° 超过 {config.MICRO_LOOP_ANGLE_MAX_DEG}°"

    scale = result.get("checker_scale")
    if scale is None or not math.isfinite(float(scale)) or abs(float(scale) - 1.0) > float(
        config.MICRO_LOOP_SCALE_TOL
    ):
        return False, f"尺度 {scale} 偏离 1 超过 {config.MICRO_LOOP_SCALE_TOL}"

    dx_mm = result.get("checker_dx_mm")
    dy_mm = result.get("checker_dy_mm")
    if dx_mm is None or dy_mm is None or not (
        math.isfinite(float(dx_mm)) and math.isfinite(float(dy_mm))
    ):
        return False, "位移为非法值"
    shift_um = max(abs(float(dx_mm)), abs(float(dy_mm))) * 1000.0
    if shift_um > float(config.MICRO_LOOP_MAX_PLAUSIBLE_SHIFT_UM):
        return False, f"位移 {shift_um:.1f} μm 超过物理不可能值"

    return True, ""


def _tracker_process(tracker: Any, frame: np.ndarray) -> tuple[dict[str, Any], Any, Any]:
    """
    走一帧棋盘格识别：模糊 → 固定 RNG → CheckerboardTracker.process。

    这里刻意不复用 camera.preprocess_frame：它在本项目当前配置下
    （CAMERA_MATRIX 与 VISION_ROI 都是 None）实际只做「灰度 → 高斯模糊 → 画质指标」，
    而画质指标是 4 次全幅遍历（实测约 40 ms/帧），对闭环毫无用处。
    识别部分与 preprocess_frame 逐字等价：同样的高斯模糊核，同一个
    CheckerboardTracker.process。这不是"换了识别流程"，只是不算那份显示用的统计量。

    fixed RNG：_estimate_rigid_motion 用 estimateAffinePartial2D(method=RANSAC)，
    内点集抖动会让 88 点均值位移跳动约 (2/88)·0.5 px·0.0685 ≈ 0.8 μm。
    固定种子让同一帧永远得到同一结果——对 3 μm 容差不是小数目。
    """

    import cv2

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        raise RuntimeError(f"闭环微动只支持 uint8 帧，实际 dtype={gray.dtype}。")
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.setRNGSeed(0)
    return tracker.process(blurred)


def continuity_jump_um(
    visual_position_um: float,
    previous_axis_position_um: float,
    vision_to_axis_sign: float,
) -> float:
    """在同一机器人轴坐标系内计算相邻测量的跳变量。"""

    sign = float(vision_to_axis_sign)
    if sign not in (-1.0, 1.0):
        raise ValueError("vision_to_axis_sign 必须是 +1 或 -1。")
    current_axis_um = sign * float(visual_position_um)
    return abs(current_axis_um - float(previous_axis_position_um))


@dataclass
class RawClip:
    """一段临时 RAW 及其旁车信息。"""

    raw_path: Path
    meta: dict[str, Any]
    frame_rows: list[dict[str, Any]]

    @property
    def frame_count(self) -> int:
        return int(self.meta.get("frame_count", len(self.frame_rows)))

    @property
    def width(self) -> int:
        return int(self.meta["width"])

    @property
    def height(self) -> int:
        return int(self.meta["height"])


def open_raw_clip(raw_path: Path, *, require_sidecar: bool = True) -> RawClip:
    """
    打开一段临时 RAW 及其旁车元数据和逐帧时间戳。

    输入：RAW 路径。
    输出：RawClip。

    实验作用：形状一律从相机写的元数据读，不靠猜；猜错会让整体错位。
    """

    raw_path = Path(raw_path)
    meta_path = raw_path.with_name(raw_path.stem + "_raw_meta.json")
    frames_path = raw_path.with_name(raw_path.stem + "_frames.csv")
    if require_sidecar and not meta_path.exists():
        raise FileNotFoundError(f"找不到 {meta_path.name}，无法确定 RAW 的形状。")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    rows: list[dict[str, Any]] = []
    if frames_path.exists():
        rows = read_rows_csv(frames_path)
    return RawClip(raw_path=raw_path, meta=meta, frame_rows=rows)


def iter_clip_frames(
    clip: RawClip, *, every_n: int = 1, start_index: int = 0
) -> Iterator[tuple[int, dict[str, Any], np.ndarray]]:
    """
    逐帧读出一段 RAW。

    输入：RawClip、抽帧步长、起始帧序号。
    输出：迭代 (段内帧序号, 逐帧时间戳行, 帧数组)。

    用 memmap 而不是把整段读进内存：一段 1.6 s 的全幅片段是 600 MB，
    而父进程同时还要持有上一段的结果。

    start_index 是在线测量的时间预算开关：识别一帧全幅要 0.165 s，
    而"运动后的稳定位置"只存在于窗口末尾，前面的瞬态帧对闭环决策没有贡献。
    从 start_index 起读，等于把"分析整段"换成"只分析尾部"。
    """

    frame_count = clip.frame_count
    mapped = np.memmap(
        clip.raw_path,
        dtype=np.uint8,
        mode="r",
        shape=(frame_count, clip.height, clip.width),
    )
    try:
        for index in range(max(0, int(start_index)), frame_count, max(1, int(every_n))):
            row = clip.frame_rows[index] if index < len(clip.frame_rows) else {}
            yield index, row, np.asarray(mapped[index])
    finally:
        del mapped
        gc.collect()


def _row_time_ns(row: dict[str, Any]) -> int:
    """从逐帧时间戳行里取出 host_ns（子进程与父进程共享同一时钟）。"""

    value = row.get("host_ns")
    if value in (None, ""):
        return 0
    return int(float(value))


def scan_corners(
    clip: RawClip, *, every_n: int = 1, start_index: int = 0
) -> list[tuple[int, Any]]:
    """
    用一次性 tracker 扫出每一帧的原始角点，用于挑参考帧。

    输入：RawClip、抽帧步长、起始帧序号。
    输出：[(段内帧序号, 角点数组)]，只含识别成功的帧。

    注意这里只用 process() 返回的第二个元素（当前帧角点），
    不用它的位移——一次性 tracker 会在自己看到的第一帧上锁存参考，
    而那一帧完全可能是离群帧。这一步的目的就是不要被它左右。
    """

    from camera import CheckerboardTracker

    survey = CheckerboardTracker()
    found: list[tuple[int, Any]] = []
    for index, _, frame in iter_clip_frames(clip, every_n=every_n, start_index=start_index):
        _, corners, _ = _tracker_process(survey, frame)
        if corners is not None:
            found.append((index, corners))
    return found


def pick_medoid_frame(found: Sequence[tuple[int, Any]]) -> int:
    """
    从识别成功的帧里挑出"最不可能是离群点"的一帧（medoid）。

    输入：[(帧序号, 角点)]。
    输出：medoid 的段内帧序号。

    做法：取所有帧角点的质心，再挑质心离全体质心中位数最近的那一帧。
    为什么不能用"第一帧"：参考一旦锁存就永不更新，且没有公开的 setter，
    所以任何一帧被选为参考，它带来的误差会成为整组的永久零点。
    """

    if not found:
        raise RuntimeError("参考片段里没有任何一帧识别成功。")

    import cv2

    centroids = np.asarray(
        [np.asarray(corners, dtype=float).reshape(-1, 2).mean(axis=0) for _, corners in found],
        dtype=float,
    )
    median_centroid = np.median(centroids, axis=0)
    distances = np.linalg.norm(centroids - median_centroid, axis=1)
    best = int(np.argmin(distances))
    del cv2
    return int(found[best][0])


class GroupVisionMeter:
    """
    一个目标组一个实例的视觉测量器。

    CheckerboardTracker 只在**第一次成功识别**时锁存参考角点与 mm_per_pixel，
    此后永不更新，也没有公开的 setter。所以：
    - 跨片段必须复用同一个实例，否则每段都会重新定义零点；
    - 参考帧用 medoid 挑，不让"碰巧第一帧"变成整组的永久零点。
    """

    def __init__(self, *, label: str = "") -> None:
        from camera import CheckerboardTracker

        self.label = label
        self.tracker = CheckerboardTracker()
        self.ref_found_by: str | None = None
        self.ref_us: dict[str, float] = {"x": 0.0, "y": 0.0}
        self.sigma_ref_um: float = float("nan")
        self.est_sigma_um: float = float("nan")
        self.reference_frame_index: int | None = None
        self.n_primed_frames = 0

    # ---- 参考建立 -------------------------------------------------------

    def prime(self, clip: RawClip, *, every_n: int = 1, frames: int | None = None) -> dict[str, Any]:
        """
        从一段静止片段建立本组的视觉零点。

        输入：参考片段、抽帧步长、只用末尾多少帧（None = 整段）。
        输出：含零点、噪声底、测量不确定度的字典。

        流程：扫角点 → 挑 medoid → 让正式 tracker 以该帧为第一帧锁存 →
        再走一遍，用中位数定零点、用块自助定测量不确定度。

        frames 同样是时间预算开关。零点需要的是"足够好的中位数"，不是整段：
        24 帧的中位数不确定度约 0.26 μm，已经好于一次闭环测量本身，
        而整段扫两遍在 5 s 静止基线上要 110 s。
        """

        start_index = 0
        if frames is not None:
            start_index = max(0, clip.frame_count - max(2, int(frames)))

        found = scan_corners(clip, every_n=every_n, start_index=start_index)
        if not found:
            raise RuntimeError(f"参考片段里没有任何一帧识别成功（{clip.raw_path.name}）。")
        medoid_index = pick_medoid_frame(found)

        # 正式 tracker 必须"第一眼"就看到 medoid 帧，才能把参考锁在它上面。
        primed = False
        for index, _, frame in iter_clip_frames(
            clip, every_n=every_n, start_index=start_index
        ):
            if index != medoid_index:
                continue
            result, _, _ = _tracker_process(self.tracker, frame)
            if result.get("checker_is_valid"):
                self.ref_found_by = result.get("checker_found_by")
                primed = True
            break
        if not primed:
            raise RuntimeError("medoid 参考帧在正式 tracker 上识别失败，参考无法建立。")

        self.reference_frame_index = medoid_index
        measurement = self.measure_clip(clip, every_n=every_n, tail_frames=frames)
        accepted = [
            row for row in measurement.frame_rows if str(row.get("accepted")) == "True"
        ]
        if not measurement.frame_rows:
            raise RuntimeError("参考片段没有任何帧。")

        accept_ratio = len(accepted) / len(measurement.frame_rows)
        if accept_ratio < 0.6:
            raise RuntimeError(
                f"参考片段合格帧比例只有 {accept_ratio:.1%}，低于 60%。"
                "零点建立不可信，本组不继续。"
            )

        dx_values = [float(row["checker_dx_mm"]) * 1000.0 for row in accepted]
        dy_values = [float(row["checker_dy_mm"]) * 1000.0 for row in accepted]
        self.ref_us = {
            "x": float(np.median(dx_values)) if dx_values else 0.0,
            "y": float(np.median(dy_values)) if dy_values else 0.0,
        }
        spread = max(mad_sigma(dx_values), mad_sigma(dy_values))
        if math.isfinite(spread) and spread > float(config.MICRO_LOOP_MAX_PLAUSIBLE_SHIFT_UM) / 100.0:
            raise RuntimeError(
                f"参考片段自身抖动 {spread:.1f} μm 过大，零点不可信，本组不继续。"
            )
        self.sigma_ref_um = float(spread)
        self.n_primed_frames = len(accepted)

        # 这里的 block bootstrap 量的就是"一次闭环测量的不确定度"，
        # 所以窗口必须取 MICRO_LOOP_MEASURE_FRAMES（在线实际分析多少帧），
        # 而不是时间意义上的尾部窗口——两者在"只分析尾部"的架构下不是一回事，
        # 用错会让窗口比样本还长，退化成一个样本、标准差恒为 0。
        self.est_sigma_um = estimate_measurement_sigma_um(
            dx_values if len(dx_values) >= len(dy_values) else dy_values,
            int(config.MICRO_LOOP_MEASURE_FRAMES),
            every_n=every_n,
        )
        return {
            "label": self.label,
            "reference_frame_index": medoid_index,
            "ref_found_by": self.ref_found_by,
            "reference_us": dict(self.ref_us),
            "sigma_ref_um": self.sigma_ref_um,
            "est_sigma_um": self.est_sigma_um,
            "primed_valid_frames": self.n_primed_frames,
            "primed_total_frames": len(measurement.frame_rows),
            "accept_ratio": accept_ratio,
        }

    @staticmethod
    def _fps(clip: RawClip) -> float:
        return float(
            clip.meta.get("camera_actual_fps")
            or config.EXPECTED_VISION_FPS
            or 132.0
        )

    def _to_reference_us(self, dx_mm: Any, dy_mm: Any) -> dict[str, float]:
        """把一帧的原始位移换算成相对本组零点的 μm 坐标。"""

        return {
            "x": float(dx_mm) * 1000.0 - self.ref_us["x"],
            "y": float(dy_mm) * 1000.0 - self.ref_us["y"],
        }

    # ---- 轴向映射：整个实验唯一的一处 -----------------------------------

    @staticmethod
    def axis_value_um(
        position_us: dict[str, float], robot_axis: str, gains: "ProbeGains"
    ) -> float:
        """
        把一帧的视觉位置映射成"机器人测试轴正方向的位移"标量。

        输入：视觉位置 {"x": μm, "y": μm}、机器人轴名、探针结果。
        输出：已经选好视觉轴、并且已经归一到机器人轴正向的标量（μm）。

        为什么必须只有这一处：选轴（robot Y 走的是视觉 y 还是 x）与符号
        （视觉正向是否等于机器人正向）是两个**必须一起用**的映射，任何一处
        漏掉其中一个都会得到"看起来合理、实则反了"的数。上一版就是把
        "选轴"留给调用点、结果测量层硬编码读了视觉 x，于是 robot Y 的六个
        目标全部测的是视觉 x 的漂移；而符号没归一，让 sign=−1 的正常轴
        在闭环联锁里被判成"方向错误"直接中止。

        把两者一起封在这里之后，调用方拿到的数天然满足
        "正数 = 机器人测试轴正方向"，联锁、命令律、误差定义全都直接用它，
        不再各自做符号运算。
        """

        vision_axis = gains.vision_axis.get(robot_axis)
        if vision_axis not in ("x", "y"):
            raise KeyError(f"探针结果里没有机器人轴 {robot_axis!r} 的视觉轴映射。")
        return float(gains.sign[robot_axis]) * float(position_us[vision_axis])

    @classmethod
    def axis_measured_um(
        cls,
        measurement: "ClipMeasurement",
        robot_axis: str,
        gains: "ProbeGains",
    ) -> float | None:
        """
        把一次片段测量映射成"机器人测试轴正方向的稳定位置"（μm）。

        输入：片段测量、机器人轴名、探针结果。
        输出：标量；测量不可用时返回 None。

        measured_um 是视觉 x 的尾窗中位数、transverse_um 是视觉 y 的
        尾窗中位数——两个都已经算好了，这里只负责按探针结果挑一个并归一符号，
        不引入任何新的数值处理。
        """

        if measurement is None or not measurement.is_usable:
            return None
        vision_axis = gains.vision_axis.get(robot_axis)
        if vision_axis not in ("x", "y"):
            raise KeyError(f"探针结果里没有机器人轴 {robot_axis!r} 的视觉轴映射。")
        if measurement.vision_axis != vision_axis:
            # 这不是"容错"，是必须硬抛：测量用的轴与探针指定的轴不一致时，
            # 拿到的数看起来完全合理，但它描述的是另一条轴的漂移。
            raise ValueError(
                f"机器人 {robot_axis} 轴应当测视觉 {vision_axis}，"
                f"但本次测量用的是视觉 {measurement.vision_axis}。"
            )
        if measurement.measured_um is None:
            return None
        value = float(measurement.measured_um)
        if not math.isfinite(value):
            return None
        return float(gains.sign[robot_axis]) * value

    @classmethod
    def axis_positions_um(
        cls,
        measurement: "ClipMeasurement",
        gains: "ProbeGains",
        *,
        robot_axes: Sequence[str] = ("X", "Y"),
    ) -> dict[str, float | None]:
        """
        **一次**测量同时给出两条机器人轴的稳定位置（μm，各自已归一符号）。

        输入：片段测量、探针结果、要取的机器人轴。
        输出：{机器人轴: 位置 μm}；测量不可用时每个轴都是 None。

        与 axis_measured_um 的分工：后者是常规入口，带"测量用的视觉轴必须与
        探针一致"的硬校验，因此一次测量只能供一条机器人轴使用。建零点时两条
        轴都要一个锚点，若为此把同一段片段扫两遍，识别开销直接翻倍；
        而 measure_clip 一次识别本来就同时算出**两条**视觉轴的尾窗中位数
        （主轴的进 measured_um，另一条进 transverse_um），所以这里不需要第二遍。

        为什么"另一条视觉轴"是唯一确定的：探针门禁已保证两个机器人轴映射到
        **不同**的视觉轴（否则像平面近似侧视，探针会具名中止）。本函数再检查
        一次这个前提——两条轴要求同一条视觉轴时直接抛错，不会静默给一个错数。

        局限（如实写在接口上）：transverse_um 没有自己独立的 n_tail 与
        sigma_tail，尾部是否静止由主轴的门槛统一把关。两条轴的帧集完全相同，
        所以这个共享的门是合理的；但这个数因此适合当**锚点**，
        不适合单独当成一次带不确定度的测量去下结论。
        """

        wanted = {axis: gains.vision_axis.get(axis) for axis in robot_axes}
        for axis, vision_axis in wanted.items():
            if vision_axis not in ("x", "y"):
                raise KeyError(f"探针结果里没有机器人轴 {axis!r} 的视觉轴映射。")
        if len(set(wanted.values())) != len(wanted):
            raise ValueError(
                f"两个机器人轴映射到了同一条视觉轴（{wanted}），"
                "无法从一次测量里同时取出它们的位置。"
            )

        if measurement is None or not measurement.is_usable:
            return {axis: None for axis in robot_axes}

        positions: dict[str, float | None] = {}
        for axis, vision_axis in wanted.items():
            # 主轴用 measured_um，另一条用 transverse_um——两者都是同一批
            # 合格帧在各自视觉轴上的尾窗中位数。
            raw = (
                measurement.measured_um
                if vision_axis == measurement.vision_axis
                else measurement.transverse_um
            )
            if raw is None or not math.isfinite(float(raw)):
                positions[axis] = None
                continue
            positions[axis] = float(gains.sign[axis]) * float(raw)
        return positions

    # ---- 片段测量 -------------------------------------------------------

    def measure_clip(
        self,
        clip: RawClip,
        *,
        every_n: int = 1,
        iteration_id: str = "",
        prev_measured_um: float | None = None,
        vision_to_axis_sign: float | None = None,
        tail_frames: int | None = None,
        tail_window_s: float | None = None,
        vision_axis: str = "x",
    ) -> "ClipMeasurement":
        """
        逐帧测量一段临时 RAW。

        输入：片段、抽帧步长、迭代标识、上一轮的稳定位置（用于连续性联锁）、
              只分析末尾多少帧（None = 整段）。
        输出：ClipMeasurement。

        tail_frames 是在线时长的主要开关：识别一帧全幅 1936×1464 实测 0.165 s，
        所以"分析多少帧"几乎就是"这一步花多久"，而机械等待只占 4%。
        闭环要的是**运动停稳之后**的位置，它就落在窗口末尾，
        所以默认只分析最后 MICRO_LOOP_MEASURE_FRAMES 帧；窗口本身照录不误，
        RAW 全量保留，前面那些瞬态帧留给事后离线复查。

        连续性联锁：prev_measured_um 是已经归一到机器人轴正方向的位置；
        当前视觉位置必须先乘 vision_to_axis_sign，二者统一坐标系后再比较。
        单次迭代最大合法变化就是 100 μm 限幅再加上漂移，
        超过 3 倍限幅一定是检测出了问题。这类帧标为不可信并剔除，
        **绝不用来发命令**。这是控制安全门，不是"识别率低就重测"——
        质量指标只记录，不参与实验控制。
        """

        positions: list[dict[str, float]] = []
        times_ns: list[int] = []
        rows: list[dict[str, Any]] = []
        axis_positions: list[float] = []
        axis_times: list[int] = []
        valid_count = 0
        jump_limit_um = float(config.MICRO_LOOP_MAX_ITER_JUMP_UM)
        if prev_measured_um is not None:
            if vision_to_axis_sign is None:
                raise ValueError(
                    "使用 prev_measured_um 连续性联锁时必须提供 vision_to_axis_sign。"
                )
            sign = float(vision_to_axis_sign)
            if sign not in (-1.0, 1.0):
                raise ValueError("vision_to_axis_sign 必须是 +1 或 -1。")

        start_index = 0
        if tail_frames is not None:
            start_index = max(0, clip.frame_count - max(1, int(tail_frames)))

        # 连续性与峰值都只看"被测的那条视觉轴"。选轴由调用方按探针结果给出，
        # 不能在这里写死 x——robot Y 走的是视觉 y，写死 x 会让 Y 的整条链
        # 都在测另一条轴的漂移，而且看起来完全正常（上一版的实测缺陷）。
        if vision_axis not in ("x", "y"):
            raise ValueError(f"vision_axis 只能是 'x' 或 'y'，收到 {vision_axis!r}。")

        for index, frame_row, frame in iter_clip_frames(
            clip, every_n=every_n, start_index=start_index
        ):
            result, _, _ = _tracker_process(self.tracker, frame)
            accepted, reason = _accept_frame(result, ref_found_by=self.ref_found_by)
            host_ns = _row_time_ns(frame_row)

            dx_mm = result.get("checker_dx_mm")
            dy_mm = result.get("checker_dy_mm")
            position: dict[str, float] | None = None
            if accepted:
                position = self._to_reference_us(dx_mm, dy_mm)
                if prev_measured_um is not None and math.isfinite(prev_measured_um):
                    jump_um = continuity_jump_um(
                        position[vision_axis],
                        float(prev_measured_um),
                        float(vision_to_axis_sign),
                    )
                    if jump_um > jump_limit_um:
                        accepted = False
                        reason = (
                            f"相对上一轮稳定位置跳变 "
                            f"{jump_um:.1f} μm，"
                            f"超过 {jump_limit_um:g} μm"
                        )
                        position = None

            if accepted and position is not None:
                valid_count += 1
                positions.append(position)
                times_ns.append(host_ns)
                axis_positions.append(position[vision_axis])
                axis_times.append(host_ns)

            rows.append(
                {
                    "iteration_id": iteration_id,
                    "frame_index": index,
                    "frame_id": frame_row.get("frame_id", ""),
                    "host_ns": host_ns,
                    "camera_timestamp_raw": frame_row.get("camera_timestamp_raw", ""),
                    "checker_dx_mm": dx_mm,
                    "checker_dy_mm": dy_mm,
                    "checker_is_valid": result.get("checker_is_valid"),
                    "accepted": accepted,
                    "checker_quality": result.get("checker_quality"),
                    "checker_residual_px": result.get("checker_residual_px"),
                    "checker_angle_deg": result.get("checker_angle_deg"),
                    "checker_scale": result.get("checker_scale"),
                    "checker_found_by": result.get("checker_found_by"),
                    "reject_reason": reason,
                }
            )

        tail_window_s = float(
            config.MICRO_LOOP_TAIL_WINDOW_S
            if tail_window_s is None
            else tail_window_s
        )
        if not math.isfinite(tail_window_s) or tail_window_s <= 0:
            raise ValueError("tail_window_s 必须是有限正数。")
        measured = median_of_tail(axis_positions, axis_times, tail_window_s)
        cutoff_ns = (axis_times[-1] - int(tail_window_s * 1e9)) if axis_times else 0
        n_tail = sum(1 for t in axis_times if t >= cutoff_ns)

        tail_values = [v for v, t in zip(axis_positions, axis_times) if t >= cutoff_ns]
        sigma_tail = float(np.std(tail_values)) if len(tail_values) >= 2 else float("nan")

        # 横向轴 = 另一条视觉轴。它的用途只有一个：让操作者能看见
        # "被测轴之外还剩多少没被补偿的横向位移"（探针的交叉耦合告警也用它）。
        other_axis = "y" if vision_axis == "x" else "x"
        transverse = median_of_tail(
            [p[other_axis] for p in positions], times_ns, tail_window_s
        )
        peak_um = max((abs(v) for v in axis_positions), default=float("nan"))

        for row in rows:
            row["in_tail_window"] = bool(
                row.get("accepted") is True and int(row.get("host_ns") or 0) >= cutoff_ns
            )
            row["t_rel_s"] = (
                round((int(row["host_ns"]) - times_ns[0]) / 1e9, 6) if times_ns else ""
            )

        return ClipMeasurement(
            raw_path=clip.raw_path,
            frame_count=len(rows),
            n_valid=valid_count,
            measured_um=measured,
            transverse_um=transverse,
            n_tail=n_tail,
            sigma_tail_um=sigma_tail,
            peak_axis_um=peak_um,
            positions=positions,
            times_ns=times_ns,
            frame_rows=rows,
            first_row_ns=int(times_ns[0]) if times_ns else 0,
            last_row_ns=int(times_ns[-1]) if times_ns else 0,
            truncated=bool(clip.meta.get("truncated", False)),
            vision_axis=vision_axis,
        )


@dataclass
class ClipMeasurement:
    """
    一段临时 RAW 的逐帧测量结果。

    measured_um 是**被测视觉轴**（vision_axis）的尾窗中位数，
    transverse_um 是另一条视觉轴的尾窗中位数。
    vision_axis 记录在案，是为了让"测的是哪条轴"这件事可追溯：
    调用方按探针结果选轴，选错时能被查出来，而不是静默给一个错数。
    """

    raw_path: Path
    frame_count: int
    n_valid: int
    measured_um: float | None
    transverse_um: float | None
    n_tail: int
    sigma_tail_um: float
    peak_axis_um: float
    positions: list[dict[str, float]]
    times_ns: list[int]
    frame_rows: list[dict[str, Any]]
    first_row_ns: int
    last_row_ns: int
    truncated: bool = False
    vision_axis: str = "x"

    @property
    def detection_rate(self) -> float:
        """棋盘格有效识别率（含被接受判据剔除的帧）。"""

        if not self.frame_rows:
            return 0.0
        valid = sum(1 for row in self.frame_rows if row.get("checker_is_valid") is True)
        return valid / len(self.frame_rows)

    @property
    def accept_rate(self) -> float:
        """通过全部接受判据的比例。"""

        if not self.frame_rows:
            return 0.0
        accepted = sum(1 for row in self.frame_rows if row.get("accepted") is True)
        return accepted / len(self.frame_rows)

    @property
    def rejection_summary(self) -> str:
        """压缩记录本段各类拒绝原因，RAW 删除后仍可追查。"""

        counts: dict[str, int] = {}
        for row in self.frame_rows:
            if row.get("accepted") is True:
                continue
            reason = str(row.get("reject_reason") or "未注明原因")
            counts[reason] = counts.get(reason, 0) + 1
        return "；".join(f"{reason} ×{count}" for reason, count in counts.items())

    @property
    def tail_static(self) -> bool:
        """
        尾部窗口自身是否足够静止。

        这是"这一轮测到的是停稳后的位置"的直接证据。编码器侧的稳定判据可能
        超时降级，而这里看的是**视觉数据本身**——闭环本来就该以视觉为准。
        """

        if self.n_tail < int(config.MICRO_LOOP_MIN_TAIL_FRAMES):
            return False
        if not math.isfinite(self.sigma_tail_um):
            return False
        return self.sigma_tail_um <= float(config.MICRO_LOOP_TAIL_SIGMA_MAX_UM)

    @property
    def is_usable(self) -> bool:
        """
        本轮测量能不能用来发命令。

        四个条件缺一不可：没被缓冲写满截断、尾部有足够多的合格帧、
        尾部确实静止、以及拿到的是一个有限的数值。
        """

        if self.truncated or self.measured_um is None:
            return False
        if not math.isfinite(float(self.measured_um)):
            return False
        return self.tail_static


@dataclass
class RobotXYMeasurement:
    """同一批视觉帧经完整 2×2 变换后的机器人 XY 稳定位置。"""

    x_um: float | None
    y_um: float | None
    n_tail: int
    sigma_x_um: float
    sigma_y_um: float
    is_usable: bool
    reason: str

    def as_dict(self) -> dict[str, float | None]:
        return {"X": self.x_um, "Y": self.y_um}


def robot_xy_from_measurement(
    measurement: ClipMeasurement,
    gains: ProbeGains,
    *,
    tail_window_s: float | None = None,
) -> RobotXYMeasurement:
    """把一次片段的尾窗位置通过完整 G⁻¹ 换成机器人 X/Y 坐标。"""

    window_s = float(
        config.MICRO_LOOP_TAIL_WINDOW_S
        if tail_window_s is None
        else tail_window_s
    )
    if not measurement.times_ns:
        return RobotXYMeasurement(None, None, 0, math.nan, math.nan, False, "没有合格视觉帧")
    cutoff_ns = int(measurement.times_ns[-1]) - int(window_s * 1e9)
    robot_positions = [
        visual_to_robot_xy(position, gains)
        for position, host_ns in zip(measurement.positions, measurement.times_ns)
        if int(host_ns) >= cutoff_ns
    ]
    count = len(robot_positions)
    if count:
        x_values = np.asarray([item["X"] for item in robot_positions], dtype=float)
        y_values = np.asarray([item["Y"] for item in robot_positions], dtype=float)
        x_um = float(np.median(x_values))
        y_um = float(np.median(y_values))
        sigma_x = float(np.std(x_values)) if count >= 2 else math.nan
        sigma_y = float(np.std(y_values)) if count >= 2 else math.nan
    else:
        x_um = y_um = None
        sigma_x = sigma_y = math.nan

    reasons: list[str] = []
    if measurement.truncated:
        reasons.append("相机缓冲截断")
    if count < int(config.MICRO_LOOP_MIN_TAIL_FRAMES):
        reasons.append(
            f"尾窗合格帧 {count} < {int(config.MICRO_LOOP_MIN_TAIL_FRAMES)}"
        )
    sigma_max = max(sigma_x, sigma_y) if all(
        math.isfinite(value) for value in (sigma_x, sigma_y)
    ) else math.nan
    if not math.isfinite(sigma_max):
        reasons.append("尾窗标准差不是有限数")
    elif sigma_max > float(config.MICRO_LOOP_TAIL_SIGMA_MAX_UM):
        reasons.append(
            f"机器人坐标尾窗标准差 {sigma_max:.2f} μm > "
            f"{float(config.MICRO_LOOP_TAIL_SIGMA_MAX_UM):.2f} μm"
        )
    if x_um is None or y_um is None or not all(
        math.isfinite(float(value)) for value in (x_um, y_um)
    ):
        reasons.append("没有有限的机器人 XY 位置")
    return RobotXYMeasurement(
        x_um=x_um,
        y_um=y_um,
        n_tail=count,
        sigma_x_um=sigma_x,
        sigma_y_um=sigma_y,
        is_usable=not reasons,
        reason="；".join(reasons),
    )


def last_accepted_frame_index(measurement: "ClipMeasurement") -> int | None:
    """取最后一个通过接受判据的帧序号（即"停稳后"的代表帧）。"""

    for row in reversed(measurement.frame_rows):
        if row.get("accepted") is True:
            return int(row["frame_index"])
    return None


def peak_frame_index(
    measurement: "ClipMeasurement",
    *,
    axis: str = "X",
    reference_um: float = 0.0,
    target_um: float | None = None,
) -> int | None:
    """
    取本次迭代里偏离目标最远的那一帧（过冲或最大误差的代表帧）。

    输入：逐帧测量、被测轴、本组视觉零点、目标位置。
    输出：段内帧序号；没有合格帧时 None。

    偏离要相对**目标**而不是相对参考零点算：参考零点离目标可能有 50 μm，
    按它算的话每一帧的"峰值"都是同一帧，证据图就失去意义了。
    """

    key = "checker_dx_mm" if str(axis).upper() == "X" else "checker_dy_mm"
    best: tuple[float, int] | None = None
    for row in measurement.frame_rows:
        if row.get("accepted") is not True:
            continue
        value = row.get(key)
        if value in (None, ""):
            continue
        position = float(value) * 1000.0 - float(reference_um)
        deviation = (
            abs(position - float(target_um)) if target_um is not None else abs(position)
        )
        if best is None or deviation > best[0]:
            best = (deviation, int(row["frame_index"]))
    return None if best is None else best[1]


def save_evidence_png(clip: RawClip, frame_index: int, path: Path) -> bool:
    """
    保存一张全分辨率 PNG 证据帧（绝不用 JPEG）。

    输入：片段、段内帧序号、目标路径。
    输出：是否保存成功。
    """

    import cv2

    try:
        for index, _, frame in iter_clip_frames(clip, every_n=1):
            if index != int(frame_index):
                continue
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return bool(cv2.imwrite(str(path), frame))
        return False
    except Exception:
        return False


def analyze_static_clip(
    clip: RawClip,
    clip_measurement: ClipMeasurement,
    *,
    fps: float,
    record_start_ns: int,
    record_stop_ns: int,
) -> dict[str, Any]:
    """
    把静止基线的逐帧位移时序压成一份可以判读的统计。

    输入：片段、逐帧测量、帧率、录制起止。
    输出：含均值/中位数/标准差/峰峰值/帧数/识别率的字典。

    实验作用：确认去掉 MJPG 之后，全分辨率静止画面本身的微振动与视觉噪声有多少。
    这个数直接决定后面 3 μm 容差到底是不是在筛噪声。
    """

    accepted = [row for row in clip_measurement.frame_rows if row.get("accepted") is True]
    dx = np.asarray([float(row["checker_dx_mm"]) * 1000.0 for row in accepted], dtype=float)
    dy = np.asarray([float(row["checker_dy_mm"]) * 1000.0 for row in accepted], dtype=float)
    expected = expected_frames(record_start_ns, record_stop_ns, fps)
    # actual_frames 是相机真正录到的帧数；clip_measurement.frame_count 只是
    # 为节省在线时间而抽样分析的帧数，二者不能混为一谈。
    actual = clip.frame_count
    tail_frames = max(1, int(round(float(config.MICRO_LOOP_TAIL_WINDOW_S) * float(fps))))

    def stats(values: np.ndarray) -> dict[str, Any]:
        if values.size == 0:
            return {
                "mean_um": None,
                "median_um": None,
                "std_um": None,
                "mad_sigma_um": None,
                "peak_to_peak_um": None,
            }
        return {
            "mean_um": float(values.mean()),
            "median_um": float(np.median(values)),
            "std_um": float(values.std()),
            "mad_sigma_um": mad_sigma(values.tolist()),
            "peak_to_peak_um": float(values.max() - values.min()),
        }

    return {
        "kind": "MICRO_CLOSED_LOOP_STATIC_BASELINE",
        "clip": str(clip_measurement.raw_path),
        "width": clip.width,
        "height": clip.height,
        "codec": "UNCOMPRESSED_RAW",
        "actual_frames": actual,
        "expected_frames": expected,
        "frame_ratio": (actual / expected) if expected else None,
        "valid_chessboard_frames": len(accepted),
        "chessboard_detection_rate": clip_measurement.detection_rate,
        "chessboard_accept_rate": clip_measurement.accept_rate,
        "x": stats(dx),
        "y": stats(dy),
        "est_sigma_um": estimate_measurement_sigma_um(dx.tolist(), tail_frames),
        "record_start_ns": int(record_start_ns),
        "record_stop_ns": int(record_stop_ns),
        "duration_s": (int(record_stop_ns) - int(record_start_ns)) / 1e9,
    }


# =============================================================================
# 3. 命令接缝：唯一的 μm → m 转换点
# =============================================================================

def enforce_fixed_safety_envelope(
    pose: Sequence[float],
    center_pose: Sequence[float],
    *,
    label: str,
    radius_m: float | None = None,
) -> float:
    """要求 TCP 位于本次启动位置周围的固定球形安全包络内。"""

    if len(pose) < 3 or len(center_pose) < 3:
        raise RuntimeError(f"{label} 缺少完整 XYZ，无法执行固定安全包络检查。")
    position = np.asarray(pose[:3], dtype=float)
    center = np.asarray(center_pose[:3], dtype=float)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(center)):
        raise RuntimeError(f"{label} 或安全包络中心包含 NaN/无穷值，拒绝运动。")

    radius = float(
        config.MICRO_LOOP_FIXED_SAFETY_RADIUS_M if radius_m is None else radius_m
    )
    if not math.isfinite(radius) or radius <= 0:
        raise RuntimeError("固定安全包络半径必须是有限正数。")

    delta = position - center
    distance = float(np.linalg.norm(delta))
    if distance > radius + 1e-12:
        raise RuntimeError(
            f"{label} 距本次启动 TCP {distance * 1000.0:.2f} mm，超过固定安全半径 "
            f"{radius * 1000.0:.1f} mm；XYZ 偏移分别为 "
            f"({delta[0] * 1000.0:+.2f}, {delta[1] * 1000.0:+.2f}, "
            f"{delta[2] * 1000.0:+.2f}) mm。已拒绝运动。"
        )
    return distance

def _wait_until_position_stable(
    robot: Any,
    stop_event: Any,
    write_state: Callable[[dict[str, Any]], None],
    *,
    timeout_s: float,
    description: str,
) -> dict[str, Any] | None:
    """
    基于位置的稳定判据。超时返回 None，由调用方降级处理，不抛错。

    输入：机器人、停止事件、状态写盘回调、超时、描述。
    输出：稳定时的最后一条状态；超时返回 None。

    为什么不用 robot.motion_in_progress()：它带固定 200 ms 宽限期，
    是 5 μm 运动全程（约 14 ms）的 14 倍，对微动零信息量。

    为什么超时不致命：窗口取 5 μm 而不是既有微动实验的 1 μm，
    因为 CB3 的 actual_tcp_pose 由关节编码器换算，笛卡尔分辨率本身
    可能就有几 μm，1 μm 窗口可能永远无法满足。即便如此，超时也只降级为
    "按固定时间等一等"，真正的有效性由**视觉片段自身的窗内标准差**判定——
    闭环本来就该以视觉为准，编码器稳定性只是旁证。
    """

    window_ns = int(float(config.MICRO_LOOP_STABLE_WINDOW_MS) * 1_000_000)
    window_m = float(config.MICRO_LOOP_STABLE_WINDOW_UM) / 1_000_000.0
    speed_limit_m_s = float(config.MICRO_LOOP_STABLE_SPEED_MM_S) / 1000.0
    hold_ns = int(float(config.MICRO_LOOP_STABLE_HOLD_SECONDS) * 1_000_000_000)

    samples: deque[tuple[int, np.ndarray]] = deque()
    stable_since: int | None = None
    deadline = time.perf_counter() + float(timeout_s)

    while True:
        if stop_event.is_set():
            raise RuntimeError(f"{description}期间收到停止请求。")
        if time.perf_counter() >= deadline:
            return None

        state = robot.read_state()
        write_state(state)
        now_ns = int(state["host_ns"])
        position = np.asarray(state["actual_tcp_pose"][:3], dtype=float)
        samples.append((now_ns, position))
        while samples and samples[0][0] < now_ns - window_ns:
            samples.popleft()

        gap_m = max(
            (float(np.linalg.norm(position - other)) for _, other in samples), default=0.0
        )
        speed_values = state.get("actual_tcp_speed") or []
        speed_m_s = (
            float(np.linalg.norm(np.asarray(speed_values[:3], dtype=float)))
            if len(speed_values) >= 3
            else math.inf
        )

        if gap_m <= window_m and speed_m_s <= speed_limit_m_s:
            if stable_since is None:
                stable_since = now_ns
            elif now_ns - stable_since >= hold_ns:
                return state
        else:
            stable_since = None

        time.sleep(0.002)


def send_cartesian_micro_correction(
    robot: Any,
    axis: str,
    delta_um: float,
    *,
    safety_center_pose: Sequence[float],
    stop_event: Any,
    write_state: Callable[[dict[str, Any]], None],
    settle_timeout_s: float | None = None,
) -> dict[str, Any]:
    """
    把一个 μm 级笛卡尔修正发下去。**这是本模块唯一的 μm → m 转换点。**

    输入：机器人、轴名（"X"/"Y"）、有符号修正量（μm）、停止事件、状态写盘回调。
    输出：含 tcp_before/tcp_after/achieved_um/transverse_um/status 的字典。

    为什么要单独封成一层：将来若要比较 MoveL 与 servoj，只需要替换这一个函数。
    它现在用的是与既有微动实验完全相同的链路——URRobot 的 MoveL，
    两点 Waypoint，validate_trajectory + verify_controller_safety_limits——
    不切换到任何新的执行接口。

    两个必须做对的细节：
    1. 轨迹必须是**两点**。robot.execute_trajectory 对长度 ≤1 的轨迹静默 no-op
       （它发送 trajectory[1:]），一点轨迹等于什么都没发。
    2. 发完必须验证命令真的下去了。execute_trajectory 在"成功"和"静默忽略"
       两条路径上都返回 None，唯一能区分的是 motion_command_time 有没有推进。
       检查这一项，才能把"控制器没理我"和"我根本没发出去"分开。
    """

    from robot import Waypoint, validate_trajectory

    axis = str(axis).upper()
    if axis not in AXIS_INDEX:
        raise ValueError(f"未知轴 {axis!r}，只能是 X 或 Y。")

    axis_index = AXIS_INDEX[axis]
    delta_m = float(delta_um) / 1_000_000.0
    if delta_m == 0.0:
        raise ValueError("修正量为 0 不应调用本函数；死区判断在 command_for_error 里做。")

    # 只读一次位姿：它同时充当记录用的 tcp_before、轨迹起点来源和 achieved 断言基线。
    # 读两次会让"实际位移"不等于指令 Δ——两次 RTDE 读之间隔几十微秒，
    # 而本实验的步长只有 5 μm。
    settled = _wait_until_position_stable(
        robot,
        stop_event,
        write_state,
        timeout_s=float(config.MICRO_LOOP_PRE_STABLE_TIMEOUT_S),
        description="微动前的稳定等待",
    )
    if settled is None:
        time.sleep(float(config.MICRO_LOOP_MOTION_SETTLE_SECONDS))

    before_state = robot.read_state()
    write_state(before_state)
    tcp_before = [float(value) for value in before_state["actual_tcp_pose"]]

    current = list(tcp_before)
    target = list(current)
    target[axis_index] = current[axis_index] + delta_m
    trajectory = [
        Waypoint(
            "P0",
            current,
            float(config.MICRO_LOOP_SPEED_MM_S) / 1000.0,
            float(config.MICRO_LOOP_ACCELERATION_M_S2),
            0.0,
        ),
        Waypoint(
            "P1",
            target,
            float(config.MICRO_LOOP_SPEED_MM_S) / 1000.0,
            float(config.MICRO_LOOP_ACCELERATION_M_S2),
            0.0,
        ),
    ]
    # 最高优先级的软件包络：中心固定为本次 micro_closed_loop 启动 TCP，
    # 绝不能像普通 relative 校验那样随每一步当前位置重新锚定。
    enforce_fixed_safety_envelope(
        current, safety_center_pose, label="微动轨迹起点"
    )
    enforce_fixed_safety_envelope(
        target, safety_center_pose, label="微动轨迹目标"
    )
    if len(trajectory) != 2:
        raise RuntimeError("内部错误：微动轨迹必须是两点。")
    validate_trajectory(trajectory, for_real_robot=True, pose_source="relative")
    robot.verify_controller_safety_limits(trajectory)

    command_start_ns = time.perf_counter_ns()
    stamp_before = robot.motion_command_time
    robot.execute_trajectory(trajectory)
    if robot.motion_command_time == stamp_before:
        raise RuntimeError(
            "moveL 被静默忽略：execute_trajectory 没有推进 motion_command_time。"
            "这通常意味着轨迹点被丢弃（例如只有一个路径点），本步未真正执行。"
        )

    timeout_s = (
        float(config.MICRO_LOOP_STEP_TIMEOUT_S)
        if settle_timeout_s is None
        else float(settle_timeout_s)
    )
    period = 1.0 / float(config.ROBOT_RECORD_HZ)
    deadline = time.perf_counter() + timeout_s
    while robot.motion_in_progress():
        if stop_event.is_set():
            robot.stop_motion()
            raise RuntimeError("闭环微动收到停止请求，已调用 stopL。")
        if time.perf_counter() >= deadline:
            robot.stop_motion()
            raise TimeoutError("单步微动超过超时上限，已调用 stopL。")
        write_state(robot.read_state())
        time.sleep(period)

    after_state = _wait_until_position_stable(
        robot,
        stop_event,
        write_state,
        timeout_s=float(config.MICRO_LOOP_POST_STABLE_TIMEOUT_S),
        description="微动后的稳定等待",
    )
    settle_ok = after_state is not None
    if after_state is None:
        time.sleep(float(config.MICRO_LOOP_MOTION_SETTLE_SECONDS))
        after_state = robot.read_state()
        write_state(after_state)

    tcp_after = [float(value) for value in after_state["actual_tcp_pose"]]
    achieved_um = (tcp_after[axis_index] - tcp_before[axis_index]) * 1_000_000.0
    transverse = [
        tcp_after[index] - tcp_before[index] for index in range(3) if index != axis_index
    ]
    transverse_um = float(np.linalg.norm(np.asarray(transverse, dtype=float))) * 1_000_000.0

    status, note = evaluate_achieved_robot_side(float(delta_um), achieved_um)
    return {
        "command_start_ns": command_start_ns,
        "command_end_ns": int(after_state["host_ns"]),
        "tcp_before": tcp_before,
        "tcp_after": tcp_after,
        "achieved_um": achieved_um,
        "expected_um": float(delta_um),
        "transverse_um": transverse_um,
        "settle_ok": settle_ok,
        "status": status,
        "note": note,
    }


def evaluate_achieved_robot_side(delta_um: float, achieved_um: float) -> tuple[str, str]:
    """
    在机器人一侧判断这一步是否真的按指令走了。

    输入：指令位移、实测位移（都由 actual_tcp_pose 得出，μm）。
    输出：(状态码, 说明)。

    与视觉侧的 measured_um 并排，就得到本次实验最有用的三分表：
      机器人没动 + 视觉没看到 → 控制器忽略了命令（最小有效运动尺度）
      机器人动了 + 视觉没看到 → 机械柔性/夹具间隙吸收，或视觉噪声底
      机器人动了 + 视觉看到但量不对 → 增益误差或交叉耦合
    """

    magnitude = abs(float(delta_um))
    got = abs(float(achieved_um))
    if magnitude <= 0.0:
        return STEP_VALID, ""

    low = max(float(config.MICRO_LOOP_ACHIEVED_FLOOR_UM),
              float(config.MICRO_LOOP_ACHIEVED_MIN_RATIO) * magnitude)
    high = float(config.MICRO_LOOP_ACHIEVED_MAX_RATIO) * magnitude + float(
        config.MICRO_LOOP_ACHIEVED_SLACK_UM
    )
    if got < low:
        return STEP_SMALL, f"实测位移 {achieved_um:+.2f} μm 低于下限 {low:.2f} μm。"
    if got > high:
        return STEP_LARGE, f"实测位移 {achieved_um:+.2f} μm 高于上限 {high:.2f} μm。"
    if got >= float(config.MICRO_LOOP_ACHIEVED_FLOOR_UM) and (
        (float(achieved_um) > 0.0) != (float(delta_um) > 0.0)
    ):
        return STEP_SIGN, (
            f"指令 {delta_um:+.2f} μm 但实测 {achieved_um:+.2f} μm，方向相反；"
            "检查基座/工具坐标系或轴符号。"
        )
    return STEP_VALID, ""


# =============================================================================
# 4. 临时文件管理（Windows 句柄是真实风险）
# =============================================================================

def temp_workspace(output_root: Path) -> tuple[Path, Path]:
    """
    返回临时文件目录和回收目录。

    输入：输出根目录。
    输出：(临时目录, 回收目录)。

    两者都放在输出根目录下（与运行目录同一个卷），
    这样 os.replace 是原子的、不跨卷，而且**运行目录里永远不会出现视频文件**——
    即使进程崩溃，运行目录也是干净的。
    """

    output_root = Path(output_root)
    return output_root / TEMP_DIR_NAME, output_root / TRASH_DIR_NAME


def prepare_temp_workspace(output_root: Path) -> tuple[Path, Path]:
    """
    实验开始时建立并清空临时工作区。

    输入：输出根目录。
    输出：(临时目录, 回收目录)。

    只在**启动时**清空。循环中途绝不清空——那会把别人正在用的文件删掉。
    """

    temp_dir, trash_dir = temp_workspace(output_root)
    for path in (temp_dir, trash_dir):
        path.mkdir(parents=True, exist_ok=True)
    sweep_directory(trash_dir)
    return temp_dir, trash_dir


def sweep_directory(path: Path) -> int:
    """尽力删除目录下的所有文件，返回释放的字节数。"""

    released = 0
    path = Path(path)
    if not path.exists():
        return 0
    for entry in path.iterdir():
        try:
            if entry.is_file():
                released += int(entry.stat().st_size)
                entry.unlink()
        except OSError:
            continue
    return released


def delete_video_payloads(root: Path) -> tuple[int, list[Path]]:
    """递归删除一次运行目录内的大体积视频载荷，保留 CSV/JSON 等结果旁车。"""

    root = Path(root)
    if not root.exists():
        return 0, []
    root_resolved = root.resolve()
    suffixes = {".raw", ".avi", ".mp4", ".mkv"}
    released = 0
    failed: list[Path] = []
    for entry in root.rglob("*"):
        if not entry.is_file() or entry.suffix.lower() not in suffixes:
            continue
        try:
            resolved = entry.resolve()
            if not resolved.is_relative_to(root_resolved):
                failed.append(entry)
                continue
            size = int(entry.stat().st_size)
            removed = False
            for _attempt in range(10):
                try:
                    entry.unlink()
                    removed = True
                    released += size
                    break
                except PermissionError:
                    gc.collect()
                    time.sleep(0.1)
            if not removed:
                failed.append(entry)
        except OSError:
            failed.append(entry)
    return released, failed


def directory_bytes(path: Path) -> int:
    """统计目录下所有文件的总字节数。"""

    total = 0
    path = Path(path)
    if not path.exists():
        return 0
    for entry in path.iterdir():
        try:
            if entry.is_file():
                total += int(entry.stat().st_size)
        except OSError:
            continue
    return total


def drop_temp_files(paths: Sequence[Path], trash_dir: Path, *, log: Callable[[str], None] | None = None) -> None:
    """
    把临时文件移进回收目录，而不是直接删除。

    输入：文件路径列表、回收目录、可选日志回调。

    为什么用移动而不是 unlink：np.memmap 持有文件句柄，CPython 不保证
    在 del / 关闭之后立刻释放，WinError 32（文件被占用）会随机出现。
    在一个 200 多次迭代的循环里，这意味着第 137 次把整轮弄死。
    移到同卷的回收目录是原子的、不依赖句柄是否已经释放。
    回收目录在实验结束时统一清空。

    调用前必须已经关闭所有句柄，并且已经把该轮结果 flush + fsync 落盘——
    否则中途崩溃会同时丢掉数据和文件。
    """

    trash_dir = Path(trash_dir)
    trash_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        moved = False
        for attempt in range(10):
            try:
                os.replace(str(path), str(trash_dir / path.name))
                moved = True
                break
            except PermissionError:
                gc.collect()
                time.sleep(0.1)
            except OSError:
                gc.collect()
                time.sleep(0.1)
        if not moved:
            if log is not None:
                log(f"临时文件无法释放（句柄未释放）：{path}")
            continue

    quota_bytes = int(float(config.MICRO_LOOP_TRASH_QUOTA_GB) * (1024 ** 3))
    used = directory_bytes(trash_dir)
    if used > quota_bytes:
        raise RuntimeError(
            f"临时文件回收目录已达 {format_gib(used)}，超过配额 "
            f"{float(config.MICRO_LOOP_TRASH_QUOTA_GB):g} GiB。"
            "这说明临时文件没有按预期释放，继续下去会写满磁盘，已安全终止。"
        )


def clip_temp_paths(temp_dir: Path, stem: str) -> dict[str, Path]:
    """
    一段临时录制的三个文件路径。

    输入：临时目录、文件名主干。
    输出：{raw, meta, frames}。
    """

    temp_dir = Path(temp_dir)
    return {
        "raw": temp_dir / f"{stem}.raw",
        "meta": temp_dir / f"{stem}_raw_meta.json",
        "frames": temp_dir / f"{stem}_frames.csv",
    }


def next_clip_stem(run_id: str, label: str, counter: int) -> str:
    """生成唯一的临时片段名，避免任何形式的覆盖。"""

    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(label))
    return f"{run_id}_{safe}_{int(counter):05d}"


# =============================================================================
# 5. 相机进程：内存缓冲 → 未压缩 RAW（无编码器、无手电筒门禁）
# =============================================================================

def _closed_loop_camera_status(status_queue: Any, event: str, **extra: Any) -> None:
    """向主进程发一条相机状态确认。"""

    status_queue.put(
        {"source": "camera", "event": event, "host_ns": time.perf_counter_ns(), **extra},
        timeout=1.0,
    )


def micro_closed_loop_camera_worker(
    error_queue: Any,
    command_queue: Any,
    status_queue: Any,
    stop_event: Any,
) -> None:
    """
    闭环微动的相机子进程：一次连接，按窗口把帧缓冲进内存再落盘。

    输入：错误队列、命令队列、状态队列、停止事件。
    输出：每个窗口一个 `{stem}.raw` + `{stem}_frames.csv` + `{stem}_raw_meta.json`（都是临时文件）。

    命令：
    - OPEN_WINDOW：开始把帧写进内存缓冲；
    - CLOSE_WINDOW：停止采集，把缓冲落盘成未压缩 RAW 并回报 WINDOW_SAVED；
    - SHUTDOWN：收尾退出。

    为什么不是边采边写盘：实测全幅 1936×1464 实时写盘每帧均值 2.27 ms、
    峰值 7.38 ms，而相机周期 7.56 ms —— 会丢 6.11% 的帧。memcpy 进内存约 0.3 ms，
    余量 25 倍。所以采集阶段只做"取帧 + 拷贝"，识别与落盘全部挪到片段结束之后。

    为什么在片段之间仍然持续取帧：MVS SDK 的节点队列会积压，
    下一次开录时就会拿到运动之前拍到的陈旧帧。持续取帧并丢弃是唯一可靠的做法，
    与 batch_camera_worker 的空闲预览一致。
    """

    import cv2

    from camera import HikCameraSource, _resize_for_preview

    buffer: np.ndarray | None = None
    active: dict[str, Any] | None = None
    previous_frame_id: int | None = None
    preview_last_ns = 0
    preview_period_ns = int(1_000_000_000 / float(config.MICRO_LOOP_PREVIEW_FPS))
    preview_window_name = "UR10 micro closed-loop camera"

    def close_window() -> None:
        """封口当前窗口：把内存缓冲落盘成未压缩 RAW，写旁车文件，回报主程序。"""

        nonlocal active, buffer, previous_frame_id
        if active is None:
            raise RuntimeError("没有可关闭的相机窗口。")

        width = int(active["width"])
        height = int(active["height"])
        frame_count = int(active["count"])
        raw_path: Path = active["raw_path"]
        frames_path: Path = active["frames_path"]
        meta_path: Path = active["meta_path"]

        # WINDOW_SAVED 是“这一段的所有文件已经可读”的承诺。必须先把逐帧 CSV
        # 显式刷盘并关闭，不能依赖 active 清空后的对象回收时机。
        active["frames_file"].flush()
        active["frames_file"].close()

        with raw_path.open("xb") as raw_file:
            if frame_count and buffer is not None:
                raw_file.write(memoryview(buffer[:frame_count]).cast("B"))
        bytes_written = int(raw_path.stat().st_size)
        expected_bytes = frame_count * width * height
        if bytes_written != expected_bytes:
            raise RuntimeError(
                f"RAW 文件大小与帧数不一致：{raw_path.name} 实际 {bytes_written} 字节，"
                f"按 {frame_count} 帧 × {width}×{height} 应为 {expected_bytes} 字节。"
            )

        meta = {
            "kind": "MICRO_CLOSED_LOOP_RAW_META",
            "clip_label": active["clip_label"],
            "raw_path": str(raw_path),
            "width": width,
            "height": height,
            "dtype": "uint8",
            "frame_count": frame_count,
            "bytes_written": bytes_written,
            "expected_bytes": expected_bytes,
            "first_frame_id": active["first_frame_id"],
            "last_frame_id": active["last_frame_id"],
            "record_start_ns": active["record_start_ns"],
            "record_stop_ns": active["record_stop_ns"],
            "camera_actual_fps": active["camera_fps"],
            "buffer_capacity": int(active["capacity"]),
            "truncated": bool(active["truncated"]),
            "dropped_after_full": int(active["dropped"]),
            "readme": (
                "离线读取：np.memmap(raw_path, dtype=np.uint8, mode='r', "
                f"shape=(frame_count, {height}, {width}))。"
                "未压缩位精确数据，不含任何有损编码；本文件是临时文件，处理完即删。"
            ),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
        )

        _closed_loop_camera_status(
            status_queue,
            "WINDOW_SAVED",
            clip_label=active["clip_label"],
            raw_path=str(raw_path),
            frames_path=str(frames_path),
            meta_path=str(meta_path),
            width=width,
            height=height,
            frame_count=frame_count,
            bytes_written=bytes_written,
            first_frame_id=active["first_frame_id"],
            last_frame_id=active["last_frame_id"],
            record_start_ns=active["record_start_ns"],
            record_stop_ns=active["record_stop_ns"],
            camera_actual_fps=active["camera_fps"],
            truncated=bool(active["truncated"]),
            dropped_after_full=int(active["dropped"]),
        )
        active = None
        # 落盘可能耗时上百毫秒；下一帧属于窗口间空闲期，不把落盘耗时误判成丢帧。
        previous_frame_id = None

    def snapshot_window(command: dict[str, Any]) -> None:
        """复制当前窗口末尾若干帧供父进程计算本轮 command，原窗口继续录制。"""

        if active is None or buffer is None:
            raise RuntimeError("没有可快照的活动相机窗口。")
        requested = max(1, int(command.get("frame_count", config.MICRO_LOOP_MEASURE_FRAMES)))
        count = int(active["count"])
        if count < requested:
            raise RuntimeError(
                f"活动窗口只有 {count} 帧，不能生成要求 {requested} 帧的运动前快照。"
            )
        start = count - requested
        stem = str(command["stem"])
        paths = clip_temp_paths(Path(command["temp_dir"]), stem)
        for key in ("raw", "meta", "frames"):
            if paths[key].exists():
                raise FileExistsError(f"拒绝覆盖相机快照：{paths[key]}")

        with paths["raw"].open("xb") as raw_file:
            raw_file.write(memoryview(buffer[start:count]).cast("B"))
        source_rows = list(active["frame_rows"])[start:count]
        snapshot_rows = [
            {**row, "segment_frame_index": index}
            for index, row in enumerate(source_rows)
        ]
        write_rows_csv(paths["frames"], snapshot_rows, RAW_FRAME_COLUMNS)
        bytes_written = int(paths["raw"].stat().st_size)
        meta = {
            "kind": "MICRO_CLOSED_LOOP_RAW_SNAPSHOT_META",
            "clip_label": str(command.get("clip_label", "before_snapshot")),
            "source_clip_label": active["clip_label"],
            "raw_path": str(paths["raw"]),
            "width": int(active["width"]),
            "height": int(active["height"]),
            "dtype": "uint8",
            "frame_count": requested,
            "bytes_written": bytes_written,
            "expected_bytes": requested * int(active["width"]) * int(active["height"]),
            "first_frame_id": snapshot_rows[0]["frame_id"],
            "last_frame_id": snapshot_rows[-1]["frame_id"],
            "record_start_ns": snapshot_rows[0]["host_ns"],
            "record_stop_ns": snapshot_rows[-1]["host_ns"],
            "camera_actual_fps": active["camera_fps"],
            "buffer_capacity": requested,
            "truncated": False,
            "dropped_after_full": 0,
            "source_start_index": start,
            "source_stop_index": count - 1,
            "readme": "本文件是活动命令窗口中、机器人命令前最后若干帧的临时副本。",
        }
        paths["meta"].write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        _closed_loop_camera_status(
            status_queue,
            "WINDOW_SNAPSHOT",
            clip_label=meta["clip_label"],
            source_clip_label=active["clip_label"],
            raw_path=str(paths["raw"]),
            frames_path=str(paths["frames"]),
            meta_path=str(paths["meta"]),
            frame_count=requested,
            first_frame_id=meta["first_frame_id"],
            last_frame_id=meta["last_frame_id"],
            record_start_ns=meta["record_start_ns"],
            record_stop_ns=meta["record_stop_ns"],
        )

    try:
        if config.SHOW_PREVIEW:
            cv2.namedWindow(preview_window_name)

        with HikCameraSource() as source:
            iterator = iter(source)
            fps = float(source.actual_camera_fps or config.EXPECTED_VISION_FPS or 30.0)

            warmup_start_ns: int | None = None
            first = None
            warmup_frames = 0
            print(
                f"[状态] 工业相机正在预热 {config.MICRO_LOOP_CAMERA_WARMUP_SECONDS:g} 秒，"
                "初始化预览并排空启动帧",
                flush=True,
            )
            while first is None:
                packet = next(iterator)
                if stop_event.is_set():
                    return
                warmup_frames += 1
                if warmup_start_ns is None:
                    warmup_start_ns = int(packet.host_ns)
                if (
                    config.SHOW_PREVIEW
                    and int(packet.host_ns) - preview_last_ns >= preview_period_ns
                ):
                    preview_last_ns = int(packet.host_ns)
                    cv2.imshow(
                        preview_window_name,
                        _resize_for_preview(
                            packet.frame, max_width=int(config.MICRO_LOOP_PREVIEW_MAX_WIDTH)
                        ),
                    )
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        raise RuntimeError("相机预览窗口收到 q/Esc。")
                if (
                    int(packet.host_ns) - warmup_start_ns
                    >= int(float(config.MICRO_LOOP_CAMERA_WARMUP_SECONDS) * 1_000_000_000)
                ):
                    first = packet

            full_height, full_width = int(first.frame.shape[0]), int(first.frame.shape[1])
            _closed_loop_camera_status(
                status_queue,
                "READY",
                actual_camera_fps=fps,
                full_width=full_width,
                full_height=full_height,
                warmup_frames=warmup_frames,
            )
            print(
                f"[状态] 工业相机预热完成（排空 {warmup_frames} 帧），"
                f"实际帧率约 {fps:.3f} fps，全幅 {full_width}×{full_height}"
                "（不裁剪、不缩放，保留完整分辨率）",
                flush=True,
            )

            for packet in (item for pair in ((first,), iterator) for item in pair):
                if stop_event.is_set():
                    break
                frame = packet.frame

                while True:
                    try:
                        command = command_queue.get_nowait()
                    except Empty:
                        break
                    action = str(command.get("action", ""))
                    if action == "SHUTDOWN":
                        stop_event.set()
                        break
                    if action == "OPEN_WINDOW":
                        if active is not None:
                            raise RuntimeError("上一段相机窗口尚未关闭。")
                        paths = clip_temp_paths(Path(command["temp_dir"]), str(command["stem"]))
                        for key in ("raw", "meta", "frames"):
                            if paths[key].exists():
                                raise FileExistsError(f"拒绝覆盖相机输出：{paths[key]}")
                        capacity = int(command["capacity_frames"])
                        if buffer is None or int(buffer.shape[0]) < capacity:
                            # 只分配一次，之后所有窗口复用同一块内存。
                            buffer = np.empty((capacity, full_height, full_width), dtype=np.uint8)
                        frames_file = paths["frames"].open(
                            "x", encoding="utf-8-sig", newline="", buffering=262_144
                        )
                        frames_writer = csv.DictWriter(
                            frames_file, fieldnames=list(RAW_FRAME_COLUMNS)
                        )
                        frames_writer.writeheader()
                        active = {
                            "clip_label": str(command.get("clip_label", "")),
                            "raw_path": paths["raw"],
                            "meta_path": paths["meta"],
                            "frames_path": paths["frames"],
                            "frames_file": frames_file,
                            "frames_writer": frames_writer,
                            "width": full_width,
                            "height": full_height,
                            "capacity": capacity,
                            "count": 0,
                            "truncated": False,
                            "dropped": 0,
                            "first_frame_id": None,
                            "last_frame_id": None,
                            "record_start_ns": None,
                            "record_stop_ns": None,
                            "camera_fps": fps,
                            "frame_rows": [],
                        }
                        previous_frame_id = None
                        _closed_loop_camera_status(
                            status_queue,
                            "WINDOW_OPENED",
                            clip_label=active["clip_label"],
                            width=full_width,
                            height=full_height,
                            capacity=capacity,
                        )
                    elif action == "CLOSE_WINDOW":
                        close_window()
                    elif action == "SNAPSHOT_WINDOW":
                        snapshot_window(command)
                    else:
                        raise ValueError(f"未知相机命令：{action!r}")

                if active is not None and buffer is not None:
                    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if gray.dtype != np.uint8:
                        raise RuntimeError(f"闭环微动只支持 uint8 帧，实际 dtype={gray.dtype}。")
                    if gray.shape != (full_height, full_width):
                        raise RuntimeError(
                            f"帧尺寸在实验中途变化：{gray.shape} vs "
                            f"({full_height}, {full_width})，数据不可用。"
                        )
                    # 唯一在采集路径上的操作：一次 memcpy。不做灰度转换之外的任何处理。
                    if int(active["count"]) >= int(active["capacity"]):
                        # 缓冲写满：**不是致命错误，也绝不静默丢弃**。
                        # RAW 与逐帧 CSV 必须等长，所以两者一起停写，并如实计数上报。
                        # 上层看到 truncated 就把这一轮标成 MEASUREMENT_LOST 继续跑，
                        # 因为正常窗口（约 2 s）离上限（6 s）还有 3 倍余量，
                        # 真写满说明机械臂在反复拖长稳定等待，那是被测现象的一部分。
                        if not active["truncated"]:
                            active["truncated"] = True
                            print(
                                f"[状态] 相机缓冲写满（{active['capacity']} 帧），"
                                f"本段后续帧不再记录；这一轮将标记为测量丢失。",
                                flush=True,
                            )
                        active["dropped"] = int(active["dropped"]) + 1
                    else:
                        buffer[int(active["count"])] = gray
                        frame_row = {
                            "segment_frame_index": int(active["count"]),
                            "frame_id": int(packet.frame_id),
                            "host_ns": int(packet.host_ns),
                            "camera_timestamp_raw": packet.camera_timestamp_raw,
                        }
                        active["frames_writer"].writerow(frame_row)
                        active["frame_rows"].append(frame_row)
                        if active["first_frame_id"] is None:
                            active["first_frame_id"] = int(packet.frame_id)
                            active["record_start_ns"] = int(packet.host_ns)
                        active["last_frame_id"] = int(packet.frame_id)
                        active["record_stop_ns"] = int(packet.host_ns)
                        active["count"] = int(active["count"]) + 1
                    previous_frame_id = int(packet.frame_id)

                if (
                    config.SHOW_PREVIEW
                    and int(packet.host_ns) - preview_last_ns >= preview_period_ns
                ):
                    preview_last_ns = int(packet.host_ns)
                    cv2.imshow(
                        preview_window_name,
                        _resize_for_preview(
                            frame, max_width=int(config.MICRO_LOOP_PREVIEW_MAX_WIDTH)
                        ),
                    )
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        raise RuntimeError("相机预览窗口收到 q/Esc。")
    except StopIteration:
        try:
            error_queue.put("闭环微动相机没有返回任何图像。", timeout=1.0)
        except Exception:
            pass
        stop_event.set()
    except Exception as exc:
        try:
            error_queue.put(f"闭环微动相机进程异常：{type(exc).__name__}: {exc}", timeout=1.0)
        except Exception:
            pass
        stop_event.set()
    finally:
        if active is not None:
            try:
                active["frames_file"].close()
            except Exception:
                pass
        cv2.destroyAllWindows()


# =============================================================================
# 6. 机器人进程：单点 moveL 微动 + 基于位置的稳定判据
# =============================================================================

def _closed_loop_robot_status(status_queue: Any, event: str, **extra: Any) -> None:
    """向主进程发一条机器人状态确认。"""

    status_queue.put(
        {"source": "robot", "event": event, "host_ns": time.perf_counter_ns(), **extra},
        timeout=1.0,
    )


def micro_closed_loop_robot_worker(
    error_queue: Any,
    command_queue: Any,
    status_queue: Any,
    stop_event: Any,
    run_dir: Path,
) -> None:
    """
    闭环微动的机器人子进程：一次连接，按命令执行单轴微动。

    输入：错误/命令/状态队列、停止事件、运行目录。
    输出：`robot_log.csv`（125 Hz 全速率状态）和每条命令的确认消息。

    命令：
    - STABILIZE：等位置稳定（不运动），用于每组开始前把画面定住；
    - MOVE：执行一次 μm 级修正（走 send_cartesian_micro_correction 这一层）；
    - RETURN_START：低速返回实验初始位姿并稳定；
    - SHUTDOWN：收尾退出。
    """

    from robot import (
        URRobot,
        _dashboard_safety_status_is_normal,
        _rtde_safety_mode_is_normal,
        _tcp_speed_norm_m_s,
    )

    robot = URRobot()
    start_pose: list[float] | None = None
    active_file: Any = None
    period = 1.0 / float(config.ROBOT_RECORD_HZ)

    def write_state(state: dict[str, Any]) -> None:
        """把一条 125 Hz 状态写进 robot_log.csv。"""

        if start_pose is not None:
            pose_for_guard = list(state.get("actual_tcp_pose") or [])
            try:
                enforce_fixed_safety_envelope(
                    pose_for_guard,
                    start_pose,
                    label="运动中实测 TCP",
                )
            except Exception:
                # 实测位置一旦越过固定包络，先请求受控停止，再把异常交给上层收尾。
                robot.stop_motion()
                raise

        if active_file is None:
            return
        pose = list(state.get("actual_tcp_pose") or [None] * 6)
        speed = list(state.get("actual_tcp_speed") or [None] * 6)
        active_file.write(
            ",".join(
                str(value)
                for value in (
                    int(state["host_ns"]),
                    state.get("robot_timestamp_s"),
                    state.get("robot_mode"),
                    state.get("safety_mode"),
                    *pose,
                    *speed,
                )
            )
            + "\n"
        )

    def stabilise(description: str) -> bool:
        """等位置稳定；超时降级为固定等待并如实回报。"""

        settled = _wait_until_position_stable(
            robot,
            stop_event,
            write_state,
            timeout_s=float(config.MICRO_LOOP_STABLE_TIMEOUT_S),
            description=description,
        )
        if settled is None:
            time.sleep(float(config.MICRO_LOOP_MOTION_SETTLE_SECONDS))
            return False
        return True

    try:
        if not config.ROBOT_RELATIVE_MOTION_ENABLED:
            raise PermissionError(
                "ROBOT_RELATIVE_MOTION_ENABLED=False，XY 微动闭环测试被锁定。"
            )

        robot.connect(require_control=True)
        state = robot.read_state()
        write_state(state)

        dashboard_normal = _dashboard_safety_status_is_normal(robot.dashboard_info)
        rtde_normal = _rtde_safety_mode_is_normal(state)
        if dashboard_normal is False or rtde_normal is False:
            raise RuntimeError("机器人安全状态不是 NORMAL，拒绝开始自动微动。")
        if _tcp_speed_norm_m_s(state) > float(config.ROBOT_EXPERIMENT_STOP_SPEED_MM_S) / 1000.0:
            raise RuntimeError("机器人当前不处于静止状态，拒绝开始自动微动。")

        start_pose = [float(value) for value in robot.current_tcp_pose()]
        enforce_fixed_safety_envelope(
            start_pose, start_pose, label="固定安全包络中心"
        )
        active_file = (Path(run_dir) / "robot_log.csv").open(
            "x", encoding="utf-8", newline="", buffering=262_144
        )
        active_file.write(
            ",".join(
                ["host_ns", "robot_timestamp_s", "robot_mode", "safety_mode"]
                + [f"tcp_{name}" for name in ("x", "y", "z", "rx", "ry", "rz")]
                + [f"tcp_speed_{name}" for name in ("x", "y", "z", "rx", "ry", "rz")]
            )
            + "\n"
        )
        active_file.flush()

        _closed_loop_robot_status(
            status_queue,
            "READY",
            start_pose=start_pose,
            safety=state.get("safety_mode"),
            fixed_safety_radius_m=float(config.MICRO_LOOP_FIXED_SAFETY_RADIUS_M),
        )
        print(
            f"[状态] UR10 已连接并确认静止，实验初始 TCP 位姿 "
            f"({start_pose[0]:.6f}, {start_pose[1]:.6f}, {start_pose[2]:.6f}) m；"
            f"固定球形安全包络半径 "
            f"{float(config.MICRO_LOOP_FIXED_SAFETY_RADIUS_M) * 100.0:.1f} cm",
            flush=True,
        )

        while not stop_event.is_set():
            try:
                command = command_queue.get(timeout=0.2)
            except Empty:
                # 即使当前没有命令，也以 5 Hz 监控实际 TCP；包络中心始终是启动位姿。
                write_state(robot.read_state())
                continue
            action = str(command.get("action", ""))

            if action == "SHUTDOWN":
                break

            if action == "STABILIZE":
                ok = stabilise("闭环微动前的稳定等待")
                _closed_loop_robot_status(
                    status_queue,
                    "STABILIZED",
                    settle_ok=ok,
                    pose=[float(value) for value in robot.current_tcp_pose()],
                )
                continue

            if action == "MOVE":
                axis = str(command["axis"])
                delta_um = float(command["delta_um"])
                result = send_cartesian_micro_correction(
                    robot,
                    axis,
                    delta_um,
                    safety_center_pose=start_pose,
                    stop_event=stop_event,
                    write_state=write_state,
                    # 探针走 1 mm，是微动步的 20~200 倍，必须给它更长的超时，
                    # 否则会每次都被判超时并退化成"靠固定 settle 猜"。
                    settle_timeout_s=command.get("settle_timeout_s"),
                )
                _closed_loop_robot_status(
                    status_queue,
                    "MOVE_DONE",
                    axis=axis,
                    delta_um=delta_um,
                    achieved_um=result["achieved_um"],
                    tcp_before=result["tcp_before"],
                    tcp_after=result["tcp_after"],
                    transverse_um=result["transverse_um"],
                    settle_ok=result["settle_ok"],
                    command_start_ns=result["command_start_ns"],
                    command_end_ns=result["command_end_ns"],
                    command_status=result["status"],
                    note=result["note"],
                )
                continue

            if action == "RETURN_START":
                if start_pose is None:
                    raise RuntimeError("尚未记录实验初始位姿，无法返回。")
                from robot import Waypoint, validate_trajectory

                current = [float(value) for value in robot.current_tcp_pose()]
                trajectory = [
                    Waypoint(
                        "P0",
                        current,
                        float(config.MICRO_LOOP_RETURN_SPEED_MM_S) / 1000.0,
                        float(config.MICRO_LOOP_ACCELERATION_M_S2),
                        0.0,
                    ),
                    Waypoint(
                        "P1",
                        list(start_pose),
                        float(config.MICRO_LOOP_RETURN_SPEED_MM_S) / 1000.0,
                        float(config.MICRO_LOOP_ACCELERATION_M_S2),
                        0.0,
                    ),
                ]
                enforce_fixed_safety_envelope(
                    current, start_pose, label="返回初始位姿的轨迹起点"
                )
                enforce_fixed_safety_envelope(
                    start_pose, start_pose, label="返回初始位姿的轨迹目标"
                )
                validate_trajectory(trajectory, for_real_robot=True, pose_source="relative")
                robot.verify_controller_safety_limits(trajectory)
                robot.execute_trajectory(trajectory)
                deadline = time.perf_counter() + float(config.ROBOT_MOTION_TIMEOUT_S)
                while robot.motion_in_progress():
                    if stop_event.is_set():
                        robot.stop_motion()
                        break
                    if time.perf_counter() >= deadline:
                        robot.stop_motion()
                        raise TimeoutError("返回实验初始位姿超时，已调用 stopL。")
                    write_state(robot.read_state())
                    time.sleep(period)
                settle_ok = stabilise("返回实验初始位姿后的稳定等待")
                final_pose = [float(value) for value in robot.current_tcp_pose()]
                enforce_fixed_safety_envelope(
                    final_pose, start_pose, label="返回后的实测 TCP"
                )
                drift_mm = float(
                    np.linalg.norm(
                        np.asarray(final_pose[:3], dtype=float)
                        - np.asarray(start_pose[:3], dtype=float)
                    )
                    * 1000.0
                )
                _closed_loop_robot_status(
                    status_queue,
                    "RETURNED",
                    pose=final_pose,
                    drift_mm=drift_mm,
                    settle_ok=settle_ok,
                )
                continue

            raise ValueError(f"未知机器人命令：{action!r}")

    except Exception as exc:
        try:
            error_queue.put(f"闭环微动机器人进程异常：{type(exc).__name__}: {exc}", timeout=1.0)
        except Exception:
            pass
        stop_event.set()
    finally:
        if active_file is not None:
            try:
                active_file.close()
            except Exception:
                pass
        try:
            robot.stop_motion()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
