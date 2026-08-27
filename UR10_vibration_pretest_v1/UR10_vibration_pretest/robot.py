"""
UR10 轨迹生成、安全检查、离线 dry-run 和 ur_rtde 真机适配。

从用户操作看，本文件承担两类任务：
1. 离线阶段：把 config.py 里的 A/B/C 点变成轨迹，检查数值是否离谱，生成 dry-run 报告。
2. 真机阶段：连接 UR，读取状态，必要时发送 moveL 轨迹，并把机器人状态写入实验日志。

读代码时建议按“会不会让机械臂运动”分层看：
- Waypoint、build_trajectory()、validate_trajectory()、estimate_trajectory() 只是内存计算，不连接机器人。
- run_robot_dry_run() 只生成报告和轨迹图，不导入 ur_rtde，不可能发运动命令。
- URRobot.connect(require_control=False) 只读状态；require_control=True 才创建可发送运动命令的接口。
- run_robot_test() 和 robot_worker() 才可能接触真机运动，它们会先经过配置、安全检查和人工确认。

安全边界：
- dry-run 只能说明代码里的坐标、速度、线段长度没有明显数字错误，不代表现场一定安全。
- 示例点位绝不能直接发给 UR10；实机前必须用示教器确认点位和工作区。
- 只有 ROBOT_POSES_CONFIRMED=True、控制器安全检查通过、起点检查通过、操作者再次确认后，真机运动才会继续。
"""

from __future__ import annotations

import json
import math
import re
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Full
from typing import Any

import numpy as np

import config


# =============================================================================
# 1. 统一轨迹数据结构：把 config.py 里的点位包装成后续流程共同理解的路径点
# =============================================================================

@dataclass(slots=True)
class Waypoint:
    """
    一条机器人 TCP 轨迹中的单个路径点。

    输入来源：
    - config.POINT_A / POINT_B / POINT_C 提供 pose；
    - config.LINEAR_SPEED_M_S、LINEAR_ACCELERATION_M_S2、BLEND_RADIUS_M 提供运动参数。

    输出去向：
    - dry-run 报告会把 Waypoint 写成人能检查的 JSON；
    - 真机运动时 execute_trajectory() 会把 Waypoint 转成 ur_rtde 的 moveL path 格式。

    实验作用：
    - 它把“末端要到哪里”和“用多快速度去”放在同一个对象里，避免轨迹生成、
      安全检查和真机执行各自理解一套格式。
    - pose 顺序固定为 [x, y, z, rx, ry, rz]，位置单位 m，姿态为旋转向量 rad。
    """

    name: str
    pose: list[float]
    speed_m_s: float
    acceleration_m_s2: float
    blend_radius_m: float = 0.0


def build_trajectory() -> list[Waypoint]:
    """
    根据 config.TRAJECTORY_TYPE 把 A/B/C 点组装成一条轨迹清单。

    输入：
    - TRAJECTORY_TYPE 决定使用 static、line 还是 l_shape；
    - POINT_A/B/C 提供末端位姿；
    - 速度、加速度和交融半径来自 config.py。

    输出：
    - list[Waypoint]，供 dry-run、robot_test 或 experiment 后续检查/执行。

    实验作用：
    - static 只返回 A 点，用于静止记录；
    - line 返回 A→B，用于直线运动工况；
    - l_shape 返回 A→B→C，B 点可带 blend 半径，用于转弯工况。
    这里还没有连接机器人，只是在内存里生成“将来可能要走的路径清单”。
    """

    # 本段创建起点 A。输入是 config.POINT_A；输出是轨迹清单中的第一个 Waypoint。
    # 真机模式要求操作者已经把 TCP 放到 A 附近，后续不会自动发送“回到 A”的盲目运动。
    point_a = Waypoint(
        "A",
        list(config.POINT_A),
        config.LINEAR_SPEED_M_S,
        config.LINEAR_ACCELERATION_M_S2,
        0.0,
    )

    # 本段处理静止工况。输出只有 A 点，表示机器人不执行后续位移，只配合记录静止数据。
    if config.TRAJECTORY_TYPE == "static":
        return [point_a]

    # 本段创建 B 点。line 工况中 B 是终点；l_shape 工况中 B 是中间转弯点。
    # 只有 l_shape 才给 B 点使用 blend 半径，让轨迹更接近圆滑过渡。
    point_b = Waypoint(
        "B",
        list(config.POINT_B),
        config.LINEAR_SPEED_M_S,
        config.LINEAR_ACCELERATION_M_S2,
        config.BLEND_RADIUS_M if config.TRAJECTORY_TYPE == "l_shape" else 0.0,
    )
    # 本段处理直线工况。输出 A→B，适合匀速直线段的振动预实验。
    if config.TRAJECTORY_TYPE == "line":
        return [point_a, point_b]

    # 本段创建 C 点。C 是 L 形轨迹终点，终点 blend 必须为 0，避免机器人不精确到达终点。
    point_c = Waypoint(
        "C",
        list(config.POINT_C),
        config.LINEAR_SPEED_M_S,
        config.LINEAR_ACCELERATION_M_S2,
        0.0,
    )
    if config.TRAJECTORY_TYPE == "l_shape":
        return [point_a, point_b, point_c]

    raise ValueError(f"未知 TRAJECTORY_TYPE：{config.TRAJECTORY_TYPE}")


def _position(pose: list[float]) -> np.ndarray:
    """
    从 UR 的 6 维 TCP 位姿中取出 xyz 位置。

    输入：pose = [x, y, z, rx, ry, rz]。
    输出：NumPy 向量 [x, y, z]。

    实验作用：轨迹长度、工作区范围、起点误差这些检查只关心末端位置，
    不在这里处理姿态 rx/ry/rz。
    """

    # 本段把普通列表变成 NumPy 向量，方便后续做相减和求长度。
    return np.asarray(pose[:3], dtype=float)


def _segment_length(start: Waypoint, end: Waypoint) -> float:
    """
    计算相邻两个 TCP 路径点之间的直线距离。

    输入：相邻的 Waypoint，例如 A 和 B。
    输出：TCP 在笛卡尔空间中的距离，单位 m。

    实验作用：这不是机器人关节真实运动长度，但可以快速发现点位是否离得离谱，
    例如单位写错、示教坐标抄错或 A/B/C 相距过远。
    """

    # 本段做几何距离计算。输入是两个 xyz；输出是位移向量的模长。
    return float(np.linalg.norm(_position(end.pose) - _position(start.pose)))


def validate_trajectory(
    trajectory: list[Waypoint],
    *,
    for_real_robot: bool,
    pose_source: str = "absolute",
) -> list[str]:
    """
    对一条轨迹做启动前的纯软件安全检查。

    输入：
    - trajectory：build_trajectory() 生成的 Waypoint 清单；
    - for_real_robot：这次检查是否准备进入真机流程。

    输出：
    - 检查通过时返回 messages，供终端和 dry-run 报告展示；
    - 发现格式、工作区、线段长度、交融半径或真机确认问题时抛错。

    实验作用：
    - dry-run 时用于帮你发现轨迹数字是否明显不合理；
    - 真机前用于拦住未确认点位或明显危险配置。
    它仍然看不见桌子、夹具、电缆和人员，所以不能替代现场安全确认。
    """

    if not trajectory:
        raise ValueError("轨迹不能为空。")

    # 本段准备检查结果说明。它不是给机器人用的，而是给终端和报告中人读的。
    messages: list[str] = []
    axes = ("x", "y", "z")

    # 本段逐个检查路径点本身。
    # 输入：每个 Waypoint；输出：确认 pose、速度、加速度、blend 和 xyz 工作区都基本合理。
    for waypoint in trajectory:
        if len(waypoint.pose) != 6:
            raise ValueError(f"{waypoint.name} 点位姿不是 6 个数。")

        if not all(math.isfinite(value) for value in waypoint.pose):
            raise ValueError(f"{waypoint.name} 点包含无穷值或 NaN。")

        if waypoint.speed_m_s <= 0 or waypoint.acceleration_m_s2 <= 0:
            raise ValueError(f"{waypoint.name} 点的速度和加速度必须为正数。")

        if waypoint.blend_radius_m < 0:
            raise ValueError(f"{waypoint.name} 点的交融半径不能为负数。")

        # 本段只检查 TCP 的 xyz 是否在软件工作区内。
        # 姿态 rx/ry/rz 不在这里设统一范围，因为不同末端工具和安装方式差异很大。
        for axis_index, axis_name in enumerate(axes):
            lower, upper = config.WORKSPACE_LIMITS_M[axis_name]
            value = waypoint.pose[axis_index]
            if not lower <= value <= upper:
                raise ValueError(
                    f"{waypoint.name} 点 {axis_name}={value:.4f} m "
                    f"超出软件工作区 [{lower:.4f}, {upper:.4f}] m。"
                )

        messages.append(
            f"{waypoint.name}: xyz={waypoint.pose[:3]} m, "
            f"speed={waypoint.speed_m_s:.3f} m/s, "
            f"acc={waypoint.acceleration_m_s2:.3f} m/s², "
            f"blend={waypoint.blend_radius_m:.4f} m"
        )

    # 本段检查相邻点之间的路径段。
    # 输入：A→B、B→C 等相邻点对；输出：每段长度，或在线段过短/过长时停止。
    # 实验作用：两个点单独都在工作区内，不代表它们之间的距离适合当前预实验。
    lengths: list[float] = []
    for start, end in zip(trajectory[:-1], trajectory[1:]):
        length = _segment_length(start, end)
        lengths.append(length)

        if length <= 1e-6:
            raise ValueError(f"{start.name}→{end.name} 两点位置重合。")
        max_segment_length_m = (
            config.ROBOT_EXPERIMENT_MAX_SEGMENT_MM / 1000.0
            if pose_source == "relative"
            else config.MAX_SEGMENT_LENGTH_M
        )
        if length > max_segment_length_m:
            raise ValueError(
                f"{start.name}→{end.name} 长 {length:.4f} m，"
                f"超过允许上限 {max_segment_length_m:.4f} m。"
            )
        messages.append(f"{start.name}→{end.name}: 线段长度 {length:.4f} m")

    # 本段检查终点是否被错误设置了 blend。
    # 实验作用：终点带 blend 时机器人可能不会精确到达最后指定点，所以这里直接禁止。
    if trajectory[-1].blend_radius_m != 0:
        raise ValueError("最后一个轨迹点的 blend_radius_m 必须为 0。")

    # 本段检查中间点的交融半径是否过大。
    # 输入：中间点 blend 和前后两段长度；输出：允许圆滑过渡或拒绝过大的 blend。
    # 实验作用：blend 太大可能吞掉整段路径，让实际轨迹偏离你以为的 A→B→C。
    for index in range(1, len(trajectory) - 1):
        radius = trajectory[index].blend_radius_m
        allowed = 0.5 * min(lengths[index - 1], lengths[index])
        if radius >= allowed:
            raise ValueError(
                f"{trajectory[index].name} 点交融半径 {radius:.4f} m 过大；"
                f"根据相邻线段，本程序要求小于 {allowed:.4f} m。"
            )

    # 本段只在真机流程中启用。
    # 输入：for_real_robot 和 ROBOT_POSES_CONFIRMED；输出：允许真机继续或拒绝示例点位。
    # 实验作用：dry-run 可以用示例点学习流程，但真机绝不能用未确认点位。
    if pose_source not in {"absolute", "relative"}:
        raise ValueError(f"未知 pose_source={pose_source!r}。")

    if for_real_robot and pose_source == "absolute" and not config.ROBOT_POSES_CONFIRMED:
        raise PermissionError(
            "ROBOT_POSES_CONFIRMED=False。示例位姿只能用于 dry_run；"
            "请先用示教器确认 A/B/C 和真实工作区，再明确改为 True。"
        )

    if (
        for_real_robot
        and pose_source == "relative"
        and not config.ROBOT_RELATIVE_MOTION_ENABLED
    ):
        raise PermissionError(
            "ROBOT_RELATIVE_MOTION_ENABLED=False。三个相对运动实验默认禁止真机运动；"
            "请在实验室完成通信测试和现场安全确认后再明确改为 True。"
        )

    return messages


# =============================================================================
# 2. 轨迹时间估计和 dry-run 输出：把轨迹变成可人工检查的报告
# =============================================================================

def _trapezoid_time(distance: float, speed: float, acceleration: float) -> float:
    """
    粗略估计一段直线运动需要多久。

    输入：线段长度、目标速度、加速度。
    输出：一维梯形/三角速度模型下的估计时间，单位 s。

    实验作用：dry-run 报告用它检查运动时长数量级是否合理。
    UR 控制器实际还会考虑姿态、交融、关节限制和内部规划，所以这个结果不是精确预测。
    """

    # 本段计算达到目标速度前需要的时间和距离。
    # 后面用它判断本段运动是否有足够距离进入匀速阶段。
    acceleration_time = speed / acceleration
    acceleration_distance = 0.5 * acceleration * acceleration_time**2

    # 本段处理短距离运动。输出是“三角速度曲线”时间：加速后立刻减速，没有匀速段。
    if 2.0 * acceleration_distance >= distance:
        return 2.0 * math.sqrt(distance / acceleration)

    # 本段处理较长距离运动。输出是“加速 -> 匀速 -> 减速”的梯形速度曲线时间。
    cruise_distance = distance - 2.0 * acceleration_distance
    return 2.0 * acceleration_time + cruise_distance / speed


def estimate_trajectory(trajectory: list[Waypoint]) -> dict[str, Any]:
    """
    把轨迹清单转换成 dry-run 报告中的统计信息。

    输入：Waypoint 清单。
    输出：普通 dict，包含每段长度、粗略时间、总长度、总时长和 xyz 包围盒。

    实验作用：这些信息服务“人检查轨迹数量级”，不是给机器人控制器执行。
    你可以通过总长度、时间和 xyz_min/max 快速判断点位是否落在预期实验区域。
    """

    segments: list[dict[str, Any]] = []

    # 本段逐段估算路径。输入是相邻点对；输出是 segments 中的一条段信息。
    for start, end in zip(trajectory[:-1], trajectory[1:]):
        distance = _segment_length(start, end)
        duration = _trapezoid_time(
            distance,
            end.speed_m_s,
            end.acceleration_m_s2,
        )
        # 本段把一段运动整理成 JSON 友好的字段，后面直接写入 trajectory_report.json。
        segments.append(
            {
                "from": start.name,
                "to": end.name,
                "distance_m": distance,
                "estimated_duration_s": duration,
                "speed_m_s": end.speed_m_s,
                "acceleration_m_s2": end.acceleration_m_s2,
                "end_blend_radius_m": end.blend_radius_m,
            }
        )

    # 本段取出所有路径点位置，用于计算整条轨迹覆盖的 xyz 范围。
    positions = np.asarray([waypoint.pose[:3] for waypoint in trajectory], dtype=float)

    # 本段输出整条轨迹的汇总。它是 dry-run 的“检查摘要”，不是运动命令。
    return {
        "trajectory_type": config.TRAJECTORY_TYPE,
        "for_real_robot": False,
        "robot_poses_confirmed": config.ROBOT_POSES_CONFIRMED,
        "segments": segments,
        "total_distance_m": float(sum(item["distance_m"] for item in segments)),
        "estimated_motion_duration_s": float(
            sum(item["estimated_duration_s"] for item in segments)
        ),
        "xyz_min_m": positions.min(axis=0).tolist(),
        "xyz_max_m": positions.max(axis=0).tolist(),
    }


def _direction_sign(value: str, axis: str) -> int:
    """把 UI/命令行中的 +X/-X/+Y/-Y 转成符号。"""

    expected = {f"+{axis}": 1, f"-{axis}": -1}
    if value not in expected:
        raise ValueError(f"{axis} 方向必须是 {sorted(expected)} 之一，实际为 {value!r}。")
    return expected[value]


def _finite_positive(name: str, value: float) -> float:
    """校验一个有限正数参数。"""

    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是有限正数。")
    return value


def _validate_speed_mm_s(name: str, value: float) -> float:
    """校验 UI 输入速度，并返回 m/s。"""

    speed_mm_s = _finite_positive(name, value)
    if not (
        config.ROBOT_EXPERIMENT_MIN_SPEED_MM_S
        <= speed_mm_s
        <= config.ROBOT_EXPERIMENT_MAX_SPEED_MM_S
    ):
        raise ValueError(
            f"{name}={speed_mm_s:.3f} mm/s 超出允许范围 "
            f"[{config.ROBOT_EXPERIMENT_MIN_SPEED_MM_S:.3f}, "
            f"{config.ROBOT_EXPERIMENT_MAX_SPEED_MM_S:.3f}] mm/s。"
        )
    return speed_mm_s / 1000.0


def _offset_pose(start_pose: list[float], dx_m: float, dy_m: float) -> list[float]:
    """基于本次实际 A 点生成只改 X/Y、姿态保持不变的新 TCP 位姿。"""

    pose = list(start_pose)
    if len(pose) != 6:
        raise ValueError("当前 TCP 位姿必须是 6 个数。")
    pose[0] += float(dx_m)
    pose[1] += float(dy_m)
    return pose


def _relative_parameters_for_json(parameters: dict[str, Any]) -> dict[str, Any]:
    """把相对运动参数整理成 JSON 友好的纯数据。"""

    return {
        key: (float(value) if isinstance(value, (int, float)) else value)
        for key, value in parameters.items()
    }


def build_relative_motion_trajectory(
    experiment_mode: str,
    start_pose: list[float],
    parameters: dict[str, Any],
) -> tuple[list[Waypoint], dict[str, Any]]:
    """
    根据本次实际 TCP 起点 A 和 UI 参数生成三种相对运动轨迹。

    输出的最后一个 Waypoint 始终回到 A，且所有点保持 A 的 Rx/Ry/Rz。
    """

    acceleration = _finite_positive(
        "acceleration_m_s2",
        float(parameters.get("acceleration_m_s2", config.ROBOT_EXPERIMENT_ACCELERATION_M_S2)),
    )
    x_direction = str(parameters.get("x_direction", "+X"))
    y_direction = str(parameters.get("y_direction", "+Y"))
    sign_x = _direction_sign(x_direction, "X")
    sign_y = _direction_sign(y_direction, "Y")
    blend_m = float(parameters.get("blend_mm", config.ROBOT_RELATIVE_BLEND_MM)) / 1000.0
    if not math.isfinite(blend_m) or blend_m < 0:
        raise ValueError("blend_mm 必须是有限非负数。")

    point_a = Waypoint(
        "A",
        list(start_pose),
        config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S / 1000.0,
        acceleration,
        0.0,
    )
    waypoints: list[Waypoint]
    computed: dict[str, Any] = {
        "experiment_mode": experiment_mode,
        "start_pose_a": list(start_pose),
        "x_direction": x_direction,
        "y_direction": y_direction,
        "acceleration_m_s2": acceleration,
        "input_parameters": _relative_parameters_for_json(parameters),
    }

    if experiment_mode == "x_line_experiment":
        speed_m_s = _validate_speed_mm_s(
            "speed_mm_s",
            float(parameters.get("speed_mm_s", config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S)),
        )
        total_time_s = _finite_positive(
            "total_time_s",
            float(parameters.get("total_time_s", config.ROBOT_EXPERIMENT_DEFAULT_TOTAL_TIME_S)),
        )
        if total_time_s > config.ROBOT_EXPERIMENT_MAX_TOTAL_TIME_S:
            raise ValueError("total_time_s 超出允许上限。")
        distance_m = speed_m_s * total_time_s / 2.0
        point_b = Waypoint(
            "B",
            _offset_pose(start_pose, sign_x * distance_m, 0.0),
            speed_m_s,
            acceleration,
            0.0,
        )
        point_a_return = Waypoint("A_return", list(start_pose), speed_m_s, acceleration, 0.0)
        waypoints = [point_a, point_b, point_a_return]
        computed.update(
            {
                "nominal_total_motion_time_s": total_time_s,
                "one_way_distance_m": distance_m,
                "dx_m": sign_x * distance_m,
                "dy_m": 0.0,
                "point_b": point_b.pose,
            }
        )
    elif experiment_mode == "xy_line_experiment":
        speed_m_s = _validate_speed_mm_s(
            "speed_mm_s",
            float(parameters.get("speed_mm_s", config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S)),
        )
        total_time_s = _finite_positive(
            "total_time_s",
            float(parameters.get("total_time_s", config.ROBOT_EXPERIMENT_DEFAULT_TOTAL_TIME_S)),
        )
        angle_deg = float(parameters.get("angle_deg", config.ROBOT_EXPERIMENT_DEFAULT_ANGLE_DEG))
        if (
            not math.isfinite(angle_deg)
            or not 0.0 <= angle_deg <= config.ROBOT_EXPERIMENT_MAX_ANGLE_DEG
        ):
            raise ValueError(
                f"angle_deg 必须在 0 到 {config.ROBOT_EXPERIMENT_MAX_ANGLE_DEG:.1f} 之间。"
            )
        if total_time_s > config.ROBOT_EXPERIMENT_MAX_TOTAL_TIME_S:
            raise ValueError("total_time_s 超出允许上限。")
        distance_m = speed_m_s * total_time_s / 2.0
        angle_rad = math.radians(angle_deg)
        dx_m = sign_x * distance_m * math.cos(angle_rad)
        dy_m = sign_y * distance_m * math.sin(angle_rad)
        point_b = Waypoint(
            "B",
            _offset_pose(start_pose, dx_m, dy_m),
            speed_m_s,
            acceleration,
            0.0,
        )
        point_a_return = Waypoint("A_return", list(start_pose), speed_m_s, acceleration, 0.0)
        waypoints = [point_a, point_b, point_a_return]
        computed.update(
            {
                "nominal_total_motion_time_s": total_time_s,
                "one_way_distance_m": distance_m,
                "angle_deg": angle_deg,
                "dx_m": dx_m,
                "dy_m": dy_m,
                "point_b": point_b.pose,
            }
        )
    elif experiment_mode == "xy_l_experiment":
        x_speed_m_s = _validate_speed_mm_s(
            "x_speed_mm_s",
            float(parameters.get("x_speed_mm_s", config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S)),
        )
        y_speed_m_s = _validate_speed_mm_s(
            "y_speed_mm_s",
            float(parameters.get("y_speed_mm_s", config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S)),
        )
        x_time_s = _finite_positive(
            "x_one_way_time_s",
            float(parameters.get("x_one_way_time_s", config.ROBOT_EXPERIMENT_DEFAULT_X_ONE_WAY_TIME_S)),
        )
        y_time_s = _finite_positive(
            "y_one_way_time_s",
            float(parameters.get("y_one_way_time_s", config.ROBOT_EXPERIMENT_DEFAULT_Y_ONE_WAY_TIME_S)),
        )
        if max(x_time_s, y_time_s) > config.ROBOT_EXPERIMENT_MAX_TOTAL_TIME_S:
            raise ValueError("L 折线单段时间超出允许上限。")
        dx_m = sign_x * x_speed_m_s * x_time_s
        dy_m = sign_y * y_speed_m_s * y_time_s
        point_b = Waypoint(
            "B",
            _offset_pose(start_pose, dx_m, 0.0),
            x_speed_m_s,
            acceleration,
            blend_m,
        )
        point_c = Waypoint(
            "C",
            _offset_pose(start_pose, dx_m, dy_m),
            y_speed_m_s,
            acceleration,
            0.0,
        )
        point_b_return = Waypoint(
            "B_return",
            point_b.pose,
            y_speed_m_s,
            acceleration,
            0.0,
        )
        point_a_return = Waypoint("A_return", list(start_pose), x_speed_m_s, acceleration, 0.0)
        waypoints = [point_a, point_b, point_c, point_b_return, point_a_return]
        computed.update(
            {
                "nominal_total_motion_time_s": 2.0 * (x_time_s + y_time_s),
                "dx_m": dx_m,
                "dy_m": dy_m,
                "point_b": point_b.pose,
                "point_c": point_c.pose,
                "blend_m": blend_m,
            }
        )
    else:
        raise ValueError(f"未知相对运动模式：{experiment_mode}")

    validate_trajectory(waypoints, for_real_robot=False, pose_source="relative")
    return waypoints, computed


def _plot_trajectory(trajectory: list[Waypoint], output_path: Path) -> None:
    """
    把 dry-run 轨迹画成三维 TCP 路径示意图。

    输入：Waypoint 清单和 PNG 输出路径。
    输出：trajectory_3d.png。

    实验作用：帮助你直观看出 A/B/C 的相对位置和路径方向。
    图里的线只是 TCP 轨迹，不是机械臂连杆，也不包含桌面、夹具、电缆，因此不能作为真机安全依据。
    """

    # 本段只在真正需要画图时加载 matplotlib，避免普通轨迹计算被绘图库依赖拖住。
    import matplotlib

    # Agg 后端不需要桌面窗口，适合脚本直接保存 PNG。
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 本段把 Waypoint 转成三维坐标数组，供 matplotlib 画线和标点。
    positions = np.asarray([waypoint.pose[:3] for waypoint in trajectory], dtype=float)

    # 本段创建图像并画出 TCP 折线路径。
    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(111, projection="3d")

    axes.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        marker="o",
        linewidth=2,
    )
    # 本段在每个路径点旁标注 A/B/C，帮助人检查路径方向。
    for waypoint, position in zip(trajectory, positions):
        axes.text(position[0], position[1], position[2], f" {waypoint.name}")

    axes.set_xlabel("X / m")
    axes.set_ylabel("Y / m")
    axes.set_zlabel("Z / m")
    axes.set_title("TCP dry-run trajectory")
    axes.grid(True)
    # 本段调整布局，尽量避免坐标轴标签被裁掉。
    figure.tight_layout()

    # 本段输出 PNG 并释放绘图对象，避免多次运行时占用内存。
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_robot_dry_run() -> Path:
    """
    用户选择 robot_dry_run 后进入这里。

    输入：config.py 中的轨迹类型、A/B/C 点、速度、加速度、工作区限制。
    输出：robot_dry_run_时间戳 文件夹，其中包含 trajectory_report.json 和 trajectory_3d.png。

    实验作用：这是最安全的机器人相关入口。
    它只在软件里生成轨迹、检查数值、保存报告，不导入 ur_rtde，不连接 UR，不可能发送运动命令。
    """

    # 本段把配置参数变成轨迹清单。输入是 TRAJECTORY_TYPE 和 A/B/C；输出是 Waypoint 列表。
    trajectory = build_trajectory()

    # 本段做 dry-run 级别检查。for_real_robot=False 表示允许示例点用于软件演示，
    # 但仍检查格式、线段长度、工作区和 blend 是否明显不合理。
    messages = validate_trajectory(trajectory, for_real_robot=False)

    # 本段生成给人检查的轨迹统计摘要。
    estimate = estimate_trajectory(trajectory)

    # 本段创建本次 dry-run 的输出目录，避免覆盖旧报告。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.OUTPUT_ROOT / f"robot_dry_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 本段组装 JSON 报告。输入是轨迹、检查说明和估计摘要；输出是完整 dry-run 记录。
    report = {
        "kind": "ROBOT_DRY_RUN",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "warning": (
            "这些坐标只通过软件数值检查，没有验证真实可达性、奇异位形、"
            "桌面、夹具、电缆和人员安全。"
        ),
        "waypoints": [asdict(waypoint) for waypoint in trajectory],
        "checks": messages,
        "estimate": estimate,
    }

    # 本段保存机器和人都能读的 JSON 报告。
    report_path = output_dir / "trajectory_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 本段保存人眼更容易理解的三维轨迹示意图。
    _plot_trajectory(trajectory, output_dir / "trajectory_3d.png")

    # 本段在终端打印摘要，方便不打开文件也能确认这次 dry-run 的关键结果。
    print("[轨迹 dry-run] 数值检查通过，但这不等于真机安全检查通过。")
    for message in messages:
        print(f"  - {message}")
    print(
        f"[轨迹 dry-run] 总长度 {estimate['total_distance_m']:.4f} m，"
        f"粗估运动时间 {estimate['estimated_motion_duration_s']:.3f} s。"
    )
    print(f"[轨迹 dry-run] 报告：{report_path}")
    return output_dir


# =============================================================================
# 3. UR Dashboard 只读信息：连接控制器文本接口，但只查询状态
# =============================================================================

def _read_socket_line(connection: socket.socket) -> str:
    """
    从 UR Dashboard socket 读取一行文本回复。

    输入：已经连接到 29999 端口的 socket。
    输出：去掉换行后的字符串。

    实验作用：Dashboard 协议是“发一行命令，收一行回复”的文本协议。
    这里仅服务版本和状态读取，不解析也不发送运动命令。
    """

    # 本段准备接收缓冲区。socket 收到的是 bytes，后面才解码成字符串。
    chunks = bytearray()

    # 本段持续读取，直到遇到换行、连接关闭或达到最大长度。
    # 8192 字节是保护上限，避免异常连接持续发送内容导致内存增长。
    while len(chunks) < 8192:
        data = connection.recv(1024)

        # 空 bytes 通常表示连接被对方关闭。
        if not data:
            break

        # 本段把本次收到的数据追加到总缓冲区。
        chunks.extend(data)

        # Dashboard 正常回复以换行结束，看到换行就可以停止读取。
        if b"\n" in data:
            break

    # 本段把 UR 返回的字节转换成 Python 字符串；异常字符用替代符，避免读取状态时直接崩溃。
    return chunks.decode("utf-8", errors="replace").strip()


def read_dashboard_information() -> dict[str, str]:
    """
    通过 UR Dashboard 端口读取控制器版本和状态。

    输入：config.ROBOT_HOST、ROBOT_DASHBOARD_PORT、连接超时。
    输出：包含 greeting、Polyscope 版本、机器人模式、安全状态和控制器代际的 dict。

    实验作用：在 robot_test 或 experiment 开始前确认连到的是预期 UR 控制器。
    这一步只发只读状态查询，不发送运动命令；若端口不可用，RTDE 后续仍可尝试连接。
    """

    # 本段准备默认返回结构。即使某些读取失败，日志里的 dashboard 字段也尽量稳定。
    information = {
        "greeting": "",
        "polyscope_version": "",
        "robot_mode": "",
        "safety_status": "",
        "controller_generation": "unknown",
    }

    # 本段建立 Dashboard 文本连接。输入是 ROBOT_HOST 和端口；输出是可收发文本命令的 socket。
    with socket.create_connection(
        (config.ROBOT_HOST, int(config.ROBOT_DASHBOARD_PORT)),
        timeout=float(config.ROBOT_CONNECT_TIMEOUT_S),
    ) as connection:
        # 本段设置读写超时，避免网线/IP 错误时程序一直卡住。
        connection.settimeout(float(config.ROBOT_CONNECT_TIMEOUT_S))

        # 本段读取连接后控制器主动发来的 greeting。
        information["greeting"] = _read_socket_line(connection)

        # 本段定义只读状态查询命令。key 是保存字段名，command 是发给 Dashboard 的文本。
        commands = {
            "polyscope_version": "PolyscopeVersion",
            "robot_mode": "robotmode",
            "safety_status": "safetystatus",
        }
        for key, command in commands.items():
            # 本段发送一条只读命令，并读取一行回复。
            connection.sendall((command + "\n").encode("ascii"))
            information[key] = _read_socket_line(connection)

    # 本段从 PolyscopeVersion 文本中提取主版本号，用于粗略判断 CB3 或 e/UR 系列。
    version_text = information["polyscope_version"]
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", version_text)
    if match:
        major = int(match.group(1))
        if major == 3:
            information["controller_generation"] = "CB3"
        elif major >= 5:
            information["controller_generation"] = "e-Series/UR-Series"
        else:
            information["controller_generation"] = f"unclassified-major-{major}"

    return information


# =============================================================================
# 4. ur_rtde 适配器：把第三方库包装成项目内部固定的机器人接口
# =============================================================================

class URRobot:
    """
    项目内部和 UR 真机通信的统一包装类。

    输入：
    - config.py 中的 ROBOT_HOST、RTDE 频率和安全参数；
    - 上游生成并检查过的 Waypoint 轨迹。

    输出：
    - read_state() 返回可写入 JSON 的机器人状态；
    - execute_trajectory() 把轨迹提交给 UR 控制器；
    - stop_motion()/disconnect() 负责异常或正常收尾。

    实验作用：把第三方 ur_rtde 的字段名和连接细节集中在一个类里。
    以后若更换通信库，主要改这里；轨迹生成、主程序和分析文件不需要知道底层差异。
    """

    def __init__(self) -> None:
        self.receive: Any = None
        self.control: Any = None
        self.dashboard_info: dict[str, str] = {}
        self.motion_command_time: float | None = None

    def connect(self, *, require_control: bool) -> None:
        """
        建立机器人通信连接，并按需要决定是否创建运动控制接口。

        输入：
        - require_control=False：只建立 RTDE Receive，用于读取状态；
        - require_control=True：额外建立 RTDE Control，后续才可能发送 moveL。

        输出：
        - self.dashboard_info 保存 Dashboard 状态；
        - self.receive 保存只读状态接口；
        - self.control 在需要运动控制时保存控制接口。

        实验作用：把“只读连接”和“可运动连接”分开。robot_test 默认只读；
        experiment 和显式允许的 robot_test 才会创建 control 接口。
        """

        try:
            # 本段先读取 Dashboard 状态。输入是 UR 的文本接口；输出是控制器版本和安全状态信息。
            # 这一步失败不一定代表 RTDE 失败，所以这里只记录警告并继续尝试 RTDE。
            self.dashboard_info = read_dashboard_information()
        except Exception as exc:
            self.dashboard_info = {
                "controller_generation": "unknown",
                "dashboard_error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[机器人警告] Dashboard 信息读取失败：{exc}")

        # 本段在真机连接时才导入 rtde_receive。
        # 实验作用：robot_dry_run 不会因为未安装 ur_rtde 而启动失败。
        try:
            import rtde_receive
        except ImportError as exc:
            raise RuntimeError(
                "真机模式需要 ur-rtde。请执行 pip install -r requirements.txt，"
                "或先切换到 robot_dry_run。"
            ) from exc

        # 本段建立只读状态接口。输出的 receive 只能读取关节角、TCP 位姿、速度等状态。
        self.receive = rtde_receive.RTDEReceiveInterface(
            config.ROBOT_HOST,
            float(config.ROBOT_RTDE_FREQUENCY),
        )

        if require_control:
            # 本段只有 require_control=True 时执行。
            # Control 接口能够发送运动命令，所以只有真机运动确实需要时才创建。
            try:
                import rtde_control
            except ImportError as exc:
                self.disconnect()
                raise RuntimeError("无法导入 rtde_control，ur-rtde 安装可能不完整。") from exc

            self.control = rtde_control.RTDEControlInterface(
                config.ROBOT_HOST,
                float(config.ROBOT_RTDE_FREQUENCY),
            )

    @staticmethod
    def _safe_call(
        target: Any,
        method_name: str,
        default: Any,
    ) -> Any:
        """
        安全读取不同 ur_rtde 版本中可能存在差异的状态字段。

        输入：目标对象、方法名和默认值。
        输出：方法调用结果；方法不存在或调用失败时返回默认值。

        实验作用：关节角、TCP 位姿这些核心字段必须成功读取；
        但某些次要状态字段在不同 ur_rtde 版本中可能不存在，缺失时记录 None，而不是伪造数据。
        """

        method = getattr(target, method_name, None)
        if method is None:
            return default
        try:
            return method()
        except Exception:
            return default

    def read_state(self) -> dict[str, Any]:
        """
        读取一条机器人状态，并整理成可写入 JSON 的普通字典。

        输入：RTDE Receive 当前读取到的机器人状态。
        输出：kind="ROBOT" 的 dict，包含主机时间戳、机器人时间戳、关节角、TCP 位姿、TCP 速度等。

        实验作用：完整实验中 robot_worker 会按 ROBOT_RECORD_HZ 反复调用它，
        把机器人状态和视觉帧写入同一个 run_log.txt，供后续对时和振动分析使用。
        这里会把 NumPy 数组或库对象转换成 list[float]、float、None 等 JSON 友好类型。
        """

        if self.receive is None:
            raise RuntimeError("尚未连接 RTDE Receive。")

        # 本段读取最核心的机器人状态。
        # actual_q 是 6 个关节角；tcp_pose 是末端位姿；tcp_speed 是末端当前速度。
        actual_q = self.receive.getActualQ()
        tcp_pose = self.receive.getActualTCPPose()
        tcp_speed = self.receive.getActualTCPSpeed()

        return {
            "kind": "ROBOT",
            "host_ns": time.perf_counter_ns(),
            "robot_timestamp_s": float(
                self._safe_call(self.receive, "getTimestamp", math.nan)
            ),
            "actual_q_rad": [float(value) for value in actual_q],
            "actual_tcp_pose": [float(value) for value in tcp_pose],
            "actual_tcp_speed": [float(value) for value in tcp_speed],
            "target_tcp_pose": [
                float(value)
                for value in self._safe_call(self.receive, "getTargetTCPPose", [])
            ],
            "speed_scaling": float(
                self._safe_call(self.receive, "getSpeedScaling", math.nan)
            ),
            "robot_mode": self._safe_call(self.receive, "getRobotMode", None),
            "safety_mode": self._safe_call(self.receive, "getSafetyMode", None),
            "runtime_state": self._safe_call(self.receive, "getRuntimeState", None),
            "robot_status_bits": self._safe_call(
                self.receive,
                "getRobotStatus",
                None,
            ),
            "safety_status_bits": self._safe_call(
                self.receive,
                "getSafetyStatusBits",
                None,
            ),
        }

    def verify_controller_safety_limits(self, trajectory: list[Waypoint]) -> None:
        """
        让 UR 控制器自己判断每个目标位姿是否在安全限制内。

        输入：即将执行的 Waypoint 轨迹。
        输出：全部点通过时无返回值；任一点不可达或超限制时抛错。

        实验作用：这是软件 xyz 工作区之外的第二层检查。
        它更接近控制器真实判断，但仍不能识别桌面、夹具、电缆和人员。
        """

        if self.control is None:
            raise RuntimeError("控制连接未建立，无法调用控制器安全检查。")

        # 本段取出 ur_rtde 提供的控制器安全检查方法。
        # 如果当前库版本没有这个方法，为了安全，不允许跳过后继续运动。
        checker = getattr(self.control, "isPoseWithinSafetyLimits", None)
        if checker is None:
            raise RuntimeError(
                "当前 ur_rtde 版本没有 isPoseWithinSafetyLimits；"
                "为了安全，程序不允许跳过该检查后运动。"
            )

        # 本段逐点询问控制器。输入是每个 Waypoint 的 pose；输出是控制器层面的通过/拒绝。
        for waypoint in trajectory:
            if not bool(checker(waypoint.pose)):
                raise RuntimeError(
                    f"UR 控制器判定 {waypoint.name} 点不可达或超出当前安全限制。"
                )

    def current_tcp_pose(self) -> list[float]:
        """
        读取当前 TCP 位姿。

        输入：RTDE Receive 当前状态。
        输出：[x, y, z, rx, ry, rz]。

        实验作用：用于起点检查，只读状态，不引发任何运动。
        """

        if self.receive is None:
            raise RuntimeError("尚未连接 RTDE Receive。")
        return [float(value) for value in self.receive.getActualTCPPose()]

    def verify_at_start(self, start_pose: list[float]) -> None:
        """
        确认机器人当前 TCP 已经在轨迹起点附近。

        输入：轨迹起点 pose，通常是 POINT_A。
        输出：足够接近时无返回值；距离超过 START_POSE_TOLERANCE_M 时抛错。

        实验作用：要求操作者先用示教器把 TCP 放到 A 点附近。
        程序不会为了“自动回起点”而先执行一段未经观察的运动。
        """

        # 本段比较当前 TCP 和目标起点的 xyz 距离。姿态不在这里做距离范数比较。
        current = np.asarray(self.current_tcp_pose()[:3], dtype=float)
        target = np.asarray(start_pose[:3], dtype=float)
        error = float(np.linalg.norm(current - target))

        if error > config.START_POSE_TOLERANCE_M:
            raise RuntimeError(
                f"当前 TCP 与 A 点相差 {error * 1000.0:.2f} mm，"
                f"超过允许值 {config.START_POSE_TOLERANCE_M * 1000.0:.2f} mm。"
                "请先用示教器低速移动到安全起点，不要让程序盲目回位。"
            )

    def execute_trajectory(self, trajectory: list[Waypoint]) -> None:
        """
        把已检查通过的轨迹提交给 UR 控制器执行。

        输入：Waypoint 清单，第一项应是人工确认过的起点 A。
        输出：UR 控制器接受异步 moveL 路径；函数返回后运动可能仍在继续。

        实验作用：一次提交完整 moveL 路径，让 UR 控制器按自身周期插补。
        因此相机 132 fps 与 RTDE 125/500 Hz 不需要逐帧互相“对齐发送”，只需共享时间戳。
        """

        if self.control is None:
            raise RuntimeError("控制连接未建立，不能执行轨迹。")

        if len(trajectory) <= 1:
            return

        # 本段把内部 Waypoint 转成 ur_rtde 的 moveL path 格式。
        # 第一项 A 是已经人工到达的起点；真正发送的是后续 B/C，避免重复命令 A。
        # 单个 path 点格式：[x, y, z, rx, ry, rz, speed, acceleration, blend]。
        path: list[list[float]] = []
        for waypoint in trajectory[1:]:
            path.append(
                list(waypoint.pose)
                + [
                    float(waypoint.speed_m_s),
                    float(waypoint.acceleration_m_s2),
                    float(waypoint.blend_radius_m),
                ]
            )

        # 本段提交异步 moveL。
        # 第二个参数 True 表示命令发出后 Python 不会卡在 moveL 里，
        # 后续由 motion_in_progress() 轮询完成状态，同时还能继续读取机器人状态。
        accepted = bool(self.control.moveL(path, True))
        if not accepted:
            raise RuntimeError("UR 控制器拒绝了异步 moveL 路径。")
        self.motion_command_time = time.perf_counter()

    def motion_in_progress(self) -> bool:
        """
        判断异步 moveL 是否仍在执行。

        输入：RTDE Control 的异步进度接口，必要时退回到 TCP 速度判断。
        输出：True 表示运动仍在进行；False 表示当前没有异步运动。

        实验作用：robot_worker 用它决定何时写入 motion_finished，并通知 main.py 进入后记录阶段。
        """

        if self.control is None:
            return False

        progress_method = getattr(self.control, "getAsyncOperationProgress", None)
        if progress_method is not None:
            progress = int(progress_method())
            if progress >= 0:
                return True

            # 本段处理刚提交命令后的短暂同步间隙。
            # 刚提交后立刻得到 -1 不应误判为已经完成，否则 motion_finished 会过早写入。
            if (
                self.motion_command_time is not None
                and time.perf_counter() - self.motion_command_time < 0.20
            ):
                return True
            return False

        # 本段是极旧 ur_rtde 版本的保守退路：没有异步进度接口时，用 TCP 速度判断是否仍在动。
        # 正式使用前更建议升级 ur_rtde，让完成判断来自控制器异步状态。
        speed = np.asarray(self.receive.getActualTCPSpeed(), dtype=float)
        return float(np.linalg.norm(speed[:3])) > 1e-4

    def stop_motion(self) -> None:
        """
        异常退出时请求 UR 做受控停止。

        输入：RTDE Control 当前连接。
        输出：调用 stopL(1.0)，不发送新的目标位姿。

        实验作用：发生超时、主程序停止或异常时，尽量用受控减速停止，而不是继续执行旧轨迹。
        """

        if self.control is None:
            return
        try:
            self.control.stopL(1.0)
        except Exception as exc:
            print(f"[机器人警告] stopL 调用失败：{exc}")

    def disconnect(self) -> None:
        """
        断开机器人通信接口。

        输入：当前可能存在的 control 和 receive 连接。
        输出：接口 disconnect，并把对象引用清空。

        实验作用：正常结束或异常退出时释放 RTDE 连接，避免下次运行被旧连接占用。
        """

        for interface_name in ("control", "receive"):
            interface = getattr(self, interface_name)
            if interface is not None:
                try:
                    interface.disconnect()
                except Exception:
                    pass
                setattr(self, interface_name, None)


# =============================================================================
# 5. 真机前的人工确认：程序层面的最后一道防误触
# =============================================================================

def require_operator_confirmation(purpose: str) -> None:
    """
    在真机动作前要求操作者输入完整确认短语。

    输入：
    - purpose：这次确认对应的动作说明；
    - config.OPERATOR_CONFIRM_TEXT：必须逐字输入的确认文本。

    输出：
    - 文本一致时函数正常返回；
    - 文本不一致时抛出 PermissionError，取消本次运动。

    实验作用：降低误按回车或误触发脚本导致真机运动的风险。
    它不能替代示教器急停、防护区和现场风险评估，只是程序层最后一道防误触。
    """

    # 本段允许配置关闭人工确认。正式实验中建议保持开启。
    if not config.REQUIRE_OPERATOR_CONFIRMATION:
        return

    # 本段把即将发生的真机动作打印出来，让操作者知道自己正在确认什么。
    print()
    print(f"[安全确认] 即将进行：{purpose}")
    print("[安全确认] 请确认工作区无人、路径无夹具/桌面/线缆干涉，急停可立即触及。")

    # 本段要求输入完整短语，而不是简单 y/n，降低误触风险。
    typed = input(f"[安全确认] 请输入：{config.OPERATOR_CONFIRM_TEXT}\n> ").strip()

    # 本段执行确认判断。输入不完全一致就取消本次运动。
    if typed != config.OPERATOR_CONFIRM_TEXT:
        raise PermissionError("确认文本不一致，本次运动已取消。")


# =============================================================================
# 6. robot_test：用户单独验证 UR 连接，默认只读状态不运动
# =============================================================================

def _dashboard_safety_status_is_normal(dashboard_info: dict[str, str]) -> bool | None:
    """从 Dashboard 只读回复中判断 safety status 是否明确为 NORMAL。"""

    status = dashboard_info.get("safety_status")
    if not status:
        return None
    upper = status.upper()
    if "NORMAL" in upper:
        return True
    if "PROTECTIVE" in upper or "SAFEGUARD" in upper or "FAULT" in upper:
        return False
    return None


def _rtde_safety_mode_is_normal(state: dict[str, Any]) -> bool | None:
    """保守判断 RTDE safety mode；缺字段时返回未知。"""

    safety_mode = state.get("safety_mode")
    if safety_mode is None:
        return None
    try:
        # ur_rtde 常见枚举中 1 表示 NORMAL；未知非 1 值按异常处理。
        return int(safety_mode) == 1
    except (TypeError, ValueError):
        return None


def run_robot_connection_test() -> Path:
    """
    只读检测 UR 通信稳定性。

    本函数只调用 robot.connect(require_control=False) 和 read_state()，不会构造轨迹，
    不创建 RTDEControlInterface，也不会调用 move/stop 类运动接口。
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.OUTPUT_ROOT / f"robot_connection_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "robot_connection_log.txt"
    summary_path = output_dir / "connection_summary.json"

    robot = URRobot()
    samples: list[dict[str, Any]] = []
    read_errors: list[str] = []
    started_ns = time.perf_counter_ns()
    success = False
    failure_reason = ""
    status_printed = False

    try:
        robot.connect(require_control=False)
        duration_s = float(config.ROBOT_CONNECTION_TEST_SECONDS)
        period = 1.0 / float(config.ROBOT_RECORD_HZ)
        deadline = time.perf_counter() + duration_s
        next_sample = time.perf_counter()

        while time.perf_counter() < deadline:
            try:
                samples.append(robot.read_state())
            except Exception as exc:
                read_errors.append(f"{type(exc).__name__}: {exc}")
            next_sample += period
            time.sleep(max(0.0, next_sample - time.perf_counter()))

        elapsed_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
        actual_hz = len(samples) / elapsed_s if elapsed_s > 0 else 0.0
        expected_reads = max(1, int(duration_s * float(config.ROBOT_RECORD_HZ)))
        enough_reads = len(samples) >= max(1, int(expected_reads * 0.8))
        dashboard_safety_ok = _dashboard_safety_status_is_normal(robot.dashboard_info)
        rtde_safety_ok = _rtde_safety_mode_is_normal(samples[-1]) if samples else None

        if read_errors:
            failure_reason = f"读取状态异常 {len(read_errors)} 次。"
        elif not samples:
            failure_reason = "没有成功读取任何机器人状态。"
        elif not enough_reads:
            failure_reason = (
                f"成功读取次数 {len(samples)} 低于预期 {expected_reads} 的 80%。"
            )
        elif dashboard_safety_ok is False or rtde_safety_ok is False:
            failure_reason = "机器人安全状态不是 NORMAL。"
        else:
            success = True

        summary = {
            "kind": "ROBOT_CONNECTION_TEST_SUMMARY",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "robot_host": config.ROBOT_HOST,
            "duration_s": duration_s,
            "target_read_hz": config.ROBOT_RECORD_HZ,
            "successful_read_count": len(samples),
            "actual_read_hz": actual_hz,
            "read_error_count": len(read_errors),
            "read_errors": read_errors[:10],
            "dashboard": robot.dashboard_info,
            "dashboard_safety_ok": dashboard_safety_ok,
            "rtde_safety_ok": rtde_safety_ok,
            "success": success,
            "failure_reason": failure_reason,
            "first_state": samples[0] if samples else None,
            "last_state": samples[-1] if samples else None,
        }

        with log_path.open("w", encoding="utf-8", buffering=1) as file:
            file.write(json.dumps({"kind": "META", **summary}, ensure_ascii=False) + "\n")
            for sample in samples:
                file.write(json.dumps(sample, ensure_ascii=False, allow_nan=True) + "\n")

        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )

        if not success:
            print("[状态] 机械臂通信检测失败", flush=True)
            status_printed = True
            raise RuntimeError(failure_reason or "机械臂通信检测失败。")

        print("[状态] 机械臂通信检测通过", flush=True)
        status_printed = True
        print(f"[机器人] 通信检测记录：{log_path}", flush=True)
        return output_dir
    except Exception as exc:
        if not summary_path.exists():
            failure_reason = f"{type(exc).__name__}: {exc}"
            summary = {
                "kind": "ROBOT_CONNECTION_TEST_SUMMARY",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "robot_host": config.ROBOT_HOST,
                "duration_s": config.ROBOT_CONNECTION_TEST_SECONDS,
                "target_read_hz": config.ROBOT_RECORD_HZ,
                "successful_read_count": len(samples),
                "actual_read_hz": 0.0,
                "read_error_count": len(read_errors),
                "read_errors": read_errors[:10],
                "dashboard": robot.dashboard_info,
                "success": False,
                "failure_reason": failure_reason,
                "first_state": samples[0] if samples else None,
                "last_state": samples[-1] if samples else None,
            }
            log_path.write_text(
                json.dumps({"kind": "META", **summary}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
                encoding="utf-8",
            )
        if not status_printed:
            print("[状态] 机械臂通信检测失败", flush=True)
        raise
    finally:
        robot.disconnect()


def run_robot_test() -> Path:
    """
    用户选择 robot_test 后进入这里。

    输入：
    - 机器人 IP、RTDE 频率、A/B/C 点和 robot_test 安全开关；
    - 当 ROBOT_TEST_ALLOW_MOTION=True 时，还需要点位确认和人工确认。

    输出：
    - robot_test_时间戳 文件夹；
    - robot_test_log.txt，其中保存 META 和若干条 ROBOT 状态记录。

    实验作用：
    - 默认 False：只连接 Dashboard/RTDE Receive，读取关节角和 TCP 状态，不创建控制接口；
    - 手动 True：在轨迹检查、控制器安全检查、起点检查和人工确认后，执行低速 A→B 测试。
    """

    # 本段先把 config.py 中的 A/B/C 和 TRAJECTORY_TYPE 变成轨迹对象。
    # 即使默认不运动，也要检查这些配置是否基本合理，避免日志里留下明显坏配置。
    trajectory = build_trajectory()

    # 本段根据是否允许运动选择检查强度。
    # False 时只做数值检查；True 时额外要求 ROBOT_POSES_CONFIRMED=True。
    validate_trajectory(
        trajectory,
        for_real_robot=bool(config.ROBOT_TEST_ALLOW_MOTION),
    )

    # 本段创建本次 robot_test 的输出目录和日志路径，避免覆盖旧状态记录。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.OUTPUT_ROOT / f"robot_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "robot_test_log.txt"

    # 本段创建机器人通信包装对象。真正连接在 robot.connect() 中发生。
    robot = URRobot()
    try:
        # 本段建立连接。默认只创建 receive 读取状态；只有允许运动时才创建 control。
        robot.connect(require_control=bool(config.ROBOT_TEST_ALLOW_MOTION))

        # 本段把 Dashboard 信息打印出来，帮助确认连接到的是预期控制器。
        print(
            "[机器人] 控制器代际判断："
            f"{robot.dashboard_info.get('controller_generation', 'unknown')}"
        )
        print(f"[机器人] Dashboard：{robot.dashboard_info}")

        # 本段准备状态缓存。输入来自 robot.read_state()；输出最终写入 robot_test_log.txt。
        samples: list[dict[str, Any]] = []

        # 本段计算只读采样节奏。输出 sample_count 和 period，后续按 ROBOT_RECORD_HZ 读取状态。
        sample_count = max(1, int(config.PRE_RECORD_SECONDS * config.ROBOT_RECORD_HZ))
        period = 1.0 / config.ROBOT_RECORD_HZ
        next_time = time.perf_counter()

        # 本段先读一段状态，不发送任何运动命令。
        # 实验作用：确认 RTDE Receive 能稳定返回关节角、TCP 位姿和速度。
        for _ in range(sample_count):
            samples.append(robot.read_state())
            next_time += period
            time.sleep(max(0.0, next_time - time.perf_counter()))

        # 本段只在终端打印第一条摘要；完整状态序列会在后面写入日志。
        first = samples[0]
        print(f"[机器人] 当前关节角 rad：{first['actual_q_rad']}")
        print(f"[机器人] 当前 TCP：{first['actual_tcp_pose']}")

        # 本段只有显式打开 ROBOT_TEST_ALLOW_MOTION 时才进入。
        # 这是 robot_test 中唯一可能让真机运动的分支。
        if config.ROBOT_TEST_ALLOW_MOTION:
            if len(trajectory) < 2:
                raise ValueError("robot_test 允许运动时，TRAJECTORY_TYPE 不能是 static。")

            # 本段把完整轨迹缩减为 A→B 低速测试路径。
            # 输入：原轨迹的 A/B 点；输出：使用 ROBOT_TEST_SPEED/ACCEL 的 test_path。
            # 实验作用：robot_test 只验证最小运动闭环，不直接跑完整 experiment 轨迹。
            test_path = [
                Waypoint(
                    "A",
                    trajectory[0].pose,
                    config.ROBOT_TEST_SPEED_M_S,
                    config.ROBOT_TEST_ACCELERATION_M_S2,
                    0.0,
                ),
                Waypoint(
                    "B",
                    trajectory[1].pose,
                    config.ROBOT_TEST_SPEED_M_S,
                    config.ROBOT_TEST_ACCELERATION_M_S2,
                    0.0,
                ),
            ]
            # 本段再次检查缩减后的 A→B 测试路径，确保低速测试本身也满足真机条件。
            validate_trajectory(test_path, for_real_robot=True)

            # 本段调用 UR 控制器自己的安全限制判断。
            robot.verify_controller_safety_limits(test_path)

            # 本段要求当前 TCP 已经人工放在 A 点附近，避免程序先盲目回起点。
            robot.verify_at_start(test_path[0].pose)

            # 本段要求终端人工确认，防止误触发低速测试运动。
            require_operator_confirmation("UR10 A→B 低速单机测试")

            # 本段发送异步 moveL，并在运动期间持续读取状态。
            # 输出：samples 中追加运动过程的机器人状态；超时时 stopL 并报错。
            robot.execute_trajectory(test_path)
            deadline = time.perf_counter() + config.ROBOT_MOTION_TIMEOUT_S
            while robot.motion_in_progress():
                if time.perf_counter() > deadline:
                    robot.stop_motion()
                    raise TimeoutError("低速测试运动超时，已调用 stopL。")
                samples.append(robot.read_state())
                time.sleep(period)

        # 本段把 robot_test 的全部结果写入日志。
        # 输入：META、Dashboard 信息和 samples；输出：robot_test_log.txt 中逐行 JSON。
        with log_path.open("w", encoding="utf-8") as file:
            meta = {
                "kind": "META",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "robot_test",
                "motion_allowed": config.ROBOT_TEST_ALLOW_MOTION,
                "dashboard": robot.dashboard_info,
            }
            file.write(json.dumps(meta, ensure_ascii=False, allow_nan=True) + "\n")
            for record in samples:
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")

        print(f"[机器人] robot_test 完成，记录：{log_path}")
        return output_dir
    finally:
        # 本段是 robot_test 的统一收尾。
        # 无论正常结束还是异常退出，都尝试 stopL 和断开 RTDE，避免连接残留。
        robot.stop_motion()
        robot.disconnect()


# =============================================================================
# 7. 完整实验中的机器人子进程：按主进程信号记录状态并执行轨迹
# =============================================================================

def _put_record(record_queue: Any, record: dict[str, Any], stop_event: Any) -> None:
    """
    把机器人记录送入完整实验的统一日志队列。

    输入：record_queue、要写入的记录、stop_event。
    输出：记录进入队列；如果队列满，则置位 stop_event 并抛错。

    实验作用：队列满说明写日志进程跟不上。此时继续产生无法落盘、无法对时的数据意义不大，
    所以选择停止实验，而不是让机器人数据悄悄丢失。
    """

    try:
        record_queue.put(record, timeout=1.0)
    except Full as exc:
        stop_event.set()
        raise RuntimeError("记录队列已满，机器人数据无法写入。") from exc


def _record_robot_event(record_queue: Any, stop_event: Any, name: str, **extra: Any) -> None:
    """给相对运动实验写入统一事件。"""

    _put_record(
        record_queue,
        {
            "kind": "EVENT",
            "host_ns": time.perf_counter_ns(),
            "name": name,
            **extra,
        },
        stop_event,
    )


def _tcp_speed_norm_m_s(state: dict[str, Any]) -> float:
    """从一条机器人状态记录计算 TCP 线速度模长。"""

    speed = state.get("actual_tcp_speed") or []
    if len(speed) < 3:
        return math.inf
    return float(np.linalg.norm(np.asarray(speed[:3], dtype=float)))


def _wait_until_tcp_stopped(
    robot: URRobot,
    threshold_m_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    """等待 TCP 线速度降到阈值以下，并返回最后一条状态。"""

    deadline = time.perf_counter() + timeout_s
    last_state = robot.read_state()
    while time.perf_counter() < deadline:
        last_state = robot.read_state()
        if _tcp_speed_norm_m_s(last_state) <= threshold_m_s:
            return last_state
        time.sleep(0.02)
    raise TimeoutError("等待机器人停止超时。")


def relative_motion_robot_worker(
    record_queue: Any,
    error_queue: Any,
    experiment_mode: str,
    parameters: dict[str, Any],
    run_dir: Path,
    stop_event: Any,
    robot_ready: Any,
    motion_start: Any,
    motion_done: Any,
) -> None:
    """
    三个新相对运动实验共用的机器人子进程。

    它在 motion_start 置位前只连接、读状态、计算和验证轨迹，不发送运动命令。
    """

    robot = URRobot()
    motion_started = False
    normal_motion_finished = False
    robot_log_path = Path(run_dir) / "robot_log.txt"
    parameters_path = Path(run_dir) / "experiment_parameters.json"
    period = 1.0 / float(config.ROBOT_RECORD_HZ)
    stop_speed_m_s = float(config.ROBOT_EXPERIMENT_STOP_SPEED_MM_S) / 1000.0

    try:
        _record_robot_event(record_queue, stop_event, "hardware_preparation_started")
        if not config.ROBOT_RELATIVE_MOTION_ENABLED:
            raise PermissionError(
                "ROBOT_RELATIVE_MOTION_ENABLED=False，三个相对运动实验默认禁止真机运动。"
            )
        robot.connect(require_control=True)
        first_state = robot.read_state()
        dashboard_safety_ok = _dashboard_safety_status_is_normal(robot.dashboard_info)
        rtde_safety_ok = _rtde_safety_mode_is_normal(first_state)
        if dashboard_safety_ok is False or rtde_safety_ok is False:
            raise RuntimeError("机器人安全状态不是 NORMAL，拒绝运动。")
        if _tcp_speed_norm_m_s(first_state) > stop_speed_m_s:
            raise RuntimeError("当前 TCP 速度未接近停止，拒绝计算相对运动起点。")

        start_pose = [float(value) for value in first_state["actual_tcp_pose"]]
        trajectory, computed = build_relative_motion_trajectory(
            experiment_mode,
            start_pose,
            parameters,
        )
        validate_trajectory(trajectory, for_real_robot=True, pose_source="relative")
        robot.verify_controller_safety_limits(trajectory)

        experiment_parameters = {
            **computed,
            "kind": "RELATIVE_MOTION_PARAMETERS",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "robot_host": config.ROBOT_HOST,
            "waypoints": [asdict(waypoint) for waypoint in trajectory],
            "return_warning_mm": config.ROBOT_RETURN_WARNING_MM,
            "flash_gate": {
                "brightness_baseline_seconds": config.BRIGHTNESS_BASELINE_SECONDS,
                "flash_wait_timeout_seconds": config.FLASH_WAIT_TIMEOUT_SECONDS,
                "flash_min_duration_seconds": config.FLASH_MIN_DURATION_SECONDS,
                "vision_recovery_stable_seconds": config.VISION_RECOVERY_STABLE_SECONDS,
                "flash_recovery_timeout_seconds": config.FLASH_RECOVERY_TIMEOUT_SECONDS,
            },
            "industrial_camera_actual_fps": None,
        }
        parameters_path.write_text(
            json.dumps(experiment_parameters, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )

        _record_robot_event(
            record_queue,
            stop_event,
            "robot_ready",
            start_pose=start_pose,
            waypoint_count=len(trajectory),
        )
        print("[状态] 正在准备机械臂", flush=True)
        robot_ready.set()

        with robot_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            log_file.write(
                json.dumps(
                    {
                        "kind": "META",
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "mode": experiment_mode,
                        "robot_host": config.ROBOT_HOST,
                        "dashboard": robot.dashboard_info,
                    },
                    ensure_ascii=False,
                    allow_nan=True,
                )
                + "\n"
            )

            next_sample = time.perf_counter()
            while not motion_start.is_set():
                if stop_event.is_set():
                    return
                state = robot.read_state()
                log_file.write(json.dumps(state, ensure_ascii=False, allow_nan=True) + "\n")
                next_sample += period
                time.sleep(max(0.0, next_sample - time.perf_counter()))

            if stop_event.is_set():
                return

            _record_robot_event(
                record_queue,
                stop_event,
                "motion_command_sent",
                trajectory_mode=experiment_mode,
            )
            print("[状态] 机械臂运动中", flush=True)
            robot.execute_trajectory(trajectory)
            motion_started = True
            motion_deadline = time.perf_counter() + float(config.ROBOT_MOTION_TIMEOUT_S)

            while robot.motion_in_progress():
                if stop_event.is_set():
                    robot.stop_motion()
                    break
                if time.perf_counter() > motion_deadline:
                    robot.stop_motion()
                    raise TimeoutError("相对运动执行超过超时上限，已调用 stopL。")
                state = robot.read_state()
                log_file.write(json.dumps(state, ensure_ascii=False, allow_nan=True) + "\n")
                next_sample += period
                time.sleep(max(0.0, next_sample - time.perf_counter()))

            stopped_state = _wait_until_tcp_stopped(robot, stop_speed_m_s, 5.0)
            log_file.write(json.dumps(stopped_state, ensure_ascii=False, allow_nan=True) + "\n")
            if stop_event.is_set():
                _record_robot_event(
                    record_queue,
                    stop_event,
                    "robot_stopped_confirmed",
                    tcp_speed_m_s=_tcp_speed_norm_m_s(stopped_state),
                    interrupted=True,
                )
                return
            current_xyz = np.asarray(stopped_state["actual_tcp_pose"][:3], dtype=float)
            start_xyz = np.asarray(start_pose[:3], dtype=float)
            error_xyz_m = current_xyz - start_xyz
            error_norm_m = float(np.linalg.norm(error_xyz_m))
            warning = error_norm_m * 1000.0 > float(config.ROBOT_RETURN_WARNING_MM)
            _record_robot_event(
                record_queue,
                stop_event,
                "robot_stopped_confirmed",
                tcp_speed_m_s=_tcp_speed_norm_m_s(stopped_state),
            )
            _record_robot_event(
                record_queue,
                stop_event,
                "robot_return_error_measured",
                error_xyz_m=error_xyz_m.tolist(),
                error_norm_m=error_norm_m,
                warning=warning,
            )
            _record_robot_event(record_queue, stop_event, "motion_finished")
            print("[状态] 机械臂已回到初始位置", flush=True)
            normal_motion_finished = True
            motion_done.set()

            while not stop_event.is_set():
                state = robot.read_state()
                log_file.write(json.dumps(state, ensure_ascii=False, allow_nan=True) + "\n")
                next_sample += period
                time.sleep(max(0.0, next_sample - time.perf_counter()))
    except Exception as exc:
        try:
            error_queue.put(
                f"相对运动机器人进程异常：{type(exc).__name__}: {exc}",
                timeout=1.0,
            )
        except Full:
            pass
        stop_event.set()
    finally:
        if motion_started and not normal_motion_finished:
            robot.stop_motion()
        robot.disconnect()


def robot_worker(
    record_queue: Any,
    error_queue: Any,
    start_event: Any,
    stop_event: Any,
    robot_ready: Any,
    motion_done: Any,
) -> None:
    """
    完整实验中的机器人子进程入口。

    输入：
    - record_queue：把 ROBOT/EVENT 记录交给写日志进程；
    - error_queue：把机器人异常交回主进程；
    - start_event：主进程确认相机和机器人都 ready、且操作者确认后才置位；
    - stop_event：主进程或异常路径要求停止时置位；
    - robot_ready：本进程完成连接和安全检查后置位；
    - motion_done：本进程判断运动结束后置位。

    输出：
    - run_log.txt 中持续出现 ROBOT 状态记录和 motion_command_sent/motion_finished 事件；
    - 若出错，则 error_queue 中出现错误文字，并触发 stop_event。

    实验作用：这是完整 experiment 中唯一负责 UR 真机状态读取和轨迹执行的进程。
    阅读它时按时间顺序看：建轨迹 -> 真机安全检查 -> 等 start_event -> 预记录 -> 发运动命令 -> 持续记录。
    """

    robot = URRobot()
    motion_started = False

    try:
        # 本段是真机运动前的连续检查链。
        # 输入：config.py 中的轨迹和当前机器人状态；输出：robot_ready 置位或抛错停止。
        # 检查顺序：
        # 1. validate_trajectory：本程序自己的数字检查；
        # 2. robot.connect(require_control=True)：建立读取和控制连接；
        # 3. verify_controller_safety_limits：UR 控制器自己的安全限制检查；
        # 4. verify_at_start：确认当前 TCP 已在 A 点附近，不让程序盲目回起点。
        trajectory = build_trajectory()
        validate_trajectory(trajectory, for_real_robot=True)
        robot.connect(require_control=True)
        robot.verify_controller_safety_limits(trajectory)
        robot.verify_at_start(trajectory[0].pose)
        robot_ready.set()

        # 本段等待主进程统一开始信号。
        # 即使机器人已经 ready，也必须等相机 ready 和操作者确认完成后，main.py 才会设置 start_event。
        while not start_event.is_set():
            if stop_event.wait(0.05):
                return

        # 本段建立机器人子进程自己的采样时间轴。
        # 输入：start_event 触发后的当前时刻；输出：预记录结束时间 motion_due 和采样周期 period。
        # 实验作用：先静止记录 PRE_RECORD_SECONDS 秒，再提交运动命令，便于分析静止噪声基线。
        record_start = time.perf_counter()
        motion_due = record_start + config.PRE_RECORD_SECONDS
        period = 1.0 / config.ROBOT_RECORD_HZ
        next_sample = time.perf_counter()

        # 本段是机器人完整实验主循环。
        # 输入：stop_event、motion_due、robot.read_state()；输出：持续写入 ROBOT 记录和运动事件。
        while not stop_event.is_set():
            now = time.perf_counter()

            # 本段在预记录时间到达后只触发一次。
            # 输出 motion_command_sent 事件，并根据轨迹类型决定是否执行 moveL。
            if not motion_started and now >= motion_due:
                _put_record(
                    record_queue,
                    {
                        "kind": "EVENT",
                        "host_ns": time.perf_counter_ns(),
                        "name": "motion_command_sent",
                        "trajectory_type": config.TRAJECTORY_TYPE,
                    },
                    stop_event,
                )

                # static 只有一个点，不需要 moveL；line/l_shape 才执行轨迹。
                if len(trajectory) > 1:
                    robot.execute_trajectory(trajectory)
                motion_started = True

                # 本段处理 static 工况。
                # 没有真实运动命令，预记录完成后即可报告 motion_finished，主进程随后进入后记录计时。
                if len(trajectory) == 1:
                    motion_done.set()
                    _put_record(
                        record_queue,
                        {
                            "kind": "EVENT",
                            "host_ns": time.perf_counter_ns(),
                            "name": "motion_finished",
                        },
                        stop_event,
                    )

            # 本段每个采样周期都执行。
            # 输入：RTDE Receive 当前状态；输出：一条 kind="ROBOT" 记录进入统一日志队列。
            state = robot.read_state()
            _put_record(record_queue, state, stop_event)

            # 本段判断异步 moveL 是否已经结束。
            # 输出 motion_finished 事件和 motion_done 信号；之后继续采样，直到主进程结束后记录阶段。
            if motion_started and len(trajectory) > 1 and not robot.motion_in_progress():
                motion_done.set()
                _put_record(
                    record_queue,
                    {
                        "kind": "EVENT",
                        "host_ns": time.perf_counter_ns(),
                        "name": "motion_finished",
                    },
                    stop_event,
                )

                # 本段是运动完成后的状态记录。
                # 实验作用：主程序还要等 POST_RECORD_SECONDS，机器人进程在这段时间继续写状态。
                while not stop_event.is_set():
                    _put_record(record_queue, robot.read_state(), stop_event)
                    next_sample += period
                    time.sleep(max(0.0, next_sample - time.perf_counter()))
                break

            next_sample += period
            time.sleep(max(0.0, next_sample - time.perf_counter()))

            # 本段是运动超时保护。
            # 输入：运动已开始后的持续时间；输出：超时则 stopL 并把异常交给主程序处理。
            if motion_started and now - motion_due > config.ROBOT_MOTION_TIMEOUT_S:
                robot.stop_motion()
                raise TimeoutError("正式轨迹执行超过超时上限，已调用 stopL。")
    except Exception as exc:
        # 本段把机器人子进程内部异常转成主进程能读到的错误消息。
        # 输出：error_queue 中一条错误文本，并设置 stop_event 要求其他进程收尾。
        try:
            error_queue.put(
                f"机器人进程异常：{type(exc).__name__}: {exc}",
                timeout=1.0,
            )
        except Full:
            pass
        stop_event.set()
    finally:
        # 本段是机器人子进程的统一收尾。
        # 若已经发过运动但还没确认完成，先尝试 stopL；随后断开 RTDE 连接。
        if motion_started and not motion_done.is_set():
            robot.stop_motion()
        robot.disconnect()
