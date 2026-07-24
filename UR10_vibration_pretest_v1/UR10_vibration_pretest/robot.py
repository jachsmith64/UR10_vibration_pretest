"""
UR10 轨迹生成、安全检查、离线 dry-run 和 ur_rtde 真机适配。

所有可能让机械臂运动的调用都集中在本文件，并受“位姿已确认”和“操作者再次确认”两层开关保护。

给初学者的文件地图：
1. Waypoint 表示一个机器人 TCP 路径点，也就是“末端要到哪里、用多快速度去”。
2. build_trajectory() 根据 config.py 里的 A/B/C 点生成一条路径。
3. validate_trajectory() 做纯数字层面的安全检查，例如点是否超出工作区、线段是否过长。
4. run_robot_dry_run() 只做软件检查和报告，不导入 ur_rtde，也不会连接机器人。
5. URRobot 是真实机器人连接的包装类，所有 RTDE 读写都集中在这个类里。
6. run_robot_test() 和 robot_worker() 才可能接触真机；它们都必须先通过安全开关。

把这个文件分成两半会更容易读：
- 上半部分是“纸上算轨迹”：可以安全离线运行。
- 下半部分是“和真机说话”：必须确认 IP、工作区、点位和现场安全后才能运行。

安全边界：
- dry-run 不是实机安全证明，只能说明代码里的数值没有明显错误。
- 示例点位不能直接发给 UR10。
- 只有 ROBOT_POSES_CONFIRMED=True 且操作者再次确认后，真机运动才会继续。
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
# 1. 统一轨迹数据结构
# =============================================================================

@dataclass(slots=True)
class Waypoint:
    """
    一个 TCP 路径点及其运动参数。

    pose 顺序固定为 [x, y, z, rx, ry, rz]；speed、acceleration 和 blend_radius 都使用 SI 单位。

    字段解释：
    - name：点名，例如 A、B、C，主要用于日志和报错。
    - pose：UR 的 TCP 位姿。前三个数是位置 m，后三个数是旋转向量 rad。
    - speed_m_s：机器人沿路径运动的线速度，单位 m/s。
    - acceleration_m_s2：线加速度，单位 m/s²。
    - blend_radius_m：交融半径。中间点可用它圆滑过渡，终点必须为 0。
    """

    name: str
    pose: list[float]
    speed_m_s: float
    acceleration_m_s2: float
    blend_radius_m: float = 0.0


def build_trajectory() -> list[Waypoint]:
    """
    根据 TRAJECTORY_TYPE 生成 A、A→B 或 A→B→C 的统一路径。

    初学者理解方式：
    - static：只需要 A 点，用于静止记录；
    - line：从 A 到 B；
    - l_shape：从 A 到 B 再到 C，中间 B 点可以带 blend 半径。

    这里还没有连接机器人，只是在内存里组装“将来可能要走的路径清单”。
    """

    # A 点永远是轨迹起点。真机模式下程序要求机器人已经人工放到 A 附近，
    # 因此后面执行运动时不会再盲目发送“回到 A”的命令。
    point_a = Waypoint(
        "A",
        list(config.POINT_A),
        config.LINEAR_SPEED_M_S,
        config.LINEAR_ACCELERATION_M_S2,
        0.0,
    )

    # static 模式没有后续运动点，只有起点。
    if config.TRAJECTORY_TYPE == "static":
        return [point_a]

    point_b = Waypoint(
        "B",
        list(config.POINT_B),
        config.LINEAR_SPEED_M_S,
        config.LINEAR_ACCELERATION_M_S2,
        config.BLEND_RADIUS_M if config.TRAJECTORY_TYPE == "l_shape" else 0.0,
    )
    # line 模式只需要 A 和 B。
    if config.TRAJECTORY_TYPE == "line":
        return [point_a, point_b]

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
    从 6 维 UR 位姿中取出前三个位置坐标。

    UR 位姿格式是 [x, y, z, rx, ry, rz]：
    - x/y/z 决定 TCP 在空间中的位置；
    - rx/ry/rz 决定 TCP 姿态。
    计算线段长度时只需要位置，不需要姿态。
    """

    # np.asarray 让后续可以直接做向量减法和范数计算。
    return np.asarray(pose[:3], dtype=float)


def _segment_length(start: Waypoint, end: Waypoint) -> float:
    """
    计算相邻 TCP 点的直线距离，单位 m。

    这不是机器人真实关节运动长度，只是 TCP 在笛卡尔空间中的直线距离。
    用它可以快速发现点位是否离得离谱。
    """

    # 两个位置向量相减得到位移向量，np.linalg.norm 计算它的长度。
    return float(np.linalg.norm(_position(end.pose) - _position(start.pose)))


def validate_trajectory(
    trajectory: list[Waypoint],
    *,
    for_real_robot: bool,
) -> list[str]:
    """
    检查格式、工作区、线段长度和交融半径，返回便于展示的说明列表。

    软件检查只能发现数字层面的风险，无法看见桌子、夹具、电缆和人员。

    参数 for_real_robot 的意义：
    - False：用于 dry-run，只检查数值是否合理；
    - True：用于真机模式，还会要求 ROBOT_POSES_CONFIRMED=True。
    """

    if not trajectory:
        raise ValueError("轨迹不能为空。")

    # messages 不是给算法用的，而是给终端和报告看的“检查说明”。
    messages: list[str] = []
    axes = ("x", "y", "z")

    # 第一轮：逐个点检查。
    # 这里主要确认每个 Waypoint 本身没有明显错误。
    for waypoint in trajectory:
        if len(waypoint.pose) != 6:
            raise ValueError(f"{waypoint.name} 点位姿不是 6 个数。")

        if not all(math.isfinite(value) for value in waypoint.pose):
            raise ValueError(f"{waypoint.name} 点包含无穷值或 NaN。")

        if waypoint.speed_m_s <= 0 or waypoint.acceleration_m_s2 <= 0:
            raise ValueError(f"{waypoint.name} 点的速度和加速度必须为正数。")

        if waypoint.blend_radius_m < 0:
            raise ValueError(f"{waypoint.name} 点的交融半径不能为负数。")

        # 只检查 TCP 的 xyz 是否在软件工作区内。
        # 姿态 rx/ry/rz 不在这里用范围限制，因为不同工具姿态差异很大。
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

    # 第二轮：检查相邻点之间的线段。
    # 两个点单独看都在工作区内，不代表它们之间的连线长度合理。
    lengths: list[float] = []
    for start, end in zip(trajectory[:-1], trajectory[1:]):
        length = _segment_length(start, end)
        lengths.append(length)

        if length <= 1e-6:
            raise ValueError(f"{start.name}→{end.name} 两点位置重合。")
        if length > config.MAX_SEGMENT_LENGTH_M:
            raise ValueError(
                f"{start.name}→{end.name} 长 {length:.4f} m，"
                f"超过 MAX_SEGMENT_LENGTH_M={config.MAX_SEGMENT_LENGTH_M:.4f} m。"
            )
        messages.append(f"{start.name}→{end.name}: 线段长度 {length:.4f} m")

    # 只有内部路径点才允许使用 blend；终点设置 blend 会让机器人可能不到达指定终点。
    if trajectory[-1].blend_radius_m != 0:
        raise ValueError("最后一个轨迹点的 blend_radius_m 必须为 0。")

    # blend 半径不能大到超过相邻线段的一半，否则圆滑过渡可能吞掉整段路径。
    for index in range(1, len(trajectory) - 1):
        radius = trajectory[index].blend_radius_m
        allowed = 0.5 * min(lengths[index - 1], lengths[index])
        if radius >= allowed:
            raise ValueError(
                f"{trajectory[index].name} 点交融半径 {radius:.4f} m 过大；"
                f"根据相邻线段，本程序要求小于 {allowed:.4f} m。"
            )

    # 最后一关：只有真机模式才要求人为确认点位。
    # dry-run 允许使用示例点，因为它不会发送给机器人。
    if for_real_robot and not config.ROBOT_POSES_CONFIRMED:
        raise PermissionError(
            "ROBOT_POSES_CONFIRMED=False。示例位姿只能用于 dry_run；"
            "请先用示教器确认 A/B/C 和真实工作区，再明确改为 True。"
        )

    return messages


# =============================================================================
# 2. 轨迹时间估计和 dry-run 输出
# =============================================================================

def _trapezoid_time(distance: float, speed: float, acceleration: float) -> float:
    """
    用一维梯形/三角速度曲线粗估单段运动时间。

    UR 控制器会考虑姿态、交融和内部限制，所以该结果只用于发现数量级错误，不是精确预测。
    """

    # 若能加速到目标速度，acceleration_time 是从 0 加速到 speed 需要的时间。
    acceleration_time = speed / acceleration

    # acceleration_distance 是加速阶段走过的距离。
    acceleration_distance = 0.5 * acceleration * acceleration_time**2

    # 如果加速距离的两倍已经超过总距离，说明还没来得及达到目标速度就要减速。
    # 这种情况是“三角速度曲线”。
    if 2.0 * acceleration_distance >= distance:
        return 2.0 * math.sqrt(distance / acceleration)

    # 否则就是“加速 -> 匀速 -> 减速”的梯形速度曲线。
    cruise_distance = distance - 2.0 * acceleration_distance
    return 2.0 * acceleration_time + cruise_distance / speed


def estimate_trajectory(trajectory: list[Waypoint]) -> dict[str, Any]:
    """
    汇总每段长度、粗略时间、总长度和包围盒，供 dry-run 报告使用。

    返回的是普通 dict，方便直接写入 JSON。
    这些信息用于人检查轨迹数量级，而不是给机器人控制器执行。
    """

    segments: list[dict[str, Any]] = []

    # 遍历相邻点对，例如 A->B、B->C。
    for start, end in zip(trajectory[:-1], trajectory[1:]):
        # 每一段都单独估计距离和运动时间。
        distance = _segment_length(start, end)
        duration = _trapezoid_time(
            distance,
            end.speed_m_s,
            end.acceleration_m_s2,
        )
        # 把本段信息保存成字典，后面会写入 trajectory_report.json。
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

    # positions 是所有点的 xyz，用来计算整条路径的空间范围。
    positions = np.asarray([waypoint.pose[:3] for waypoint in trajectory], dtype=float)

    # 总长度和总时间只是把每段加起来。
    # xyz_min/max 可帮助你检查轨迹是否落在预期工作区附近。
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


def _plot_trajectory(trajectory: list[Waypoint], output_path: Path) -> None:
    """
    生成三维路径示意图；图只展示数值关系，不代表机械臂连杆或现场障碍物。

    注意：
    - 图里的线是 TCP 轨迹，不是机械臂各关节的实际形状；
    - 图里没有桌子、夹具、电缆；
    - 所以这张图只能帮助理解路径，不能作为真机安全依据。
    """

    # matplotlib 只在需要画图时导入，避免普通轨迹计算加载绘图库。
    import matplotlib

    # Agg 后端不需要桌面窗口，适合在脚本里直接保存 PNG。
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 取出所有路径点的 xyz 坐标。
    positions = np.asarray([waypoint.pose[:3] for waypoint in trajectory], dtype=float)

    # 创建 3D 图，并把 A/B/C 点连成折线。
    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(111, projection="3d")

    axes.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        marker="o",
        linewidth=2,
    )
    # 在每个点旁边写上名称，方便看出路径方向。
    for waypoint, position in zip(trajectory, positions):
        axes.text(position[0], position[1], position[2], f" {waypoint.name}")

    axes.set_xlabel("X / m")
    axes.set_ylabel("Y / m")
    axes.set_zlabel("Z / m")
    axes.set_title("TCP dry-run trajectory")
    axes.grid(True)
    # tight_layout 尽量避免坐标轴标签被裁掉。
    figure.tight_layout()

    # 保存 PNG 后关闭 figure，避免多次运行时占用内存。
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_robot_dry_run() -> Path:
    """
    完全不导入 ur_rtde 的轨迹检查入口。

    你可以先用它验证 A/B/C 数据格式、线段长度和交融半径，再讨论真机连接。

    这个函数适合初学者反复运行，因为它只做三件事：
    1. 按 config.py 生成轨迹；
    2. 做纯软件检查；
    3. 保存 JSON 报告和 PNG 示意图。
    """

    # 第一步：根据 TRAJECTORY_TYPE 和 POINT_A/B/C 生成路径。
    trajectory = build_trajectory()

    # 第二步：检查路径数字是否合理。
    # for_real_robot=False 表示允许示例点用于软件演示，但仍检查格式、长度和工作区。
    messages = validate_trajectory(trajectory, for_real_robot=False)

    # 第三步：计算每段距离、粗略时间、包围盒等报告字段。
    estimate = estimate_trajectory(trajectory)

    # 第四步：为本次 dry-run 创建独立输出目录。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.OUTPUT_ROOT / f"robot_dry_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 第五步：组装要写入 JSON 的报告内容。
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

    # 第六步：保存机器可读的 JSON 报告。
    report_path = output_dir / "trajectory_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 第七步：保存人眼更容易理解的三维轨迹示意图。
    _plot_trajectory(trajectory, output_dir / "trajectory_3d.png")

    # 第八步：在终端打印摘要，方便你不打开文件也能看到检查结果。
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
# 3. UR Dashboard 只读信息：判断 CB3 / e-Series 控制器代际
# =============================================================================

def _read_socket_line(connection: socket.socket) -> str:
    """
    从 Dashboard socket 读取一行文本。

    Dashboard 协议是很朴素的文本协议：发一行命令，收一行回复。
    这里不解析运动，只读取状态类文本。
    """

    # bytearray 适合一点一点追加网络收到的 bytes。
    chunks = bytearray()

    # 最多读 8192 字节，防止异常连接一直发送内容导致内存增长。
    while len(chunks) < 8192:
        # recv(1024) 表示这次最多收 1024 字节。
        data = connection.recv(1024)

        # 收到空 bytes 通常表示连接被对方关闭。
        if not data:
            break

        # 把本次收到的数据追加到总缓冲区。
        chunks.extend(data)

        # Dashboard 正常回复以换行结束，看到换行就可以停止读取。
        if b"\n" in data:
            break

    # UR 返回的是字节，decode 后才是 Python 字符串。
    # errors="replace" 表示遇到异常字符时用替代字符，不让程序直接崩溃。
    return chunks.decode("utf-8", errors="replace").strip()


def read_dashboard_information() -> dict[str, str]:
    """
    通过 UR 的 29999 Dashboard 端口读取软件版本、机器人模式和安全状态。

    这一步不发送运动命令；若端口不可用，后续仍可尝试 RTDE，但代际会标为 unknown。
    """

    # 先准备默认字段。即使后面某些读取失败，返回字典结构也尽量稳定。
    information = {
        "greeting": "",
        "polyscope_version": "",
        "robot_mode": "",
        "safety_status": "",
        "controller_generation": "unknown",
    }

    # Dashboard 默认端口是 29999。
    # create_connection 只建立 TCP 连接，不会让机器人运动。
    with socket.create_connection(
        (config.ROBOT_HOST, int(config.ROBOT_DASHBOARD_PORT)),
        timeout=float(config.ROBOT_CONNECT_TIMEOUT_S),
    ) as connection:
        # 给读写都设置超时，避免网线/IP 错误时程序一直卡住。
        connection.settimeout(float(config.ROBOT_CONNECT_TIMEOUT_S))

        # 连接成功后，Dashboard 会先发一行 greeting。
        information["greeting"] = _read_socket_line(connection)

        # 这些命令都是只读状态查询。
        # key 是我们保存到字典里的名字，command 是发给 UR Dashboard 的文本。
        commands = {
            "polyscope_version": "PolyscopeVersion",
            "robot_mode": "robotmode",
            "safety_status": "safetystatus",
        }
        for key, command in commands.items():
            # Dashboard 命令需要以换行结尾。
            connection.sendall((command + "\n").encode("ascii"))

            # 每发一个命令，就读回一行回复。
            information[key] = _read_socket_line(connection)

    # 从 PolyscopeVersion 文本中提取主版本号，用来粗略判断控制器代际。
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
# 4. ur_rtde 适配器
# =============================================================================

class URRobot:
    """
    把第三方 ur_rtde 的字段名封装成项目内部固定接口。

    以后若更换通信库，主要改这个类；轨迹生成、主程序和分析文件不需要知道底层差异。
    """

    def __init__(self) -> None:
        self.receive: Any = None
        self.control: Any = None
        self.dashboard_info: dict[str, str] = {}
        self.motion_command_time: float | None = None

    def connect(self, *, require_control: bool) -> None:
        """
        先尝试读取 Dashboard，再按需要创建 RTDE Receive 和 Control 连接。

        require_control=False 时只建立“读取状态”的连接，用于更安全的 robot_test。
        require_control=True 时才建立“发送控制命令”的连接，后续才可能执行 moveL。
        """

        try:
            # Dashboard 是 UR 控制器的文本接口，可读取版本、安全状态等信息。
            # 这一步失败不一定代表 RTDE 失败，所以这里只记录警告并继续尝试 RTDE。
            self.dashboard_info = read_dashboard_information()
        except Exception as exc:
            self.dashboard_info = {
                "controller_generation": "unknown",
                "dashboard_error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[机器人警告] Dashboard 信息读取失败：{exc}")

        # 导入放在函数内，确保 robot_dry_run 不会因为未安装 ur_rtde 而启动失败。
        try:
            import rtde_receive
        except ImportError as exc:
            raise RuntimeError(
                "真机模式需要 ur-rtde。请执行 pip install -r requirements.txt，"
                "或先切换到 robot_dry_run。"
            ) from exc

        # Receive 接口只读机器人状态，例如关节角、TCP 位姿、速度。
        self.receive = rtde_receive.RTDEReceiveInterface(
            config.ROBOT_HOST,
            float(config.ROBOT_RTDE_FREQUENCY),
        )

        if require_control:
            # Control 接口能发送运动命令，所以只有真机运动确实需要时才创建。
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
        """不同 ur_rtde 版本偶尔缺少次要状态字段；缺失时记录 None，不伪造数据。"""

        method = getattr(target, method_name, None)
        if method is None:
            return default
        try:
            return method()
        except Exception:
            return default

    def read_state(self) -> dict[str, Any]:
        """
        读取一条可序列化的 UR 状态，所有列表都转为普通 float。

        为什么要“可序列化”：
        实验日志是一行一条 JSON，NumPy 数组或某些库对象不能直接写进 JSON，
        所以这里把它们提前变成 list[float]、float、None 这些普通类型。
        """

        if self.receive is None:
            raise RuntimeError("尚未连接 RTDE Receive。")

        # 这三个是最核心的机器人状态：
        # actual_q：6 个关节角；
        # tcp_pose：末端 TCP 的 [x,y,z,rx,ry,rz]；
        # tcp_speed：末端当前速度。
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
        调用 UR 控制器自身的逆解与安全限制检查。

        这比软件 xyz 包围盒更强，但仍不能识别环境中的桌子或夹具。
        """

        if self.control is None:
            raise RuntimeError("控制连接未建立，无法调用控制器安全检查。")

        # UR 控制器自己知道当前安全平面、关节限制和可达性。
        # 这个检查比我们自己写的 xyz 范围更接近真实控制器判断。
        checker = getattr(self.control, "isPoseWithinSafetyLimits", None)
        if checker is None:
            raise RuntimeError(
                "当前 ur_rtde 版本没有 isPoseWithinSafetyLimits；"
                "为了安全，程序不允许跳过该检查后运动。"
            )

        for waypoint in trajectory:
            if not bool(checker(waypoint.pose)):
                raise RuntimeError(
                    f"UR 控制器判定 {waypoint.name} 点不可达或超出当前安全限制。"
                )

    def current_tcp_pose(self) -> list[float]:
        """只读取当前 TCP 位姿，不引发运动。"""

        if self.receive is None:
            raise RuntimeError("尚未连接 RTDE Receive。")
        return [float(value) for value in self.receive.getActualTCPPose()]

    def verify_at_start(self, start_pose: list[float]) -> None:
        """
        要求操作者已经用示教器把 TCP 放到 A 点附近。

        程序不会为了“自动回起点”而先执行一段未经观察的运动。
        """

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
        一次提交完整 moveL 路径，让 UR 控制器按自身周期插补。

        因此相机 132 fps 与 RTDE 125/500 Hz 不需要逐帧互相“对齐发送”，只需共享时间戳。
        """

        if self.control is None:
            raise RuntimeError("控制连接未建立，不能执行轨迹。")

        if len(trajectory) <= 1:
            return

        # 第一项 A 是已经人工到达的起点；真正发送的是后续 B/C，避免重复命令 A。
        # ur_rtde 的 moveL path 格式是：
        # [x, y, z, rx, ry, rz, speed, acceleration, blend]
        # 所以这里把 Waypoint 拆成第三方库需要的列表。
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

        # 第二个参数 True 表示异步执行：
        # 命令发出后 Python 不会卡在 moveL 里，而是由 motion_in_progress() 轮询完成状态。
        accepted = bool(self.control.moveL(path, True))
        if not accepted:
            raise RuntimeError("UR 控制器拒绝了异步 moveL 路径。")
        self.motion_command_time = time.perf_counter()

    def motion_in_progress(self) -> bool:
        """异步操作进度大于等于 0 表示仍在执行，-1 通常表示当前无异步操作。"""

        if self.control is None:
            return False

        progress_method = getattr(self.control, "getAsyncOperationProgress", None)
        if progress_method is not None:
            progress = int(progress_method())
            if progress >= 0:
                return True

            # 控制与接收接口存在极短同步间隙，刚提交后立即得到 -1 不应误判为已经完成。
            if (
                self.motion_command_time is not None
                and time.perf_counter() - self.motion_command_time < 0.20
            ):
                return True
            return False

        # 极旧版本没有异步进度时，用 TCP 速度作保守退路；正式使用前应升级 ur_rtde。
        speed = np.asarray(self.receive.getActualTCPSpeed(), dtype=float)
        return float(np.linalg.norm(speed[:3])) > 1e-4

    def stop_motion(self) -> None:
        """异常退出时使用受控 stopL，不发送新目标点。"""

        if self.control is None:
            return
        try:
            self.control.stopL(1.0)
        except Exception as exc:
            print(f"[机器人警告] stopL 调用失败：{exc}")

    def disconnect(self) -> None:
        """先断开控制再断开接收，任一对象不存在时也可以安全调用。"""

        for interface_name in ("control", "receive"):
            interface = getattr(self, interface_name)
            if interface is not None:
                try:
                    interface.disconnect()
                except Exception:
                    pass
                setattr(self, interface_name, None)


# =============================================================================
# 5. 真机前的人工确认
# =============================================================================

def require_operator_confirmation(purpose: str) -> None:
    """
    在终端要求完整输入确认短语，避免误按一个回车就启动。

    这个确认不能替代示教器急停、防护区和现场风险评估，只是程序层最后一道防误触。
    """

    # 如果配置里关闭了人工确认，就直接返回。
    # 正式实验中建议保持开启。
    if not config.REQUIRE_OPERATOR_CONFIRMATION:
        return

    # 终端打印明确说明，让操作者知道接下来要做什么。
    print()
    print(f"[安全确认] 即将进行：{purpose}")
    print("[安全确认] 请确认工作区无人、路径无夹具/桌面/线缆干涉，急停可立即触及。")

    # 要求输入完整短语，而不是简单 y/n，降低误触风险。
    typed = input(f"[安全确认] 请输入：{config.OPERATOR_CONFIRM_TEXT}\n> ").strip()

    # 输入不完全一致就取消本次运动。
    if typed != config.OPERATOR_CONFIRM_TEXT:
        raise PermissionError("确认文本不一致，本次运动已取消。")


# =============================================================================
# 6. robot_test：先读状态，默认绝不运动
# =============================================================================

def run_robot_test() -> Path:
    """
    单独验证 Dashboard、RTDE 版本识别和状态读取。

    ROBOT_TEST_ALLOW_MOTION=False 时不创建控制连接，更不会上传或执行运动脚本。

    这个函数分两种情况：
    - 默认 False：只连接读取接口，采样一小段状态并保存；
    - 手动 True：在更多安全检查后执行 A->B 低速测试。
    """

    # 先按 config.py 生成轨迹。即使默认不运动，也要检查配置是否基本合理。
    trajectory = build_trajectory()

    # 只有 ROBOT_TEST_ALLOW_MOTION=True 时，才按真机运动标准要求点位确认。
    validate_trajectory(
        trajectory,
        for_real_robot=bool(config.ROBOT_TEST_ALLOW_MOTION),
    )

    # 每次 robot_test 也单独创建输出目录，避免覆盖旧状态记录。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.OUTPUT_ROOT / f"robot_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "robot_test_log.txt"

    # URRobot 负责封装 Dashboard、RTDE Receive 和 RTDE Control。
    robot = URRobot()
    try:
        # 默认只需要 receive 读取状态；只有允许运动时才创建 control。
        robot.connect(require_control=bool(config.ROBOT_TEST_ALLOW_MOTION))

        # 打印 Dashboard 信息，方便确认连到的是预期控制器。
        print(
            "[机器人] 控制器代际判断："
            f"{robot.dashboard_info.get('controller_generation', 'unknown')}"
        )
        print(f"[机器人] Dashboard：{robot.dashboard_info}")

        # samples 保存若干条机器人状态，最后写入 robot_test_log.txt。
        samples: list[dict[str, Any]] = []

        # 采样时长借用 PRE_RECORD_SECONDS；采样频率由 ROBOT_RECORD_HZ 控制。
        sample_count = max(1, int(config.PRE_RECORD_SECONDS * config.ROBOT_RECORD_HZ))
        period = 1.0 / config.ROBOT_RECORD_HZ
        next_time = time.perf_counter()

        # 先读一段静止状态，不发送运动命令。
        for _ in range(sample_count):
            samples.append(robot.read_state())
            next_time += period
            time.sleep(max(0.0, next_time - time.perf_counter()))

        # 终端只打印第一条，完整状态序列会写入日志。
        first = samples[0]
        print(f"[机器人] 当前关节角 rad：{first['actual_q_rad']}")
        print(f"[机器人] 当前 TCP：{first['actual_tcp_pose']}")

        # 只有显式打开 ROBOT_TEST_ALLOW_MOTION，才会进入下面的低速运动测试。
        if config.ROBOT_TEST_ALLOW_MOTION:
            if len(trajectory) < 2:
                raise ValueError("robot_test 允许运动时，TRAJECTORY_TYPE 不能是 static。")

            # 单机测试只保留 A→B，并强制改用更低的测试速度。
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
            # 运动前再次对缩减后的 A->B 测试路径做检查。
            validate_trajectory(test_path, for_real_robot=True)

            # 调用控制器自己的安全限制判断。
            robot.verify_controller_safety_limits(test_path)

            # 要求当前 TCP 已经人工放在 A 点附近。
            robot.verify_at_start(test_path[0].pose)

            # 终端人工确认，防止误触发。
            require_operator_confirmation("UR10 A→B 低速单机测试")

            # 发送异步 moveL 后，通过 motion_in_progress() 轮询直到完成或超时。
            robot.execute_trajectory(test_path)
            deadline = time.perf_counter() + config.ROBOT_MOTION_TIMEOUT_S
            while robot.motion_in_progress():
                if time.perf_counter() > deadline:
                    robot.stop_motion()
                    raise TimeoutError("低速测试运动超时，已调用 stopL。")
                samples.append(robot.read_state())
                time.sleep(period)

        # 无论是否运动，都把本次读取到的状态写成逐行 JSON。
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
        # 不管正常结束还是异常退出，都尝试停止运动并断开连接。
        robot.stop_motion()
        robot.disconnect()


# =============================================================================
# 7. 完整实验中的机器人子进程
# =============================================================================

def _put_record(record_queue: Any, record: dict[str, Any], stop_event: Any) -> None:
    """队列满说明记录进程跟不上；停止实验比继续产生无法对时的数据更安全。"""

    try:
        record_queue.put(record, timeout=1.0)
    except Full as exc:
        stop_event.set()
        raise RuntimeError("记录队列已满，机器人数据无法写入。") from exc


def robot_worker(
    record_queue: Any,
    error_queue: Any,
    start_event: Any,
    stop_event: Any,
    robot_ready: Any,
    motion_done: Any,
) -> None:
    """
    完整实验中的机器人进程。

    它先做软件检查、控制器安全检查和起点检查，报告 ready 后等待统一开始信号。

    这个函数只在 RUN_MODE="experiment" 时由 main.py 创建为子进程。
    阅读它时可以按时间顺序看：
    1. 建轨迹并做安全检查；
    2. 连接 UR；
    3. 等主程序发 start_event；
    4. 先记录 PRE_RECORD_SECONDS 秒静止数据；
    5. 发送运动命令；
    6. 持续记录直到主程序要求停止。
    """

    robot = URRobot()
    motion_started = False

    try:
        # 真机运动前的三层检查：
        # 1. validate_trajectory：本程序自己的数字检查；
        # 2. verify_controller_safety_limits：UR 控制器自己的安全限制检查；
        # 3. verify_at_start：确认当前 TCP 已在 A 点附近，不让程序盲目回起点。
        trajectory = build_trajectory()
        validate_trajectory(trajectory, for_real_robot=True)
        robot.connect(require_control=True)
        robot.verify_controller_safety_limits(trajectory)
        robot.verify_at_start(trajectory[0].pose)
        robot_ready.set()

        # 连接和检查都成功后，仍然不立刻运动。
        # 必须等 main.py 在相机也 ready、操作者确认后发出 start_event。
        while not start_event.is_set():
            if stop_event.wait(0.05):
                return

        # record_start 是统一开始时刻。
        # 先静止记录 PRE_RECORD_SECONDS 秒，再提交运动命令，便于分析静止噪声基线。
        record_start = time.perf_counter()
        motion_due = record_start + config.PRE_RECORD_SECONDS
        period = 1.0 / config.ROBOT_RECORD_HZ
        next_sample = time.perf_counter()

        # 主循环按 ROBOT_RECORD_HZ 尽量稳定采样。
        while not stop_event.is_set():
            now = time.perf_counter()

            # 到达预记录时间后，只发送一次运动命令。
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

                # static 工况没有运动命令，预记录完成就直接报告“运动阶段完成”。
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

            # 不管是否已经开始运动，都持续读取机器人状态并写入日志。
            state = robot.read_state()
            _put_record(record_queue, state, stop_event)

            # 异步 moveL 完成后，记录 motion_finished 事件。
            # 之后继续采样，直到 main.py 完成 POST_RECORD_SECONDS 计时。
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

                # 运动完成后仍保持状态采集，直到主程序等完 POST_RECORD_SECONDS 再设置 stop。
                while not stop_event.is_set():
                    _put_record(record_queue, robot.read_state(), stop_event)
                    next_sample += period
                    time.sleep(max(0.0, next_sample - time.perf_counter()))
                break

            next_sample += period
            time.sleep(max(0.0, next_sample - time.perf_counter()))

            # 超时是兜底保护：如果控制器迟迟不报告完成，就先 stopL，再抛出错误。
            if motion_started and now - motion_due > config.ROBOT_MOTION_TIMEOUT_S:
                robot.stop_motion()
                raise TimeoutError("正式轨迹执行超过超时上限，已调用 stopL。")
    except Exception as exc:
        try:
            error_queue.put(
                f"机器人进程异常：{type(exc).__name__}: {exc}",
                timeout=1.0,
            )
        except Full:
            pass
        stop_event.set()
    finally:
        if motion_started and not motion_done.is_set():
            robot.stop_motion()
        robot.disconnect()
