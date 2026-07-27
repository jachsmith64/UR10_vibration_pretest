"""
读取实验 TXT/JSONL，完成时间对齐、去趋势、振动指标和图表输出。

这个文件不导入相机 SDK 或 ur_rtde，因此可以在任何装有科学计算依赖的电脑上离线运行。

给初学者的文件地图：
1. load_run_file() 读取实验日志，把每一行 JSON 按 META、EVENT、VISION、ROBOT、ERROR 分开。
2. extract_vision_series() 从视觉记录中取出时间和位移，只保留有效帧。
3. resample_uniform() 把不完全等间隔的采样点插值成等间隔序列，方便做频谱。
4. detrend_motion() / detrend_piecewise() 去掉“整体移动趋势”，留下我们关心的微小振动。
5. calculate_spectrum() 计算 FFT、Welch 功率谱、主频和不同频带的能量。
6. plot_*() 系列函数负责把结果画成图。
7. run_analysis() 是本文件的总入口，会按顺序调用上面的步骤并保存摘要。

这份分析代码的核心思想：
机器人末端在移动时，图像里看到的位移包含两部分：
- 大的、慢的整体运动趋势；
- 小的、快的振动。
分析时要先把慢趋势剥离掉，再对剩下的残差做 RMS、主频、恢复时间等统计。

安全边界：
- 本文件只读已有日志并写分析结果。
- 它不会打开相机、不会连接 UR，也不会发送任何运动指令。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
from matplotlib.axes import Axes

# Agg 不需要桌面窗口，适合从 VS Code、服务器或批处理稳定保存 PNG。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import config


# =============================================================================
# 1. 读取和整理逐行 JSON 记录
# =============================================================================

@dataclass(slots=True)
class LoadedRun:
    """
    按记录类型拆开的实验内容，并保留无法解析行的位置供排错。

    实验日志是一行一个 JSON 字典。不同 kind 代表不同来源：
    - META：本次运行的基本信息；
    - EVENT：实验流程事件；
    - VISION：视觉算法逐帧结果；
    - ROBOT：机器人逐帧/逐周期状态；
    - ERROR：运行中记录到的错误。
    """

    source_path: Path
    meta: list[dict[str, Any]]
    events: list[dict[str, Any]]
    vision: list[dict[str, Any]]
    robot: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    malformed_lines: list[int]


def _find_latest_run_file() -> Path:
    """
    在 outputs 中选择修改时间最新的正式或视觉测试记录。

    自动选择会在终端打印实际文件；需要固定某次实验时，应在配置或命令行明确给出路径。
    """

    # 这三个文件名分别对应正式实验、视觉测试、机器人单机测试。
    patterns = ("**/run_log.txt", "**/vision_results.txt", "**/robot_test_log.txt")
    candidates: list[Path] = []

    # glob("**/xxx") 会递归搜索 outputs 下所有子目录。
    for pattern in patterns:
        candidates.extend(config.OUTPUT_ROOT.glob(pattern))

    # 只保留真实文件，排除同名目录等异常情况。
    candidates = [path for path in candidates if path.is_file()]

    # 如果没有任何可分析文件，就给出清晰错误。
    if not candidates:
        raise FileNotFoundError(
            f"{config.OUTPUT_ROOT} 下没有 run_log.txt 或 vision_results.txt。"
        )
    # 选择最近修改的文件，通常就是刚刚运行得到的结果。
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_analysis_file(requested: Path | None = None) -> Path:
    """
    按“命令行参数→config.py→最新记录”的优先级解析分析目标。

    requested 来自 main.py 的 --analysis-file。
    如果用户显式指定，就优先用它；否则再看 config.ANALYSIS_FILE；
    两者都没有时，才自动寻找最新记录。
    """

    # 命令行参数优先级高于 config.py。
    candidate = requested or config.ANALYSIS_FILE

    # 用户没有指定时，自动从 outputs 中找最新记录。
    if candidate is None:
        resolved = _find_latest_run_file()
        print(f"[分析] 未指定文件，自动选择最新记录：{resolved}")
        return resolved

    # expanduser 支持 ~ 这样的用户目录写法。
    candidate = Path(candidate).expanduser()

    # 相对路径按项目目录解释，而不是按终端当前目录解释。
    if not candidate.is_absolute():
        candidate = (config.PROJECT_DIR / candidate).resolve()

    # 最后确认文件真的存在。
    if not candidate.exists():
        raise FileNotFoundError(f"分析文件不存在：{candidate}")
    return candidate


def load_run_file(path: Path) -> LoadedRun:
    """
    每行独立解析，单行损坏不会让整个实验完全打不开。

    malformed_lines 会写入摘要，提醒你回看磁盘写入或人工编辑造成的问题。
    """

    # buckets 按 kind 分类存储日志行。
    # 这样后面的分析函数不用每次都在全部记录里搜索。
    buckets: dict[str, list[dict[str, Any]]] = {
        "META": [],
        "EVENT": [],
        "VISION": [],
        "ROBOT": [],
        "ERROR": [],
    }
    # malformed 保存无法解析或结构异常的行号。
    malformed: list[int] = []

    # 逐行读取日志。即使某一行坏了，也尽量保留其他正常行。
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            # 去掉首尾空白；空行直接跳过。
            text = raw_line.strip()
            if not text:
                continue

            # 尝试把当前行从 JSON 文本转成 Python 对象。
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                malformed.append(line_number)
                continue

            # 本项目约定每行必须是一个字典；其他类型视为异常行。
            if not isinstance(record, dict):
                malformed.append(line_number)
                continue

            # kind 决定该记录属于哪个桶。
            kind = str(record.get("kind", "")).upper()
            if kind in buckets:
                buckets[kind].append(record)
            else:
                malformed.append(line_number)

    # 把分类后的内容打包成 dataclass，后续传参更清楚。
    loaded = LoadedRun(
        source_path=Path(path),
        meta=buckets["META"],
        events=buckets["EVENT"],
        vision=buckets["VISION"],
        robot=buckets["ROBOT"],
        errors=buckets["ERROR"],
        malformed_lines=malformed,
    )

    # 没有视觉记录就无法分析末端振动，所以这里直接停止。
    if not loaded.vision:
        raise ValueError(
            f"{path} 中没有 VISION 记录，无法计算相机测得的末端振动。"
        )
    return loaded


# =============================================================================
# 2. 视觉序列、事件和机器人序列
# =============================================================================

def _finite_float(value: Any) -> float:
    """
    把 None、字符串和 JSON 中的 NaN 统一转成浮点 NaN，便于 NumPy 后续筛选。

    日志来自 JSON，字段可能缺失、为 None、为字符串或为 NaN。
    与其让后续每个函数都处理这些情况，不如在入口统一清洗。
    """

    try:
        # float() 可以把 int、float、可转数字的字符串统一变成浮点数。
        result = float(value)
    except (TypeError, ValueError):
        # 不能转成数字时，用 NaN 表示“无效数值”。
        return math.nan

    # 即使能转成 float，也要排除 inf 和 nan。
    return result if math.isfinite(result) else math.nan


def extract_vision_series(
    records: list[dict[str, Any]],
    method: str,
    axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提取时间、位移和质量，只保留该方法明确标记有效且数值有限的帧。

    host_ns 是相机与 UR 共同的电脑时基，绝不使用“第 N 帧对应第 N 条机器人记录”的假设。
    """

    # 两种视觉方法的字段前缀不同：
    # circles 对应 circle_dx_mm，checkerboard 对应 checker_dx_mm。
    prefix = "circle" if method == "circles" else "checker"

    # 先用 list 收集有效点，最后再统一转 NumPy 数组。
    time_values: list[float] = []
    displacement_values: list[float] = []
    quality_values: list[float] = []

    # 逐条检查 VISION 记录。
    for record in records:
        # 只保留该方法明确标记有效的帧。
        if not bool(record.get(f"{prefix}_is_valid", False)):
            continue

        # 读取并清洗时间、位移、质量分。
        host_ns = _finite_float(record.get("host_ns"))
        analysis_time_s = _finite_float(record.get("analysis_time_s"))
        dx = _finite_float(record.get(f"{prefix}_dx_mm"))
        dy = _finite_float(record.get(f"{prefix}_dy_mm"))
        quality = _finite_float(record.get(f"{prefix}_quality"))

        # 根据 config.ANALYSIS_AXIS 选择要分析的方向。
        if axis == "x":
            displacement = dx
        elif axis == "y":
            displacement = dy
        elif axis == "magnitude":
            displacement = math.hypot(dx, dy)
        else:
            raise ValueError(f"未知分析轴：{axis}")

        # 只有时间和位移都有效，才把这一帧纳入分析。
        if math.isfinite(host_ns) and math.isfinite(displacement):
            # vision_test 用图片/视频自身时间，正式 experiment 则由 camera_worker 写入共同 host 时基。
            time_values.append(
                analysis_time_s if math.isfinite(analysis_time_s) else host_ns * 1e-9
            )
            displacement_values.append(displacement)
            quality_values.append(quality)

    # 后续 Welch 频谱和滤波至少需要一些点，太少时结果没有意义。
    if len(time_values) < 8:
        raise ValueError(
            f"{method} 在 {axis} 方向只有 {len(time_values)} 个有效点，"
            "至少需要 8 个才能进行基本振动分析。"
        )

    # 转成 NumPy 数组，便于排序、插值和数学计算。
    time_array = np.asarray(time_values, dtype=float)
    displacement_array = np.asarray(displacement_values, dtype=float)
    quality_array = np.asarray(quality_values, dtype=float)

    # 日志通常按时间写入，但仍显式排序，防止异常写入顺序影响分析。
    order = np.argsort(time_array)
    time_array = time_array[order]
    displacement_array = displacement_array[order]
    quality_array = quality_array[order]

    # 同一 host_ns 理论上不应重复；若发生，只保留第一次以保证插值时间严格递增。
    unique_time, unique_indices = np.unique(time_array, return_index=True)
    return (
        unique_time,
        displacement_array[unique_indices],
        quality_array[unique_indices],
    )


def event_times_seconds(events: list[dict[str, Any]]) -> dict[str, float]:
    """
    把每种事件第一次出现的 host_ns 转为秒；重复事件保留第一次以避免覆盖真实起点。

    事件时间用于划分分析窗口，例如：
    - motion_command_sent：机器人运动命令发出；
    - motion_finished：机器人报告运动完成。
    """

    result: dict[str, float] = {}

    # 先按 host_ns 排序，确保“第一次出现”是真的最早事件。
    for event in sorted(events, key=lambda item: _finite_float(item.get("host_ns"))):
        # name 是事件名称；host_ns 是事件发生时的电脑时钟。
        name = str(event.get("name", ""))
        host_ns = _finite_float(event.get("host_ns"))

        # 只保留名称非空、时间有效、且尚未出现过的事件。
        if name and math.isfinite(host_ns) and name not in result:
            result[name] = host_ns * 1e-9
    return result


def _window_mask(
    time_s: np.ndarray,
    start_s: float | None,
    end_s: float | None,
) -> np.ndarray:
    """
    建立闭区间掩码；缺少某端事件时自动延伸到数据首尾。

    返回数组和 time_s 一样长：
    - True 表示该时间点在窗口内；
    - False 表示该时间点不属于窗口。
    """

    # 如果没有开始事件，就从数据第一帧开始。
    start = time_s[0] if start_s is None else start_s

    # 如果没有结束事件，就延伸到数据最后一帧。
    end = time_s[-1] if end_s is None else end_s

    # NumPy 可以一次性比较整个数组，得到布尔掩码。
    return (time_s >= start) & (time_s <= end)


def select_analysis_windows(
    time_s: np.ndarray,
    events: dict[str, float],
) -> dict[str, np.ndarray]:
    """
    把完整视觉时间序列拆成实验分析窗口。

    输入：
    - time_s：每帧视觉测量结果的时间；
    - events：实验事件时间，例如 experiment_started、motion_command_sent、motion_finished。

    输出：
    - full：整段有效视觉记录；
    - baseline：运动命令前的静止段；
    - motion：机器人运动命令发出到运动完成；
    - steady_motion：粗略裁掉 motion 两端后的中间段，用于匀速段预分析；
    - post：运动完成后的残余振动段。

    实验作用：
    同一条位移曲线在不同工况下要分析不同片段。这里先把“可选片段”准备好，
    后续 run_analysis() 再根据 config.ANALYSIS_PRIMARY_WINDOW 选择真正用于 RMS 和频谱的主窗口。
    """

    # 本段读取窗口边界事件。
    # 输入来自日志里的 EVENT 记录；缺失实验开始/结束时用数据首尾兜底，避免离线视觉测试无法分析。
    experiment_start = events.get("experiment_started", float(time_s[0]))
    motion_start = events.get("motion_command_sent")
    motion_end = events.get("motion_finished")
    experiment_end = events.get("experiment_finished", float(time_s[-1]))

    # 本段处理没有机器人运动事件的离线数据。
    # vision_test、合成图片或手动移动视频通常没有 motion_start/motion_end。
    # 这种情况下无法可靠划分静止/运动/残余段，因此把完整序列作为 motion 和 steady_motion。
    if motion_start is None or motion_end is None:
        return {
            "full": np.ones_like(time_s, dtype=bool),
            "baseline": np.zeros_like(time_s, dtype=bool),
            "motion": np.ones_like(time_s, dtype=bool),
            "steady_motion": np.ones_like(time_s, dtype=bool),
            "post": np.zeros_like(time_s, dtype=bool),
        }

    # 本段从 motion 中粗略裁出 steady_motion。
    # 输入是运动开始/结束事件；输出是去掉两端加减速影响后的中间窗口。
    # 它不是精确的速度闭环识别，只是为“匀速段优先分析”提供一个保守的预实验窗口。
    motion_duration = max(0.0, motion_end - motion_start)
    trim = config.STEADY_MOTION_TRIM_FRACTION * motion_duration
    steady_start = motion_start + trim
    steady_end = motion_end - trim
    if steady_start >= steady_end:
        steady_start = motion_start
        steady_end = motion_end

    # 本段输出所有分析窗口的布尔掩码。
    # 每个掩码和 time_s 一样长，True 表示该帧属于对应窗口。
    return {
        "full": _window_mask(time_s, experiment_start, experiment_end),
        "baseline": _window_mask(time_s, experiment_start, motion_start),
        "motion": _window_mask(time_s, motion_start, motion_end),
        "steady_motion": _window_mask(time_s, steady_start, steady_end),
        "post": _window_mask(time_s, motion_end, experiment_end),
    }


def extract_robot_tcp_series(
    robot_records: list[dict[str, Any]],
    axis: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    提取 UR 编码器推算的 TCP 位置并换成 mm。

    magnitude 使用相对第一条机器人记录的三维位移长度，而视觉 magnitude 是二维图像平面位移。
    """

    # 没有机器人记录时，调用方会跳过视觉/机器人对比图。
    if not robot_records:
        return None

    # 机器人 TCP 位姿前三项是 x、y、z；这里只支持 x/y 单轴直接对比。
    axis_index = {"x": 0, "y": 1}

    # 先用 list 收集有效记录。
    time_values: list[float] = []
    positions: list[list[float]] = []

    # 逐条读取 ROBOT 记录。
    for record in robot_records:
        # host_ns 是电脑时基，actual_tcp_pose 是 UR 返回的 TCP 位姿。
        host_ns = _finite_float(record.get("host_ns"))
        pose = record.get("actual_tcp_pose")

        # 时间无效、位姿不是列表、或位姿长度不足时跳过。
        if not math.isfinite(host_ns) or not isinstance(pose, list) or len(pose) < 3:
            continue

        # 只取 xyz，并把每一项清洗成有限浮点数。
        xyz = [_finite_float(value) for value in pose[:3]]
        if all(math.isfinite(value) for value in xyz):
            time_values.append(host_ns * 1e-9)
            positions.append(xyz)

    # 少于两个点无法形成曲线。
    if len(time_values) < 2:
        return None

    # 转成 NumPy 数组便于切片和排序。
    time_array = np.asarray(time_values, dtype=float)
    position_array = np.asarray(positions, dtype=float)

    # x/y 单轴直接取对应坐标并从 m 换成 mm。
    if axis in axis_index:
        values_mm = position_array[:, axis_index[axis]] * 1000.0
    else:
        # magnitude 用相对第一条记录的三维距离。
        relative = position_array - position_array[0]
        values_mm = np.linalg.norm(relative, axis=1) * 1000.0

    # 按时间排序，返回时间和位移。
    order = np.argsort(time_array)
    return time_array[order], values_mm[order]


# =============================================================================
# 3. 不等间隔数据重采样和去趋势
# =============================================================================

def resample_uniform(
    time_s: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """
    把不完全等间隔的视觉位移数据转换成均匀时间轴。

    输入：
    - time_s：原始时间戳，可能来自图片 timestamps.csv、视频时间戳或电脑时钟；
    - values：对应时间上的位移序列。

    输出：
    - uniform_time：按中位采样间隔生成的均匀时间轴；
    - uniform_values：插值到均匀时间轴上的位移；
    - sample_rate_hz：由真实时间戳估计的采样率；
    - timing：采样间隔质量指标，用于判断是否掉帧或帧率异常。

    实验作用：
    FFT、Welch 和数字滤波要求数据近似等间隔。这里不宣称增加真实带宽，
    只是把已有测量点整理成后续频域算法能处理的形式。
    """

    if len(time_s) != len(values):
        raise ValueError("时间与位移数组长度不一致。")

    # 本段从原始时间戳估计真实采样节奏。
    # 使用相邻时间差而不是固定 config.IMAGE_FOLDER_FPS，是为了发现实际帧率是否偏离 132 fps。
    delta_t = np.diff(time_s)
    valid_delta = delta_t[delta_t > 0]
    if len(valid_delta) == 0:
        raise ValueError("时间戳没有递增，无法估计采样率。")

    # 本段用中位间隔作为重采样步长。
    # 中位数比平均数更不容易被偶发卡顿或个别大间隔带偏。
    median_dt = float(np.median(valid_delta))
    sample_rate_hz = 1.0 / median_dt

    # 本段生成频域分析使用的均匀序列。
    # 输入是原始点；输出是同一物理量在均匀时间轴上的插值版本。
    uniform_time = np.arange(time_s[0], time_s[-1] + 0.5 * median_dt, median_dt)
    uniform_values = np.interp(uniform_time, time_s, values)

    # 本段把采样质量写成机器可读指标。
    # 后续 summary 会展示这些值；如果 max_gap 或估计帧率异常，会额外生成采样率警告。
    timing = {
        "sample_rate_hz": sample_rate_hz,
        "median_dt_s": median_dt,
        "dt_std_s": float(np.std(valid_delta)),
        "max_gap_s": float(np.max(valid_delta)),
        "original_sample_count": int(len(time_s)),
        "uniform_sample_count": int(len(uniform_time)),
    }
    return uniform_time, uniform_values, sample_rate_hz, timing


def sampling_warnings(timing: dict[str, float]) -> list[str]:
    """
    根据实际采样时间判断是否需要提醒用户。

    输入：resample_uniform() 估计出的采样时间质量；
    输出：中文警告列表。正常接近 132 fps 时返回空列表，不打扰用户。

    实验作用：
    你的相机标称 132 fps，但真实采集可能略有偏差。这里不要求精确等于 132；
    只有估计帧率明显偏离，或出现异常大间隔时，才提示你检查相机导出、timestamps.csv 或掉帧问题。
    """

    warnings: list[str] = []
    expected = config.EXPECTED_VISION_FPS
    actual = float(timing["sample_rate_hz"])

    # 本段检查“整体采样率是否明显偏离标称值”。
    # 输入是实际估计帧率和配置里的期望帧率；输出最多一条偏差警告。
    if expected is not None:
        tolerance = max(
            float(config.FPS_WARNING_ABSOLUTE_TOLERANCE_HZ),
            abs(float(expected)) * float(config.FPS_WARNING_RELATIVE_TOLERANCE),
        )
        if abs(actual - float(expected)) > tolerance:
            warnings.append(
                f"实测采样率约 {actual:.3f} Hz，和标称 {float(expected):.3f} Hz "
                f"偏差超过 {tolerance:.3f} Hz。"
            )

    # 本段检查“局部是否出现异常大间隔”。
    # 即使平均 fps 正常，个别大间隔也可能影响频谱和恢复时间，所以单独提醒。
    median_dt = float(timing["median_dt_s"])
    max_gap = float(timing["max_gap_s"])
    if median_dt > 0 and max_gap > median_dt * float(config.FRAME_GAP_WARNING_FACTOR):
        warnings.append(
            f"最大相邻采样间隔 {max_gap:.6f} s，超过中位间隔 "
            f"{median_dt:.6f} s 的 {float(config.FRAME_GAP_WARNING_FACTOR):.2f} 倍。"
        )

    return warnings


def _savgol_window_points(sample_rate_hz: float, sample_count: int) -> int:
    """把秒制趋势窗口换成不超过数据长度的奇数点数。"""

    desired = int(round(config.SAVGOL_WINDOW_SECONDS * sample_rate_hz))
    desired = max(desired, config.SAVGOL_POLYORDER + 3)
    if desired % 2 == 0:
        desired += 1

    maximum = sample_count if sample_count % 2 == 1 else sample_count - 1
    return min(desired, maximum)


def detrend_motion(
    values: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回正常运动趋势和其上的振动残差。

    去趋势方法会直接影响低频结论，所以摘要中会明确记录所用方法和参数。

    例子：
    如果机器人从 A 点慢慢移动到 B 点，视觉位移里会有一个很大的“整体移动”。
    但我们真正关心的是叠在整体移动上的小抖动，所以要把慢变化趋势先减掉。
    """

    values = np.asarray(values, dtype=float)

    if config.DETREND_METHOD == "linear":
        # linear：假设整体趋势近似一条直线，适合匀速直线运动的粗分析。
        residual = np.asarray(signal.detrend(values, type="linear"), dtype=float)
        trend = values - residual
        return trend, residual

    if config.DETREND_METHOD == "savgol":
        # savgol：用平滑曲线表示慢趋势，比直线更能跟随缓慢弯曲变化。
        window = _savgol_window_points(sample_rate_hz, len(values))
        if window <= config.SAVGOL_POLYORDER:
            raise ValueError("数据太短，无法使用当前 Savitzky-Golay 参数。")
        trend = np.asarray(
            signal.savgol_filter(
                values,
                window_length=window,
                polyorder=config.SAVGOL_POLYORDER,
                mode="interp",
            ),
            dtype=float,
        )
        return trend, values - trend

    if config.DETREND_METHOD == "highpass":
        # highpass：直接保留高于某个频率的成分，适合你明确知道低频都不是关注目标时使用。
        nyquist = 0.5 * sample_rate_hz
        if not 0 < config.HIGHPASS_CUTOFF_HZ < nyquist:
            raise ValueError(
                f"高通截止频率 {config.HIGHPASS_CUTOFF_HZ} Hz 必须小于奈奎斯特频率 "
                f"{nyquist:.3f} Hz。"
            )
        sos = signal.butter(
            config.HIGHPASS_ORDER,
            config.HIGHPASS_CUTOFF_HZ,
            btype="highpass",
            fs=sample_rate_hz,
            output="sos",
        )
        residual = np.asarray(signal.sosfiltfilt(sos, values), dtype=float)
        return values - residual, residual

    raise ValueError(f"未知 DETREND_METHOD：{config.DETREND_METHOD}")


def detrend_piecewise(
    time_s: np.ndarray,
    values: np.ndarray,
    sample_rate_hz: float,
    events: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    对“运动前静止—运动—运动后静止”分别去趋势，再拼回完整时域图。

    若整段只减一条直线，正常的“静止—斜坡—静止”会被误当成巨大低频振动，尤其会破坏恢复时间。
    """

    motion_start = events.get("motion_command_sent")
    motion_end = events.get("motion_finished")
    if motion_start is None or motion_end is None:
        return detrend_motion(values, sample_rate_hz)

    masks = (
        time_s < motion_start,
        (time_s >= motion_start) & (time_s <= motion_end),
        time_s > motion_end,
    )
    trend = np.full_like(values, np.nan, dtype=float)
    residual = np.full_like(values, np.nan, dtype=float)

    for mask in masks:
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue

        segment = values[mask]
        if count >= 8:
            segment_trend, segment_residual = detrend_motion(segment, sample_rate_hz)
        else:
            # 极短区间无法稳定滤波，用区间均值作为趋势比凭空外推更保守。
            segment_trend = np.full_like(segment, float(np.mean(segment)))
            segment_residual = segment - segment_trend

        trend[mask] = segment_trend
        residual[mask] = segment_residual

    # 事件时间正好落在浮点间隙时，使用原值作为趋势并把残差设零，避免图中出现 NaN 断点。
    missing = ~np.isfinite(residual)
    trend[missing] = values[missing]
    residual[missing] = 0.0
    return trend, residual


# =============================================================================
# 4. 时域、频域和恢复时间指标
# =============================================================================

def calculate_spectrum(
    residual: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, Any]:
    """
    同时计算单边 FFT 幅值谱和 Welch 功率谱密度。

    主频与频带能量用 Welch PSD，时域正弦幅值查看可使用 FFT amplitude。

    简单理解：
    - 时域图告诉你“什么时候抖得大”；
    - 频谱图告诉你“主要以多少 Hz 在抖”。
    """

    # 先去掉平均值，让频谱不要被一个直流偏置占据。
    centered = np.asarray(residual, dtype=float) - float(np.mean(residual))

    # 加窗可以减少有限长度数据在 FFT 中产生的边缘泄漏。
    window = np.hanning(len(centered))
    fft_values = np.fft.rfft(centered * window)
    fft_frequency = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    amplitude = np.abs(fft_values) * (2.0 / max(float(np.sum(window)), 1e-12))
    if len(amplitude):
        amplitude[0] *= 0.5

    # Welch 会把数据分段求平均，频谱更稳，但点太少时不能可靠计算。
    nperseg = min(1024, len(centered))
    if nperseg < 8:
        raise ValueError("有效点少于 8，无法计算 Welch 频谱。")
    welch_frequency, psd = signal.welch(
        centered,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,  # pyright: ignore[reportArgumentType]
        scaling="density",
    )

    # 主频只在用户关心的频率范围里找，并且不能超过奈奎斯特频率。
    upper_limit = min(config.DOMINANT_FREQ_MAX_HZ, 0.5 * sample_rate_hz)
    search_mask = (
        (welch_frequency >= config.DOMINANT_FREQ_MIN_HZ)
        & (welch_frequency <= upper_limit)
    )
    if np.any(search_mask):
        local_index = int(np.argmax(psd[search_mask]))
        dominant_frequency = float(welch_frequency[search_mask][local_index])
        dominant_psd = float(psd[search_mask][local_index])
    else:
        dominant_frequency = math.nan
        dominant_psd = math.nan

    # NumPy 2.x 提供 trapezoid；旧版可退回 trapz。不能把 np.trapz 直接写成 getattr 默认值，
    # 因为 Python 会先求值默认参数，而较新的 NumPy 版本已经移除了该旧名称。
    integrate = cast(Any, getattr(np, "trapezoid", getattr(np, "trapz", None)))
    if integrate is None:
        raise RuntimeError("当前 NumPy 版本缺少 trapezoid/trapz 积分函数。")
    band_energy: dict[str, float] = {}
    for low_hz, high_hz in config.FREQUENCY_BANDS_HZ:
        mask = (welch_frequency >= low_hz) & (welch_frequency < high_hz)
        key = f"{low_hz:g}-{high_hz:g}Hz"
        band_energy[key] = (
            float(integrate(psd[mask], welch_frequency[mask]))
            if np.count_nonzero(mask) >= 2
            else 0.0
        )

    return {
        "fft_frequency_hz": fft_frequency,
        "fft_amplitude_mm": amplitude,
        "welch_frequency_hz": welch_frequency,
        "welch_psd_mm2_per_hz": psd,
        "dominant_frequency_hz": dominant_frequency,
        "dominant_psd_mm2_per_hz": dominant_psd,
        "band_energy_mm2": band_energy,
        "frequency_resolution_hz": float(sample_rate_hz / len(centered)),
    }


def calculate_time_metrics(residual: np.ndarray) -> dict[str, float]:
    """
    给出最常用的峰峰值、RMS、标准差和最大绝对残差，单位均为 mm。

    这些指标都基于“去趋势后的残差”，不是原始位移。
    它们描述的是振动强弱，而不是机器人从 A 到 B 的整体运动距离。
    """

    # 确保输入是浮点数组，避免整型数组参与平方时出现不必要的类型问题。
    residual = np.asarray(residual, dtype=float)

    # ptp 是 peak-to-peak，即最大值减最小值。
    # RMS 是均方根，常用于描述振动能量大小。
    return {
        "peak_to_peak_mm": float(np.ptp(residual)),
        "rms_mm": float(np.sqrt(np.mean(residual**2))),
        "std_mm": float(np.std(residual)),
        "max_abs_mm": float(np.max(np.abs(residual))),
        "mean_mm": float(np.mean(residual)),
    }


def calculate_recovery_time(
    uniform_time_s: np.ndarray,
    residual: np.ndarray,
    events: dict[str, float],
    baseline_mask: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, float | None]:
    """
    运动完成后包络连续低于阈值一段时间，才认定恢复。

    没有 motion_finished 事件或记录尾部太短时返回 None，不用最后一帧冒充恢复。

    恢复时间的含义：
    从 motion_finished 开始计时，直到振动包络连续一段时间低于阈值。
    这样可以避免某一个瞬间刚好低于阈值就误判为已经恢复。
    """

    # 没有运动完成事件，就不知道从什么时候开始找恢复时间。
    motion_finished = events.get("motion_finished")
    if motion_finished is None:
        return {
            "recovery_time_s": None,
            "recovery_threshold_mm": None,
            "baseline_rms_mm": None,
        }

    # baseline_mask 表示运动前静止区间。
    # 若静止区间点数足够，用它估计系统背景噪声 RMS。
    if np.count_nonzero(baseline_mask) >= 4:
        baseline_rms = float(
            np.sqrt(np.mean(np.asarray(residual)[baseline_mask] ** 2))
        )
    else:
        baseline_rms = 0.0

    # 阈值取“绝对阈值”和“基线 RMS 若干倍”中的较大值。
    # 这样既不过分相信噪声很小的情况，也能适应噪声较大的实验。
    threshold = max(
        config.RECOVERY_ABSOLUTE_THRESHOLD_MM,
        config.RECOVERY_BASELINE_FACTOR * baseline_rms,
    )
    # 找到 motion_finished 在均匀时间轴中的位置。
    start_index = int(np.searchsorted(uniform_time_s, motion_finished, side="left"))

    # 运动后剩余数据太短时，不足以判断是否真的恢复。
    if len(uniform_time_s) - start_index < 4:
        return {
            "recovery_time_s": None,
            "recovery_threshold_mm": float(threshold),
            "baseline_rms_mm": float(baseline_rms),
        }

    # 只对运动后的残差求包络，避免运动段的大振幅通过 Hilbert 非局部边缘效应污染恢复起点。
    post_residual = np.asarray(residual[start_index:], dtype=float)
    # Hilbert 包络可以近似表示振动振幅随时间的变化。
    envelope = np.abs(signal.hilbert(post_residual))

    # 连续保持低于阈值的最少样本数。
    hold_samples = max(1, int(math.ceil(config.RECOVERY_HOLD_SECONDS * sample_rate_hz)))

    # 从运动结束后的每一个位置开始尝试，寻找第一个连续低于阈值的窗口。
    recovery_time: float | None = None
    for post_index in range(0, len(envelope) - hold_samples + 1):
        if np.all(envelope[post_index : post_index + hold_samples] <= threshold):
            absolute_index = start_index + post_index
            recovery_time = float(uniform_time_s[absolute_index] - motion_finished)
            break

    return {
        "recovery_time_s": recovery_time,
        "recovery_threshold_mm": float(threshold),
        "baseline_rms_mm": float(baseline_rms),
    }


def compare_vision_methods(
    records: list[dict[str, Any]],
    axis: str,
) -> dict[str, float | int | None]:
    """
    在两种方法的共同时间区间插值比较，不要求相同帧都识别成功。

    RMSE 小表示位移接近，相关系数高表示波形形状一致；两者不能单独证明绝对精度。
    """

    # 尝试分别提取圆点法和棋盘格法的有效序列。
    # 如果某种方法有效点太少，extract_vision_series 会抛 ValueError。
    try:
        circle_time, circle_value, _ = extract_vision_series(records, "circles", axis)
        checker_time, checker_value, _ = extract_vision_series(
            records,
            "checkerboard",
            axis,
        )
    except ValueError:
        return {
            "common_sample_count": 0,
            "circle_checker_rmse_mm": None,
            "circle_checker_correlation": None,
        }

    # 两种方法不一定在完全相同的时间范围都有结果。
    # 这里只比较它们共同覆盖的时间区间。
    common_start = max(circle_time[0], checker_time[0])
    common_end = min(circle_time[-1], checker_time[-1])

    # 使用圆点法的时间点作为公共时间轴。
    mask = (circle_time >= common_start) & (circle_time <= common_end)
    common_time = circle_time[mask]

    # 点太少时 RMSE 和相关系数都没有参考价值。
    if len(common_time) < 3:
        return {
            "common_sample_count": int(len(common_time)),
            "circle_checker_rmse_mm": None,
            "circle_checker_correlation": None,
        }

    # 将棋盘格法插值到圆点法的时间点上，才能逐点相减。
    checker_interpolated = np.interp(common_time, checker_time, checker_value)
    circle_common = circle_value[mask]

    # RMSE 表示两条曲线在数值上平均差多少毫米。
    difference = circle_common - checker_interpolated
    rmse = float(np.sqrt(np.mean(difference**2)))

    # 相关系数表示两条曲线形状是否同步。
    # 若某条曲线几乎不变，标准差为 0，则相关系数没有意义。
    if np.std(circle_common) > 0 and np.std(checker_interpolated) > 0:
        correlation = float(np.corrcoef(circle_common, checker_interpolated)[0, 1])
    else:
        correlation = math.nan

    return {
        "common_sample_count": int(len(common_time)),
        "circle_checker_rmse_mm": rmse,
        "circle_checker_correlation": correlation,
    }


# =============================================================================
# 5. 绘图
# =============================================================================

def _relative_time(time_s: np.ndarray, origin_s: float) -> np.ndarray:
    """
    所有图统一以第一条有效视觉记录为 0 s，避免显示很大的 perf_counter 秒数。

    perf_counter 秒数本身没有直观意义；减去 origin 后，
    图上的横轴就变成“从本次记录开始过了几秒”。
    """

    # np.asarray 保证 time_s 可以是 list 或数组，输出都是 NumPy 数组。
    return np.asarray(time_s, dtype=float) - origin_s


def _mark_events(
    axes: Axes,
    events: dict[str, float],
    origin_s: float,
) -> None:
    """
    在时域图上标出发送运动与完成时刻，帮助判断振动发生在哪一阶段。

    这只是画图辅助，不参与任何数值计算。
    """

    # 给不同事件固定颜色，便于多张图之间保持一致。
    colors = {
        "motion_command_sent": "tab:red",
        "motion_finished": "tab:green",
    }
    # 如果日志中存在某个事件，就在图上画一条竖线。
    for name, color in colors.items():
        if name in events:
            axes.axvline(
                events[name] - origin_s,
                color=color,
                linestyle="--",
                linewidth=1.2,
                label=name,
            )


def plot_time_domain(
    output_path: Path,
    original_time_s: np.ndarray,
    original_values: np.ndarray,
    uniform_time_s: np.ndarray,
    trend: np.ndarray,
    residual: np.ndarray,
    events: dict[str, float],
    method: str,
    axis: str,
) -> None:
    """
    保存“时间域”总览图：上半部分看原始位移，下半部分看振动残差。

    这张图通常是分析结果里最先看的图：
    - 上图帮助判断相机测到的整体位移趋势是否合理；
    - 下图帮助判断去掉整体运动后，剩下的微小振动大不大；
    - 竖虚线标出机器人开始运动和运动结束的时刻。
    """

    # 把本次记录第一帧作为图上的 0 秒，避免横轴显示很大的系统计时数字。
    origin = float(original_time_s[0])

    # 创建上下两个子图，共享同一个 x 轴，方便直接对齐观察。
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # 上图第一条线：原始视觉位移。
    # 它包含机器人整体移动趋势，也包含叠加在上面的微小振动。
    axes[0].plot(
        _relative_time(original_time_s, origin),
        original_values,
        linewidth=0.9,
        label="measured displacement",
    )

    # 上图第二条线：根据配置选定方法估计出来的慢趋势。
    # 原始位移减去这条趋势后，才得到下图中的 residual。
    axes[0].plot(
        _relative_time(uniform_time_s, origin),
        trend,
        linewidth=1.5,
        label=f"{config.DETREND_METHOD} trend",
    )

    # 设置上图坐标轴、网格和事件标记，让图不只是曲线，还能读出实验阶段。
    axes[0].set_ylabel("Displacement / mm")
    axes[0].grid(True, alpha=0.3)
    _mark_events(axes[0], events, origin)
    axes[0].legend(loc="best")

    # 下图只画 residual，也就是“去掉慢趋势后的剩余部分”。
    # 这部分才是后续 RMS、峰峰值和频谱分析关注的对象。
    axes[1].plot(
        _relative_time(uniform_time_s, origin),
        residual,
        color="tab:orange",
        linewidth=0.9,
    )

    # 零线用于判断残差是在 0 附近振动，还是仍存在明显偏移。
    axes[1].axhline(0.0, color="black", linewidth=0.7)

    # 设置下图坐标轴、网格和同样的事件竖线。
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Vibration residual / mm")
    axes[1].grid(True, alpha=0.3)
    _mark_events(axes[1], events, origin)

    # 总标题记录使用的视觉方法和分析方向，避免以后打开图片时忘记配置。
    figure.suptitle(f"Vision displacement and vibration — {method}, axis={axis}")

    # tight_layout 尽量避免标题、坐标轴文字和图像内容互相挤压。
    figure.tight_layout()

    # 保存 PNG 后立刻关闭 figure，避免批量分析时内存里堆积很多图对象。
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_spectrum(
    output_path: Path,
    spectrum: dict[str, Any],
    sample_rate_hz: float,
) -> None:
    """
    保存“频谱”图：上半部分是 FFT 幅值，下半部分是 Welch 功率谱密度。

    初学者可以这样理解：
    - 时间域图告诉你“什么时候振”；
    - 频谱图告诉你“主要以多少 Hz 在振”；
    - FFT 幅值直观，Welch PSD 更稳定，适合看噪声中的频率成分。
    """

    # 奈奎斯特频率是当前采样率理论上能表示的最高频率。
    # 图上限不能超过奈奎斯特频率，否则会显示没有物理意义的范围。
    nyquist = 0.5 * sample_rate_hz
    upper = min(config.DOMINANT_FREQ_MAX_HZ, nyquist)

    # 创建两个上下排列的频域子图，并共用频率横轴。
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # 第一幅图画 FFT 幅值谱，用于快速看出哪些频率分量比较强。
    axes[0].plot(
        spectrum["fft_frequency_hz"],
        spectrum["fft_amplitude_mm"],
        linewidth=1.0,
    )

    # 限制频率范围，避免高频空白或无关频段压缩主要区域。
    axes[0].set_ylabel("FFT amplitude / mm")
    axes[0].set_xlim(0.0, upper)
    axes[0].grid(True, alpha=0.3)

    # 第二幅图画 Welch PSD。
    # semilogy 表示 y 轴使用对数刻度，更容易同时观察强信号和弱信号。
    axes[1].semilogy(
        spectrum["welch_frequency_hz"],
        np.maximum(spectrum["welch_psd_mm2_per_hz"], 1e-18),
        linewidth=1.0,
    )

    # np.maximum(..., 1e-18) 是为了避免对数坐标遇到 0 后无法绘图。
    axes[1].set_xlabel("Frequency / Hz")
    axes[1].set_ylabel("PSD / mm²/Hz")
    axes[1].set_xlim(0.0, upper)
    axes[1].grid(True, which="both", alpha=0.3)

    # 主频来自 calculate_spectrum() 的自动搜索结果，写在标题里便于快速查看。
    dominant = spectrum["dominant_frequency_hz"]
    figure.suptitle(f"Vibration spectrum — dominant {dominant:.3f} Hz")

    # 保存图片并释放 Matplotlib 图对象。
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_method_comparison(
    output_path: Path,
    records: list[dict[str, Any]],
    axis: str,
) -> bool:
    """
    保存“圆点法 vs 棋盘格法”对比图。

    这个图用于检查两套视觉算法是否给出相近趋势。
    如果某一种方法有效点太少，就返回 False，避免生成一张看似成功但没有信息量的图。
    """

    # 分别提取圆点法和棋盘格法的有效位移序列。
    # 任意一种方法提取失败，说明当前日志不适合做这张对比图。
    try:
        circle_time, circle_value, _ = extract_vision_series(records, "circles", axis)
        checker_time, checker_value, _ = extract_vision_series(
            records,
            "checkerboard",
            axis,
        )
    except ValueError:
        return False

    # 两条曲线共用一个横轴原点，便于直接比较波形出现的时间。
    origin = min(circle_time[0], checker_time[0])

    # 创建单幅图即可，因为这里比较的是两种方法测到的同一类位移。
    figure, axes = plt.subplots(figsize=(11, 4.8))

    # 第一条线：圆点阵列跟踪结果。
    axes.plot(
        _relative_time(circle_time, origin),
        circle_value,
        linewidth=0.9,
        label="circles",
    )

    # 第二条线：棋盘格角点跟踪结果。
    # alpha 稍低一点，重叠时仍能看清两条曲线。
    axes.plot(
        _relative_time(checker_time, origin),
        checker_value,
        linewidth=0.9,
        label="checkerboard",
        alpha=0.8,
    )

    # 设置坐标、标题、网格和图例，让图片脱离上下文也能读懂。
    axes.set_xlabel("Time / s")
    axes.set_ylabel("Displacement / mm")
    axes.set_title(f"Circle vs checkerboard — axis={axis}")
    axes.grid(True, alpha=0.3)
    axes.legend()

    # 保存图片，关闭 figure，并用 True 告诉调用者“这张图确实生成了”。
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return True


def plot_vision_robot_alignment(
    output_path: Path,
    vision_time_s: np.ndarray,
    vision_values: np.ndarray,
    robot_series: tuple[np.ndarray, np.ndarray] | None,
    axis: str,
) -> bool:
    """
    用共同 host_ns 画视觉与 UR TCP，不强行假设两者零点和方向定义完全一致。

    两条曲线都减去各自第一点，只比较相对变化；外参未标定前不能把它当作绝对误差。
    """

    # 如果日志里没有 ROBOT 记录，说明这次数据只有视觉结果，无法画机器人对齐图。
    if robot_series is None:
        return False

    # 解包机器人时间序列，并寻找视觉与机器人共同覆盖的时间段。
    robot_time_s, robot_values = robot_series
    common_start = max(vision_time_s[0], robot_time_s[0])
    common_end = min(vision_time_s[-1], robot_time_s[-1])

    # 没有重叠时间段时，两条曲线无法在同一横轴上进行比较。
    if common_end <= common_start:
        return False

    # 图上的 0 秒取共同时间段起点。
    origin = common_start

    # 视觉和机器人各自减去第一点，只比较“变化量”。
    # 这样可以避开未标定外参造成的绝对坐标零点差异。
    vision_relative = vision_values - vision_values[0]
    robot_relative = robot_values - robot_values[0]

    # 创建单幅图，把视觉和 UR TCP 相对位移画在一起。
    figure, axes = plt.subplots(figsize=(11, 4.8))

    # 第一条线：视觉算法估计的相对位移。
    axes.plot(
        _relative_time(vision_time_s, origin),
        vision_relative,
        linewidth=0.9,
        label="vision relative",
    )

    # 第二条线：UR 控制器反馈的 TCP 相对位移。
    axes.plot(
        _relative_time(robot_time_s, origin),
        robot_relative,
        linewidth=0.9,
        label="UR TCP relative",
        alpha=0.8,
    )

    # 只显示两者共同存在的时间范围，避免曲线外推造成误解。
    axes.set_xlim(0.0, common_end - common_start)

    # 设置坐标轴、标题、网格和图例。
    axes.set_xlabel("Time / s")
    axes.set_ylabel("Relative displacement / mm")
    axes.set_title(f"Host-clock alignment — axis={axis}")
    axes.grid(True, alpha=0.3)
    axes.legend()

    # 保存图片并关闭 figure；返回 True 表示图已成功生成。
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return True


# =============================================================================
# 6. 摘要写入与分析总入口
# =============================================================================

def _format_optional(value: float | int | None, digits: int = 6) -> str:
    """
    把可选数值转成摘要文件里的字符串。

    分析里有些指标可能真的算不出来，例如：
    - 没有 motion_finished 事件，就无法计算恢复时间；
    - 两种视觉方法共同样本太少，就无法计算相关系数；
    - 输入本身是 NaN 或 inf，也不能当作有效数值写入摘要。

    这时写 unavailable 比写 0 更安全，因为 0 会让人误以为结果真的等于零。
    """

    # None 表示“没有这个结果”，直接写成 unavailable。
    if value is None:
        return "unavailable"

    # 尝试把 int/float/字符串数字统一转成 float，方便判断是否有限。
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        # 如果不是数字，就保留它本来的文字形式。
        return str(value)

    # NaN 和 inf 都不是可解释的实验结果，摘要中统一显示 unavailable。
    if not math.isfinite(numeric):
        return "unavailable"

    # 正常数字按指定小数位格式化，避免摘要里数字长得过于杂乱。
    return f"{numeric:.{digits}f}"


def write_summary(
    output_path: Path,
    loaded: LoadedRun,
    timing: dict[str, float],
    timing_warnings: list[str],
    selected_window: str,
    window_counts: dict[str, int],
    time_metrics: dict[str, float],
    spectrum: dict[str, Any],
    recovery: dict[str, float | None],
    comparison: dict[str, float | int | None],
    valid_quality: np.ndarray,
    primary_count: int,
) -> None:
    """
    写出一份中文分析摘要 TXT，方便人工快速复查本次实验结果。

    这个文件的目标不是给程序再次读取，而是给人看：
    - 写清楚源数据来自哪个日志；
    - 写清楚使用了哪种视觉方法、哪个方向、哪种去趋势方法；
    - 把时域、频域、恢复时间和方法一致性指标按章节列出来；
    - 如果日志中本来就记录了 ERROR，也一并放进摘要。
    """

    # lines 是最终 TXT 的每一行。
    # 先写标题、来源和本次分析使用的关键配置。
    lines = [
        "UR10 末端振动预实验分析摘要",
        "=" * 56,
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"源文件：{loaded.source_path}",
        f"视觉方法：{config.ANALYSIS_VISION_METHOD}",
        f"分析方向：{config.ANALYSIS_AXIS}",
        f"主分析窗口：{selected_window}",
        f"去趋势方法：{config.DETREND_METHOD}",
        "",

        # 第一节检查数据完整性，帮助判断这次分析的数据基础是否可靠。
        "一、数据完整性",
        f"视觉总记录数：{len(loaded.vision)}",
        f"本方法有效记录数：{len(valid_quality)}",
        f"主分析区间点数：{primary_count}",
        f"机器人记录数：{len(loaded.robot)}",
        f"事件记录数：{len(loaded.events)}",
        f"错误记录数：{len(loaded.errors)}",
        f"无法解析的行：{loaded.malformed_lines or '无'}",
        f"平均识别质量：{_format_optional(float(np.nanmean(valid_quality)), 4)}",
        f"窗口点数：{window_counts}",
        "",

        # 第二节记录采样时间质量。
        # 如果采样间隔波动很大，频域分析的可信度会受影响。
        "二、采样时间",
        f"估计采样率：{timing['sample_rate_hz']:.6f} Hz",
        f"中位采样间隔：{timing['median_dt_s']:.9f} s",
        f"采样间隔标准差：{timing['dt_std_s']:.9f} s",
        f"最大相邻间隔：{timing['max_gap_s']:.9f} s",
        f"采样率警告：{'; '.join(timing_warnings) if timing_warnings else '无'}",
        "",

        # 第三节是时域振动强度指标。
        # 它们都来自去趋势残差，不是机器人整体移动距离。
        "三、去趋势后时域指标",
        f"峰峰值：{time_metrics['peak_to_peak_mm']:.6f} mm",
        f"RMS：{time_metrics['rms_mm']:.6f} mm",
        f"标准差：{time_metrics['std_mm']:.6f} mm",
        f"最大绝对残差：{time_metrics['max_abs_mm']:.6f} mm",
        "",

        # 第四节是频域指标，主要用于判断振动集中在哪些频率。
        "四、频域指标",
        f"主频：{_format_optional(spectrum['dominant_frequency_hz'], 6)} Hz",
        f"频率分辨率：{spectrum['frequency_resolution_hz']:.6f} Hz",
    ]

    # 频带能量来自 config.FREQUENCY_BANDS_HZ。
    # 使用循环写入，方便以后在 config.py 里增减频段，而不用改这里的正文结构。
    for band_name, energy in spectrum["band_energy_mm2"].items():
        lines.append(f"{band_name} 频带能量：{energy:.10e} mm²")

    # 追加恢复时间、两种视觉方法一致性，以及阅读摘要时必须注意的解释提醒。
    lines.extend(
        [
            "",
            "五、恢复时间",
            (
                "恢复时间："
                f"{_format_optional(recovery['recovery_time_s'], 6)} s"
            ),
            (
                "恢复阈值："
                f"{_format_optional(recovery['recovery_threshold_mm'], 6)} mm"
            ),
            (
                "静止基线 RMS："
                f"{_format_optional(recovery['baseline_rms_mm'], 6)} mm"
            ),
            "",
            "六、圆点法与棋盘格法一致性",
            f"共同样本数：{comparison['common_sample_count']}",
            (
                "两方法 RMSE："
                f"{_format_optional(comparison['circle_checker_rmse_mm'], 6)} mm"
            ),
            (
                "两方法相关系数："
                f"{_format_optional(comparison['circle_checker_correlation'], 6)}"
            ),
            "",
            "解释提醒",
            "1. 去趋势残差不是相机原始位移，低频结果会受到所选趋势模型影响。",
            "2. 自动得到的主频必须同时查看频谱图、静止噪声和识别质量，不能只看一个数字。",
            "3. 视觉与 UR TCP 未完成坐标外参标定前，只能比较共同时间上的相对变化。",
        ]
    )

    # 如果原始日志里有 ERROR 记录，摘要末尾完整列出，便于回头排查采集过程。
    if loaded.errors:
        lines.extend(["", "记录中的错误："])
        for error in loaded.errors:
            lines.append(f"- {error.get('message', error)}")

    # 最后一次性写入文件。
    # 用 UTF-8 保存，保证中文摘要在 VS Code 和大多数编辑器中都能正常打开。
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(analysis_file: Path | None = None) -> Path:
    """
    完成一次离线振动分析并返回新建的结果文件夹。

    输入：
    - run_log.txt 或 vision_results.txt；
    - config.py 中选择的视觉方法、分析方向、主窗口和去趋势方法。

    输出：
    - 时域图、频谱图、方法对比图、视觉/机器人对齐图；
    - analysis_summary.txt 给人快速复查；
    - analysis_metrics.json 给后续批量比较。

    实验作用：
    这里把视觉模块输出的“逐帧位移数据”转换成“振动结论”。
    主窗口用于计算 RMS、峰峰值和频谱；完整窗口用于展示全程和估计恢复时间。
    """

    # 本段读入已经保存的实验/视觉日志。
    # 输入可以来自命令行，也可以由配置自动选择最新文件；输出 loaded 按记录类型拆好了 META/EVENT/VISION/ROBOT。
    source_path = resolve_analysis_file(analysis_file)
    loaded = load_run_file(source_path)

    # 本段从逐帧 VISION 记录中抽出要分析的位移曲线。
    # 输入是完整视觉日志；输出是某一种方法、某一个方向的 time/value/quality 序列。
    # 例如 circles + x 表示“圆点法测得的 x 方向位移”。
    vision_time, vision_values, quality = extract_vision_series(
        loaded.vision,
        config.ANALYSIS_VISION_METHOD,
        config.ANALYSIS_AXIS,
    )

    # 本段把同一条位移曲线切成不同实验阶段。
    # 输入是视觉时间轴和 EVENT 事件；输出是 full/baseline/motion/steady_motion/post 等窗口掩码。
    events = event_times_seconds(loaded.events)
    windows = select_analysis_windows(vision_time, events)

    # 本段决定“哪一段真正拿去算主指标”。
    # 输入是所有候选窗口；输出是 primary_time/primary_values。
    # 如果所选窗口点数太少，会退回 full，避免用几帧数据硬算频谱。
    selected_window = config.ANALYSIS_PRIMARY_WINDOW
    primary_mask = windows[selected_window]
    if np.count_nonzero(primary_mask) < 8:
        print(f"[分析警告] {selected_window} 区间有效点不足 8，改用全部有效视觉数据。")
        selected_window = "full"
        primary_mask = windows["full"]

    window_counts = {
        name: int(np.count_nonzero(mask))
        for name, mask in windows.items()
    }

    primary_time = vision_time[primary_mask]
    primary_values = vision_values[primary_mask]

    # 本段保护主分析的最低数据量。
    # 少于 8 个点时，重采样、去趋势和频谱都没有稳定意义，因此直接报错。
    if len(primary_time) < 8:
        raise ValueError("最终主分析区间仍少于 8 个有效点。")

    # 本段把主窗口位移转换成“振动残差”。
    # 输入是原始位移曲线；输出是均匀时间轴、趋势项和 residual。
    # residual 才是后续 RMS/频谱真正关心的抖动，不是机器人正常运动的大位移。
    uniform_time, uniform_values, sample_rate_hz, timing = resample_uniform(
        primary_time,
        primary_values,
    )
    timing_warnings = sampling_warnings(timing)
    for warning in timing_warnings:
        print(f"[分析警告] {warning}")

    trend, residual = detrend_motion(uniform_values, sample_rate_hz)

    # 本段把残差转换成主分析结论。
    # time_metrics 描述“抖得多大”；spectrum 描述“主要以多少 Hz 在抖”。
    time_metrics = calculate_time_metrics(residual)
    spectrum = calculate_spectrum(residual, sample_rate_hz)

    # 本段准备恢复时间所需的完整序列。
    # 主窗口可能只包含 motion 或 steady_motion；恢复时间必须同时看运动前基线和运动后尾段。
    full_mask = windows["full"]
    full_time = vision_time[full_mask]
    full_values = vision_values[full_mask]

    full_uniform_time, full_uniform_values, full_rate, _ = resample_uniform(
        full_time,
        full_values,
    )

    # 本段对完整序列分段去趋势。
    # 输入是全程位移；输出是全程趋势和全程残差。
    # 分段处理能减少“静止-运动-静止”三段互相拉扯趋势线的问题。
    full_trend, full_residual = detrend_piecewise(
        full_uniform_time,
        full_uniform_values,
        full_rate,
        events,
    )

    # 本段提取运动前静止基线。
    # 输入是实验开始和运动命令事件；输出 baseline_uniform_mask。
    # 恢复时间阈值会参考基线 RMS，避免把静态视觉噪声误判成残余振动。
    baseline_start = events.get("experiment_started", float(full_uniform_time[0]))
    baseline_end = events.get("motion_command_sent")
    baseline_uniform_mask = _window_mask(
        full_uniform_time,
        baseline_start,
        baseline_end,
    )

    if baseline_end is None:
        baseline_uniform_mask[:] = False

    # 本段计算运动后的恢复时间。
    # 输入是全程残差、事件时间和基线掩码；输出包含恢复时间、阈值和基线 RMS。
    # 事件不足时函数返回 None，不会凭空猜一个恢复时间。
    recovery = calculate_recovery_time(
        full_uniform_time,
        full_residual,
        events,
        baseline_uniform_mask,
        full_rate,
    )

    # 本段准备辅助对比数据。
    # 输入仍然只来自日志文件；输出用于画圆点/棋盘格一致性图，以及视觉/UR TCP 对齐图。
    # 这里不会连接机器人，也不会读取相机。
    comparison = compare_vision_methods(loaded.vision, config.ANALYSIS_AXIS)
    robot_series = extract_robot_tcp_series(loaded.robot, config.ANALYSIS_AXIS)

    # 本段创建本次分析结果文件夹。
    # 同一个原始日志可以反复用不同窗口、方向或去趋势参数分析；每次结果都单独保存。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = source_path.parent / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 本段输出时域总览图。
    # 输入是完整原始位移、完整趋势、完整残差和事件时间；输出用于检查抖动发生在哪个实验阶段。
    plot_time_domain(
        output_dir / "01_time_domain.png",
        full_time,
        full_values,
        full_uniform_time,
        full_trend,
        full_residual,
        events,
        config.ANALYSIS_VISION_METHOD,
        config.ANALYSIS_AXIS,
    )

    # 本段输出频域图。
    # 输入是主窗口 residual 的频谱结果；输出用于查看主频和频带能量。
    plot_spectrum(
        output_dir / "02_spectrum.png",
        spectrum,
        sample_rate_hz,
    )

    # 本段输出圆点法/棋盘格法一致性图。
    # 如果其中一种方法数据不足，绘图函数会跳过有效曲线，避免制造误导性对比。
    plot_method_comparison(
        output_dir / "03_circle_checker_comparison.png",
        loaded.vision,
        config.ANALYSIS_AXIS,
    )

    # 本段输出视觉与 UR TCP 对齐图。
    # 没有 ROBOT 记录时不会生成有效曲线；这张图只用于时间关系和趋势对比，不代表已完成外参标定。
    plot_vision_robot_alignment(
        output_dir / "04_vision_robot_alignment.png",
        full_time,
        full_values,
        robot_series,
        config.ANALYSIS_AXIS,
    )

    # 本段输出给人看的中文摘要。
    # 输入是本次分析的核心指标、采样率警告、窗口点数和错误记录；输出用于快速判断结果是否可信。
    write_summary(
        output_dir / "analysis_summary.txt",
        loaded,
        timing,
        timing_warnings,
        selected_window,
        window_counts,
        time_metrics,
        spectrum,
        recovery,
        comparison,
        quality,
        len(primary_time),
    )

    # 机器可读指标便于以后批量比较不同速度、轨迹或控制方法，不必再从 TXT 反向解析数字。
    # JSON 和 TXT 摘要保存的是同一批核心结论，只是面向“程序读取”和“人工阅读”两种用途。
    metrics_json = {
        "source_file": str(source_path),
        "analysis_method": config.ANALYSIS_VISION_METHOD,
        "analysis_axis": config.ANALYSIS_AXIS,
        "analysis_primary_window": selected_window,
        "window_counts": window_counts,
        "detrend_method": config.DETREND_METHOD,
        "timing": timing,
        "timing_warnings": timing_warnings,
        "time_metrics": time_metrics,
        "dominant_frequency_hz": spectrum["dominant_frequency_hz"],
        "band_energy_mm2": spectrum["band_energy_mm2"],
        "recovery": recovery,
        "method_comparison": comparison,
    }

    # ensure_ascii=False 保留中文；allow_nan=True 允许 JSON 中记录 NaN，方便保留无法计算的数值状态。
    (output_dir / "analysis_metrics.json").write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    # 在终端打印最关键的三项结果，方便命令行运行时不用先打开 summary。
    print(
        f"[分析] 峰峰值 {time_metrics['peak_to_peak_mm']:.6f} mm，"
        f"RMS {time_metrics['rms_mm']:.6f} mm，"
        f"主频 {spectrum['dominant_frequency_hz']:.3f} Hz。"
    )

    # 打印输出目录，让用户能马上找到图表和摘要。
    print(f"[分析] 结果文件夹：{output_dir}")
    return output_dir


if __name__ == "__main__":
    # 允许直接运行 analyze.py，但推荐统一从 main.py 的 analyze 模式进入。
    # validate_config("analyze") 会检查分析相关配置，但不会连接相机或机器人。
    config.validate_config("analyze")

    # 直接运行本文件时，不传 analysis_file，让 resolve_analysis_file() 按配置自动选择。
    run_analysis()
