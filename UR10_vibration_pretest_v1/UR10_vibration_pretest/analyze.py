"""
离线读取实验 TXT/JSONL，并把逐帧记录转换成振动分析结果。

从用户操作看，本文件只在 RUN_MODE="analyze" 或 main.py 调用 run_analysis() 时工作。
它不会打开相机、不会连接 UR，也不会发送任何运动指令，只读取已经保存好的日志。

数据流可以这样理解：
1. resolve_analysis_file() 决定要分析哪一个 run_log.txt / vision_results.txt。
2. load_run_file() 逐行读取 JSON，并按 META、EVENT、VISION、ROBOT、ERROR 分类。
3. extract_vision_series() 从 VISION 记录中抽出某种视觉方法、某个方向的位移曲线。
4. select_analysis_windows() 根据 EVENT 把曲线切成 baseline、motion、steady_motion、post 等阶段。
5. resample_uniform() 把真实时间戳下的不等间隔点整理成均匀时间轴，供滤波和频谱使用。
6. detrend_motion() / detrend_piecewise() 去掉机器人正常移动的慢趋势，留下振动残差。
7. calculate_*() 计算时域指标、频域指标、恢复时间和两种视觉方法一致性。
8. plot_*() 与 write_summary() 把图、中文摘要和 JSON 指标写入 analysis_时间戳 文件夹。

这份分析代码的核心物理思路：
视觉位移 = 机器人正常运动造成的慢变化 + 末端振动造成的小幅快速变化。
因此分析时先剥离慢趋势，再对残差计算 RMS、主频、频带能量和恢复时间。
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

# 本文件只保存图，不弹出图形窗口。
# Agg 后端不需要桌面环境，适合 VS Code、服务器或批处理稳定生成 PNG。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import config


# =============================================================================
# 1. 读取和整理逐行 JSON 记录：先把混合日志拆成后续函数能理解的几类数据
# =============================================================================

@dataclass(slots=True)
class LoadedRun:
    """
    一次实验日志被解析后的分类结果。

    输入来源：run_log.txt、vision_results.txt 或 robot_test_log.txt 中的一行行 JSON。

    输出去向：
    - meta 给摘要说明本次运行配置；
    - events 用于划分运动前、运动中、运动后窗口；
    - vision 是振动分析的主要数据源；
    - robot 用于和视觉曲线做时间对齐参考；
    - errors/malformed_lines 用于提醒这次采集或日志文件是否异常。

    实验作用：把“混在一个文件里的多来源记录”拆成清楚的容器，
    后续分析函数就不用反复在全部日志中搜索 kind。
    """

    source_path: Path
    meta: list[dict[str, Any]]
    events: list[dict[str, Any]]
    vision: list[dict[str, Any]]
    robot: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    malformed_lines: list[int]


@dataclass(slots=True)
class AnalysisSegment:
    """
    描述一段已经被自动切出来的实验阶段。

    输入来源：EVENT 时间戳、ROBOT TCP 速度，或视觉位移变化。
    输出去向：select_analysis_windows() 把它转换成布尔窗口，run_analysis() 再分别计算单段和合并指标。

    实验作用：把“整条曲线”拆成更接近真实工况的静止段、运动段和匀速段。
    例如反复启停实验中会出现 motion_001、static_002、motion_002，而不是只剩一段笼统 motion。
    """

    name: str
    kind: str
    start_s: float
    end_s: float
    source: str

    def to_json(self) -> dict[str, float | str]:
        """把片段对象转换成 JSON 摘要能直接保存的字典。"""

        return {
            "name": self.name,
            "kind": self.kind,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.end_s - self.start_s,
            "source": self.source,
        }


def _find_latest_run_file() -> Path:
    """
    在 outputs 中自动寻找最近一次可分析记录。

    输入：config.OUTPUT_ROOT 下的历史输出目录。
    输出：修改时间最新的 run_log.txt、vision_results.txt 或 robot_test_log.txt。

    实验作用：方便刚跑完实验后直接 analyze，不必手动复制路径。
    如果需要固定复查某次实验，应在 config.ANALYSIS_FILE 或命令行 --analysis-file 明确指定。
    """

    # 本段列出可被分析器识别的日志文件名。
    # 正式实验、视觉测试、机器人单机测试分别使用不同文件名。
    patterns = ("**/run_log.txt", "**/vision_results.txt", "**/robot_test_log.txt")
    candidates: list[Path] = []

    # 本段递归搜索 outputs 下所有历史结果目录。
    for pattern in patterns:
        candidates.extend(config.OUTPUT_ROOT.glob(pattern))

    # 本段只保留真实文件，排除同名目录等异常情况。
    candidates = [path for path in candidates if path.is_file()]

    # 本段处理没有可分析文件的情况，直接给出清晰错误。
    if not candidates:
        raise FileNotFoundError(
            f"{config.OUTPUT_ROOT} 下没有 run_log.txt 或 vision_results.txt。"
        )
    # 本段选择最近修改的文件，通常就是刚刚运行得到的结果。
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_analysis_file(requested: Path | None = None) -> Path:
    """
    决定这次 analyze 到底读取哪个日志文件。

    输入：
    - requested：main.py 从 --analysis-file 传进来的路径；
    - config.ANALYSIS_FILE；
    - outputs 下的最新记录。

    输出：最终存在的绝对 Path。

    实验作用：同一套分析代码既能“自动分析刚刚跑完的记录”，也能“指定复查某一次旧实验”。
    优先级为：命令行参数 -> config.py -> 自动寻找最新记录。
    """

    # 本段应用优先级：命令行参数优先于 config.py。
    candidate = requested or config.ANALYSIS_FILE

    # 本段处理用户没有指定文件的情况，自动从 outputs 中找最新记录。
    if candidate is None:
        resolved = _find_latest_run_file()
        print(f"[分析] 未指定文件，自动选择最新记录：{resolved}")
        return resolved

    # 本段把用户写法整理成 Path。expanduser 支持 ~ 这样的用户目录写法。
    candidate = Path(candidate).expanduser()

    # 本段统一相对路径的基准目录，避免从不同终端位置运行时解析到不同文件。
    if not candidate.is_absolute():
        candidate = (config.PROJECT_DIR / candidate).resolve()

    # 本段确认目标文件真实存在，避免后续打开文件时才报更难懂的错误。
    if not candidate.exists():
        raise FileNotFoundError(f"分析文件不存在：{candidate}")
    return candidate


def load_run_file(path: Path) -> LoadedRun:
    """
    读取逐行 JSON 日志，并按 kind 分类。

    输入：run_log.txt / vision_results.txt / robot_test_log.txt。
    输出：LoadedRun，其中不同 kind 已拆入不同列表，坏行号保存在 malformed_lines。

    实验作用：实验日志可能因为异常退出、磁盘写入或人工编辑出现个别坏行。
    这里逐行独立解析，单行损坏不会让整个实验完全打不开；坏行会写入摘要提醒复查。
    """

    # 本段准备分类桶。输入是一条条 JSON 记录；输出是按 kind 分类的列表。
    buckets: dict[str, list[dict[str, Any]]] = {
        "META": [],
        "EVENT": [],
        "VISION": [],
        "ROBOT": [],
        "ERROR": [],
    }
    # 本段保存无法解析或结构异常的行号，后续写入中文摘要。
    malformed: list[int] = []

    # 本段逐行读取日志。每一行独立处理，尽量保留其他正常行。
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            # 空行没有实验信息，直接跳过。
            text = raw_line.strip()
            if not text:
                continue

            # 本段尝试把当前行从 JSON 文本转成 Python 对象。
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                malformed.append(line_number)
                continue

            # 本项目约定每行必须是一个字典；其他类型视为异常行。
            if not isinstance(record, dict):
                malformed.append(line_number)
                continue

            # 本段按 kind 把记录放入对应桶；未知 kind 说明日志结构不符合本项目约定。
            kind = str(record.get("kind", "")).upper()
            if kind in buckets:
                buckets[kind].append(record)
            else:
                malformed.append(line_number)

    # 本段把分类后的内容打包成 dataclass，后续传参更清楚。
    loaded = LoadedRun(
        source_path=Path(path),
        meta=buckets["META"],
        events=buckets["EVENT"],
        vision=buckets["VISION"],
        robot=buckets["ROBOT"],
        errors=buckets["ERROR"],
        malformed_lines=malformed,
    )

    # 本段确认有视觉数据。没有 VISION 记录就无法分析相机测得的末端振动。
    if not loaded.vision:
        raise ValueError(
            f"{path} 中没有 VISION 记录，无法计算相机测得的末端振动。"
        )
    return loaded


# =============================================================================
# 2. 视觉序列、事件和机器人序列：从分类日志里抽出可分析的时间曲线
# =============================================================================

def _finite_float(value: Any) -> float:
    """
    把日志字段统一清洗成“有限浮点数或 NaN”。

    输入：JSON 字段值，可能是 int、float、字符串、None、NaN 或 inf。
    输出：可用时返回 float；不可用时返回 math.nan。

    实验作用：日志字段可能缺失或异常。与其让每个分析函数都重复判断，
    不如在入口统一清洗，后续只需要用 math.isfinite() 判断是否可参与计算。
    """

    try:
        # 本段把 int、float、可转数字的字符串统一变成浮点数。
        result = float(value)
    except (TypeError, ValueError):
        # 不能转成数字时，用 NaN 表示“无效数值”。
        return math.nan

    # 本段排除 inf 和 nan，保证后续有效数据都是有限实数。
    return result if math.isfinite(result) else math.nan


def extract_vision_series(
    records: list[dict[str, Any]],
    method: str,
    axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 VISION 记录中提取一条可用于振动分析的位移曲线。

    输入：
    - records：load_run_file() 得到的 VISION 记录；
    - method：circles 或 checkerboard；
    - axis：x、y 或 magnitude。

    输出：
    - time_array：每个有效视觉点的时间；
    - displacement_array：对应方向的位移，单位 mm；
    - quality_array：识别质量分。

    实验作用：只保留该方法明确标记有效、且时间和位移都有效的帧。
    对正式 experiment，host_ns 是相机与 UR 共同的电脑时基；分析绝不假设“第 N 帧对应第 N 条机器人记录”。
    """

    # 本段确定字段前缀。不同视觉方法在 JSON 中使用不同字段名。
    prefix = "circle" if method == "circles" else "checker"

    # 本段先用 list 收集有效点，最后再统一转 NumPy 数组，便于排序和插值。
    time_values: list[float] = []
    displacement_values: list[float] = []
    quality_values: list[float] = []

    # 本段逐条检查 VISION 记录。输入是一帧视觉结果；输出可能追加一个有效采样点。
    for record in records:
        # 只保留该方法明确标记有效的帧，避免把识别失败结果混进振动曲线。
        if not bool(record.get(f"{prefix}_is_valid", False)):
            continue

        # 本段读取并清洗时间、位移、质量分。
        host_ns = _finite_float(record.get("host_ns"))
        analysis_time_s = _finite_float(record.get("analysis_time_s"))
        dx = _finite_float(record.get(f"{prefix}_dx_mm"))
        dy = _finite_float(record.get(f"{prefix}_dy_mm"))
        quality = _finite_float(record.get(f"{prefix}_quality"))

        # 本段根据用户选择的分析方向，把 dx/dy 转成一条标量位移曲线。
        if axis == "x":
            displacement = dx
        elif axis == "y":
            displacement = dy
        elif axis == "magnitude":
            displacement = math.hypot(dx, dy)
        else:
            raise ValueError(f"未知分析轴：{axis}")

        # 本段决定这一帧是否真正进入分析。
        # vision_test 优先使用 analysis_time_s；正式 experiment 可用 host_ns 与机器人记录对时。
        if math.isfinite(displacement) and (
            math.isfinite(analysis_time_s) or math.isfinite(host_ns)
        ):
            if math.isfinite(analysis_time_s):
                time_values.append(analysis_time_s)
            else:
                time_values.append(host_ns * 1e-9)
            displacement_values.append(displacement)
            quality_values.append(quality)

    # 本段保护最低数据量。点太少时，滤波、RMS 和频谱都没有稳定意义。
    if len(time_values) < 8:
        raise ValueError(
            f"{method} 在 {axis} 方向只有 {len(time_values)} 个有效点，"
            "至少需要 8 个才能进行基本振动分析。"
        )

    # 本段转成 NumPy 数组，便于排序、插值和数学计算。
    time_array = np.asarray(time_values, dtype=float)
    displacement_array = np.asarray(displacement_values, dtype=float)
    quality_array = np.asarray(quality_values, dtype=float)

    # 本段显式按时间排序。日志通常有序，但异常写入顺序不应影响分析。
    order = np.argsort(time_array)
    time_array = time_array[order]
    displacement_array = displacement_array[order]
    quality_array = quality_array[order]

    # 本段去掉重复时间点，保证后续插值时横轴严格递增。
    # 同一时间理论上不应重复；若发生，只保留第一次。
    unique_time, unique_indices = np.unique(time_array, return_index=True)
    return (
        unique_time,
        displacement_array[unique_indices],
        quality_array[unique_indices],
    )


def event_times_seconds(events: list[dict[str, Any]]) -> dict[str, float]:
    """
    把 EVENT 记录转换成“事件名 -> 发生时间秒数”的字典。

    输入：load_run_file() 得到的 EVENT 记录。
    输出：每种事件第一次出现的 host_ns 秒值。

    实验作用：事件时间用于划分分析窗口，例如运动命令发出、运动完成、实验开始和结束。
    重复事件保留第一次，以避免后续重复写入覆盖真实起点。
    """

    result: dict[str, float] = {}

    # 本段先按 host_ns 排序，确保“第一次出现”是真的最早事件。
    for event in sorted(events, key=lambda item: _finite_float(item.get("host_ns"))):
        # 本段读取事件名称和电脑时钟。
        name = str(event.get("name", ""))
        host_ns = _finite_float(event.get("host_ns"))

        # 本段只保留名称非空、时间有效、且尚未出现过的事件。
        if name and math.isfinite(host_ns) and name not in result:
            result[name] = host_ns * 1e-9
    return result


def event_times_sequence(events: list[dict[str, Any]]) -> dict[str, list[float]]:
    """
    把 EVENT 记录转换成“事件名 -> 所有发生时间”的字典。

    输入：load_run_file() 得到的 EVENT 记录。
    输出：每种事件的完整时间列表，单位秒，并按发生顺序排列。

    实验作用：event_times_seconds() 只保留第一次事件，适合旧版单段实验；
    本函数保留重复事件，适合反复启停、L 形多段运动和以后更复杂的预实验。
    """

    result: dict[str, list[float]] = {}
    for event in sorted(events, key=lambda item: _finite_float(item.get("host_ns"))):
        name = str(event.get("name", ""))
        host_ns = _finite_float(event.get("host_ns"))
        if name and math.isfinite(host_ns):
            result.setdefault(name, []).append(host_ns * 1e-9)
    return result


def _window_mask(
    time_s: np.ndarray,
    start_s: float | None,
    end_s: float | None,
) -> np.ndarray:
    """
    根据开始/结束时间建立一个分析窗口掩码。

    输入：完整时间轴、窗口开始时间、窗口结束时间。
    输出：与 time_s 一样长的布尔数组，True 表示该点属于窗口。

    实验作用：后续 baseline、motion、post 都用这种掩码从同一条位移曲线中切出不同阶段。
    缺少某端事件时自动延伸到数据首尾，让纯 vision_test 也能被分析。
    """

    # 本段补齐缺失边界。没有开始事件就从第一帧开始，没有结束事件就到最后一帧。
    start = time_s[0] if start_s is None else start_s
    end = time_s[-1] if end_s is None else end_s

    # 本段一次性比较整个时间数组，得到窗口内/外标记。
    return (time_s >= start) & (time_s <= end)


def extract_robot_speed_series(
    robot_records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    从 ROBOT 记录中提取 TCP 线速度大小。

    输入：robot_worker 写入的 ROBOT 状态记录。
    输出：时间数组和 TCP 线速度模长，单位 m/s；没有足够速度记录时返回 None。

    实验作用：用于自动识别机器人是否真的在运动。它比只看“有没有发过运动命令”更适合
    反复启停、加减速、路径转弯或机器人还没真正开始动的情况。
    """

    time_values: list[float] = []
    speed_values: list[float] = []

    for record in robot_records:
        host_ns = _finite_float(record.get("host_ns"))
        tcp_speed = record.get("actual_tcp_speed")
        if not math.isfinite(host_ns) or not isinstance(tcp_speed, list) or len(tcp_speed) < 3:
            continue

        linear_speed = [_finite_float(value) for value in tcp_speed[:3]]
        if all(math.isfinite(value) for value in linear_speed):
            time_values.append(host_ns * 1e-9)
            speed_values.append(float(np.linalg.norm(linear_speed)))

    if len(time_values) < 2:
        return None

    time_array = np.asarray(time_values, dtype=float)
    speed_array = np.asarray(speed_values, dtype=float)
    order = np.argsort(time_array)
    sorted_time = time_array[order]
    sorted_speed = speed_array[order]
    unique_time, unique_indices = np.unique(sorted_time, return_index=True)
    return unique_time, sorted_speed[unique_indices]


def _merge_intervals(
    intervals: list[tuple[float, float]],
    max_gap_s: float,
) -> list[tuple[float, float]]:
    """
    合并间隔很小的同类时间段。

    输入：若干起止时间，以及允许忽略的短暂停顿时间。
    输出：合并后的起止时间。

    实验作用：机器人速度在零附近短暂抖一下，不应把一次连续运动切成很多碎片。
    """

    if not intervals:
        return []

    intervals = sorted(intervals)
    merged: list[tuple[float, float]] = [intervals[0]]
    for start_s, end_s in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start_s - previous_end <= max_gap_s:
            merged[-1] = (previous_start, max(previous_end, end_s))
        else:
            merged.append((start_s, end_s))
    return merged


def _motion_intervals_from_robot_speed(
    robot_records: list[dict[str, Any]],
    data_start_s: float,
    data_end_s: float,
) -> list[tuple[float, float]]:
    """
    用机器人 TCP 速度把全程切出运动段。

    输入：ROBOT 速度记录和视觉数据覆盖的时间范围。
    输出：若干 motion 起止时间。

    实验作用：这一步回答“机器人实际什么时候在动”。反复加速、减速、停止时，
    它会按真实速度切出多段 motion，而不是只相信第一条 motion_command_sent 和 motion_finished。
    """

    speed_series = extract_robot_speed_series(robot_records)
    if speed_series is None:
        return []

    robot_time, robot_speed = speed_series
    on_threshold = float(config.ROBOT_MOTION_ON_SPEED_M_S)
    off_threshold = float(config.ROBOT_MOTION_OFF_SPEED_M_S)

    intervals: list[tuple[float, float]] = []
    in_motion = False
    start_s = data_start_s

    for current_time, current_speed in zip(robot_time, robot_speed):
        if current_time < data_start_s or current_time > data_end_s:
            continue

        if not in_motion and current_speed >= on_threshold:
            start_s = float(current_time)
            in_motion = True
        elif in_motion and current_speed <= off_threshold:
            end_s = float(current_time)
            if end_s - start_s >= float(config.SEGMENT_MIN_MOTION_SECONDS):
                intervals.append((start_s, end_s))
            in_motion = False

    if in_motion:
        end_s = data_end_s
        if end_s - start_s >= float(config.SEGMENT_MIN_MOTION_SECONDS):
            intervals.append((start_s, end_s))

    return _merge_intervals(intervals, float(config.SEGMENT_MERGE_GAP_SECONDS))


def _motion_intervals_from_events(
    event_sequence: dict[str, list[float]],
    data_start_s: float,
    data_end_s: float,
) -> list[tuple[float, float]]:
    """
    用运动命令事件把全程切出运动段。

    输入：所有 motion_command_sent 和 motion_finished 时间。
    输出：按顺序配对后的 motion 起止时间。

    实验作用：当没有机器人速度记录时，事件仍然能提供“程序计划中的运动区间”。
    它不判断机器人速度是否稳定，只负责把命令发出到完成之间视作运动段。
    """

    starts = event_sequence.get("motion_command_sent", [])
    ends = event_sequence.get("motion_finished", [])
    intervals: list[tuple[float, float]] = []
    end_index = 0

    for start_s in starts:
        while end_index < len(ends) and ends[end_index] <= start_s:
            end_index += 1
        if end_index >= len(ends):
            break

        end_s = ends[end_index]
        end_index += 1
        start_s = max(float(start_s), data_start_s)
        end_s = min(float(end_s), data_end_s)
        if end_s - start_s >= float(config.SEGMENT_MIN_MOTION_SECONDS):
            intervals.append((start_s, end_s))

    return _merge_intervals(intervals, float(config.SEGMENT_MERGE_GAP_SECONDS))


def _motion_intervals_from_vision_velocity(
    time_s: np.ndarray,
    values: np.ndarray | None,
) -> list[tuple[float, float]]:
    """
    用视觉位移变化速度粗略切出运动段。

    输入：视觉时间和位移曲线。
    输出：基于位移速度异常升高得到的候选 motion 段。

    实验作用：这是没有 ROBOT、没有 EVENT 时的兜底方法。它只能用于离线辅助复查，
    因为真实振动也会改变视觉位移速度，不能像机器人速度那样明确代表运动指令。
    """

    if values is None or len(time_s) < 8 or len(values) != len(time_s):
        return []

    delta_t = np.diff(time_s)
    delta_v = np.diff(values)
    valid = delta_t > 0
    if np.count_nonzero(valid) < 4:
        return []

    velocity_time = (time_s[:-1] + time_s[1:]) * 0.5
    velocity = np.zeros_like(delta_v, dtype=float)
    velocity[valid] = np.abs(delta_v[valid] / delta_t[valid])

    baseline = float(np.nanmedian(velocity[valid]))
    spread = float(np.nanmedian(np.abs(velocity[valid] - baseline)))
    threshold = baseline + float(config.VISION_MOTION_VELOCITY_FACTOR) * max(spread, 1e-9)
    moving = velocity > threshold

    intervals: list[tuple[float, float]] = []
    start_s: float | None = None
    for current_time, is_moving in zip(velocity_time, moving):
        if is_moving and start_s is None:
            start_s = float(current_time)
        elif not is_moving and start_s is not None:
            end_s = float(current_time)
            if end_s - start_s >= float(config.SEGMENT_MIN_MOTION_SECONDS):
                intervals.append((start_s, end_s))
            start_s = None

    if start_s is not None:
        end_s = float(time_s[-1])
        if end_s - start_s >= float(config.SEGMENT_MIN_MOTION_SECONDS):
            intervals.append((start_s, end_s))

    return _merge_intervals(intervals, float(config.SEGMENT_MERGE_GAP_SECONDS))


def _steady_interval_from_robot_speed(
    robot_records: list[dict[str, Any]],
    motion_start_s: float,
    motion_end_s: float,
) -> tuple[float, float] | None:
    """
    在一个运动段内部，用 TCP 速度寻找更像匀速的子段。

    输入：ROBOT 速度记录和一个 motion 起止时间。
    输出：steady_motion 起止时间；无法可靠判断时返回 None。

    实验作用：运动段两端常包含加减速，抖动分析时可能想单独看中间匀速部分。
    这里用“速度接近中位速度、加速度较小”判断，而不是固定裁掉一段时间。
    """

    speed_series = extract_robot_speed_series(robot_records)
    if speed_series is None:
        return None

    robot_time, robot_speed = speed_series
    mask = (robot_time >= motion_start_s) & (robot_time <= motion_end_s)
    if np.count_nonzero(mask) < 5:
        return None

    segment_time = robot_time[mask]
    segment_speed = robot_speed[mask]
    valid_speed = segment_speed[segment_speed > float(config.ROBOT_MOTION_OFF_SPEED_M_S)]
    if len(valid_speed) < 5:
        return None

    target_speed = float(np.median(valid_speed))
    speed_band = max(
        target_speed * float(config.ROBOT_STEADY_SPEED_RELATIVE_TOLERANCE),
        float(config.ROBOT_MOTION_ON_SPEED_M_S),
    )

    acceleration = np.gradient(segment_speed, segment_time)
    steady_mask = (
        (segment_speed > float(config.ROBOT_MOTION_OFF_SPEED_M_S))
        & (np.abs(segment_speed - target_speed) <= speed_band)
        & (np.abs(acceleration) <= float(config.ROBOT_STEADY_ACCELERATION_M_S2))
    )

    if np.count_nonzero(steady_mask) < 3:
        return None

    steady_times = segment_time[steady_mask]
    steady_start = float(steady_times[0])
    steady_end = float(steady_times[-1])
    if steady_end - steady_start < float(config.SEGMENT_MIN_MOTION_SECONDS):
        return None
    return steady_start, steady_end


def _trimmed_steady_interval(
    motion_start_s: float,
    motion_end_s: float,
) -> tuple[float, float]:
    """
    在没有可靠速度匀速判断时，按比例裁掉运动段两端。

    输入：motion 起止时间。
    输出：粗略 steady_motion 起止时间。

    实验作用：这是旧逻辑的保守兜底。它不能真正识别匀速，只是假设中间段比两端更接近匀速。
    """

    duration = max(0.0, motion_end_s - motion_start_s)
    trim = float(config.STEADY_MOTION_TRIM_FRACTION) * duration
    steady_start = motion_start_s + trim
    steady_end = motion_end_s - trim
    if steady_start >= steady_end:
        return motion_start_s, motion_end_s
    return steady_start, steady_end


def _build_segments_from_motion_intervals(
    motion_intervals: list[tuple[float, float]],
    data_start_s: float,
    data_end_s: float,
    source: str,
    robot_records: list[dict[str, Any]] | None,
) -> list[AnalysisSegment]:
    """
    把 motion 起止时间扩展成静止段、运动段和匀速段。

    输入：已经识别出的 motion 区间，以及全程数据边界。
    输出：AnalysisSegment 列表。

    实验作用：它把“机器人动过哪些时间”整理成分析真正需要的窗口：
    每段运动单独算、所有运动合并算、运动之间的静止段也保留下来供对照。
    """

    segments: list[AnalysisSegment] = []
    cursor = data_start_s
    static_index = 1
    motion_index = 1

    for motion_start_s, motion_end_s in sorted(motion_intervals):
        motion_start_s = max(motion_start_s, data_start_s)
        motion_end_s = min(motion_end_s, data_end_s)
        if motion_end_s <= motion_start_s:
            continue

        if motion_start_s - cursor >= float(config.SEGMENT_MIN_STATIC_SECONDS):
            segments.append(
                AnalysisSegment(
                    name=f"static_{static_index:03d}",
                    kind="static",
                    start_s=cursor,
                    end_s=motion_start_s,
                    source=source,
                )
            )
            static_index += 1

        segments.append(
            AnalysisSegment(
                name=f"motion_{motion_index:03d}",
                kind="motion",
                start_s=motion_start_s,
                end_s=motion_end_s,
                source=source,
            )
        )

        steady_interval = None
        if robot_records:
            steady_interval = _steady_interval_from_robot_speed(
                robot_records,
                motion_start_s,
                motion_end_s,
            )
        if steady_interval is None:
            steady_interval = _trimmed_steady_interval(motion_start_s, motion_end_s)

        steady_start_s, steady_end_s = steady_interval
        if steady_end_s > steady_start_s:
            segments.append(
                AnalysisSegment(
                    name=f"steady_motion_{motion_index:03d}",
                    kind="steady_motion",
                    start_s=steady_start_s,
                    end_s=steady_end_s,
                    source=source,
                )
            )

        cursor = max(cursor, motion_end_s)
        motion_index += 1

    if data_end_s - cursor >= float(config.SEGMENT_MIN_STATIC_SECONDS):
        segments.append(
            AnalysisSegment(
                name=f"static_{static_index:03d}",
                kind="static",
                start_s=cursor,
                end_s=data_end_s,
                source=source,
            )
        )

    return segments


def select_analysis_windows(
    time_s: np.ndarray,
    events: dict[str, float],
    *,
    event_sequence: dict[str, list[float]] | None = None,
    robot_records: list[dict[str, Any]] | None = None,
    vision_values: np.ndarray | None = None,
    return_segments: bool = False,
) -> dict[str, np.ndarray]:
    """
    把完整视觉时间序列拆成实验分析窗口。

    输入：
    - time_s：每帧视觉测量结果的时间；
    - events：每种事件第一次出现的时间，保留旧版单段分析兼容；
    - event_sequence：每种事件的完整时间列表，用于多段运动命令配对；
    - robot_records：机器人状态记录，用于按真实 TCP 速度分段；
    - vision_values：视觉位移曲线，只在没有机器人速度和事件时兜底。

    输出：
    - full：整段有效视觉记录；
    - motion_001/static_001/steady_motion_001：自动切出的单个实验片段；
    - motion_all/static_all/steady_motion_all：同类片段合并窗口；
    - baseline/motion/steady_motion/post：保留旧名字，方便已有配置继续使用。

    实验作用：
    同一条位移曲线在不同工况下要分析不同片段。这里优先用机器人速度识别真实启停；
    没有速度时用事件配对；再没有事件时才用视觉位移变化兜底。
    后续 run_analysis() 会对主窗口、单段窗口和合并窗口分别输出指标。
    """

    # 本段先确定可分析的全程边界。
    # 输入是视觉数据首尾和 experiment_started/experiment_finished；输出是 full 窗口范围。
    data_start_s = float(time_s[0])
    data_end_s = float(time_s[-1])
    experiment_start = events.get("experiment_started", data_start_s)
    experiment_end = events.get("experiment_finished", data_end_s)
    data_start_s = max(data_start_s, experiment_start)
    data_end_s = min(data_end_s, experiment_end)

    event_sequence = event_sequence or {}
    robot_records = robot_records or []
    source_choice = str(config.ANALYSIS_SEGMENTATION_SOURCE)

    # 本段决定用哪一种证据切分运动段。
    # auto 的优先级是：机器人真实速度 -> 运动事件 -> 视觉位移速度兜底。
    motion_intervals: list[tuple[float, float]] = []
    segmentation_source = "none"

    if source_choice in {"auto", "robot_speed"}:
        motion_intervals = _motion_intervals_from_robot_speed(
            robot_records,
            data_start_s,
            data_end_s,
        )
        if motion_intervals:
            segmentation_source = "robot_speed"

    if not motion_intervals and source_choice in {"auto", "events"}:
        motion_intervals = _motion_intervals_from_events(
            event_sequence,
            data_start_s,
            data_end_s,
        )
        if motion_intervals:
            segmentation_source = "events"

    if not motion_intervals and source_choice in {"auto", "vision_velocity"}:
        motion_intervals = _motion_intervals_from_vision_velocity(time_s, vision_values)
        if motion_intervals:
            segmentation_source = "vision_velocity"

    full_mask = _window_mask(time_s, data_start_s, data_end_s)

    # 本段处理完全切不出运动段的离线数据。
    # 输入可能是纯静止视觉测试；输出保持旧版 motion/steady_motion 可用，避免分析流程中断。
    if not motion_intervals:
        windows = {
            "full": full_mask,
            "baseline": np.zeros_like(time_s, dtype=bool),
            "motion": full_mask.copy(),
            "steady_motion": full_mask.copy(),
            "post": np.zeros_like(time_s, dtype=bool),
        }
        segments: list[AnalysisSegment] = [
            AnalysisSegment(
                name="motion_001",
                kind="motion",
                start_s=data_start_s,
                end_s=data_end_s,
                source=segmentation_source,
            )
        ]
    else:
        # 本段把 motion 区间扩展成静止、运动、匀速三类片段，再转成窗口掩码。
        # 输出既有 motion_001 这种单段窗口，也有 motion_all 这种合并窗口。
        segments = _build_segments_from_motion_intervals(
            motion_intervals,
            data_start_s,
            data_end_s,
            segmentation_source,
            robot_records,
        )
        windows = {"full": full_mask}

        for segment in segments:
            windows[segment.name] = _window_mask(time_s, segment.start_s, segment.end_s)

        for kind in ("static", "motion", "steady_motion"):
            masks = [
                windows[segment.name]
                for segment in segments
                if segment.kind == kind
            ]
            windows[f"{kind}_all"] = (
                np.logical_or.reduce(masks)
                if masks
                else np.zeros_like(time_s, dtype=bool)
            )

        # 本段保留旧窗口名。
        # baseline 是第一段运动前静止，post 是最后一段运动后静止；motion/steady_motion 是同类合并。
        static_segments = [segment for segment in segments if segment.kind == "static"]
        motion_segments = [segment for segment in segments if segment.kind == "motion"]
        steady_segments = [
            segment for segment in segments if segment.kind == "steady_motion"
        ]

        first_motion = motion_segments[0] if motion_segments else None
        last_motion = motion_segments[-1] if motion_segments else None

        baseline_segment = next(
            (
                segment
                for segment in static_segments
                if first_motion is not None and segment.end_s <= first_motion.start_s
            ),
            None,
        )
        post_segment = next(
            (
                segment
                for segment in reversed(static_segments)
                if last_motion is not None and segment.start_s >= last_motion.end_s
            ),
            None,
        )

        windows["baseline"] = (
            windows[baseline_segment.name]
            if baseline_segment is not None
            else np.zeros_like(time_s, dtype=bool)
        )
        windows["post"] = (
            windows[post_segment.name]
            if post_segment is not None
            else np.zeros_like(time_s, dtype=bool)
        )
        windows["motion"] = windows.get("motion_all", full_mask.copy())
        windows["steady_motion"] = (
            windows["steady_motion_all"]
            if steady_segments
            else windows["motion"]
        )

    if return_segments:
        return cast(Any, (windows, segments))
    return windows


def select_analysis_segments(
    time_s: np.ndarray,
    events: dict[str, float],
    *,
    event_sequence: dict[str, list[float]] | None = None,
    robot_records: list[dict[str, Any]] | None = None,
    vision_values: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], list[AnalysisSegment]]:
    """
    返回分析窗口和窗口背后的片段说明。

    输入：和 select_analysis_windows() 相同。
    输出：窗口掩码字典，以及 motion/static/steady_motion 片段清单。

    实验作用：run_analysis() 需要窗口来计算，也需要片段说明写进摘要和 JSON，
    这样你能看见程序到底把哪几段判断成运动、静止或匀速。
    """

    return cast(
        tuple[dict[str, np.ndarray], list[AnalysisSegment]],
        select_analysis_windows(
            time_s,
            events,
            event_sequence=event_sequence,
            robot_records=robot_records,
            vision_values=vision_values,
            return_segments=True,
        ),
    )


def extract_robot_tcp_series(
    robot_records: list[dict[str, Any]],
    axis: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    从 ROBOT 记录中提取 UR 控制器反馈的 TCP 位移曲线。

    输入：robot_worker 或 robot_test 写入的 ROBOT 记录，以及分析方向 axis。
    输出：机器人时间序列和 TCP 位置/相对位移序列，单位 mm；没有足够记录时返回 None。

    实验作用：用于画“视觉 vs UR TCP”对齐图，检查两套记录是否在同一时间段变化。
    注意 magnitude 使用相对第一条机器人记录的三维位移长度，而视觉 magnitude 是二维图像平面位移。
    外参未标定前，这个对比只能看相对变化和时间对齐，不能当绝对误差。
    """

    # 本段处理没有机器人记录的情况，例如纯 vision_test。
    if not robot_records:
        return None

    # 本段定义 x/y 单轴对应 TCP pose 的哪一列。
    axis_index = {"x": 0, "y": 1}

    # 本段先用 list 收集有效记录，最后再转 NumPy 数组。
    time_values: list[float] = []
    positions: list[list[float]] = []

    # 本段逐条读取 ROBOT 记录。输入是一条机器人状态；输出可能追加一个有效 TCP 点。
    for record in robot_records:
        # host_ns 是电脑时基，actual_tcp_pose 是 UR 返回的 TCP 位姿。
        host_ns = _finite_float(record.get("host_ns"))
        pose = record.get("actual_tcp_pose")

        # 本段跳过时间无效、位姿不是列表、或位姿长度不足的记录。
        if not math.isfinite(host_ns) or not isinstance(pose, list) or len(pose) < 3:
            continue

        # 本段只取 xyz，并把每一项清洗成有限浮点数。
        xyz = [_finite_float(value) for value in pose[:3]]
        if all(math.isfinite(value) for value in xyz):
            time_values.append(host_ns * 1e-9)
            positions.append(xyz)

    # 本段保护最低数据量。少于两个点无法形成曲线。
    if len(time_values) < 2:
        return None

    # 本段转成 NumPy 数组，便于切片和排序。
    time_array = np.asarray(time_values, dtype=float)
    position_array = np.asarray(positions, dtype=float)

    # 本段把机器人坐标从 m 换成 mm，并按用户选择得到一条标量曲线。
    if axis in axis_index:
        values_mm = position_array[:, axis_index[axis]] * 1000.0
    else:
        # magnitude 用相对第一条记录的三维距离。
        relative = position_array - position_array[0]
        values_mm = np.linalg.norm(relative, axis=1) * 1000.0

    # 本段按时间排序，返回时间和位移。
    order = np.argsort(time_array)
    return time_array[order], values_mm[order]


# =============================================================================
# 3. 不等间隔数据重采样和去趋势：把原始位移变成可做频谱的振动残差
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
    """
    把 Savitzky-Golay 的秒制窗口换成滤波器需要的样本点数。

    输入：实际采样率和当前数据点数。
    输出：不超过数据长度、且为奇数的窗口点数。

    实验作用：config.py 中用秒描述窗口更符合实验直觉；
    scipy 滤波函数需要样本点数，所以这里根据实际帧率完成转换。
    """

    # 本段先按实际采样率把秒数转换成点数，并保证至少比多项式阶数大。
    desired = int(round(config.SAVGOL_WINDOW_SECONDS * sample_rate_hz))
    desired = max(desired, config.SAVGOL_POLYORDER + 3)
    if desired % 2 == 0:
        desired += 1

    # 本段保证窗口不超过当前数据长度，且仍然是奇数。
    maximum = sample_count if sample_count % 2 == 1 else sample_count - 1
    return min(desired, maximum)


def detrend_motion(
    values: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从位移序列中分离“慢趋势”和“振动残差”。

    输入：
    - values：某个分析窗口内的原始位移，单位 mm；
    - sample_rate_hz：由真实时间戳估计出的采样率。

    输出：
    - trend：机器人正常运动或慢漂移形成的趋势；
    - residual：values - trend，也就是后续 RMS 和频谱使用的振动残差。

    实验作用：如果机器人从 A 点移动到 B 点，视觉位移里会有一个大的整体移动。
    我们真正关心的是叠在整体移动上的小抖动，所以先把慢变化趋势剥离。
    去趋势方法会直接影响低频结论，因此摘要中会明确记录所用方法和参数。
    """

    values = np.asarray(values, dtype=float)

    if config.DETREND_METHOD == "linear":
        # 本段使用线性去趋势。
        # 输入是原始位移；输出是一条直线趋势和围绕直线的残差。
        # 实验作用：适合匀速直线段粗分析，但不适合明显弯曲或分段静止-运动-静止的全程曲线。
        residual = np.asarray(signal.detrend(values, type="linear"), dtype=float)
        trend = values - residual
        return trend, residual

    if config.DETREND_METHOD == "savgol":
        # 本段使用 Savitzky-Golay 平滑趋势。
        # 输入是原始位移和窗口长度；输出是一条可缓慢弯曲的趋势线。
        # 实验作用：比直线更能跟随缓慢路径变化，但窗口太短会把真实振动也当趋势扣掉。
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
        # 本段使用高通滤波直接保留高于截止频率的成分。
        # 输入是原始位移和截止频率；输出是高通后的 residual。
        # 实验作用：适合明确知道低频都不是关注目标时使用；否则可能滤掉真实低频振动。
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
    对完整实验的不同阶段分别去趋势，再拼回全程残差。

    输入：
    - 全程均匀时间轴和位移；
    - 采样率；
    - motion_command_sent / motion_finished 等事件。

    输出：
    - 全程 trend；
    - 全程 residual。

    实验作用：完整实验常见形态是“运动前静止 -> 运动中斜坡/曲线 -> 运动后静止”。
    若整段只减一条直线，两段静止会被误当成巨大低频变化，尤其会破坏恢复时间判断。
    """

    # 本段读取运动开始和结束事件。没有事件时无法分段，只能退回普通去趋势。
    motion_start = events.get("motion_command_sent")
    motion_end = events.get("motion_finished")
    if motion_start is None or motion_end is None:
        return detrend_motion(values, sample_rate_hz)

    # 本段把全程分成运动前、运动中、运动后三段。
    masks = (
        time_s < motion_start,
        (time_s >= motion_start) & (time_s <= motion_end),
        time_s > motion_end,
    )
    # 本段预分配输出数组。每段去趋势后再填回对应位置。
    trend = np.full_like(values, np.nan, dtype=float)
    residual = np.full_like(values, np.nan, dtype=float)

    for mask in masks:
        # 本段逐段处理。长段使用配置的去趋势方法，极短段用均值作为保守趋势。
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

    # 本段补齐可能遗漏的点。
    # 如果事件时间正好落在浮点间隙导致某些点没被任一段覆盖，就使用原值作为趋势并把残差设零。
    missing = ~np.isfinite(residual)
    trend[missing] = values[missing]
    residual[missing] = 0.0
    return trend, residual


# =============================================================================
# 4. 时域、频域和恢复时间指标：把振动残差转成可比较的数字结论
# =============================================================================

def calculate_spectrum(
    residual: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, Any]:
    """
    从振动残差中计算频域结果。

    输入：
    - residual：去趋势后的振动残差，单位 mm；
    - sample_rate_hz：主分析窗口的估计采样率。

    输出：
    - FFT 单边幅值谱；
    - Welch 功率谱密度；
    - 主频、主频 PSD、频带能量和频率分辨率。

    实验作用：时域图回答“什么时候抖得大”，频谱回答“主要以多少 Hz 在抖”。
    主频与频带能量使用更稳定的 Welch PSD；FFT amplitude 主要用于直观看正弦幅值。
    """

    # 本段去掉平均值，让频谱不要被直流偏置占据。
    centered = np.asarray(residual, dtype=float) - float(np.mean(residual))

    # 本段计算单边 FFT 幅值谱。
    # 加窗可以减少有限长度数据在 FFT 中产生的边缘泄漏。
    window = np.hanning(len(centered))
    fft_values = np.fft.rfft(centered * window)
    fft_frequency = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    amplitude = np.abs(fft_values) * (2.0 / max(float(np.sum(window)), 1e-12))
    if len(amplitude):
        amplitude[0] *= 0.5

    # 本段计算 Welch PSD。
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

    # 本段在用户关心的频率范围内寻找主频。
    # 上限不能超过奈奎斯特频率，否则该频段没有物理意义。
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

    # 本段准备频带能量积分函数。
    # NumPy 2.x 提供 trapezoid；旧版可退回 trapz。
    # 不能把 np.trapz 直接写成 getattr 默认值，因为 Python 会先求值默认参数。
    integrate = cast(Any, getattr(np, "trapezoid", getattr(np, "trapz", None)))
    if integrate is None:
        raise RuntimeError("当前 NumPy 版本缺少 trapezoid/trapz 积分函数。")
    # 本段按 config.FREQUENCY_BANDS_HZ 汇总各频带能量，便于后续跨工况比较。
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
    计算振动残差的时域强度指标。

    输入：去趋势后的 residual，单位 mm。
    输出：峰峰值、RMS、标准差、最大绝对残差和均值。

    实验作用：这些指标描述振动强弱，不描述机器人从 A 到 B 的整体运动距离。
    其中 RMS 常用于比较不同工况下的振动能量，峰峰值更容易受偶发尖峰影响。
    """

    # 本段确保输入是浮点数组，避免整型数组参与平方时出现不必要的类型问题。
    residual = np.asarray(residual, dtype=float)

    # 本段输出一组常用时域指标。
    # ptp 是 peak-to-peak，即最大值减最小值；RMS 是均方根。
    return {
        "peak_to_peak_mm": float(np.ptp(residual)),
        "rms_mm": float(np.sqrt(np.mean(residual**2))),
        "std_mm": float(np.std(residual)),
        "max_abs_mm": float(np.max(np.abs(residual))),
        "mean_mm": float(np.mean(residual)),
    }


def calculate_window_metric_summary(
    time_s: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """
    对某一个分析窗口单独计算简版振动指标。

    输入：完整视觉时间、完整位移和一个窗口掩码。
    输出：该窗口的点数、时长、RMS、峰峰值、主频和频带能量；点数不足时说明 unavailable。

    实验作用：run_analysis() 的主指标只对应一个主窗口；本函数让 motion_001、motion_002、
    steady_motion_all、static_all 等窗口也能各自输出结果，方便比较不同启停段是否表现一致。
    """

    sample_count = int(np.count_nonzero(mask))
    if sample_count < 8:
        return {
            "available": False,
            "sample_count": sample_count,
            "reason": "窗口有效点少于 8，无法稳定计算频谱和 RMS。",
        }

    window_time = time_s[mask]
    window_values = values[mask]
    duration_s = float(window_time[-1] - window_time[0]) if len(window_time) else 0.0

    try:
        uniform_time, uniform_values, sample_rate_hz, timing = resample_uniform(
            window_time,
            window_values,
        )
        _, residual = detrend_motion(uniform_values, sample_rate_hz)
        time_metrics = calculate_time_metrics(residual)
        spectrum = calculate_spectrum(residual, sample_rate_hz)
    except ValueError as exc:
        return {
            "available": False,
            "sample_count": sample_count,
            "duration_s": duration_s,
            "reason": str(exc),
        }

    return {
        "available": True,
        "sample_count": sample_count,
        "duration_s": duration_s,
        "timing": timing,
        "time_metrics": time_metrics,
        "dominant_frequency_hz": spectrum["dominant_frequency_hz"],
        "frequency_resolution_hz": spectrum["frequency_resolution_hz"],
        "band_energy_mm2": spectrum["band_energy_mm2"],
        "uniform_sample_count": int(len(uniform_time)),
    }


def calculate_recovery_time(
    uniform_time_s: np.ndarray,
    residual: np.ndarray,
    events: dict[str, float],
    baseline_mask: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, float | None]:
    """
    估计运动结束后振动恢复到阈值以内需要多久。

    输入：
    - full_uniform_time_s：全程均匀时间轴；
    - residual：全程去趋势残差；
    - events：运动完成等事件时间；
    - baseline_mask：运动前静止区间；
    - sample_rate_hz：全程序列采样率。

    输出：恢复时间、恢复阈值和基线 RMS；缺少必要条件时恢复时间为 None。

    实验作用：从 motion_finished 开始计时，直到振动包络连续一段时间低于阈值。
    这样可以避免某一个瞬间刚好低于阈值就误判为已经恢复。
    没有 motion_finished 事件或记录尾部太短时返回 None，不用最后一帧冒充恢复。
    """

    # 本段处理没有运动完成事件的情况。没有起点就不能定义“恢复用了多久”。
    motion_finished = events.get("motion_finished")
    if motion_finished is None:
        return {
            "recovery_time_s": None,
            "recovery_threshold_mm": None,
            "baseline_rms_mm": None,
        }

    # 本段估计静止基线噪声 RMS。
    # 输入是运动前静止区间；输出 baseline_rms，用于动态调整恢复阈值。
    if np.count_nonzero(baseline_mask) >= 4:
        baseline_rms = float(
            np.sqrt(np.mean(np.asarray(residual)[baseline_mask] ** 2))
        )
    else:
        baseline_rms = 0.0

    # 本段计算恢复阈值。
    # 取绝对阈值和基线 RMS 若干倍中的较大值，既不过分相信极小噪声，也能适应噪声较大的实验。
    threshold = max(
        config.RECOVERY_ABSOLUTE_THRESHOLD_MM,
        config.RECOVERY_BASELINE_FACTOR * baseline_rms,
    )
    # 本段找到 motion_finished 在均匀时间轴中的位置，后续只看运动结束后的残差。
    start_index = int(np.searchsorted(uniform_time_s, motion_finished, side="left"))

    # 本段处理后记录太短的情况。数据不足时不硬给恢复时间。
    if len(uniform_time_s) - start_index < 4:
        return {
            "recovery_time_s": None,
            "recovery_threshold_mm": float(threshold),
            "baseline_rms_mm": float(baseline_rms),
        }

    # 本段只对运动后的残差求包络。
    # 这样避免运动段的大振幅通过 Hilbert 非局部边缘效应污染恢复起点。
    post_residual = np.asarray(residual[start_index:], dtype=float)

    # Hilbert 包络近似表示振动振幅随时间的变化。
    envelope = np.abs(signal.hilbert(post_residual))

    # 本段把“连续低于阈值的秒数”换成样本数。
    hold_samples = max(1, int(math.ceil(config.RECOVERY_HOLD_SECONDS * sample_rate_hz)))

    # 本段寻找第一个连续低于阈值的窗口。
    # 输出 recovery_time 是相对 motion_finished 的秒数，而不是绝对时间戳。
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
    比较圆点法和棋盘格法给出的位移曲线是否一致。

    输入：同一份 VISION 记录和分析方向 axis。
    输出：共同样本数、两方法 RMSE、两方法相关系数。

    实验作用：compare 模式下，两种视觉方法会同时保存结果。
    这里在共同时间区间插值比较，不要求同一帧两种方法都识别成功。
    RMSE 小表示数值接近，相关系数高表示波形形状一致；两者不能单独证明绝对精度。
    """

    # 本段尝试分别提取圆点法和棋盘格法的有效序列。
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

    # 本段寻找两种方法共同覆盖的时间区间。
    # 实验作用：避免一条曲线在另一条没有数据的区域被外推比较。
    common_start = max(circle_time[0], checker_time[0])
    common_end = min(circle_time[-1], checker_time[-1])

    # 本段使用圆点法的时间点作为公共时间轴。
    mask = (circle_time >= common_start) & (circle_time <= common_end)
    common_time = circle_time[mask]

    # 本段保护最低共同样本量。点太少时 RMSE 和相关系数都没有参考价值。
    if len(common_time) < 3:
        return {
            "common_sample_count": int(len(common_time)),
            "circle_checker_rmse_mm": None,
            "circle_checker_correlation": None,
        }

    # 本段把棋盘格法插值到圆点法时间点上，才能逐点相减。
    checker_interpolated = np.interp(common_time, checker_time, checker_value)
    circle_common = circle_value[mask]

    # 本段计算 RMSE，表示两条曲线在数值上平均差多少毫米。
    difference = circle_common - checker_interpolated
    rmse = float(np.sqrt(np.mean(difference**2)))

    # 本段计算相关系数，表示两条曲线形状是否同步。
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
# 5. 绘图：把分析结果变成人能快速判断的图片
# =============================================================================

def _relative_time(time_s: np.ndarray, origin_s: float) -> np.ndarray:
    """
    把绝对秒数转换成图上更容易读的相对时间。

    输入：原始时间数组和原点时间。
    输出：time_s - origin_s。

    实验作用：host_ns/perf_counter 秒数本身没有直观意义。
    图上统一显示“从本次记录开始过了几秒”，更方便对照运动开始和结束事件。
    """

    # 本段保证输入可以是 list 或数组，输出统一为 NumPy 数组。
    return np.asarray(time_s, dtype=float) - origin_s


def _mark_events(
    axes: Axes,
    events: dict[str, float],
    origin_s: float,
) -> None:
    """
    在时域图上标出关键实验事件。

    输入：Matplotlib 坐标轴、事件时间字典、图上时间原点。
    输出：运动命令发出和运动完成的竖虚线。

    实验作用：帮助判断振动峰值发生在运动前、运动中还是运动后。
    这只是画图辅助，不参与任何数值计算。
    """

    # 本段给不同事件固定颜色，便于多张图之间保持一致。
    colors = {
        "motion_command_sent": "tab:red",
        "motion_finished": "tab:green",
    }
    # 本段只标注日志中真实存在的事件。
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
    保存时域总览图。

    输入：
    - original_time_s/original_values：原始有效视觉位移；
    - uniform_time_s/trend/residual：重采样和去趋势后的结果；
    - events：运动命令和运动完成等事件；
    - method/axis：图标题中记录的分析配置。

    输出：01_time_domain.png。

    实验作用：这是分析结果里通常最先看的图。
    上图检查整体位移趋势是否合理；下图检查去掉整体运动后剩余振动是否明显；
    竖虚线帮助判断振动发生在哪个实验阶段。
    """

    # 本段把本次记录第一帧作为图上 0 秒。
    origin = float(original_time_s[0])

    # 本段创建上下两个子图，共享同一个 x 轴，方便直接对齐观察。
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # 本段画原始视觉位移。它包含机器人整体移动趋势，也包含叠加在上面的微小振动。
    axes[0].plot(
        _relative_time(original_time_s, origin),
        original_values,
        linewidth=0.9,
        label="measured displacement",
    )

    # 本段画估计出的慢趋势。原始位移减去这条趋势后，才得到下图中的 residual。
    axes[0].plot(
        _relative_time(uniform_time_s, origin),
        trend,
        linewidth=1.5,
        label=f"{config.DETREND_METHOD} trend",
    )

    # 本段设置上图坐标轴、网格和事件标记，让图不只是曲线，还能读出实验阶段。
    axes[0].set_ylabel("Displacement / mm")
    axes[0].grid(True, alpha=0.3)
    _mark_events(axes[0], events, origin)
    axes[0].legend(loc="best")

    # 本段画 residual，也就是“去掉慢趋势后的剩余部分”。
    # 这部分才是后续 RMS、峰峰值和频谱分析关注的对象。
    axes[1].plot(
        _relative_time(uniform_time_s, origin),
        residual,
        color="tab:orange",
        linewidth=0.9,
    )

    # 本段画零线，用于判断残差是在 0 附近振动，还是仍存在明显偏移。
    axes[1].axhline(0.0, color="black", linewidth=0.7)

    # 本段设置下图坐标轴、网格和同样的事件竖线。
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Vibration residual / mm")
    axes[1].grid(True, alpha=0.3)
    _mark_events(axes[1], events, origin)

    # 本段在总标题记录视觉方法和分析方向，避免以后打开图片时忘记配置。
    figure.suptitle(f"Vision displacement and vibration — {method}, axis={axis}")

    # 本段调整布局并保存 PNG。
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_spectrum(
    output_path: Path,
    spectrum: dict[str, Any],
    sample_rate_hz: float,
) -> None:
    """
    保存频域图。

    输入：calculate_spectrum() 的输出和采样率。
    输出：02_spectrum.png。

    实验作用：上半部分 FFT 幅值直观显示频率成分，下半部分 Welch PSD 更稳定，
    适合在噪声中判断主要频率和频带能量。
    """

    # 本段确定频率横轴上限。图上限不能超过奈奎斯特频率，否则没有物理意义。
    nyquist = 0.5 * sample_rate_hz
    upper = min(config.DOMINANT_FREQ_MAX_HZ, nyquist)

    # 本段创建两个上下排列的频域子图，并共用频率横轴。
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # 本段画 FFT 幅值谱，用于快速看出哪些频率分量比较强。
    axes[0].plot(
        spectrum["fft_frequency_hz"],
        spectrum["fft_amplitude_mm"],
        linewidth=1.0,
    )

    # 本段限制频率范围，避免无关频段压缩主要区域。
    axes[0].set_ylabel("FFT amplitude / mm")
    axes[0].set_xlim(0.0, upper)
    axes[0].grid(True, alpha=0.3)

    # 本段画 Welch PSD。semilogy 使用对数 y 轴，更容易同时观察强信号和弱信号。
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

    # 本段把自动搜索出的主频写在标题里，便于快速查看。
    dominant = spectrum["dominant_frequency_hz"]
    figure.suptitle(f"Vibration spectrum — dominant {dominant:.3f} Hz")

    # 本段保存图片并释放 Matplotlib 图对象。
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_method_comparison(
    output_path: Path,
    records: list[dict[str, Any]],
    axis: str,
) -> bool:
    """
    保存圆点法和棋盘格法的位移曲线对比图。

    输入：VISION 记录和分析方向。
    输出：03_method_comparison.png；无法生成时返回 False。

    实验作用：检查两套视觉算法是否给出相近趋势。
    如果某一种方法有效点太少，就不生成图，避免一张看似成功但没有信息量的图片误导判断。
    """

    # 本段分别提取圆点法和棋盘格法的有效位移序列。
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

    # 本段让两条曲线共用横轴原点，便于直接比较波形出现时间。
    origin = min(circle_time[0], checker_time[0])

    # 本段创建单幅图，因为这里比较的是两种方法测到的同一类位移。
    figure, axes = plt.subplots(figsize=(11, 4.8))

    # 本段画圆点阵列跟踪结果。
    axes.plot(
        _relative_time(circle_time, origin),
        circle_value,
        linewidth=0.9,
        label="circles",
    )

    # 本段画棋盘格角点跟踪结果。alpha 稍低一点，重叠时仍能看清两条曲线。
    axes.plot(
        _relative_time(checker_time, origin),
        checker_value,
        linewidth=0.9,
        label="checkerboard",
        alpha=0.8,
    )

    # 本段设置坐标、标题、网格和图例，让图片脱离上下文也能读懂。
    axes.set_xlabel("Time / s")
    axes.set_ylabel("Displacement / mm")
    axes.set_title(f"Circle vs checkerboard — axis={axis}")
    axes.grid(True, alpha=0.3)
    axes.legend()

    # 本段保存图片并返回 True，告诉调用者“这张图确实生成了”。
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
    保存视觉位移与 UR TCP 位移的时间对齐图。

    输入：
    - 视觉时间和视觉位移；
    - robot_series：UR TCP 时间和位移；
    - axis：分析方向。

    输出：04_vision_robot_alignment.png；无法生成时返回 False。

    实验作用：用共同 host_ns 检查视觉和机器人记录在时间上是否大致对齐。
    两条曲线都减去各自第一点，只比较相对变化；外参未标定前不能把它当作绝对误差。
    """

    # 本段处理没有 ROBOT 记录的情况，例如纯 vision_test。
    if robot_series is None:
        return False

    # 本段解包机器人时间序列，并寻找视觉与机器人共同覆盖的时间段。
    robot_time_s, robot_values = robot_series
    common_start = max(vision_time_s[0], robot_time_s[0])
    common_end = min(vision_time_s[-1], robot_time_s[-1])

    # 本段处理没有重叠时间段的情况，此时无法在同一横轴上比较。
    if common_end <= common_start:
        return False

    # 本段把图上的 0 秒设为共同时间段起点。
    origin = common_start

    # 本段让视觉和机器人各自减去第一点，只比较“变化量”。
    # 这样可以避开未标定外参造成的绝对坐标零点差异。
    vision_relative = vision_values - vision_values[0]
    robot_relative = robot_values - robot_values[0]

    # 本段创建单幅图，把视觉和 UR TCP 相对位移画在一起。
    figure, axes = plt.subplots(figsize=(11, 4.8))

    # 本段画视觉算法估计的相对位移。
    axes.plot(
        _relative_time(vision_time_s, origin),
        vision_relative,
        linewidth=0.9,
        label="vision relative",
    )

    # 本段画 UR 控制器反馈的 TCP 相对位移。
    axes.plot(
        _relative_time(robot_time_s, origin),
        robot_relative,
        linewidth=0.9,
        label="UR TCP relative",
        alpha=0.8,
    )

    # 本段只显示两者共同存在的时间范围，避免曲线外推造成误解。
    axes.set_xlim(0.0, common_end - common_start)

    # 本段设置坐标轴、标题、网格和图例。
    axes.set_xlabel("Time / s")
    axes.set_ylabel("Relative displacement / mm")
    axes.set_title(f"Host-clock alignment — axis={axis}")
    axes.grid(True, alpha=0.3)
    axes.legend()

    # 本段保存图片并返回 True，表示图已成功生成。
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
    segments: list[AnalysisSegment] | None = None,
    window_metric_summaries: dict[str, dict[str, Any]] | None = None,
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
        ]
    )

    lines.extend(["", "七、自动分段", f"分段来源：{segments[0].source if segments else '无'}"])
    if segments:
        for segment in segments:
            lines.append(
                f"{segment.name} [{segment.kind}]："
                f"{segment.start_s:.6f} s -> {segment.end_s:.6f} s，"
                f"时长 {segment.end_s - segment.start_s:.6f} s"
            )
    else:
        lines.append("无")

    lines.extend(["", "八、分段指标"])
    if window_metric_summaries:
        for name in sorted(window_metric_summaries):
            summary = window_metric_summaries[name]
            if not bool(summary.get("available")):
                lines.append(
                    f"{name}：不可用，点数 {summary.get('sample_count', 0)}，"
                    f"原因：{summary.get('reason', '未说明')}"
                )
                continue

            metrics = cast(dict[str, float], summary["time_metrics"])
            lines.append(
                f"{name}：点数 {summary['sample_count']}，"
                f"峰峰值 {metrics['peak_to_peak_mm']:.6f} mm，"
                f"RMS {metrics['rms_mm']:.6f} mm，"
                f"主频 {_format_optional(summary['dominant_frequency_hz'], 6)} Hz"
            )
    else:
        lines.append("无")

    lines.extend(
        [
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
    event_sequence = event_times_sequence(loaded.events)
    windows, segments = select_analysis_segments(
        vision_time,
        events,
        event_sequence=event_sequence,
        robot_records=loaded.robot,
        vision_values=vision_values,
    )

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

    # 本段对每个自动窗口单独计算简版指标。
    # 输入是同一条视觉位移曲线和所有窗口掩码；输出是单段/合并窗口的 RMS、峰峰值和主频。
    # 实验作用：反复启停时可以分别看 motion_001、motion_002，也可以看 motion_all 的总体表现。
    window_metric_summaries = {
        name: calculate_window_metric_summary(vision_time, vision_values, mask)
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

    # 本段尝试输出圆点法/棋盘格法一致性图。
    # 如果其中一种方法数据不足，绘图函数会返回 False，不生成误导性的空对比图。
    plot_method_comparison(
        output_dir / "03_circle_checker_comparison.png",
        loaded.vision,
        config.ANALYSIS_AXIS,
    )

    # 本段尝试输出视觉与 UR TCP 对齐图。
    # 没有 ROBOT 记录时绘图函数会返回 False；这张图只用于时间关系和趋势对比，不代表已完成外参标定。
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
        segments,
        window_metric_summaries,
    )

    # 本段输出机器可读指标。
    # 实验作用：以后批量比较不同速度、轨迹或控制方法时，可以直接读取 JSON，
    # 不必再从中文 TXT 摘要里反向解析数字。
    metrics_json = {
        "source_file": str(source_path),
        "analysis_method": config.ANALYSIS_VISION_METHOD,
        "analysis_axis": config.ANALYSIS_AXIS,
        "analysis_primary_window": selected_window,
        "window_counts": window_counts,
        "segments": [segment.to_json() for segment in segments],
        "window_metric_summaries": window_metric_summaries,
        "detrend_method": config.DETREND_METHOD,
        "timing": timing,
        "timing_warnings": timing_warnings,
        "time_metrics": time_metrics,
        "dominant_frequency_hz": spectrum["dominant_frequency_hz"],
        "band_energy_mm2": spectrum["band_energy_mm2"],
        "recovery": recovery,
        "method_comparison": comparison,
    }

    # 本段保存 JSON 指标文件。
    # ensure_ascii=False 保留中文；allow_nan=True 允许记录 NaN，方便保留无法计算的数值状态。
    (output_dir / "analysis_metrics.json").write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    # 本段在终端打印最关键的三项结果，方便命令行运行时不用先打开 summary。
    print(
        f"[分析] 峰峰值 {time_metrics['peak_to_peak_mm']:.6f} mm，"
        f"RMS {time_metrics['rms_mm']:.6f} mm，"
        f"主频 {spectrum['dominant_frequency_hz']:.3f} Hz。"
    )

    # 本段打印输出目录，让用户能马上找到图表和摘要。
    print(f"[分析] 结果文件夹：{output_dir}")
    return output_dir


if __name__ == "__main__":
    # 本段支持直接运行 analyze.py。
    # 推荐日常仍从 main.py 的 analyze 模式进入；这里保留直接入口，方便单独调试分析脚本。
    # validate_config("analyze") 只检查分析配置，不会连接相机或机器人。
    config.validate_config("analyze")

    # 直接运行本文件时，不传 analysis_file，让 resolve_analysis_file() 按配置自动选择。
    run_analysis()
