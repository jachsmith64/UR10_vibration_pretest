"""
读取实验 TXT/JSONL，完成时间对齐、去趋势、振动指标和图表输出。

这个文件不导入相机 SDK 或 ur_rtde，因此可以在任何装有科学计算依赖的电脑上离线运行。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

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
    """按记录类型拆开的实验内容，并保留无法解析行的位置供排错。"""

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

    patterns = ("**/run_log.txt", "**/vision_results.txt", "**/robot_test_log.txt")
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(config.OUTPUT_ROOT.glob(pattern))

    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"{config.OUTPUT_ROOT} 下没有 run_log.txt 或 vision_results.txt。"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_analysis_file(requested: Path | None = None) -> Path:
    """按“命令行参数→config.py→最新记录”的优先级解析分析目标。"""

    candidate = requested or config.ANALYSIS_FILE
    if candidate is None:
        resolved = _find_latest_run_file()
        print(f"[分析] 未指定文件，自动选择最新记录：{resolved}")
        return resolved

    candidate = Path(candidate).expanduser()
    if not candidate.is_absolute():
        candidate = (config.PROJECT_DIR / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"分析文件不存在：{candidate}")
    return candidate


def load_run_file(path: Path) -> LoadedRun:
    """
    每行独立解析，单行损坏不会让整个实验完全打不开。

    malformed_lines 会写入摘要，提醒你回看磁盘写入或人工编辑造成的问题。
    """

    buckets: dict[str, list[dict[str, Any]]] = {
        "META": [],
        "EVENT": [],
        "VISION": [],
        "ROBOT": [],
        "ERROR": [],
    }
    malformed: list[int] = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            text = raw_line.strip()
            if not text:
                continue

            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                malformed.append(line_number)
                continue

            if not isinstance(record, dict):
                malformed.append(line_number)
                continue

            kind = str(record.get("kind", "")).upper()
            if kind in buckets:
                buckets[kind].append(record)
            else:
                malformed.append(line_number)

    loaded = LoadedRun(
        source_path=Path(path),
        meta=buckets["META"],
        events=buckets["EVENT"],
        vision=buckets["VISION"],
        robot=buckets["ROBOT"],
        errors=buckets["ERROR"],
        malformed_lines=malformed,
    )

    if not loaded.vision:
        raise ValueError(
            f"{path} 中没有 VISION 记录，无法计算相机测得的末端振动。"
        )
    return loaded


# =============================================================================
# 2. 视觉序列、事件和机器人序列
# =============================================================================

def _finite_float(value: Any) -> float:
    """把 None、字符串和 JSON 中的 NaN 统一转成浮点 NaN，便于 NumPy 后续筛选。"""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
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

    prefix = "circle" if method == "circles" else "checker"
    time_values: list[float] = []
    displacement_values: list[float] = []
    quality_values: list[float] = []

    for record in records:
        if not bool(record.get(f"{prefix}_is_valid", False)):
            continue

        host_ns = _finite_float(record.get("host_ns"))
        analysis_time_s = _finite_float(record.get("analysis_time_s"))
        dx = _finite_float(record.get(f"{prefix}_dx_mm"))
        dy = _finite_float(record.get(f"{prefix}_dy_mm"))
        quality = _finite_float(record.get(f"{prefix}_quality"))

        if axis == "x":
            displacement = dx
        elif axis == "y":
            displacement = dy
        elif axis == "magnitude":
            displacement = math.hypot(dx, dy)
        else:
            raise ValueError(f"未知分析轴：{axis}")

        if math.isfinite(host_ns) and math.isfinite(displacement):
            # vision_test 用图片/视频自身时间，正式 experiment 则由 camera_worker 写入共同 host 时基。
            time_values.append(
                analysis_time_s if math.isfinite(analysis_time_s) else host_ns * 1e-9
            )
            displacement_values.append(displacement)
            quality_values.append(quality)

    if len(time_values) < 8:
        raise ValueError(
            f"{method} 在 {axis} 方向只有 {len(time_values)} 个有效点，"
            "至少需要 8 个才能进行基本振动分析。"
        )

    time_array = np.asarray(time_values, dtype=float)
    displacement_array = np.asarray(displacement_values, dtype=float)
    quality_array = np.asarray(quality_values, dtype=float)

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
    """把每种事件第一次出现的 host_ns 转为秒；重复事件保留第一次以避免覆盖真实起点。"""

    result: dict[str, float] = {}
    for event in sorted(events, key=lambda item: _finite_float(item.get("host_ns"))):
        name = str(event.get("name", ""))
        host_ns = _finite_float(event.get("host_ns"))
        if name and math.isfinite(host_ns) and name not in result:
            result[name] = host_ns * 1e-9
    return result


def _window_mask(
    time_s: np.ndarray,
    start_s: float | None,
    end_s: float | None,
) -> np.ndarray:
    """建立闭区间掩码；缺少某端事件时自动延伸到数据首尾。"""

    start = time_s[0] if start_s is None else start_s
    end = time_s[-1] if end_s is None else end_s
    return (time_s >= start) & (time_s <= end)


def select_analysis_windows(
    time_s: np.ndarray,
    events: dict[str, float],
) -> dict[str, np.ndarray]:
    """
    依据事件拆出 baseline、motion、post 和 full 四个区间。

    只有 vision_test 没有运动事件时，motion 会退化为全序列，仍可分析合成或手动移动数据。
    """

    experiment_start = events.get("experiment_started", float(time_s[0]))
    motion_start = events.get("motion_command_sent")
    motion_end = events.get("motion_finished")
    experiment_end = events.get("experiment_finished", float(time_s[-1]))

    if motion_start is None or motion_end is None:
        return {
            "full": np.ones_like(time_s, dtype=bool),
            "baseline": np.zeros_like(time_s, dtype=bool),
            "motion": np.ones_like(time_s, dtype=bool),
            "post": np.zeros_like(time_s, dtype=bool),
        }

    return {
        "full": _window_mask(time_s, experiment_start, experiment_end),
        "baseline": _window_mask(time_s, experiment_start, motion_start),
        "motion": _window_mask(time_s, motion_start, motion_end),
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

    if not robot_records:
        return None

    axis_index = {"x": 0, "y": 1}
    time_values: list[float] = []
    positions: list[list[float]] = []

    for record in robot_records:
        host_ns = _finite_float(record.get("host_ns"))
        pose = record.get("actual_tcp_pose")
        if not math.isfinite(host_ns) or not isinstance(pose, list) or len(pose) < 3:
            continue

        xyz = [_finite_float(value) for value in pose[:3]]
        if all(math.isfinite(value) for value in xyz):
            time_values.append(host_ns * 1e-9)
            positions.append(xyz)

    if len(time_values) < 2:
        return None

    time_array = np.asarray(time_values, dtype=float)
    position_array = np.asarray(positions, dtype=float)

    if axis in axis_index:
        values_mm = position_array[:, axis_index[axis]] * 1000.0
    else:
        relative = position_array - position_array[0]
        values_mm = np.linalg.norm(relative, axis=1) * 1000.0

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
    按原始中位采样周期建立均匀时间轴，供 FFT、Welch 和数字滤波使用。

    原始样本仍保留用于时域图；插值不会声称增加真实测量带宽。
    """

    if len(time_s) != len(values):
        raise ValueError("时间与位移数组长度不一致。")

    delta_t = np.diff(time_s)
    valid_delta = delta_t[delta_t > 0]
    if len(valid_delta) == 0:
        raise ValueError("时间戳没有递增，无法估计采样率。")

    median_dt = float(np.median(valid_delta))
    sample_rate_hz = 1.0 / median_dt
    uniform_time = np.arange(time_s[0], time_s[-1] + 0.5 * median_dt, median_dt)
    uniform_values = np.interp(uniform_time, time_s, values)

    timing = {
        "sample_rate_hz": sample_rate_hz,
        "median_dt_s": median_dt,
        "dt_std_s": float(np.std(valid_delta)),
        "max_gap_s": float(np.max(valid_delta)),
        "original_sample_count": int(len(time_s)),
        "uniform_sample_count": int(len(uniform_time)),
    }
    return uniform_time, uniform_values, sample_rate_hz, timing


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
    """

    if config.DETREND_METHOD == "linear":
        residual = signal.detrend(values, type="linear")
        trend = values - residual
        return trend, residual

    if config.DETREND_METHOD == "savgol":
        window = _savgol_window_points(sample_rate_hz, len(values))
        if window <= config.SAVGOL_POLYORDER:
            raise ValueError("数据太短，无法使用当前 Savitzky-Golay 参数。")
        trend = signal.savgol_filter(
            values,
            window_length=window,
            polyorder=config.SAVGOL_POLYORDER,
            mode="interp",
        )
        return trend, values - trend

    if config.DETREND_METHOD == "highpass":
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
        residual = signal.sosfiltfilt(sos, values)
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
    """

    centered = np.asarray(residual, dtype=float) - float(np.mean(residual))
    window = np.hanning(len(centered))
    fft_values = np.fft.rfft(centered * window)
    fft_frequency = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    amplitude = np.abs(fft_values) * (2.0 / max(float(np.sum(window)), 1e-12))
    if len(amplitude):
        amplitude[0] *= 0.5

    nperseg = min(1024, len(centered))
    if nperseg < 8:
        raise ValueError("有效点少于 8，无法计算 Welch 频谱。")
    welch_frequency, psd = signal.welch(
        centered,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        scaling="density",
    )

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
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
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
    """给出最常用的峰峰值、RMS、标准差和最大绝对残差，单位均为 mm。"""

    residual = np.asarray(residual, dtype=float)
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
    """

    motion_finished = events.get("motion_finished")
    if motion_finished is None:
        return {
            "recovery_time_s": None,
            "recovery_threshold_mm": None,
            "baseline_rms_mm": None,
        }

    if np.count_nonzero(baseline_mask) >= 4:
        baseline_rms = float(
            np.sqrt(np.mean(np.asarray(residual)[baseline_mask] ** 2))
        )
    else:
        baseline_rms = 0.0

    threshold = max(
        config.RECOVERY_ABSOLUTE_THRESHOLD_MM,
        config.RECOVERY_BASELINE_FACTOR * baseline_rms,
    )
    start_index = int(np.searchsorted(uniform_time_s, motion_finished, side="left"))
    if len(uniform_time_s) - start_index < 4:
        return {
            "recovery_time_s": None,
            "recovery_threshold_mm": float(threshold),
            "baseline_rms_mm": float(baseline_rms),
        }

    # 只对运动后的残差求包络，避免运动段的大振幅通过 Hilbert 非局部边缘效应污染恢复起点。
    post_residual = np.asarray(residual[start_index:], dtype=float)
    envelope = np.abs(signal.hilbert(post_residual))
    hold_samples = max(1, int(math.ceil(config.RECOVERY_HOLD_SECONDS * sample_rate_hz)))

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

    common_start = max(circle_time[0], checker_time[0])
    common_end = min(circle_time[-1], checker_time[-1])
    mask = (circle_time >= common_start) & (circle_time <= common_end)
    common_time = circle_time[mask]

    if len(common_time) < 3:
        return {
            "common_sample_count": int(len(common_time)),
            "circle_checker_rmse_mm": None,
            "circle_checker_correlation": None,
        }

    checker_interpolated = np.interp(common_time, checker_time, checker_value)
    circle_common = circle_value[mask]
    difference = circle_common - checker_interpolated
    rmse = float(np.sqrt(np.mean(difference**2)))

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
    """所有图统一以第一条有效视觉记录为 0 s，避免显示很大的 perf_counter 秒数。"""

    return np.asarray(time_s, dtype=float) - origin_s


def _mark_events(
    axes: plt.Axes,
    events: dict[str, float],
    origin_s: float,
) -> None:
    """在时域图上标出发送运动与完成时刻，帮助判断振动发生在哪一阶段。"""

    colors = {
        "motion_command_sent": "tab:red",
        "motion_finished": "tab:green",
    }
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
    """上图展示原始位移与趋势，下图只展示去趋势后的微小振动。"""

    origin = float(original_time_s[0])
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(
        _relative_time(original_time_s, origin),
        original_values,
        linewidth=0.9,
        label="measured displacement",
    )
    axes[0].plot(
        _relative_time(uniform_time_s, origin),
        trend,
        linewidth=1.5,
        label=f"{config.DETREND_METHOD} trend",
    )
    axes[0].set_ylabel("Displacement / mm")
    axes[0].grid(True, alpha=0.3)
    _mark_events(axes[0], events, origin)
    axes[0].legend(loc="best")

    axes[1].plot(
        _relative_time(uniform_time_s, origin),
        residual,
        color="tab:orange",
        linewidth=0.9,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Vibration residual / mm")
    axes[1].grid(True, alpha=0.3)
    _mark_events(axes[1], events, origin)

    figure.suptitle(f"Vision displacement and vibration — {method}, axis={axis}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_spectrum(
    output_path: Path,
    spectrum: dict[str, Any],
    sample_rate_hz: float,
) -> None:
    """分别画 FFT 幅值与 Welch PSD，对数 PSD 更容易看见弱频率成分。"""

    nyquist = 0.5 * sample_rate_hz
    upper = min(config.DOMINANT_FREQ_MAX_HZ, nyquist)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(
        spectrum["fft_frequency_hz"],
        spectrum["fft_amplitude_mm"],
        linewidth=1.0,
    )
    axes[0].set_ylabel("FFT amplitude / mm")
    axes[0].set_xlim(0.0, upper)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(
        spectrum["welch_frequency_hz"],
        np.maximum(spectrum["welch_psd_mm2_per_hz"], 1e-18),
        linewidth=1.0,
    )
    axes[1].set_xlabel("Frequency / Hz")
    axes[1].set_ylabel("PSD / mm²/Hz")
    axes[1].set_xlim(0.0, upper)
    axes[1].grid(True, which="both", alpha=0.3)

    dominant = spectrum["dominant_frequency_hz"]
    figure.suptitle(f"Vibration spectrum — dominant {dominant:.3f} Hz")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_method_comparison(
    output_path: Path,
    records: list[dict[str, Any]],
    axis: str,
) -> bool:
    """两种视觉方法都有效时画在同一时间轴；任一数据不足就不生成误导性的空图。"""

    try:
        circle_time, circle_value, _ = extract_vision_series(records, "circles", axis)
        checker_time, checker_value, _ = extract_vision_series(
            records,
            "checkerboard",
            axis,
        )
    except ValueError:
        return False

    origin = min(circle_time[0], checker_time[0])
    figure, axes = plt.subplots(figsize=(11, 4.8))
    axes.plot(
        _relative_time(circle_time, origin),
        circle_value,
        linewidth=0.9,
        label="circles",
    )
    axes.plot(
        _relative_time(checker_time, origin),
        checker_value,
        linewidth=0.9,
        label="checkerboard",
        alpha=0.8,
    )
    axes.set_xlabel("Time / s")
    axes.set_ylabel("Displacement / mm")
    axes.set_title(f"Circle vs checkerboard — axis={axis}")
    axes.grid(True, alpha=0.3)
    axes.legend()
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

    if robot_series is None:
        return False

    robot_time_s, robot_values = robot_series
    common_start = max(vision_time_s[0], robot_time_s[0])
    common_end = min(vision_time_s[-1], robot_time_s[-1])
    if common_end <= common_start:
        return False

    origin = common_start
    vision_relative = vision_values - vision_values[0]
    robot_relative = robot_values - robot_values[0]

    figure, axes = plt.subplots(figsize=(11, 4.8))
    axes.plot(
        _relative_time(vision_time_s, origin),
        vision_relative,
        linewidth=0.9,
        label="vision relative",
    )
    axes.plot(
        _relative_time(robot_time_s, origin),
        robot_relative,
        linewidth=0.9,
        label="UR TCP relative",
        alpha=0.8,
    )
    axes.set_xlim(0.0, common_end - common_start)
    axes.set_xlabel("Time / s")
    axes.set_ylabel("Relative displacement / mm")
    axes.set_title(f"Host-clock alignment — axis={axis}")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return True


# =============================================================================
# 6. 摘要写入与分析总入口
# =============================================================================

def _format_optional(value: float | int | None, digits: int = 6) -> str:
    """摘要中把缺失结果明确写成 unavailable，不用 0 掩盖无法计算。"""

    if value is None:
        return "unavailable"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "unavailable"
    return f"{numeric:.{digits}f}"


def write_summary(
    output_path: Path,
    loaded: LoadedRun,
    timing: dict[str, float],
    time_metrics: dict[str, float],
    spectrum: dict[str, Any],
    recovery: dict[str, float | None],
    comparison: dict[str, float | int | None],
    valid_quality: np.ndarray,
    primary_count: int,
) -> None:
    """写成人能快速回看的中文 TXT，同时保留所有指标单位和方法名称。"""

    lines = [
        "UR10 末端振动预实验分析摘要",
        "=" * 56,
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"源文件：{loaded.source_path}",
        f"视觉方法：{config.ANALYSIS_VISION_METHOD}",
        f"分析方向：{config.ANALYSIS_AXIS}",
        f"去趋势方法：{config.DETREND_METHOD}",
        "",
        "一、数据完整性",
        f"视觉总记录数：{len(loaded.vision)}",
        f"本方法有效记录数：{len(valid_quality)}",
        f"主分析区间点数：{primary_count}",
        f"机器人记录数：{len(loaded.robot)}",
        f"事件记录数：{len(loaded.events)}",
        f"错误记录数：{len(loaded.errors)}",
        f"无法解析的行：{loaded.malformed_lines or '无'}",
        f"平均识别质量：{_format_optional(float(np.nanmean(valid_quality)), 4)}",
        "",
        "二、采样时间",
        f"估计采样率：{timing['sample_rate_hz']:.6f} Hz",
        f"中位采样间隔：{timing['median_dt_s']:.9f} s",
        f"采样间隔标准差：{timing['dt_std_s']:.9f} s",
        f"最大相邻间隔：{timing['max_gap_s']:.9f} s",
        "",
        "三、去趋势后时域指标",
        f"峰峰值：{time_metrics['peak_to_peak_mm']:.6f} mm",
        f"RMS：{time_metrics['rms_mm']:.6f} mm",
        f"标准差：{time_metrics['std_mm']:.6f} mm",
        f"最大绝对残差：{time_metrics['max_abs_mm']:.6f} mm",
        "",
        "四、频域指标",
        f"主频：{_format_optional(spectrum['dominant_frequency_hz'], 6)} Hz",
        f"频率分辨率：{spectrum['frequency_resolution_hz']:.6f} Hz",
    ]

    for band_name, energy in spectrum["band_energy_mm2"].items():
        lines.append(f"{band_name} 频带能量：{energy:.10e} mm²")

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

    if loaded.errors:
        lines.extend(["", "记录中的错误："])
        for error in loaded.errors:
            lines.append(f"- {error.get('message', error)}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(analysis_file: Path | None = None) -> Path:
    """
    完成一次分析并返回新建的结果文件夹。

    主指标使用 motion 区间；时域总图仍展示完整有效序列和运动事件。
    """

    source_path = resolve_analysis_file(analysis_file)
    loaded = load_run_file(source_path)

    vision_time, vision_values, quality = extract_vision_series(
        loaded.vision,
        config.ANALYSIS_VISION_METHOD,
        config.ANALYSIS_AXIS,
    )
    events = event_times_seconds(loaded.events)
    windows = select_analysis_windows(vision_time, events)

    primary_mask = windows["motion"]
    if np.count_nonzero(primary_mask) < 8:
        print("[分析警告] motion 区间有效点不足 8，改用全部有效视觉数据。")
        primary_mask = windows["full"]

    primary_time = vision_time[primary_mask]
    primary_values = vision_values[primary_mask]
    if len(primary_time) < 8:
        raise ValueError("最终主分析区间仍少于 8 个有效点。")

    uniform_time, uniform_values, sample_rate_hz, timing = resample_uniform(
        primary_time,
        primary_values,
    )
    trend, residual = detrend_motion(uniform_values, sample_rate_hz)
    time_metrics = calculate_time_metrics(residual)
    spectrum = calculate_spectrum(residual, sample_rate_hz)

    # 恢复时间需要运动前、运动后完整序列，因此单独对 full 区间进行同样的重采样和去趋势。
    full_mask = windows["full"]
    full_time = vision_time[full_mask]
    full_values = vision_values[full_mask]
    full_uniform_time, full_uniform_values, full_rate, _ = resample_uniform(
        full_time,
        full_values,
    )
    full_trend, full_residual = detrend_piecewise(
        full_uniform_time,
        full_uniform_values,
        full_rate,
        events,
    )

    baseline_start = events.get("experiment_started", float(full_uniform_time[0]))
    baseline_end = events.get("motion_command_sent")
    baseline_uniform_mask = _window_mask(
        full_uniform_time,
        baseline_start,
        baseline_end,
    )
    if baseline_end is None:
        baseline_uniform_mask[:] = False

    recovery = calculate_recovery_time(
        full_uniform_time,
        full_residual,
        events,
        baseline_uniform_mask,
        full_rate,
    )
    comparison = compare_vision_methods(loaded.vision, config.ANALYSIS_AXIS)
    robot_series = extract_robot_tcp_series(loaded.robot, config.ANALYSIS_AXIS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = source_path.parent / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    plot_spectrum(
        output_dir / "02_spectrum.png",
        spectrum,
        sample_rate_hz,
    )
    plot_method_comparison(
        output_dir / "03_circle_checker_comparison.png",
        loaded.vision,
        config.ANALYSIS_AXIS,
    )
    plot_vision_robot_alignment(
        output_dir / "04_vision_robot_alignment.png",
        full_time,
        full_values,
        robot_series,
        config.ANALYSIS_AXIS,
    )

    write_summary(
        output_dir / "analysis_summary.txt",
        loaded,
        timing,
        time_metrics,
        spectrum,
        recovery,
        comparison,
        quality,
        len(primary_time),
    )

    # 机器可读指标便于以后批量比较不同速度、轨迹或控制方法，不必再从 TXT 反向解析数字。
    metrics_json = {
        "source_file": str(source_path),
        "analysis_method": config.ANALYSIS_VISION_METHOD,
        "analysis_axis": config.ANALYSIS_AXIS,
        "detrend_method": config.DETREND_METHOD,
        "timing": timing,
        "time_metrics": time_metrics,
        "dominant_frequency_hz": spectrum["dominant_frequency_hz"],
        "band_energy_mm2": spectrum["band_energy_mm2"],
        "recovery": recovery,
        "method_comparison": comparison,
    }
    (output_dir / "analysis_metrics.json").write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print(
        f"[分析] 峰峰值 {time_metrics['peak_to_peak_mm']:.6f} mm，"
        f"RMS {time_metrics['rms_mm']:.6f} mm，"
        f"主频 {spectrum['dominant_frequency_hz']:.3f} Hz。"
    )
    print(f"[分析] 结果文件夹：{output_dir}")
    return output_dir


if __name__ == "__main__":
    # 允许直接运行 analyze.py，但推荐统一从 main.py 的 analyze 模式进入。
    config.validate_config("analyze")
    run_analysis()
