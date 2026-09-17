"""机械臂断电时的相机-only 静止基线测试。

本模块故意不导入 robot.py，也不创建 RTDE/Dashboard 接口。它使用与
micro_closed_loop 静止基线完全相同的相机、RAW 与棋盘格分析链路，方便把
“机械臂通电静止”与“机械臂完全断电”放在同一量尺上比较。
"""

from __future__ import annotations

import gc
import json
import math
import multiprocessing as mp
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import config
import micro_closed_loop as mcl


class OfflineStaticStop(RuntimeError):
    """操作者请求停止相机-only 静止测试。"""


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _summary_metric(summary: dict[str, Any], name: str) -> float | None:
    """兼容读取新旧静止 summary 的三个核心指标。"""

    direct = _finite(summary.get(name))
    if direct is not None:
        return direct
    key = {
        "static_sigma_um": "mad_sigma_um",
        "static_rms_um": "std_um",
        "static_peak_to_peak_um": "peak_to_peak_um",
    }[name]
    values = [
        _finite((summary.get(axis) or {}).get(key)) for axis in ("x", "y")
    ]
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def compare_power_states(
    offline_summary: dict[str, Any],
    powered_summary: dict[str, Any] | None,
    *,
    powered_summary_path: Path | None = None,
) -> dict[str, Any]:
    """比较断电与最近一次通电基线，返回透明的指标和保守判读。"""

    metrics = ("static_sigma_um", "static_rms_um", "static_peak_to_peak_um")
    if powered_summary is None:
        return {
            "kind": "OFFLINE_STATIC_POWER_COMPARISON",
            "verdict": "NO_POWERED_REFERENCE",
            "verdict_text": "没有找到既有的机械臂通电静止基线，已只保存断电结果。",
            "powered_summary_path": None,
            "metrics": {},
        }

    rows: dict[str, Any] = {}
    higher_votes = 0
    similar_votes = 0
    valid_count = 0
    for name in metrics:
        offline = _summary_metric(offline_summary, name)
        powered = _summary_metric(powered_summary, name)
        ratio = None
        delta = None
        if offline is not None and powered is not None:
            valid_count += 1
            delta = powered - offline
            ratio = powered / offline if offline > 0.0 else math.inf
            # 同时要求相对差与绝对差，避免 0.05→0.08 μm 被夸成“高 60%”。
            if ratio >= 1.5 and delta >= 0.5:
                higher_votes += 1
            if powered <= max(offline * 1.25, offline + 0.5):
                similar_votes += 1
        rows[name] = {
            "robot_powered_off": offline,
            "robot_powered_on": powered,
            "powered_minus_off_um": delta,
            "powered_to_off_ratio": ratio,
        }

    if valid_count < 2:
        verdict = "INSUFFICIENT_DATA"
        text = "有效对比指标不足，暂时不能区分视觉/环境底噪与机械臂通电振动。"
    elif higher_votes >= 2:
        verdict = "POWERED_STATE_HIGHER"
        text = (
            "至少两个稳健指标在机械臂通电时高出断电基线 50% 且超过 0.5 μm；"
            "这支持“通电/伺服状态增加了振动或漂移”，但仍建议重复通电基线确认。"
        )
    elif similar_votes == valid_count:
        verdict = "SIMILAR_BASELINES"
        text = (
            "通电与断电基线处于同一量级；当前结果没有显示机械臂通电带来明显附加振动，"
            "观测下限更可能由视觉识别、相机/支架和环境共同决定。"
        )
    else:
        verdict = "INCONCLUSIVE"
        text = (
            "部分指标升高但证据不一致，暂时不能可靠归因；建议在安装完全不变时交替做"
            "断电/通电重复测试。"
        )

    return {
        "kind": "OFFLINE_STATIC_POWER_COMPARISON",
        "verdict": verdict,
        "verdict_text": text,
        "powered_summary_path": (
            None if powered_summary_path is None else str(powered_summary_path)
        ),
        "metrics": rows,
        "important_limit": (
            "断电基线仍包含真实的桌面、地面、相机支架和空气扰动，不等于纯算法误差；"
            "只有在相机、棋盘格、曝光、光照和安装均不变时，两组差值才可归因。"
        ),
    }


def find_latest_powered_static(output_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    """寻找最近一次 micro_closed_loop 生成的有效静止基线。"""

    candidates = sorted(
        Path(output_root).glob("micro_motion_*/static/static_summary.json"),
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if _summary_metric(payload, "static_rms_um") is not None:
            return path, payload
    return None, None


def aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """把多次 5 秒断电试验压成中位数与重复范围。"""

    result: dict[str, Any] = {}
    for name in ("static_sigma_um", "static_rms_um", "static_peak_to_peak_um"):
        values = [
            value
            for item in trials
            if (value := _summary_metric(item, name)) is not None
        ]
        result[name] = float(np.median(values)) if values else None
        result[f"{name}_min"] = min(values) if values else None
        result[f"{name}_max"] = max(values) if values else None
    rates = [_finite(item.get("chessboard_detection_rate")) for item in trials]
    valid_rates = [value for value in rates if value is not None]
    result["chessboard_detection_rate"] = (
        float(np.median(valid_rates)) if valid_rates else None
    )
    return result


def _allocate_run(output_root: Path) -> tuple[str, Path]:
    base = datetime.now().strftime("offline_static_%Y%m%d_%H%M%S")
    for suffix in range(100):
        run_id = base if suffix == 0 else f"{base}_{suffix:02d}"
        run_dir = Path(output_root) / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise RuntimeError("无法分配唯一的离线静止测试输出目录。")


def run_offline_static_test(stop_request_path: Path | None = None) -> Path:
    """运行三次相机-only 静止基线；绝不连接机器人。"""

    repeats = int(config.OFFLINE_STATIC_REPEATS)
    duration_s = float(config.OFFLINE_STATIC_SECONDS)
    run_id, run_dir = _allocate_run(config.OUTPUT_ROOT)
    raw_run_dir = Path(config.MICRO_LOOP_RAW_ROOT) / run_id
    temp_dir, trash_dir = mcl.prepare_temp_workspace(raw_run_dir)
    stop_request_path = None if stop_request_path is None else Path(stop_request_path)
    if stop_request_path is not None and stop_request_path.exists():
        stop_request_path.unlink()

    log_path = run_dir / "offline_static_log.txt"

    def log(text: str) -> None:
        print(f"[断电静止测试] {text}", flush=True)
        mcl.append_log(log_path, text)

    log("本模式只连接工业相机；不会导入 robot.py，不会尝试 RTDE/Dashboard 通信。")
    log(f"计划执行 {repeats} 次，每次 {duration_s:g} 秒；每次分析完成立即删除 RAW。")

    context = mp.get_context("spawn")
    error_queue = context.Queue(maxsize=config.ERROR_QUEUE_MAXSIZE)
    command_queue = context.Queue(maxsize=16)
    status_queue = context.Queue(maxsize=64)
    stop_event = context.Event()
    camera = context.Process(
        target=mcl.micro_closed_loop_camera_worker,
        args=(error_queue, command_queue, status_queue, stop_event),
        name="offline-static-camera-only",
    )
    pending: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    stopped = False
    success = False

    def check_stop() -> None:
        if stop_request_path is not None and stop_request_path.exists():
            stop_event.set()
            raise OfflineStaticStop("收到用户停止请求。")

    def pop_error() -> str | None:
        try:
            return str(error_queue.get_nowait())
        except queue.Empty:
            return None

    def wait_status(event: str, timeout_s: float) -> dict[str, Any]:
        deadline = time.perf_counter() + timeout_s
        while True:
            check_stop()
            error = pop_error()
            if error:
                raise RuntimeError(error)
            for index, item in enumerate(pending):
                if item.get("source") == "camera" and item.get("event") == event:
                    return pending.pop(index)
            if stop_event.is_set():
                raise RuntimeError(pop_error() or f"等待相机 {event} 时相机进程已停止。")
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"等待相机 {event} 超时。")
            try:
                pending.append(status_queue.get(timeout=0.05))
            except queue.Empty:
                pass

    def safe_sleep(seconds: float) -> None:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            check_stop()
            error = pop_error()
            if error:
                raise RuntimeError(error)
            if stop_event.wait(min(0.05, deadline - time.perf_counter())):
                raise RuntimeError(pop_error() or "相机进程在录制期间停止。")

    try:
        camera.start()
        ready = wait_status("READY", float(config.WORKER_READY_TIMEOUT_S))
        fps = float(ready["actual_camera_fps"])
        width = int(ready["full_width"])
        height = int(ready["full_height"])
        capacity = int(math.ceil((duration_s + 0.5) * fps)) + 8
        estimated_bytes = capacity * width * height
        free_gb = mcl.free_gb(raw_run_dir.parent)
        required_gb = estimated_bytes / (1024.0 ** 3) + float(
            config.OFFLINE_STATIC_DISK_RESERVE_GB
        )
        if free_gb < required_gb:
            raise RuntimeError(
                f"RAW 卷只剩 {free_gb:.2f} GiB，单次断电静止测试至少需要 "
                f"{required_gb:.2f} GiB（含余量）。"
            )
        log(f"相机就绪：{width}×{height}，{fps:.3f} fps；预计单次 RAW {mcl.format_gib(estimated_bytes)}。")

        for trial_index in range(1, repeats + 1):
            stem = mcl.next_clip_stem(run_id, f"powered_off_static_{trial_index:02d}", trial_index)
            command_queue.put(
                {
                    "action": "OPEN_WINDOW",
                    "temp_dir": str(temp_dir),
                    "stem": stem,
                    "clip_label": f"powered_off_static_{trial_index:02d}",
                    "capacity_frames": capacity,
                }
            )
            wait_status("WINDOW_OPENED", 10.0)
            log(f"第 {trial_index}/{repeats} 次开始录制 {duration_s:g} 秒。")
            safe_sleep(duration_s)
            command_queue.put({"action": "CLOSE_WINDOW"})
            saved = wait_status("WINDOW_SAVED", max(30.0, duration_s + 10.0))
            if bool(saved.get("truncated")):
                raise RuntimeError(f"第 {trial_index} 次相机缓冲被截断，结果不可用。")

            paths = mcl.clip_temp_paths(temp_dir, stem)
            clip = mcl.open_raw_clip(paths["raw"])
            try:
                meter = mcl.GroupVisionMeter(label=f"offline_static_{trial_index:02d}")
                every_n = mcl.spanning_every_n(
                    clip.frame_count, int(config.MICRO_LOOP_STATIC_ANALYZE_FRAMES)
                )
                primed = meter.prime(clip, every_n=every_n, frames=None)
                measurement = meter.measure_clip(
                    clip,
                    every_n=every_n,
                    iteration_id=f"offline_static_{trial_index:02d}",
                    tail_frames=None,
                    vision_axis="x",
                )
                summary = mcl.analyze_static_clip(
                    clip,
                    measurement,
                    fps=fps,
                    record_start_ns=int(saved["record_start_ns"]),
                    record_stop_ns=int(saved["record_stop_ns"]),
                )
                for name, key in (
                    ("static_sigma_um", "mad_sigma_um"),
                    ("static_rms_um", "std_um"),
                ):
                    values = [
                        _finite(summary[axis][key]) for axis in ("x", "y")
                    ]
                    valid = [value for value in values if value is not None]
                    summary[name] = max(valid) if valid else None
                summary["static_peak_to_peak_um"] = max(
                    float(summary[axis]["peak_to_peak_um"])
                    for axis in ("x", "y")
                    if summary[axis]["peak_to_peak_um"] is not None
                )
                summary.update(
                    {
                        "kind": "ROBOT_POWERED_OFF_STATIC_TRIAL",
                        "trial_index": trial_index,
                        "robot_communication_attempted": False,
                        "robot_expected_power_state": "OFF",
                        "reference_frame_index": primed["reference_frame_index"],
                        "analyze_every_n": every_n,
                        "analyze_frames": len(measurement.frame_rows),
                    }
                )
                trial_dir = run_dir / f"trial_{trial_index:02d}"
                mcl.write_rows_csv(
                    trial_dir / "static_frames.csv",
                    measurement.frame_rows,
                    mcl.STATIC_FRAME_COLUMNS,
                )
                mcl.write_json(trial_dir / "static_summary.json", summary)
                evidence_index = mcl.last_accepted_frame_index(measurement)
                if evidence_index is not None:
                    mcl.save_evidence_png(
                        clip,
                        evidence_index,
                        trial_dir / "evidence" / "last_accepted.png",
                    )
                trials.append(summary)
                log(
                    f"第 {trial_index} 次完成：识别率 "
                    f"{summary['chessboard_detection_rate']:.1%}，sigma "
                    f"{summary['static_sigma_um']:.2f} μm，RMS "
                    f"{summary['static_rms_um']:.2f} μm，峰峰值 "
                    f"{summary['static_peak_to_peak_um']:.2f} μm。"
                )
            finally:
                del clip
                gc.collect()

            mcl.drop_temp_files(
                [paths["raw"], paths["meta"], paths["frames"]],
                trash_dir,
                log=log,
            )
            released, failed = mcl.delete_video_payloads(trash_dir)
            mcl.sweep_directory(trash_dir)
            if failed:
                raise RuntimeError(
                    f"第 {trial_index} 次 RAW 删除失败；为防止继续占盘，已停止。"
                )
            log(f"第 {trial_index} 次 RAW 已删除，释放 {mcl.format_gib(released)}。")

        aggregate = aggregate_trials(trials)
        final_summary = {
            "kind": "ROBOT_POWERED_OFF_STATIC_BASELINE",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "robot_communication_attempted": False,
            "robot_expected_power_state": "OFF",
            "camera_width": width,
            "camera_height": height,
            "camera_actual_fps": fps,
            "duration_per_trial_s": duration_s,
            "trial_count": len(trials),
            **aggregate,
            "trials": trials,
            "interpretation_limit": (
                "本结果是视觉算法 + 相机/支架 + 棋盘格安装 + 环境振动的总底噪，"
                "不是纯算法误差。应与同一安装状态下的机械臂通电静止基线比较。"
            ),
        }
        powered_path, powered = find_latest_powered_static(config.OUTPUT_ROOT)
        comparison = compare_power_states(
            final_summary, powered, powered_summary_path=powered_path
        )
        mcl.write_json(run_dir / "offline_static_summary.json", final_summary)
        mcl.write_json(run_dir / "comparison_with_latest_powered.json", comparison)
        report_lines = [
            "机械臂断电静止基线（仅工业相机）",
            f"运行目录：{run_dir}",
            f"重复：{len(trials)} × {duration_s:g} s",
            f"断电基线中位数：sigma {aggregate['static_sigma_um']:.3f} μm ｜ "
            f"RMS {aggregate['static_rms_um']:.3f} μm ｜ 峰峰值 "
            f"{aggregate['static_peak_to_peak_um']:.3f} μm",
            f"最近通电基线：{powered_path if powered_path is not None else '未找到'}",
            f"对比结论：{comparison['verdict']}｜{comparison['verdict_text']}",
            comparison["important_limit"] if "important_limit" in comparison else "",
        ]
        (run_dir / "offline_static_report.txt").write_text(
            "\n".join(line for line in report_lines if line) + "\n", encoding="utf-8"
        )
        for line in report_lines:
            if line:
                log(line)
        success = True
    except OfflineStaticStop as exc:
        stopped = True
        log(f"测试已按用户请求停止：{exc}")
    finally:
        stop_event.set()
        if camera.pid is not None:
            camera.join(timeout=float(config.WORKER_JOIN_TIMEOUT_S))
            if camera.is_alive():
                camera.terminate()
                camera.join(timeout=2.0)
        released, failed = mcl.delete_video_payloads(raw_run_dir)
        mcl.sweep_directory(temp_dir)
        mcl.sweep_directory(trash_dir)
        mcl.write_json(
            run_dir / "run_status.json",
            {
                "kind": "OFFLINE_STATIC_RUN_STATUS",
                "success": success,
                "stopped_by_user": stopped,
                "robot_communication_attempted": False,
                "completed_trials": len(trials),
                "video_bytes_deleted_in_final_cleanup": released,
                "video_cleanup_failures": [str(path) for path in failed],
            },
        )
        if stop_request_path is not None and stop_request_path.exists():
            stop_request_path.unlink()

    return run_dir
