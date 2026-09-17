"""
UR10 末端振动预实验的唯一总入口。

用户运行 python main.py 后，程序首先来到这里。本文件的职责不是实现视觉算法、
机器人轨迹或频谱分析，而是根据 config.py / 命令行选择，把任务交给对应模块。

从用户操作看，代码大致这样流动：
1. 用户选择模式：config.RUN_MODE 或命令行 --mode。
2. main() 做配置检查：先排除拼写错误、危险真机开关、明显不合理参数。
3. 单模块模式直接转交：
   - vision_test -> camera.py 读取图片/视频/相机并输出视觉结果；
   - robot_dry_run / robot_test -> robot.py 生成轨迹、检查轨迹或连接 UR；
   - analyze -> analyze.py 读取已有记录并输出分析图表。
4. experiment 模式比较特殊：main.py 会同时启动记录、相机、机器人三个子进程，
   并负责 ready、start、motion_done、stop 这些流程信号。

因此读本文件时，重点不是每个算法怎么算，而是“某个模式被选中后，谁先启动、
谁等待谁、数据从哪个队列流到哪个文件”。
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import time
from datetime import datetime, timedelta
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import config


# =============================================================================
# 1. 通用记录进程：把多个来源的数据排队写成同一个实验日志
# =============================================================================

def record_writer_worker(record_queue: Any, output_path: Path) -> None:
    """
    完整实验中的“统一写日志工人”。

    输入：
    - record_queue：相机进程、机器人进程和主进程放入的记录字典；
    - output_path：本次实验最终的 run_log.txt 路径。

    输出：
    - 每收到一个记录字典，就写成 run_log.txt 中的一行 JSON。

    实验作用：
    - 让相机数据、机器人状态、实验事件、错误信息最终进入同一个时间序列文件；
    - 多个生产者只负责把记录放进队列，不直接碰文件，避免两条记录互相穿插。
    - 收到 None 表示所有生产者已经停止，可以安全关闭文件。
    """

    # 本段准备输出文件。输入是目标路径；输出是一个已经打开、可逐行写入的日志文件。
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 本段持续消费记录队列。输入是一条条实验记录；输出是一行行 JSON 文本。
    # 行缓冲能降低异常退出时丢日志的概率，也方便实验中实时打开文件查看新记录。
    with output_path.open("w", encoding="utf-8", buffering=1) as file:
        while True:
            # 没有新记录时这里会阻塞等待，因此写日志进程不会空转占 CPU。
            record = record_queue.get()

            # None 是主进程发送的“日志可以收尾”信号，不是一条真实实验数据。
            if record is None:
                break

            # 每行是一条完整 JSON：人可以用记事本看，程序也可以逐行恢复字段。
            line = json.dumps(record, ensure_ascii=False, allow_nan=True)
            file.write(line + "\n")


def segmented_record_writer_worker(record_queue: Any, batch_dir: Path) -> None:
    """Route interleaved camera/robot/event records to each segment RUNLOG file."""

    batch_dir = Path(batch_dir)
    files: dict[str, Any] = {}
    try:
        while True:
            record = record_queue.get()
            if record is None:
                break
            segment_base = record.get("segment_base")
            if not segment_base:
                continue
            segment_base = str(segment_base)
            file = files.get(segment_base)
            if file is None:
                path = batch_dir / f"{segment_base}_RUNLOG.txt"
                file = path.open("x", encoding="utf-8", buffering=1)
                files[segment_base] = file
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
    finally:
        for file in files.values():
            file.close()


def _record_event(record_queue: Any, name: str, **extra: Any) -> None:
    """
    把主程序观察到的实验流程节点写进日志。

    输入：
    - name：事件名称，例如 all_workers_ready、experiment_started、experiment_finished；
    - extra：附加字段，例如轨迹类型或阶段信息。

    输出：
    - 一条 kind="EVENT" 的记录进入 record_queue，最后由 record_writer_worker 写入 run_log.txt。

    实验作用：
    - 给相机帧和机器人状态之间插入“实验流程标尺”，后续 analyze.py 才知道哪些时间属于静止、
      运动或后记录阶段。
    """

    # 本段给事件打上主机单调时钟。它不表示现实日期，而是用于同一台电脑内的相对对时。
    record_queue.put(
        {
            "kind": "EVENT",
            "host_ns": time.perf_counter_ns(),
            "name": name,
            **extra,
        },
        timeout=1.0,
    )


def _pop_worker_error(error_queue: Any) -> str | None:
    """
    主进程用它“顺手看一眼”子进程有没有报错。

    输入：相机进程或机器人进程写入的 error_queue。
    输出：有错误时返回错误文字；没有错误时返回 None。

    实验作用：主程序在等待 ready、等待运动完成、等待后记录结束时都要反复调用它。
    这样一旦某个子进程失败，完整实验会尽快停止，而不是继续等超时。
    """

    try:
        # 只要有一条错误，主程序上层就会把它当成本次实验失败原因。
        return str(error_queue.get_nowait())
    except queue.Empty:
        # 没有错误是正常状态，调用方会继续检查 ready、stop 或超时。
        return None


def _wait_for_worker_error(error_queue: Any, timeout_s: float = 0.5) -> str | None:
    """Allow a multiprocessing Queue feeder to publish an error before masking it."""

    immediate = _pop_worker_error(error_queue)
    if immediate:
        return immediate
    try:
        return str(error_queue.get(timeout=max(0.0, float(timeout_s))))
    except queue.Empty:
        return None


def _wait_ready(
    camera_ready: Any,
    robot_ready: Any,
    error_queue: Any,
    stop_event: Any,
) -> None:
    """
    等待完整实验的两个硬件子进程真正准备好。

    输入：
    - camera_ready：相机进程成功打开图像来源并处理过第一帧后置位。
    - robot_ready：机器人进程完成轨迹检查、连接和起点检查后置位。
    - error_queue：任一子进程失败时会写错误。
    - stop_event：主程序或子进程要求停止时会置位。

    输出：
    - 两个 ready 都置位时正常返回；
    - 任一子进程报错、stop_event 置位或等待超时时抛错。

    实验作用：确保“相机已经能取图、机器人已经准备好”之后，才允许操作者确认并开始运动。
    这个函数只等待信号，不连接相机、不连接机器人、不发运动命令。
    """

    # 本段设置等待上限。输入是配置中的秒数；输出是一个绝对截止时刻。
    deadline = time.perf_counter() + config.WORKER_READY_TIMEOUT_S

    # 本段是 ready 等待循环。它同时观察 ready、错误、停止信号和超时。
    while not (camera_ready.is_set() and robot_ready.is_set()):
        worker_error = _pop_worker_error(error_queue)
        if worker_error:
            raise RuntimeError(worker_error)

        if stop_event.is_set():
            raise RuntimeError("某个子进程在 ready 前要求停止，但没有返回更多错误信息。")

        # 超时时明确说明相机或机器人谁没准备好，便于现场排查。
        if time.perf_counter() >= deadline:
            missing = []
            if not camera_ready.is_set():
                missing.append("相机")
            if not robot_ready.is_set():
                missing.append("机器人")
            raise TimeoutError(f"等待 {'、'.join(missing)} ready 超时。")

        # 0.05 秒检查一次，既能及时响应，也不会让 CPU 忙等。
        time.sleep(0.05)


def _wait_motion_done(
    motion_done: Any,
    error_queue: Any,
    stop_event: Any,
) -> None:
    """
    完整实验运动阶段的“主程序等待逻辑”。

    输入：
    - motion_done：机器人子进程发出的“运动完成”信号；
    - error_queue：子进程异常报告；
    - stop_event：任一进程要求实验停止的信号。

    输出：
    - 机器人报告运动完成时正常返回；
    - 出错、提前停止或超时时抛错。

    实验作用：主程序本身不控制机器人每个采样点，只负责确认运动阶段是否按预期结束。
    如果机器人异常退出但没来得及置位 motion_done，这里也能通过错误、停止信号或超时退出。
    """

    # 本段把“预记录 + 最大运动时间 + 余量”合成主程序等待上限。
    deadline = (
        time.perf_counter()
        + config.PRE_RECORD_SECONDS
        + config.ROBOT_MOTION_TIMEOUT_S
        + 5.0
    )

    # 本段在运动期间持续观察完成信号、错误信号和停止信号。
    while not motion_done.is_set():
        worker_error = _pop_worker_error(error_queue)
        if worker_error:
            raise RuntimeError(worker_error)

        if stop_event.is_set():
            raise RuntimeError("实验在运动完成前收到停止信号。")

        if time.perf_counter() >= deadline:
            raise TimeoutError("主程序等待 motion_done 超时。")

        # 小睡一会儿，避免等待循环占满 CPU。
        time.sleep(0.05)


def _join_or_terminate(process: BaseProcess) -> None:
    """
    完整实验结束时回收一个子进程。

    输入：已经启动过的相机、机器人或写日志子进程。
    输出：该子进程正常退出，或在失去响应时被终止并回收。

    实验作用：优先让子进程执行自己的 finally，释放相机/RTDE/文件句柄；
    terminate 只是最后清理手段，正常流程不会靠它结束硬件连接。
    """

    # 本段先给子进程一个正常收尾窗口，让它自己关闭硬件或文件资源。
    process.join(timeout=config.WORKER_JOIN_TIMEOUT_S)

    # 本段只在子进程失去响应时执行，属于实验收尾的最后保险。
    if process.is_alive():
        print(f"[主程序警告] {process.name} 未按时退出，将终止该子进程。")
        process.terminate()

        # 终止后再 join 一次，尽量把操作系统进程资源回收干净。
        process.join(timeout=2.0)


RELATIVE_MOTION_MODES = {
    "x_line_experiment",
    "xy_line_experiment",
    "xy_l_experiment",
}


def _clear_stop_request_file(path: Path | None) -> None:
    """清理本次 run 专属停止请求文件。"""

    if path is not None and path.exists():
        path.unlink()


def _external_stop_requested(path: Path | None) -> bool:
    """检查启动器为本次 run 写入的停止请求。"""

    return bool(path is not None and path.exists())


def _wait_event_or_error(
    event: Any,
    description: str,
    error_queue: Any,
    stop_event: Any,
    stop_request_path: Path | None,
    *,
    timeout_s: float | None = None,
) -> None:
    """等待一个事件，同时响应 worker 错误和启动器停止请求。"""

    deadline = None if timeout_s is None else time.perf_counter() + timeout_s
    while not event.is_set():
        worker_error = _pop_worker_error(error_queue)
        if worker_error:
            raise RuntimeError(worker_error)
        if _external_stop_requested(stop_request_path):
            stop_event.set()
        if stop_event.is_set():
            worker_error = _pop_worker_error(error_queue)
            if worker_error:
                raise RuntimeError(worker_error)
            raise RuntimeError(f"{description} 期间收到停止请求。")
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError(f"等待 {description} 超时。")
        time.sleep(0.05)


def _sleep_with_stop_checks(
    seconds: float,
    error_queue: Any,
    stop_event: Any,
    stop_request_path: Path | None,
) -> None:
    """带错误和停止请求检查的短等待。"""

    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        worker_error = _pop_worker_error(error_queue)
        if worker_error:
            raise RuntimeError(worker_error)
        if _external_stop_requested(stop_request_path):
            stop_event.set()
        if stop_event.is_set():
            raise RuntimeError("后记录期间收到停止请求。")
        time.sleep(0.05)


def _relative_parameters_from_args(
    arguments: argparse.Namespace,
    selected_mode: str,
) -> dict[str, Any]:
    """把命令行/UI 参数整理给 robot.py 的纯轨迹函数。"""

    default_times = {
        "x_line_experiment": config.ROBOT_EXPERIMENT_DEFAULT_X_LINE_ONE_WAY_TIME_S,
        "xy_line_experiment": config.ROBOT_EXPERIMENT_DEFAULT_XY_LINE_ONE_WAY_TIME_S,
        "xy_l_experiment": config.ROBOT_EXPERIMENT_DEFAULT_L_ONE_WAY_TIME_S,
    }
    one_way_time_s = (
        float(arguments.one_way_time_s)
        if arguments.one_way_time_s is not None
        else float(default_times[selected_mode])
    )

    return {
        "speed_mm_s": arguments.speed_mm_s,
        "one_way_time_s": one_way_time_s,
        "total_time_s": arguments.total_time_s,
        "angle_deg": arguments.angle_deg,
        "x_direction": arguments.x_direction,
        "y_direction": arguments.y_direction,
        "x_speed_mm_s": arguments.x_speed_mm_s,
        "x_one_way_time_s": arguments.x_one_way_time_s,
        "y_speed_mm_s": arguments.y_speed_mm_s,
        "y_one_way_time_s": arguments.y_one_way_time_s,
        "blend_mm": arguments.blend_mm,
        "acceleration_m_s2": config.ROBOT_EXPERIMENT_ACCELERATION_M_S2,
    }


def _update_experiment_parameters_with_camera_fps(run_dir: Path) -> None:
    """相机完成后把实际 fps 回填到 experiment_parameters.json。"""

    parameters_path = run_dir / "experiment_parameters.json"
    metadata_path = run_dir / "camera" / "capture_metadata.json"
    if not parameters_path.exists() or not metadata_path.exists():
        return
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parameters["industrial_camera_actual_fps"] = metadata.get("actual_camera_fps")
    parameters_path.write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


# =============================================================================
# 2. 五种模式的高层入口：用户选一个模式，main.py 就转交给一个模块
# =============================================================================

def run_vision_test_mode() -> Path:
    """
    用户选择 vision_test 后进入这里。

    输入：config.py 中的视觉来源、视觉算法、调试图和坐标输出设置。
    输出：camera.py 创建的 vision_test_时间戳 输出目录。

    实验作用：只验证视觉链路。它会让 camera.py 读取图片/视频/相机帧、逐帧识别、
    保存 vision_results.txt 和抽样 debug 图；不会连接机器人。
    """

    # 本段是“入口转交”。只有用户真的选择 vision_test，才加载 camera.py 的视觉代码。
    from camera import run_vision_test

    # 图像读取、识别、结果保存都在 camera.py；main.py 只拿回输出目录。
    return run_vision_test()


def run_vision_capture_mode() -> Path:
    """
    用户选择 vision_capture 后进入这里。

    这个模式只高速采集海康相机原始 Mono8 帧到 RAW 文件，不做棋盘格识别，不连接机器人。
    """

    from camera import run_vision_capture

    return run_vision_capture()


def run_vision_offline_mode() -> Path:
    """
    用户选择 vision_offline 后进入这里。

    这个模式读取 vision_capture 保存的 RAW 帧，离线逐帧完整执行现有视觉识别。
    """

    from camera import run_vision_offline

    return run_vision_offline()


def run_offline_static_test_mode(arguments: argparse.Namespace) -> Path:
    """机械臂可完全断电；只启动工业相机并测三次静止底噪。"""

    from offline_static_test import run_offline_static_test

    return run_offline_static_test(arguments.stop_request_path)


def run_robot_dry_run_mode() -> Path:
    """
    用户选择 robot_dry_run 后进入这里。

    输入：config.py 中的 A/B/C 位姿、轨迹类型、速度、工作区限制。
    输出：robot.py 创建的 dry-run 报告目录。

    实验作用：只做纯软件轨迹体检。它不会创建 URRobot，不会导入 ur_rtde，
    也不会连接任何机器人 IP，适合实机前反复检查坐标是否写得离谱。
    """

    # 本段是“入口转交”。只有选择机器人相关模式，才加载 robot.py。
    from robot import run_robot_dry_run

    # 轨迹生成、工作区检查和报告保存由 robot.py 完成。
    return run_robot_dry_run()


def run_robot_connection_test_mode() -> Path:
    """只读检测机械臂通信，不创建控制接口、不发送运动命令。"""

    from robot import run_robot_connection_test

    return run_robot_connection_test()


def run_robot_test_mode() -> Path:
    """
    用户选择 robot_test 后进入这里。

    输入：机器人连接参数、轨迹端点、安全确认开关。
    输出：robot.py 创建的机器人测试输出目录。

    实验作用：单独验证 UR 连接和状态读取；当安全开关允许时，才执行低速 A→B 测试运动。
    注意：这个模式可能连接真机，是否允许运动由 ROBOT_TEST_ALLOW_MOTION 等开关共同决定。
    """

    # 本段是“入口转交”。只有明确选择 robot_test，才触碰 robot.py 的真机测试入口。
    from robot import run_robot_test

    return run_robot_test()


def run_analysis_mode(analysis_file: Path | None = None) -> Path:
    """
    用户选择 analyze 后进入这里。

    输入：已有的 run_log.txt 或 vision_results.txt，可由 --analysis-file 指定。
    输出：analyze.py 创建的时域图、频域图、summary/metrics 等分析结果。

    实验作用：把已经保存下来的视觉/机器人记录转成更容易判断振动现象的图和指标。
    它不连接相机，也不连接机器人。
    """

    # 本段是“入口转交”。analysis_file 不传时，analyze.py 会自己寻找最新记录。
    from analyze import run_analysis

    return run_analysis(analysis_file)


def run_relative_motion_experiment_mode(
    selected_mode: str,
    arguments: argparse.Namespace,
) -> Path:
    """三个相对运动实验共用的主进程协调入口。"""

    if selected_mode not in RELATIVE_MOTION_MODES:
        raise ValueError(f"不是相对运动实验模式：{selected_mode}")

    if not config.ROBOT_RELATIVE_MOTION_ENABLED:
        raise PermissionError(
            "ROBOT_RELATIVE_MOTION_ENABLED=False。请在实验室完成通信测试和现场安全确认后再改为 True。"
        )

    parameters = _relative_parameters_from_args(arguments, selected_mode)
    stop_request_path = arguments.stop_request_path
    if stop_request_path is not None and not stop_request_path.is_absolute():
        stop_request_path = (config.PROJECT_DIR / stop_request_path).resolve()
    _clear_stop_request_file(stop_request_path)

    if not arguments.ui_confirmed:
        from robot import require_operator_confirmation

        require_operator_confirmation(f"{selected_mode} 相对运动实验")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUT_ROOT / f"{selected_mode}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "experiment_events.txt"
    summary_path = run_dir / "experiment_summary.json"

    context = mp.get_context("spawn")
    record_queue = context.Queue(maxsize=config.RECORD_QUEUE_MAXSIZE)
    error_queue = context.Queue(maxsize=config.ERROR_QUEUE_MAXSIZE)

    camera_ready = context.Event()
    robot_ready = context.Event()
    capture_start = context.Event()
    vision_recovered = context.Event()
    motion_start = context.Event()
    motion_done = context.Event()
    camera_stop = context.Event()
    camera_finished = context.Event()
    stop_requested = context.Event()

    from camera import relative_motion_raw_camera_worker
    from robot import build_relative_motion_trajectory, relative_motion_robot_worker

    # 纯计算预检只验证 UI/CLI 数字，不连接机器人；真实 A 点会在 robot worker 中重算。
    build_relative_motion_trajectory(
        selected_mode,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        parameters,
    )

    writer_process = context.Process(
        target=record_writer_worker,
        args=(record_queue, events_path),
        name="relative-event-writer",
    )
    camera_process = context.Process(
        target=relative_motion_raw_camera_worker,
        args=(
            record_queue,
            error_queue,
            run_dir,
            stop_requested,
            camera_ready,
            capture_start,
            vision_recovered,
            camera_stop,
            camera_finished,
        ),
        name="relative-raw-camera",
    )
    robot_process = context.Process(
        target=relative_motion_robot_worker,
        args=(
            record_queue,
            error_queue,
            selected_mode,
            parameters,
            run_dir,
            stop_requested,
            robot_ready,
            motion_start,
            motion_done,
        ),
        name="relative-robot",
    )
    processes = [camera_process, robot_process]
    success = False
    failure_message: str | None = None

    writer_process.start()
    record_queue.put(
        {
            "kind": "META",
            "host_ns": time.perf_counter_ns(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": selected_mode,
            "robot_host": config.ROBOT_HOST,
            "stop_request_path": None if stop_request_path is None else str(stop_request_path),
        },
        timeout=1.0,
    )

    try:
        _record_event(record_queue, "hardware_preparation_started", mode=selected_mode)
        print("[状态] 正在准备工业相机", flush=True)
        camera_process.start()
        robot_process.start()

        _wait_event_or_error(
            camera_ready,
            "工业相机 ready",
            error_queue,
            stop_requested,
            stop_request_path,
            timeout_s=config.WORKER_READY_TIMEOUT_S,
        )
        _wait_event_or_error(
            robot_ready,
            "机械臂 ready",
            error_queue,
            stop_requested,
            stop_request_path,
            timeout_s=config.WORKER_READY_TIMEOUT_S,
        )

        capture_start.set()
        _wait_event_or_error(
            vision_recovered,
            "手电筒同步和视野恢复",
            error_queue,
            stop_requested,
            stop_request_path,
            timeout_s=(
                config.BRIGHTNESS_BASELINE_SECONDS
                + config.FLASH_WAIT_TIMEOUT_SECONDS
                + config.FLASH_RECOVERY_TIMEOUT_SECONDS
                + config.VISION_RECOVERY_STABLE_SECONDS
                + 2.0
            ),
        )

        if stop_requested.is_set():
            raise RuntimeError("运动前收到停止请求。")
        motion_start.set()
        _wait_event_or_error(
            motion_done,
            "机械臂运动完成",
            error_queue,
            stop_requested,
            stop_request_path,
            timeout_s=config.ROBOT_MOTION_TIMEOUT_S + 10.0,
        )

        _record_event(record_queue, "camera_post_record_started")
        print("[状态] 工业相机后记录1秒", flush=True)
        _sleep_with_stop_checks(
            float(config.POST_MOTION_RECORD_SECONDS),
            error_queue,
            stop_requested,
            stop_request_path,
        )
        camera_stop.set()
        _wait_event_or_error(
            camera_finished,
            "工业相机封口",
            error_queue,
            stop_requested,
            stop_request_path,
            timeout_s=config.WORKER_JOIN_TIMEOUT_S + 10.0,
        )
        _record_event(record_queue, "experiment_completed")
        success = True
        print("[状态] 实验完成", flush=True)
    except Exception as exc:
        failure_message = f"{type(exc).__name__}: {exc}"
        stop_requested.set()
        camera_stop.set()
        try:
            _record_event(record_queue, "experiment_aborted", message=failure_message)
        except Exception:
            pass
        print("[状态] 实验失败", flush=True)
        raise
    finally:
        stop_requested.set()
        camera_stop.set()
        for process in processes:
            if process.pid is not None:
                _join_or_terminate(process)
        record_queue.put(None, timeout=2.0)
        _join_or_terminate(writer_process)
        record_queue.close()
        error_queue.close()
        _clear_stop_request_file(stop_request_path)

        _update_experiment_parameters_with_camera_fps(run_dir)
        summary = {
            "kind": "RELATIVE_MOTION_EXPERIMENT_SUMMARY",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": selected_mode,
            "success": success,
            "failure_message": failure_message,
            "run_dir": str(run_dir),
            "camera_dir": str(run_dir / "camera"),
            "robot_log": str(run_dir / "robot_log.txt"),
            "events": str(events_path),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )

    return run_dir


class _StatusInbox:
    """Match acknowledgements from a shared multiprocessing status queue."""

    def __init__(self, status_queue: Any) -> None:
        self.queue = status_queue
        self.pending: list[dict[str, Any]] = []

    def wait(
        self,
        source: str,
        event: str,
        error_queue: Any,
        stop_event: Any,
        stop_request_path: Path | None,
        *,
        segment_base: str | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        deadline = time.perf_counter() + timeout_s
        while True:
            for index, item in enumerate(self.pending):
                if (
                    item.get("source") == source
                    and item.get("event") == event
                    and (segment_base is None or item.get("segment_base") == segment_base)
                ):
                    return self.pending.pop(index)
            worker_error = _pop_worker_error(error_queue)
            if worker_error:
                raise RuntimeError(worker_error)
            external_stop = _external_stop_requested(stop_request_path)
            if external_stop:
                stop_event.set()
                raise RuntimeError(f"等待 {source}/{event} 时收到用户停止请求。")
            if stop_event.is_set():
                worker_error = _wait_for_worker_error(error_queue)
                raise RuntimeError(worker_error or f"等待 {source}/{event} 时收到停止请求。")
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"等待 {source}/{event} 超时。")
            try:
                self.pending.append(self.queue.get(timeout=0.05))
            except queue.Empty:
                pass


def _batch_safe_sleep(
    seconds: float,
    error_queue: Any,
    stop_event: Any,
    stop_request_path: Path | None,
) -> None:
    deadline = time.perf_counter() + float(seconds)
    while time.perf_counter() < deadline:
        worker_error = _pop_worker_error(error_queue)
        if worker_error:
            raise RuntimeError(worker_error)
        external_stop = _external_stop_requested(stop_request_path)
        if external_stop:
            stop_event.set()
            raise RuntimeError("批量状态机收到用户停止请求。")
        if stop_event.wait(min(0.05, max(0.0, deadline - time.perf_counter()))):
            worker_error = _wait_for_worker_error(error_queue)
            raise RuntimeError(worker_error or "批量状态机收到停止请求。")


def _batch_segment_meta(
    path: Path,
    batch_id: str,
    row: dict[str, Any],
    segment: dict[str, Any],
    flash: dict[str, Any],
) -> None:
    payload = {
        "kind": "BATCH_SEGMENT_META",
        "batch_id": batch_id,
        "condition_id": row["condition_id"],
        "trajectory_type": row["trajectory_type"],
        "preset_id": row["preset_id"],
        "repeat_index": row["repeat_index"],
        "speed_mm_s": row["speed_mm_s"],
        "one_way_time_s": row["one_way_time_s"],
        "nominal_one_way_distance_mm": row["nominal_one_way_distance_mm"],
        "blend_radius_mm": (
            config.ROBOT_RELATIVE_BLEND_MM if row["trajectory_type"] == "l_shape" else 0.0
        ),
        "corner_mode": "BL01" if row["trajectory_type"] == "l_shape" else "STOP",
        "acceleration_m_s2": config.ROBOT_EXPERIMENT_ACCELERATION_M_S2,
        "flash": flash,
        "segment": segment,
        "vision_processing_status": "PENDING_OFFLINE",
    }
    with path.open("x", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True))


def run_batch_experiment_mode(arguments: argparse.Namespace) -> Path:
    """Run the editable plan with one camera connection and one UR connection."""

    from batch_plan import (
        allocate_batch,
        assert_outputs_available,
        enabled_plan,
        read_plan,
        segment_output_paths,
        segment_stem,
        write_segments_csv,
    )
    from camera import batch_camera_worker
    from robot import batch_robot_worker, build_relative_motion_trajectory

    if not config.ROBOT_RELATIVE_MOTION_ENABLED:
        raise PermissionError("ROBOT_RELATIVE_MOTION_ENABLED=False，批量实验被锁定。")
    if arguments.batch_plan_file is None:
        raise ValueError("batch_experiment 必须提供 --batch-plan-file。")
    if not arguments.ui_confirmed:
        from robot import require_operator_confirmation

        require_operator_confirmation("完整批量实验")

    plan = enabled_plan(read_plan(arguments.batch_plan_file))
    if not plan:
        raise ValueError("当前批量计划没有任何启用行。")
    # Pure numeric preflight: start pose is deliberately synthetic and local; no old A/B/C
    # point is imported or sent to hardware.  The worker repeats checks around the real A.
    synthetic_a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for row in plan:
        if row["trajectory_type"] == "static":
            continue
        build_relative_motion_trajectory(
            str(row["trajectory_type"]),
            synthetic_a,
            {
                "speed_mm_s": row["speed_mm_s"],
                "one_way_time_s": row["one_way_time_s"],
                "angle_deg": row.get("angle_deg", 45.0),
                "x_direction": row.get("x_direction", "+X"),
                "y_direction": row.get("y_direction", "+Y"),
                "blend_mm": config.ROBOT_RELATIVE_BLEND_MM,
                "acceleration_m_s2": config.ROBOT_EXPERIMENT_ACCELERATION_M_S2,
            },
        )
    naming_keys: set[tuple[str, str, int]] = set()
    for row in plan:
        key = (
            str(row["trajectory_type"]),
            str(row["condition_id"]),
            int(row["repeat_index"]),
        )
        if key in naming_keys:
            raise ValueError(
                "计划中有两行会生成相同的轨迹/条件/重复编号文件："
                f"{key}。请修改数值、重复序号或禁用其中一行。"
            )
        naming_keys.add(key)

    stop_request_path = arguments.stop_request_path
    if stop_request_path is not None and not stop_request_path.is_absolute():
        stop_request_path = (config.PROJECT_DIR / stop_request_path).resolve()
    _clear_stop_request_file(stop_request_path)
    batch_id, batch_dir = allocate_batch(config.OUTPUT_ROOT, arguments.batch_manual_label)
    all_paths: dict[int, dict[str, Path]] = {}
    planned_files: set[Path] = set()
    for row in plan:
        paths = segment_output_paths(batch_dir, batch_id, row)
        assert_outputs_available(paths)
        duplicates = planned_files.intersection(paths.values())
        if duplicates:
            duplicate_text = ", ".join(str(path.name) for path in sorted(duplicates))
            raise ValueError(
                "计划中存在会生成同名文件的条件/重复序号，请修改后再运行："
                + duplicate_text
            )
        planned_files.update(paths.values())
        all_paths[int(row["execution_order"])] = paths
    (batch_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    context = mp.get_context("spawn")
    record_queue = context.Queue(maxsize=config.RECORD_QUEUE_MAXSIZE)
    error_queue = context.Queue(maxsize=config.ERROR_QUEUE_MAXSIZE)
    camera_commands = context.Queue(maxsize=16)
    robot_commands = context.Queue(maxsize=16)
    status_queue = context.Queue(maxsize=64)
    stop_event = context.Event()
    writer = context.Process(
        target=segmented_record_writer_worker,
        args=(record_queue, batch_dir),
        name="batch-segment-writer",
    )
    camera = context.Process(
        target=batch_camera_worker,
        args=(record_queue, error_queue, camera_commands, status_queue, stop_event),
        name="batch-continuous-camera",
    )
    robot = context.Process(
        target=batch_robot_worker,
        args=(record_queue, error_queue, robot_commands, status_queue, stop_event),
        name="batch-continuous-robot",
    )
    inbox = _StatusInbox(status_queue)
    segment_rows: list[dict[str, Any]] = []
    current_row: dict[str, Any] | None = None
    flash: dict[str, Any] = {}
    batch_status = "RUNNING"
    failure_message: str | None = None
    robot_start_pose: list[float] | None = None

    writer.start()
    # 先让CB3独占完成Control/Receive初始化，再启动132 fps相机进程。
    # 这样RTDE握手不与相机预览、取流和OpenCV初始化争用CPU/网络调度；
    # 此时机器人只建立连接和读取静止位姿，尚未收到任何运动命令。
    print("[状态] 正在建立UR10控制和状态连接（此阶段不会运动）", flush=True)
    robot.start()
    try:
        robot_ready = inbox.wait(
            "robot", "READY", error_queue, stop_event, stop_request_path,
            timeout_s=config.WORKER_READY_TIMEOUT_S,
        )
        robot_start_pose = robot_ready["start_pose"]
        print("[状态] UR10已连接并确认静止，正在启动工业相机", flush=True)
        camera.start()
        camera_ready = inbox.wait(
            "camera", "READY", error_queue, stop_event, stop_request_path,
            timeout_s=config.WORKER_READY_TIMEOUT_S,
        )
        flash_ready = inbox.wait(
            "camera",
            "FLASH_READY",
            error_queue,
            stop_event,
            stop_request_path,
            timeout_s=(
                config.BRIGHTNESS_BASELINE_SECONDS
                + config.FLASH_WAIT_TIMEOUT_SECONDS
                + config.FLASH_RECOVERY_TIMEOUT_SECONDS
                + config.VISION_RECOVERY_STABLE_SECONDS
                + 2.0
            ),
        )
        flash = {
            "flash_system_ns": int(flash_ready["flash_system_ns"]),
            "flash_system_time": flash_ready["flash_system_time"],
            "flash_frame_id": int(flash_ready["frame_id"]),
            "camera_actual_fps": camera_ready.get("actual_camera_fps"),
        }
        flash_zero_ns = int(flash_ready["flash_system_ns"])
        print(f"[状态] 批次 {batch_id} 开始，硬件在整批期间保持连接", flush=True)

        for current_row in plan:
            paths = all_paths[int(current_row["execution_order"])]
            segment_base = segment_stem(batch_id, current_row)

            camera_commands.put(
                {
                    "action": "OPEN_SEGMENT",
                    "segment_base": segment_base,
                    "video_path": str(paths["video"]),
                    "vision_path": str(paths["vision"]),
                },
                timeout=1.0,
            )
            robot_commands.put(
                {
                    "action": "OPEN_SEGMENT",
                    "segment_base": segment_base,
                    "robot_path": str(paths["robot"]),
                },
                timeout=1.0,
            )
            inbox.wait(
                "camera", "SEGMENT_OPENED", error_queue, stop_event, stop_request_path,
                segment_base=segment_base,
            )
            inbox.wait(
                "robot", "SEGMENT_OPENED", error_queue, stop_event, stop_request_path,
                segment_base=segment_base,
            )
            record_queue.put(
                {
                    "kind": "EVENT",
                    "name": "segment_opened",
                    "host_ns": time.perf_counter_ns(),
                    "segment_base": segment_base,
                },
                timeout=1.0,
            )

            motion_started: dict[str, Any] | None = None
            motion_finished: dict[str, Any] | None = None
            if current_row["trajectory_type"] == "static":
                print("[状态] 正在记录5秒静止基线", flush=True)
                _batch_safe_sleep(
                    config.BATCH_STATIC_BASELINE_SECONDS,
                    error_queue,
                    stop_event,
                    stop_request_path,
                )
            else:
                print("[状态] 当前分段先记录1秒运动前静止画面", flush=True)
                _batch_safe_sleep(
                    config.BATCH_PRE_MOTION_SECONDS,
                    error_queue,
                    stop_event,
                    stop_request_path,
                )
                robot_commands.put(
                    {"action": "RUN_TRAJECTORY", "segment_base": segment_base, "row": current_row},
                    timeout=1.0,
                )
                motion_started = inbox.wait(
                    "robot", "MOTION_STARTED", error_queue, stop_event, stop_request_path,
                    segment_base=segment_base,
                )
                print(
                    f"[状态] 执行计划顺序 {current_row['execution_order']:02d}；"
                    f"本批启用 {len(plan)} 段；"
                    f"{current_row['trajectory_type']} {current_row['condition_id']} "
                    f"R{current_row['repeat_index']:02d}",
                    flush=True,
                )
                motion_finished = inbox.wait(
                    "robot",
                    "MOTION_FINISHED",
                    error_queue,
                    stop_event,
                    stop_request_path,
                    segment_base=segment_base,
                    timeout_s=config.ROBOT_MOTION_TIMEOUT_S + 10.0,
                )
                print("[状态] 已回到A点，继续记录1秒残余振动", flush=True)
                _batch_safe_sleep(
                    config.BATCH_POST_MOTION_SECONDS,
                    error_queue,
                    stop_event,
                    stop_request_path,
                )

            camera_commands.put(
                {"action": "CLOSE_SEGMENT", "segment_base": segment_base}, timeout=1.0
            )
            robot_commands.put(
                {"action": "CLOSE_SEGMENT", "segment_base": segment_base}, timeout=1.0
            )
            camera_closed = inbox.wait(
                "camera", "SEGMENT_CLOSED", error_queue, stop_event, stop_request_path,
                segment_base=segment_base,
            )
            inbox.wait(
                "robot", "SEGMENT_CLOSED", error_queue, stop_event, stop_request_path,
                segment_base=segment_base,
            )
            flash_wall_time = datetime.fromisoformat(str(flash["flash_system_time"]))
            camera_start_ns = int(camera_closed["system_start_ns"])
            camera_end_ns = int(camera_closed["system_end_ns"])
            system_start_time = (
                flash_wall_time
                + timedelta(seconds=(camera_start_ns - flash_zero_ns) / 1_000_000_000.0)
            ).isoformat(timespec="milliseconds")
            system_end_time = (
                flash_wall_time
                + timedelta(seconds=(camera_end_ns - flash_zero_ns) / 1_000_000_000.0)
            ).isoformat(timespec="milliseconds")
            actual_start_pose = (
                motion_started["actual_start_pose"] if motion_started else robot_start_pose
            )
            actual_end_pose = (
                motion_finished["actual_end_pose"] if motion_finished else robot_start_pose
            )
            segment = {
                "batch_id": batch_id,
                "execution_order": current_row["execution_order"],
                "condition_id": current_row["condition_id"],
                "trajectory_type": current_row["trajectory_type"],
                "preset_id": current_row["preset_id"],
                "repeat_index": current_row["repeat_index"],
                "speed_mm_s": current_row["speed_mm_s"],
                "one_way_time_s": current_row["one_way_time_s"],
                "nominal_one_way_distance_mm": current_row["nominal_one_way_distance_mm"],
                "actual_start_pose": actual_start_pose,
                "actual_end_pose": actual_end_pose,
                "system_start_time": system_start_time,
                "system_end_time": system_end_time,
                "batch_elapsed_start_s": (
                    camera_start_ns - flash_zero_ns
                ) / 1_000_000_000.0,
                "batch_elapsed_end_s": (
                    camera_end_ns - flash_zero_ns
                ) / 1_000_000_000.0,
                "first_frame_id": camera_closed["first_frame_id"],
                "last_frame_id": camera_closed["last_frame_id"],
                "flash_system_time": flash["flash_system_time"],
                "status": "COMPLETED",
                "output_video": str(paths["video"]),
                "manual_label": current_row["manual_label"],
            }
            segment_rows.append(segment)
            _batch_segment_meta(paths["meta"], batch_id, current_row, segment, flash)
            write_segments_csv(batch_dir / "segments.csv", segment_rows)

        batch_status = "COMPLETED"
        print("[状态] 批量运动与录像采集完成，机械臂停在A点", flush=True)
    except Exception as exc:
        failure_message = f"{type(exc).__name__}: {exc}"
        batch_status = "ABORTED" if _external_stop_requested(stop_request_path) else "FAILED"
        stop_event.set()
        completed_orders = {int(item["execution_order"]) for item in segment_rows}
        for row in plan:
            if int(row["execution_order"]) in completed_orders:
                continue
            paths = all_paths[int(row["execution_order"])]
            status = (
                "FAILED"
                if current_row is not None
                and int(row["execution_order"]) == int(current_row["execution_order"])
                else "ABORTED"
            )
            segment_rows.append(
                {
                    "batch_id": batch_id,
                    **{key: row.get(key) for key in (
                        "execution_order", "condition_id", "trajectory_type", "preset_id",
                        "repeat_index", "speed_mm_s", "one_way_time_s",
                        "nominal_one_way_distance_mm", "manual_label",
                    )},
                    "actual_start_pose": robot_start_pose,
                    "actual_end_pose": None,
                    "system_start_time": "",
                    "system_end_time": "",
                    "batch_elapsed_start_s": "",
                    "batch_elapsed_end_s": "",
                    "first_frame_id": "",
                    "last_frame_id": "",
                    "flash_system_time": flash.get("flash_system_time", ""),
                    "status": status,
                    "output_video": str(paths["video"]),
                }
            )
        write_segments_csv(batch_dir / "segments.csv", segment_rows)
        print(f"[状态] 批量实验{batch_status}：{failure_message}", flush=True)
        raise
    finally:
        stop_event.set()
        for command_queue in (camera_commands, robot_commands):
            try:
                command_queue.put({"action": "SHUTDOWN"}, timeout=0.2)
            except Exception:
                pass
        for process in (camera, robot):
            if process.pid is not None:
                _join_or_terminate(process)
        try:
            record_queue.put(None, timeout=2.0)
        except Exception:
            pass
        _join_or_terminate(writer)
        summary = {
            "kind": "BATCH_SUMMARY",
            "batch_id": batch_id,
            "status": batch_status,
            "failure_message": failure_message,
            "manual_label": arguments.batch_manual_label,
            "flash": flash,
            "sony_target_file": f"{batch_id}_SONY.mp4",
            "segment_count": len(plan),
            "completed_count": sum(item.get("status") == "COMPLETED" for item in segment_rows),
            "camera_restarted_between_segments": False,
            "vision_processing_status": (
                "PENDING_OFFLINE" if batch_status == "COMPLETED" else "NOT_STARTED"
            ),
        }
        (batch_dir / "batch_META.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _clear_stop_request_file(stop_request_path)

    return batch_dir


def run_micro_closed_loop_mode(arguments: argparse.Namespace) -> Path:
    """
    一键执行 XY 多目标视觉闭环逼近测试。

    输入：命令行参数（--ui-confirmed、--stop-request-path）。
    输出：本次运行目录 Path。

    流程：前置门禁 → 设备检查与预热 → 轴向/符号探针 → 5 s 静止基线
    → 建立统一视觉原点 → X/Y 各 6 个累计目标逐个闭环逼近 → 回到初始位姿
    → 汇总。每次迭代 RAW 分析落盘后立即删除，结束时还有兜底清理。

    与既有模式最大的结构差别：这里**父进程自己就是视觉处理进程**。
    迭代必须等测量结果才能决定下一步命令，再开第三个进程只增加 IPC 复杂度，
    没有任何并行收益。相机子进程只负责"取帧 + memcpy 进内存"，识别全部在父进程做完。

    关于手电筒门禁：本模式有意不构造 camera._FlashGate（batch_camera_worker 会因它
    硬抛"手电筒门禁未完成"）。闭环的时间对齐由全机共享的 perf_counter_ns 保证，
    不依赖操作者打手电筒。这是设计决定，不是安全回归。
    """

    import itertools
    import math

    import numpy as np

    import micro_closed_loop as mcl

    # -------------------------------------------------------------------------
    # 前置门禁：必须在创建任何目录之前全部通过
    # -------------------------------------------------------------------------
    if not config.ROBOT_RELATIVE_MOTION_ENABLED:
        raise PermissionError(
            "ROBOT_RELATIVE_MOTION_ENABLED=False，XY 微动闭环能力测试被锁定。"
            "本测试会反复发送真实运动命令，必须在确认真机安全后才解锁。"
        )
    if config.CONTROL_MODE == "sfc":
        raise ValueError(
            "当前版本只预留了 SFC 接口，尚未实现在线控制。为避免把开环运动误当成 SFC，"
            "XY 微动闭环测试拒绝启动。"
        )
    if config.VISION_METHOD != "checkerboard":
        raise ValueError(
            f"XY 微动闭环依赖棋盘格亚像素链路，但 VISION_METHOD={config.VISION_METHOD!r}。"
        )

    stop_request_path = arguments.stop_request_path
    if stop_request_path is not None and not stop_request_path.is_absolute():
        stop_request_path = (config.PROJECT_DIR / stop_request_path).resolve()
    _clear_stop_request_file(stop_request_path)

    # 磁盘检查在创建目录**之前**：空间不够时连目录都不应该建出来。
    free_gb = mcl.free_gb(config.OUTPUT_ROOT)
    if free_gb < float(config.MICRO_LOOP_MIN_FREE_GB_START):
        raise RuntimeError(
            f"输出卷剩余 {free_gb:.2f} GiB，低于启动门槛 "
            f"{config.MICRO_LOOP_MIN_FREE_GB_START:g} GiB。"
            "微动闭环每轮都要落一段全幅未压缩临时 RAW，空间不足时拒绝开始。"
        )

    if not arguments.ui_confirmed:
        from robot import require_operator_confirmation

        require_operator_confirmation("XY 微动闭环能力测试（会反复发送真实微动命令）")

    # -------------------------------------------------------------------------
    # 预算：时长、体积、内存。全部在创建任何目录之前算完并打印。
    # -------------------------------------------------------------------------
    plan = mcl.estimate_multi_target_run_plan()
    budget_lines = mcl.format_multi_target_budget_lines(plan)
    for line in budget_lines:
        print(line, flush=True)

    # 原始 RAW 写到独立卷（D:），CSV/JSON/汇总留在工程 outputs/。
    # 分开的理由是量级：RAW 是几十 GB，汇总只有几百 KB；放同一个卷会让
    # outputs/ 被几十 GB 二进制淹没，而 RAW 的保留策略也可能需要单独调整。
    raw_root = Path(config.MICRO_LOOP_RAW_ROOT)
    raw_free_gb = mcl.free_gb(raw_root if raw_root.exists() else raw_root.parent)
    # 每组汇总落盘后立即删该组视频，所以门禁按最大单组驻盘量 + 余量计算。
    # 整轮累计生成量仍打印供知情，但不会同时占在磁盘上。
    need_gb = plan["peak_group_bytes_worst"] / (1024.0 ** 3) + float(
        config.MICRO_LOOP_KEEP_ALL_RESERVE_GB
    )
    if raw_free_gb < need_gb:
        raise RuntimeError(
            f"原始 RAW 目标卷 {raw_root} 剩余 {raw_free_gb:.2f} GiB，"
            f"低于最大单组所需 {need_gb:.2f} GiB（最大单组 "
            f"{mcl.format_gib(int(plan['peak_group_bytes_worst']))} + 余量 "
            f"{config.MICRO_LOOP_KEEP_ALL_RESERVE_GB:g} GiB）。"
            "空间不足时在动手之前就拒绝开始。"
        )

    ram_gb = mcl.available_ram_gb()
    if math.isfinite(ram_gb) and ram_gb < float(config.MICRO_LOOP_MIN_FREE_RAM_GB):
        raise RuntimeError(
            f"可用物理内存只剩 {ram_gb:.2f} GiB，低于门槛 "
            f"{config.MICRO_LOOP_MIN_FREE_RAM_GB:g} GiB。"
            "相机缓冲要按最坏窗口一次性驻留内存（未压缩 Mono8），"
            "内存不足会在录制中途丢帧，所以拒绝开始。"
        )
    print(
        f"原始 RAW 目标卷 {raw_root} 剩余 {raw_free_gb:.2f} GiB（最大单组需 {need_gb:.2f} GiB）｜ "
        f"可用内存 {ram_gb:.2f} GiB"
        if math.isfinite(ram_gb)
        else f"原始 RAW 目标卷 {raw_root} 剩余 {raw_free_gb:.2f} GiB（最大单组需 {need_gb:.2f} GiB）｜ "
        "可用内存取不到（按未知处理，不当作充足）",
        flush=True,
    )

    run_id, run_dir = mcl.allocate_closed_loop_run(config.OUTPUT_ROOT)
    # 全部的 RAW 临时区与归档区都在 RAW 卷上，与运行目录分卷；
    # 同卷内 os.replace 才是原子的，而回收目录与临时目录必须同卷。
    raw_run_dir = raw_root / run_id
    temp_dir, trash_dir = mcl.prepare_temp_workspace(raw_run_dir)
    log_path = run_dir / "experiment_log.txt"
    keep_all_raw = bool(config.MICRO_LOOP_KEEP_ALL_RAW)

    def require_ram(where: str) -> None:
        """在开录之前再查一次内存；相机就绪后帧率已知，这次是最准的。"""

        free = mcl.available_ram_gb()
        if math.isfinite(free) and free < float(config.MICRO_LOOP_MIN_FREE_RAM_GB):
            raise RuntimeError(
                f"可用物理内存只剩 {free:.2f} GiB（{where}），低于门槛 "
                f"{config.MICRO_LOOP_MIN_FREE_RAM_GB:g} GiB，已安全终止。"
            )

    def log(text: str, *, status: bool = False) -> None:
        """同时打到 stdout（供启动器镜像到状态栏）和实验日志。"""

        print(f"[状态] {text}" if status else text, flush=True)
        mcl.append_log(log_path, text)

    def cleanup_raw_group(block_id: str, description: str, *, quiet: bool = False) -> None:
        """结果已落盘后删除这一组的视频载荷；删除失败则停止继续累积。"""

        if not bool(config.MICRO_LOOP_DELETE_VIDEO_FILES_AFTER_GROUP):
            return
        released, failed = mcl.delete_video_payloads(raw_run_dir / block_id)
        if failed:
            raise RuntimeError(
                f"{description}结束后清理视频失败，仍有 {len(failed)} 个文件被占用："
                + "，".join(str(path) for path in failed[:3])
                + "。为防止后续实验继续占满磁盘，已停止。"
            )
        if not quiet:
            log(
                f"{description}结果已落盘；该组 RAW/视频已删除，释放 "
                f"{mcl.format_gib(released)}。",
                status=True,
            )

    mcl.write_json(
        run_dir / "config.json",
        {
            "kind": "MICRO_CLOSED_LOOP_CONFIG",
            "run_id": run_id,
            "axes": list(config.MICRO_LOOP_AXES),
            "experiment": "XY_MULTI_TARGET_VISUAL_CLOSED_LOOP",
            "relative_targets_um": list(config.MICRO_TARGET_RELATIVE_STEPS_UM),
            "absolute_targets_um": list(config.MICRO_TARGET_ABSOLUTE_UM),
            "control_law": "command = error",
            "max_command_um": config.MICRO_TARGET_MAX_COMMAND_UM,
            "min_command_um": config.MICRO_LOOP_MIN_COMMAND_UM,
            "max_iter": config.MICRO_TARGET_MAX_ITER,
            "stable_tolerance_um": config.MICRO_TARGET_STABLE_TOL_UM,
            "stable_count": config.MICRO_TARGET_STABLE_COUNT,
            "small_command_um": config.MICRO_TARGET_SMALL_COMMAND_UM,
            "pre_seconds": config.MICRO_LOOP_PRE_SECONDS,
            "post_seconds": config.MICRO_LOOP_POST_SECONDS,
            "tail_window_s": config.MICRO_LOOP_TAIL_WINDOW_S,
            "max_window_seconds": config.MICRO_LOOP_MAX_WINDOW_SECONDS,
            "speed_mm_s": config.MICRO_LOOP_SPEED_MM_S,
            "acceleration_m_s2": config.MICRO_LOOP_ACCELERATION_M_S2,
            "limit_cycle_window": config.MICRO_TARGET_LIMIT_WINDOW,
            "limit_cycle_min_sign_changes": config.MICRO_TARGET_LIMIT_MIN_SIGN_CHANGES,
            "matrix_max_condition": config.MICRO_TARGET_MATRIX_MAX_CONDITION,
            "local_range_um": config.MICRO_TARGET_LOCAL_RANGE_UM,
            "probe_um": config.MICRO_LOOP_PROBE_UM,
            "probe_remeasure_attempts": config.MICRO_LOOP_PROBE_REMEASURE_ATTEMPTS,
            "probe_reference_frames": config.MICRO_LOOP_PROBE_REFERENCE_FRAMES,
            "probe_measure_frames": config.MICRO_LOOP_PROBE_MEASURE_FRAMES,
            "probe_tail_window_s": config.MICRO_LOOP_PROBE_TAIL_WINDOW_S,
            "fixed_safety_radius_m": config.MICRO_LOOP_FIXED_SAFETY_RADIUS_M,
            "delete_video_files_after_group": config.MICRO_LOOP_DELETE_VIDEO_FILES_AFTER_GROUP,
            "delete_video_files_on_exit": config.MICRO_LOOP_DELETE_VIDEO_FILES_ON_EXIT,
            "gain_min": config.MICRO_LOOP_GAIN_MIN,
            "gain_max": config.MICRO_LOOP_GAIN_MAX,
            "process_every_n_frames": config.MICRO_LOOP_PROCESS_EVERY_N_FRAMES,
            "temp_dir": str(temp_dir),
            "note": (
                "每次迭代在同一活动窗口内取得 before 快照、发送命令并取得 after；"
                "迭代结果落盘后立即删除其未压缩 RAW；"
                "正常结束、用户停止或异常退出时再做整轮兜底清理。"
            ),
        },
    )
    log(f"输出目录 {run_dir}")

    context = mp.get_context("spawn")
    error_queue = context.Queue(maxsize=config.ERROR_QUEUE_MAXSIZE)
    camera_commands = context.Queue(maxsize=16)
    robot_commands = context.Queue(maxsize=16)
    status_queue = context.Queue(maxsize=256)
    stop_event = context.Event()

    camera = context.Process(
        target=mcl.micro_closed_loop_camera_worker,
        args=(error_queue, camera_commands, status_queue, stop_event),
        name="micro-closed-loop-camera",
    )
    robot = context.Process(
        target=mcl.micro_closed_loop_robot_worker,
        args=(error_queue, robot_commands, status_queue, stop_event, run_dir),
        name="micro-closed-loop-robot",
    )
    inbox = _StatusInbox(status_queue)

    every_n = int(config.MICRO_LOOP_PROCESS_EVERY_N_FRAMES)
    clip_counter = itertools.count(1)
    master_rows: list[dict[str, Any]] = []
    temp_bytes_peak = 0
    fps = float(config.EXPECTED_VISION_FPS or 132.0)
    capacity_frames = 0
    camera_ready: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # 录制窗口原语
    # -------------------------------------------------------------------------

    def ensure_continue_disk() -> float:
        """每个窗口创建之前检查可用空间；不足时安全终止，已完成数据全部保留。"""

        free = mcl.free_gb(raw_run_dir)
        if free < float(config.MICRO_LOOP_MIN_FREE_GB_CONTINUE):
            raise RuntimeError(
                f"原始 RAW 卷剩余 {free:.2f} GiB，低于续跑门槛 "
                f"{config.MICRO_LOOP_MIN_FREE_GB_CONTINUE:g} GiB，已安全终止。"
            )
        return free

    # -------------------------------------------------------------------------
    # 时长观测：只提示，不中止
    # -------------------------------------------------------------------------

    class TimeWatch:
        """
        记录每段窗口的真实耗时，并在明显超出预计时打一条提示。

        **它不会拒绝任何一段窗口，也不会让实验提前结束。**

        上一版这里是一个带 deadline 的"准入控制"：每开一段前用已实测的平均
        段耗时外推总时长，投影超过预算就抛异常安全收尾。那个策略已按操作者
        要求删除，原因是它会产生一种不可接受的结果——机器人、相机、磁盘
        全都正常，但因为"到 10 分钟了"，X 跑完之后 Y 的闭环一个都没跑。

        保留这个类是因为**观测本身有价值**：实测平均段耗时写进日志和
        final_report，超预计时打一条提示，操作者能看见"今天比平时慢"，
        而不会因此丢掉任何实验步骤。中止只由真实故障触发（通信、相机、
        磁盘、内存、工作空间、异常漂移、方向异常），与时长无关。
        """

        def __init__(self, total_windows: int, *, notice_s: float) -> None:
            self.total_windows = max(1, int(total_windows))
            self.notice_s = float(notice_s)
            self.started_ns = time.perf_counter_ns()
            self.done = 0
            self.cost_ns = 0
            self.noticed = False
            self.reason = ""

        @property
        def elapsed_s(self) -> float:
            return (time.perf_counter_ns() - self.started_ns) / 1e9

        @property
        def mean_window_s(self) -> float:
            return float("nan") if self.done == 0 else (self.cost_ns / self.done) / 1e9

        def projected_total_s(self) -> float:
            """按已实测的平均段耗时外推的总时长；还没有样本时返回 nan。"""

            if self.done == 0:
                return float("nan")
            return self.elapsed_s + self.mean_window_s * (self.total_windows - self.done)

        def observe(self, label: str) -> None:
            """
            开一段窗口**之前**记一次进度。永远不阻止这一段开录。

            超过提示线时只打一次日志：重复打会把状态栏刷满，反而盖住
            真正需要看的异常信息。
            """

            if self.noticed or self.done == 0:
                return
            projected = self.projected_total_s()
            if math.isfinite(projected) and projected > self.notice_s:
                self.noticed = True
                self.reason = (
                    f"当前实测平均每段 {self.mean_window_s:.2f} s，"
                    f"按此速度全轮约 {projected / 60:.1f} 分钟，超过提示线 "
                    f"{self.notice_s / 60:.1f} 分钟。**继续执行全部实验**，"
                    "不跳过任何目标。"
                )
                log(f"[状态] 时长提示（不影响执行）：{self.reason}", status=True)

        def charge(self, seconds: float) -> None:
            """记一段窗口的实际耗时。"""

            self.done += 1
            self.cost_ns += int(max(0.0, float(seconds)) * 1e9)

    # 外推用**正常**段数而不是最坏段数：提示线不是截止线，误报的代价
    # （状态栏刷屏、操作者以为出了问题）比漏报大。用最坏段数会让正常一轮
    # 也几乎必然触发提示。
    watch = TimeWatch(
        int(plan["windows_normal"]), notice_s=float(config.MICRO_LOOP_TIME_NOTICE_S)
    )

    def open_window(label: str) -> str:
        """开一个录制窗口，返回本段文件名主干。"""

        nonlocal temp_bytes_peak
        ensure_continue_disk()
        require_ram(f"开录「{label}」之前")
        window_started_ns = time.perf_counter_ns()
        stem = mcl.next_clip_stem(run_id, label, next(clip_counter))
        camera_commands.put(
            {
                "action": "OPEN_WINDOW",
                "temp_dir": str(temp_dir),
                "stem": stem,
                "clip_label": label,
                "capacity_frames": capacity_frames,
            }
        )
        inbox.wait(
            "camera", "WINDOW_OPENED", error_queue, stop_event, stop_request_path,
            timeout_s=config.WORKER_READY_TIMEOUT_S,
        )
        temp_bytes_peak = max(
            temp_bytes_peak,
            capacity_frames
            * int(camera_ready["full_width"])
            * int(camera_ready["full_height"]),
        )
        open_window.started_ns = window_started_ns
        return stem

    open_window.started_ns = 0

    def close_window() -> dict[str, Any]:
        """关窗口，等相机把内存缓冲落盘成未压缩 RAW。"""

        camera_commands.put({"action": "CLOSE_WINDOW"})
        saved = inbox.wait(
            "camera", "WINDOW_SAVED", error_queue, stop_event, stop_request_path,
            timeout_s=180.0,
        )
        # 一段窗口的实际成本在段结束时入账。它既喂给时间预算的外推，
        # 也是"这一步到底慢在哪"的唯一实测来源。
        watch.charge((time.perf_counter_ns() - int(open_window.started_ns)) / 1e9)
        return saved

    def snapshot_active_window(label: str) -> dict[str, Any]:
        """复制活动命令窗口中最后 16 帧；原窗口保持打开并继续记录。"""

        stem = mcl.next_clip_stem(run_id, f"{label}_before", next(clip_counter))
        camera_commands.put(
            {
                "action": "SNAPSHOT_WINDOW",
                "temp_dir": str(temp_dir),
                "stem": stem,
                "clip_label": f"{label}_before",
                "frame_count": int(config.MICRO_LOOP_MEASURE_FRAMES),
            }
        )
        saved = inbox.wait(
            "camera",
            "WINDOW_SNAPSHOT",
            error_queue,
            stop_event,
            stop_request_path,
            timeout_s=30.0,
        )
        saved["stem"] = stem
        return saved

    def capture_window(
        label: str,
        *,
        command: dict[str, Any] | None,
        pre_seconds: float | None = None,
        post_seconds: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """
        录一段连续窗口：前段 → (可选)一条机器人命令 → 后段 → 封口。

        输入：标签、可选的机器人命令、可选的前后段时长（默认取配置）。
        输出：(相机 WINDOW_SAVED 消息, 机器人 MOVE_DONE 消息或 None)。

        一次迭代只录**一段**连续视频，而不是运动前后各一段——
        这样过冲与稳定过程的瞬态才完整留在片子里。
        """

        # 只是记一次进度，**不阻止开录**：时长不是中止条件。
        # 上一版这里是一个会抛 TimeBudgetExceeded 的准入判断，已删除。
        watch.observe(label)
        stem = open_window(label)
        pre = float(config.MICRO_LOOP_PRE_SECONDS if pre_seconds is None else pre_seconds)
        post = float(config.MICRO_LOOP_POST_SECONDS if post_seconds is None else post_seconds)
        _batch_safe_sleep(pre, error_queue, stop_event, stop_request_path)

        move_result: dict[str, Any] | None = None
        if command is not None:
            robot_commands.put(command)
            # 父进程的等待上限比机器人进程内部各段超时之和更宽松：
            # 真正的边界由机器人进程自己收口，这里只防止 IPC 永久挂死。
            move_result = inbox.wait(
                "robot", "MOVE_DONE", error_queue, stop_event, stop_request_path,
                timeout_s=30.0,
            )
        else:
            _batch_safe_sleep(
                float(config.MICRO_LOOP_MOTION_SETTLE_SECONDS),
                error_queue, stop_event, stop_request_path,
            )

        _batch_safe_sleep(post, error_queue, stop_event, stop_request_path)
        saved = close_window()
        saved["stem"] = stem
        return saved, move_result

    def read_window(
        stem: str,
        meter: mcl.GroupVisionMeter,
        *,
        vision_axis: str,
        iteration_id: str = "",
        prev_measured_um: float | None = None,
        vision_to_axis_sign: float | None = None,
        tail_frames: int | None = None,
        tail_window_s: float | None = None,
    ) -> tuple[mcl.ClipMeasurement, mcl.RawClip, list[Path]]:
        """
        打开临时 RAW 逐帧测量，返回 (测量, 片段, 待回收的临时文件路径)。

        只分析**窗口末尾** MICRO_LOOP_MEASURE_FRAMES 帧：识别一帧全幅实测
        0.165 s，"分析多少帧"几乎就是"这一步花多久"，而机械等待只占约 4%。
        闭环要的是运动停稳之后的位置，它就在窗口末尾；窗口照录不误，
        RAW 全量保留，前面的瞬态帧留给事后离线复查。

        vision_axis 必须由调用方按探针结果给出。写死 x 会让 Y 的整条链
        都在测另一条轴的漂移，而且看起来完全正常。
        """

        paths = mcl.clip_temp_paths(temp_dir, stem)
        clip = mcl.open_raw_clip(paths["raw"])
        measurement = meter.measure_clip(
            clip,
            every_n=every_n,
            iteration_id=iteration_id,
            prev_measured_um=prev_measured_um,
            vision_to_axis_sign=vision_to_axis_sign,
            tail_frames=(
                int(config.MICRO_LOOP_MEASURE_FRAMES) if tail_frames is None else tail_frames
            ),
            tail_window_s=tail_window_s,
            vision_axis=vision_axis,
        )
        return measurement, clip, [paths["raw"], paths["meta"], paths["frames"]]

    def release(
        clip: mcl.RawClip | None,
        paths: list[Path],
        *,
        block_id: str | None = None,
        keep: bool | None = None,
    ) -> None:
        """
        释放 memmap 句柄，然后按保留策略归档或回收这段 RAW。

        输入：片段对象、临时文件路径、实验块名、是否保留（None = 本轮全局策略）。

        当前组内先归档完整 RAW，保证在线分析、证据帧与汇总都已完成；调用方在
        该组结果落盘后通过 cleanup_raw_group 删除视频载荷，再进入下一组。
        归档用同卷 os.replace，是原子的且瞬间完成。
        """

        if clip is not None:
            del clip
        if not paths:
            return

        should_keep = keep_all_raw if keep is None else bool(keep)
        if should_keep and block_id:
            target = raw_run_dir / str(block_id)
            target.mkdir(parents=True, exist_ok=True)
            for path in paths:
                path = Path(path)
                if path.exists():
                    try:
                        os.replace(str(path), str(target / path.name))
                    except OSError as exc:
                        log(f"归档 {path.name} 失败（数据仍在临时目录）：{exc}")
            return

        mcl.drop_temp_files(paths, trash_dir, log=lambda text: log(text))

    # -------------------------------------------------------------------------
    # 共用的位姿换算与轴向漂移守卫
    # -------------------------------------------------------------------------

    def tcp_xy(tcp: Any) -> tuple[float, float, float] | None:
        """从 6 维 TCP 位姿里取出平移分量（m）。"""

        if tcp is None:
            return None
        try:
            values = [float(value) for value in tcp]
        except (TypeError, ValueError):
            return None
        if len(values) < 3:
            return None
        return values[0], values[1], values[2]

    def axis_drift_note(move: dict[str, Any], *, axis: str, drift: dict[str, Any]) -> str:
        """
        每次修正后的全局漂移守卫，返回要追加的告警文字（无异常时为空）。

        输入：机器人 MOVE_DONE 消息、轴名、该轴自己的漂移基准字典（被就地更新）。
        输出：预警或中止说明文字；空串表示正常。

        为什么基准是**轴向起始位姿**而不是上一段、也不是每个目标各自起算：
        robot.validate_trajectory 的 ±0.20 m 相对工作区在每次调用时都以当时
        位姿重新锚定（robot.py:215-220），所以缓慢的棘轮式走位在它眼里永远
        合法、结构上察觉不到。这里独立地拿一条轴的起点做基准就能看见。

        为什么不按目标幅值给允许量（上一版的写法）：那样 −A 目标的允许量
        只有 A，而机器人合法地走到 −A 就已经用满了额度，第一个负方向目标
        的第一次迭代就会被判"疑似棘轮式走位"而中止。**这是上一版的一个真实
        缺陷**，它会让每一个负方向目标在第一步就死掉。

        正确的不变量：`seq` 与 `alt` 两种开环模式都以净位移 ≈0 结束
        （+3/−3 与交替），闭环的每个目标最后也被纠回参考零点附近，所以
        **一条轴从头到尾的净位移应当始终很小**。于是用轴向全局限值，
        与瞬时的目标幅值无关。

        开环与闭环共用这一份实现是刻意的：两边都要防同一个棘轮式走位，
        两份拷贝迟早会漂移成两个不同的判据。
        """

        before = tcp_xy(move.get("tcp_before"))
        after = tcp_xy(move.get("tcp_after"))
        if drift.get("start") is None and before is not None:
            drift["start"] = list(before)
        origin = drift.get("start")
        if origin is None or after is None:
            return ""

        drift_um = math.dist(after, origin) * 1e6
        drift["max_um"] = max(float(drift.get("max_um", 0.0)), drift_um)
        warn_um = float(config.MICRO_LOOP_AXIS_DRIFT_WARN_UM)
        abort_um = float(config.MICRO_LOOP_AXIS_DRIFT_ABORT_UM)
        if drift_um > abort_um:
            return (
                f"{axis} 轴向累计漂移 {drift_um:.1f} μm 已超过 "
                f"{abort_um:g} μm 上限（seq/alt 与闭环都应当净位移≈0），"
                "疑似棘轮式走位"
            )
        if drift_um > warn_um and not drift.get("warned"):
            drift["warned"] = True
            log(
                f"[状态] 提示：{axis} 轴向累计漂移已达 {drift_um:.1f} μm"
                f"（警戒线 {warn_um:g} μm）",
                status=True,
            )
        return ""

    # -------------------------------------------------------------------------
    # 静止基线
    # -------------------------------------------------------------------------

    def run_static_baseline() -> dict[str, Any]:
        """5 s 全分辨率静止基线：确认去掉 MJPG 之后视觉链路的噪声底。"""

        log("静止基线开始：5 s 全分辨率未压缩录制，不裁剪、不缩放", status=True)
        saved, _ = capture_window(
            "static",
            command=None,
            pre_seconds=float(config.MICRO_LOOP_STATIC_SECONDS),
            post_seconds=0.0,
        )
        paths = mcl.clip_temp_paths(temp_dir, saved["stem"])
        clip = mcl.open_raw_clip(paths["raw"])
        block_id = "00_static"
        try:
            meter = mcl.GroupVisionMeter(label="static")
            # 静止基线要的是"整段 5 s 里画面抖了多少"，所以抽帧**跨满整段**，
            # 而不是只看尾部——只看尾部等于把 4.7 s 的记录扔掉，
            # 噪声底会被系统性低估。被抽掉的帧仍在保留的 RAW 里。
            static_every_n = mcl.spanning_every_n(
                clip.frame_count, int(config.MICRO_LOOP_STATIC_ANALYZE_FRAMES)
            )
            primed = meter.prime(
                clip,
                every_n=static_every_n,
                # every_n 已经把整段均匀压到约 40 帧；若再传 frames=40，
                # 会先截取末尾 40 帧再每 18 帧抽一次，最终只剩 3 帧。
                frames=None,
            )
            measurement = meter.measure_clip(
                clip,
                every_n=static_every_n,
                iteration_id="static",
                tail_frames=None,
                # 这里的轴选择不影响结果：analyze_static_clip 直接从逐帧行的
                # checker_dx_mm / checker_dy_mm 各算一份统计，两条轴都算。
                vision_axis="x",
            )
            summary = mcl.analyze_static_clip(
                clip,
                measurement,
                fps=fps,
                record_start_ns=int(saved["record_start_ns"]),
                record_stop_ns=int(saved["record_stop_ns"]),
            )
            summary.update(
                {
                    "reference_frame_index": primed["reference_frame_index"],
                    "ref_found_by": primed["ref_found_by"],
                    "camera_actual_fps": fps,
                    "recorded_frames": saved["frame_count"],
                    "truncated": bool(saved.get("truncated")),
                    "analyze_every_n": static_every_n,
                    "analyze_frames": len(measurement.frame_rows),
                    "raw_archive": str(raw_run_dir / block_id),
                    "raw_retained": not bool(
                        config.MICRO_LOOP_DELETE_VIDEO_FILES_AFTER_GROUP
                    ),
                    "note": (
                        "完整 5 s 未压缩 RAW 用于在线分析，本组结果落盘后按配置"
                        "立即删除；本 JSON、逐帧接受/拒绝记录和证据 PNG 保留。"
                    ),
                }
            )
            # 最终 summary 要的两个核心数：静止噪声底与静止 RMS。
            # 放在顶层而不是埋在 x/y 子字典里，是因为 GPT 端要直接读。
            for name, key in (("static_sigma_um", "mad_sigma_um"), ("static_rms_um", "std_um")):
                per_axis = [
                    float(summary[axis][key])
                    for axis in ("x", "y")
                    if summary[axis][key] is not None
                ]
                summary[name] = max(per_axis) if per_axis else None
            summary["static_peak_to_peak_um"] = max(
                float(summary[axis]["peak_to_peak_um"])
                for axis in ("x", "y")
                if summary[axis]["peak_to_peak_um"] is not None
            ) if all(summary[axis]["peak_to_peak_um"] is not None for axis in ("x", "y")) else None

            static_dir = run_dir / "static"
            mcl.write_rows_csv(
                static_dir / "static_frames.csv",
                measurement.frame_rows,
                mcl.STATIC_FRAME_COLUMNS,
            )
            mcl.write_json(static_dir / "static_summary.json", summary)

            rows = measurement.frame_rows
            picks = [
                rows[0]["frame_index"] if rows else None,
                rows[len(rows) // 2]["frame_index"] if rows else None,
                mcl.last_accepted_frame_index(measurement),
            ]
            for index, frame_index in enumerate(picks):
                if frame_index is None:
                    continue
                mcl.save_evidence_png(
                    clip, int(frame_index), static_dir / "evidence" / f"static_{index + 1:02d}.png"
                )
            log(
                f"静止基线完成：实际 {summary['actual_frames']}/预期 "
                f"{summary['expected_frames']} 帧 ｜ 分析 {len(rows)} 帧（每 "
                f"{static_every_n} 帧取 1，跨满整段）｜ X 峰峰值 "
                f"{summary['x']['peak_to_peak_um']:.2f} μm ｜ 识别率 "
                f"{summary['chessboard_detection_rate']:.1%} ｜ 测量不确定度 "
                f"{summary['est_sigma_um']:.2f} μm",
                status=True,
            )
            log(
                f"静止噪声底：sigma {summary['static_sigma_um']:.2f} μm ｜ "
                f"RMS {summary['static_rms_um']:.2f} μm ｜ 完整 RAW 将在本组落盘后立即删除",
                status=True,
            )
            return summary
        finally:
            release(clip, [paths["raw"], paths["meta"], paths["frames"]],
                    block_id=block_id, keep=True)
            cleanup_raw_group(block_id, "静止基线")

    # -------------------------------------------------------------------------
    # 轴向 / 符号探针
    # -------------------------------------------------------------------------

    # 探针的所有片段归到同一个原始数据块，便于事后一起复查。
    PROBE_BLOCK = "01_probe"

    def run_probe() -> mcl.ProbeGains:
        """
        实测视觉 ↔ 机器人基座的轴向对应、符号与增益。

        输入：无（用相机与机器人各走一段）。
        输出：ProbeGains；任一门禁不通过时抛 RuntimeError，整轮具名中止。

        这不可能从代码里推出来：视觉 +x/+y 与机器人 base X/Y 的对应关系取决于
        相机怎么装在手腕上。符号写反会让误差每一步翻倍，很快撞上 100 μm 限幅，
        所以必须实测。增益 |g| 还顺带告诉我们闭环能否在 20 次迭代内走完 50 μm。
        """

        log("轴向/符号探针开始：用 1 mm 位移实测视觉坐标系与机器人基座的对应关系", status=True)
        meter = mcl.GroupVisionMeter(label="probe")
        half = float(config.MICRO_LOOP_PROBE_REFERENCE_SECONDS) / 2.0

        def probe_move(axis: str, delta_um: float, label: str) -> dict[str, float]:
            """走一步探针位移；不合格时原地复测，返回可信的视觉位置。"""

            saved, move = capture_window(
                label,
                command={
                    "action": "MOVE",
                    "axis": axis,
                    "delta_um": float(delta_um),
                    # 探针走 MICRO_LOOP_PROBE_UM = 1 mm，是微动步的 20~200 倍。
                    # 沿用给微动的 MICRO_LOOP_STEP_TIMEOUT_S 会让它每次都超时，
                    # 然后退化成"靠固定 settle 猜"——而探针失败是具名中止。
                    "settle_timeout_s": float(config.MICRO_LOOP_PROBE_TIMEOUT_S),
                },
                pre_seconds=half,
                post_seconds=half,
            )
            move = move or {}
            max_remeasurements = int(config.MICRO_LOOP_PROBE_REMEASURE_ATTEMPTS)
            last_failure = "未知测量问题"

            for attempt in range(max_remeasurements + 1):
                attempt_label = label if attempt == 0 else f"{label}_remeasure_{attempt}"
                if attempt > 0:
                    # 复测只要求机器人保持当前位置并重新确认稳定，绝不再次发送 MOVE。
                    robot_commands.put({"action": "STABILIZE"})
                    stable = inbox.wait(
                        "robot",
                        "STABILIZED",
                        error_queue,
                        stop_event,
                        stop_request_path,
                        timeout_s=30.0,
                    )
                    log(
                        f"探针 {label} 开始第 {attempt}/{max_remeasurements} 次原地复测"
                        f"（不发送运动命令，编码器稳定={bool(stable.get('settle_ok'))}）",
                        status=True,
                    )
                    saved, _ = capture_window(
                        attempt_label,
                        command=None,
                        pre_seconds=0.0,
                        post_seconds=0.0,
                    )

                paths = mcl.clip_temp_paths(temp_dir, saved["stem"])
                clip = None
                measurement = None
                try:
                    if bool(saved.get("truncated")):
                        last_failure = "片段被相机缓冲截断"
                    else:
                        # 探针要同时读取 x/y 位置；is_usable 仍按当前既有主轴门禁判断，
                        # 复测不会改变 12 帧与 2.0 μm 的严格阈值。
                        measurement, clip, _ = read_window(
                            saved["stem"],
                            meter,
                            vision_axis="x",
                            iteration_id=attempt_label,
                            tail_frames=int(config.MICRO_LOOP_PROBE_MEASURE_FRAMES),
                            tail_window_s=float(config.MICRO_LOOP_PROBE_TAIL_WINDOW_S),
                        )
                        last = (
                            measurement.positions[-1]
                            if measurement.positions
                            else {"x": float("nan"), "y": float("nan")}
                        )
                        motion_text = (
                            f"机器人侧实测 "
                            f"{float(move.get('achieved_um', float('nan'))):+.1f} μm"
                            if attempt == 0
                            else "机器人侧无新运动"
                        )
                        log(
                            f"探针 {attempt_label}：{motion_text} ｜ 视觉 "
                            f"x {float(last['x']):+.2f} μm ｜ "
                            f"y {float(last['y']):+.2f} μm"
                        )
                        if measurement.is_usable and measurement.positions:
                            if attempt > 0:
                                log(
                                    f"探针 {label} 第 {attempt} 次原地复测合格，继续实验。",
                                    status=True,
                                )
                            return {"x": float(last["x"]), "y": float(last["y"])}

                        failures: list[str] = []
                        if measurement.truncated:
                            failures.append("片段被相机缓冲截断")
                        if measurement.measured_um is None or not math.isfinite(
                            float(measurement.measured_um)
                        ):
                            failures.append("没有有限测量值")
                        if measurement.n_tail < int(config.MICRO_LOOP_MIN_TAIL_FRAMES):
                            failures.append(
                                f"尾窗合格帧 {measurement.n_tail} < "
                                f"{int(config.MICRO_LOOP_MIN_TAIL_FRAMES)}"
                            )
                        if not math.isfinite(float(measurement.sigma_tail_um)):
                            failures.append("尾窗标准差不是有限数")
                        elif measurement.sigma_tail_um > float(
                            config.MICRO_LOOP_TAIL_SIGMA_MAX_UM
                        ):
                            failures.append(
                                f"尾窗标准差 {measurement.sigma_tail_um:.2f} μm > "
                                f"{float(config.MICRO_LOOP_TAIL_SIGMA_MAX_UM):.2f} μm"
                            )
                        last_failure = "；".join(failures) or "测量未通过可用性门禁"
                finally:
                    release(
                        clip,
                        [paths["raw"], paths["meta"], paths["frames"]],
                        block_id=PROBE_BLOCK,
                    )

                if attempt < max_remeasurements:
                    log(
                        f"探针 {attempt_label} 不合格：{last_failure}；"
                        f"将进行第 {attempt + 1}/{max_remeasurements} 次原地复测。",
                        status=True,
                    )

            raise RuntimeError(
                f"探针 {label} 初测及 {max_remeasurements} 次原地复测均不合格；"
                f"最后一次原因：{last_failure}。无法安全定符号。"
            )

        # 参考片段：建立探针坐标系零点，并把 CheckerboardTracker 的参考锁在 medoid 帧上。
        ref_saved, _ = capture_window(
            "probe_ref",
            command=None,
            pre_seconds=float(config.MICRO_LOOP_PROBE_REFERENCE_SECONDS),
            post_seconds=0.0,
        )
        ref_paths = mcl.clip_temp_paths(temp_dir, ref_saved["stem"])
        ref_clip = mcl.open_raw_clip(ref_paths["raw"])
        try:
            primed = meter.prime(
                ref_clip,
                every_n=every_n,
                frames=int(config.MICRO_LOOP_PROBE_REFERENCE_FRAMES),
            )
            log(
                f"探针参考已锁定：medoid 帧 {primed['reference_frame_index']} ｜ "
                f"合格帧比例 {primed['accept_ratio']:.1%} ｜ 测量不确定度 "
                f"{primed['est_sigma_um']:.2f} μm",
                status=True,
            )
        finally:
            release(ref_clip, [ref_paths["raw"], ref_paths["meta"], ref_paths["frames"]],
                    block_id=PROBE_BLOCK)

        reference = {"x": 0.0, "y": 0.0}
        forward: dict[str, dict[str, float]] = {}
        backward: dict[str, dict[str, float]] = {}
        for axis in ("X", "Y"):
            forward[axis] = probe_move(axis, float(config.MICRO_LOOP_PROBE_UM), f"{axis}_fwd")
            backward[axis] = probe_move(axis, -float(config.MICRO_LOOP_PROBE_UM), f"{axis}_bwd")

        gains = mcl.evaluate_probe_gains(reference, forward, backward)
        mcl.write_json(run_dir / "probe_gains.json", gains.to_json())
        for axis in ("X", "Y"):
            log(
                f"探针结果：机器人 {axis} → 视觉 {gains.vision_axis[axis]} ｜ 增益 "
                f"{gains.gain[axis][gains.vision_axis[axis]]:+.3f} ｜ 符号 "
                f"{gains.sign[axis]:+.0f} ｜ 正向/反向差 "
                f"{gains.hysteresis.get(axis, float('nan')):.1%}",
                status=True,
            )
        for note in gains.notes:
            log(f"探针提示：{note}", status=True)
        cleanup_raw_group(PROBE_BLOCK, "轴向/符号探针")
        return gains

    # -------------------------------------------------------------------------
    # 开环微动块
    # -------------------------------------------------------------------------

    def run_open_loop_phase(gains: mcl.ProbeGains, axis_drift: dict[str, Any]) -> dict[str, Any]:
        """
        开环微动整段：每个 (轴, 模式, 档位) 一块，逐块执行 6 步固定命令序列。

        输入：探针结果、轴向漂移基准持有者。
        输出：含各块统计与最小可靠档位的字典。

        为什么开环要单独一个 meter：GroupVisionMeter 的参考在第一次识别时就锁死，
        不同实例之间的坐标原点相差一个常数。把探针测到的位移与开环的位移混在
        同一个序列里链式相减，会得到一个巨大的假增量。所以开环自己建一次零点。

        为什么整个开环阶段只用**一次**建零点：seq（+3/−3）与 alt（交替）都以
        净位移 ≈0 结束，块与块之间不需要回位，链式相减在任何一步都成立。
        """

        log("=== 开环微动开始：12 块 × 6 步，逐块在线分析 ===", status=True)
        meter = mcl.GroupVisionMeter(label="open")

        # 基线窗口：既建零点，又充当第一步的"运动前位置"。
        baseline_saved, _ = capture_window(
            "open_baseline",
            command=None,
            pre_seconds=float(config.MICRO_LOOP_REFERENCE_SECONDS),
            post_seconds=0.0,
        )
        baseline_paths = mcl.clip_temp_paths(temp_dir, baseline_saved["stem"])
        baseline_clip = mcl.open_raw_clip(baseline_paths["raw"])
        try:
            primed = meter.prime(
                baseline_clip,
                every_n=every_n,
                frames=int(config.MICRO_LOOP_REFERENCE_FRAMES),
            )
            # 这里刻意**不**用常规入口 axis_measured_um：它带"测量轴必须与探针
            # 一致"的硬校验，一次测量只供一条机器人轴。零点要同时给两条轴一个
            # 锚点，若把同一段片段扫两遍，识别开销直接翻倍——而一次识别本来
            # 就同时算出了两条视觉轴的尾窗中位数。
            baseline_measure = meter.measure_clip(
                baseline_clip,
                every_n=every_n,
                iteration_id="open_baseline",
                tail_frames=int(config.MICRO_LOOP_MEASURE_FRAMES),
                vision_axis=str(gains.vision_axis[config.MICRO_LOOP_AXES[0]]),
            )
        finally:
            release(
                baseline_clip,
                [baseline_paths["raw"], baseline_paths["meta"], baseline_paths["frames"]],
                block_id="02_open_baseline",
            )

        baseline_positions = mcl.GroupVisionMeter.axis_positions_um(
            baseline_measure, gains, robot_axes=config.MICRO_LOOP_AXES
        )
        log(
            f"开环零点已建立：medoid 帧 {primed['reference_frame_index']} ｜ 合格帧比例 "
            f"{primed['accept_ratio']:.1%} ｜ 测量不确定度 {primed['est_sigma_um']:.2f} μm ｜ "
            f"基线位置 X {baseline_positions.get('X')} / Y {baseline_positions.get('Y')} μm",
            status=True,
        )
        cleanup_raw_group("02_open_baseline", "开环基线")

        chain: dict[str, float | None] = dict(baseline_positions)
        block_stats: list[dict[str, Any]] = []

        def write_summary() -> None:
            """把开环汇总落盘。每块结束都调用一次，中途收尾时也已经有数据。"""

            mcl.write_json(
                run_dir / "open_loop_summary.json",
                {
                    "kind": "MICRO_CLOSED_LOOP_OPEN_SUMMARY",
                    "axis_vision_axis": dict(gains.vision_axis),
                    "axis_sign": {key: float(value) for key, value in gains.sign.items()},
                    "blocks": block_stats,
                    "min_reliable_level_um": {
                        f"{axis}_{mode}": mcl.min_reliable_level(
                            block_stats, mode=mode, axis=axis
                        )
                        for axis in config.MICRO_LOOP_AXES
                        for mode in config.MICRO_LOOP_OPEN_MODES
                    },
                    "note": (
                        "开环只测了 5/20/50 μm 三个档位，所以这里给出的是"
                        "“已测档位里最小且稳定的那一档”，不是精确最小分辨率。"
                    ),
                },
            )

        for axis in config.MICRO_LOOP_AXES:
            for mode in config.MICRO_LOOP_OPEN_MODES:
                for level_um in config.MICRO_LOOP_AMPLITUDES_UM:
                    stats = run_open_loop_block(
                        meter,
                        axis=axis,
                        mode=mode,
                        level_um=int(level_um),
                        gains=gains,
                        chain=chain,
                        # 每轴一个基准：传整张表进去会让两条轴共用同一个 start，
                        # 于是 X 的走位会算到 Y 的漂移里，警戒线形同虚设。
                        axis_drift=axis_drift[axis],
                    )
                    block_stats.append(stats)
                    mcl.write_json(
                        mcl.block_dir(run_dir, str(stats["block_id"])) / "block_summary.json",
                        {"kind": "MICRO_CLOSED_LOOP_OPEN_BLOCK", **stats},
                    )
                    # 每块结束都刷新开环汇总：中途因预算或异常收尾时，
                    # 已经跑完的块必须已经落盘，而不是等最后统一写。
                    write_summary()
                    cleanup_raw_group(str(stats["block_id"]), f"开环块 {stats['block_id']}")

        write_summary()
        min_levels = {
            f"{axis}_{mode}": mcl.min_reliable_level(block_stats, mode=mode, axis=axis)
            for axis in config.MICRO_LOOP_AXES
            for mode in config.MICRO_LOOP_OPEN_MODES
        }
        log(
            "开环最小可靠档位（只在已测的 5/20/50 μm 里挑，推不出精确最小分辨率）："
            + " ｜ ".join(
                f"{key} → {'无' if value is None else f'{value:g} μm'}"
                for key, value in min_levels.items()
            ),
            status=True,
        )
        return {"blocks": block_stats, "min_reliable_level_um": min_levels}

    def run_open_loop_block(
        meter: mcl.GroupVisionMeter,
        *,
        axis: str,
        mode: str,
        level_um: int,
        gains: mcl.ProbeGains,
        chain: dict[str, float | None],
        axis_drift: dict[str, Any],
    ) -> dict[str, Any]:
        """
        一个开环块：按固定序列走 6 步，**每步都在线测量并立即判读**。

        输入：本阶段的视觉测量器、轴、模式、档位、探针结果、跨块链式位置、漂移基准。
        输出：该块的统计字典（同时写入逐步 CSV 与 block_summary.json）。

        用户明确要求的执行形态：测量运动前位置 → 发一次微动 → 等短时间稳定 →
        测量运动后位置 → 立即算实际位移 → 记录 → 再走下一步。
        **绝不是**"先把几十个动作跑完、录一个大视频、最后统一处理"——
        那样既拿不到"运动前位置"，也无法在异常时及时停下。
        在线只算 before/after 位置与实测增量，不做 PSD、频谱或大量出图。
        """

        block_id = mcl.open_block_id(axis, mode, level_um)
        vision_axis = str(gains.vision_axis[axis])
        steps = mcl.open_loop_steps(axis=axis, mode=mode, level_um=int(level_um))
        out_dir = mcl.block_dir(run_dir, block_id)
        shape = (
            "+Δ×3 −Δ×3（连续同向）"
            if mode == mcl.OPEN_MODE_SEQ
            else "+Δ/−Δ 交替（频繁换向）"
        )
        log(f"--- 开环块 {block_id}：{shape} ---", status=True)

        rows: list[dict[str, Any]] = []
        # 中间有测量丢失时，把丢掉的命令累计起来，等下次测到时**合并比较**。
        # 不这么做的话链条会断在一个错误的基点上，之后每一步的增量都是错的。
        pending_command_um = 0.0
        pending_steps = 0

        def persist() -> None:
            mcl.write_rows_csv(out_dir / "open_steps.csv", rows, mcl.OPEN_STEP_COLUMNS)

        for step in steps:
            commanded = float(step["commanded_increment_um"])
            step_index = int(step["step_index"])
            label = f"{block_id}_s{step_index:02d}"
            before_um = chain.get(axis)

            saved, move = capture_window(
                label,
                command={
                    "action": "MOVE",
                    "axis": axis,
                    "delta_um": commanded,
                    "settle_timeout_s": float(config.MICRO_LOOP_STEP_TIMEOUT_S),
                },
            )
            move = move or {}
            drift_note = (
                axis_drift_note(move, axis=axis, drift=axis_drift) if move else ""
            )

            measured_after: float | None = None
            measured_increment: float | None = None
            n_tail = 0
            sigma_tail = math.nan
            n_valid = 0
            accept_rate = 0.0
            reject_reasons = ""
            measured_positions: dict[str, float | None] = {}
            note = ""
            if bool(saved.get("truncated")):
                note = "相机缓冲写满，本段被截断，本步测量不可用"
                truncated_paths = mcl.clip_temp_paths(temp_dir, saved["stem"])
                release(
                    None,
                    [truncated_paths["raw"], truncated_paths["meta"], truncated_paths["frames"]],
                    block_id=block_id,
                )
            else:
                measurement, clip, paths = read_window(
                    saved["stem"],
                    meter,
                    vision_axis=vision_axis,
                    iteration_id=label,
                    prev_measured_um=before_um,
                    vision_to_axis_sign=float(gains.sign[axis]),
                )
                n_tail = measurement.n_tail
                sigma_tail = measurement.sigma_tail_um
                n_valid = measurement.n_valid
                accept_rate = measurement.accept_rate
                reject_reasons = measurement.rejection_summary
                # 一次识别同时产生两个视觉轴的位置。把两条机器人轴的链都更新，
                # 否则 X 运动造成的交叉耦合不会进入 Y 的基点，切到 Y 时第一步
                # 会把此前累计横向偏移错误地算成本步响应。
                measured_positions = mcl.GroupVisionMeter.axis_positions_um(
                    measurement, gains, robot_axes=config.MICRO_LOOP_AXES
                )
                measured_after = measured_positions.get(axis)
                release(clip, paths, block_id=block_id)

            requested = commanded + pending_command_um
            span = pending_steps + 1
            if measured_after is None:
                status = mcl.OPEN_STEP_MEASUREMENT_LOST
                pending_command_um += commanded
                pending_steps += 1
                if not note:
                    note = f"本步没有可用的视觉测量（尾窗 {n_tail} 帧）"
            elif before_um is None:
                # 链条真的断了（开环零点那次测量就失败过）。不猜、不插值：
                # 把本步位置当作新基点，并如实标注这一步的增量不可用。
                status = mcl.OPEN_STEP_MEASUREMENT_LOST
                pending_command_um = 0.0
                pending_steps = 0
                note = "本步之前没有可用基点，无法算增量；已把本步位置作为新基点"
            else:
                measured_increment = measured_after - before_um
                status, note = mcl.classify_open_step(requested, measured_increment)
                if span > 1:
                    note += f"（增量跨越 {span} 步，前面的测量丢失已合并到这一步）"
                pending_command_um = 0.0
                pending_steps = 0

            if measured_after is not None:
                for observed_axis, observed_position in measured_positions.items():
                    if observed_position is not None and math.isfinite(
                        float(observed_position)
                    ):
                        chain[observed_axis] = float(observed_position)

            if drift_note:
                status = mcl.OPEN_STEP_ABNORMAL_JUMP
                note = (note + "；" if note else "") + f"{drift_note}，已中止本块"

            rows.append(
                {
                    "block_id": block_id,
                    "axis": axis,
                    "mode": mode,
                    "target_level_um": float(level_um),
                    "step_index": step_index,
                    "direction": int(step["direction"]),
                    "commanded_increment_um": commanded,
                    "vision_axis": vision_axis,
                    "vision_before_um": "" if before_um is None else before_um,
                    "vision_after_um": "" if measured_after is None else measured_after,
                    "measured_increment_um": (
                        "" if measured_increment is None else measured_increment
                    ),
                    "robot_tcp_before": mcl._format_pose(move.get("tcp_before")),
                    "robot_tcp_after": mcl._format_pose(move.get("tcp_after")),
                    "robot_achieved_um": float(move.get("achieved_um", math.nan)),
                    "command_timestamp_ns": int(move.get("command_start_ns") or 0),
                    "measurement_timestamp_ns": int(move.get("command_end_ns") or 0),
                    "settling_time_s": float(move.get("settling_time_s", math.nan)),
                    "settle_ok": bool(move.get("settle_ok", False)),
                    "n_valid_frames": n_valid,
                    "n_tail_frames": n_tail,
                    "sigma_tail_um": sigma_tail,
                    "accept_rate": accept_rate,
                    "reject_reasons": reject_reasons,
                    "status": status,
                    "note": note,
                }
            )
            persist()
            measured_text = (
                "不可用" if measured_increment is None else f"{measured_increment:+.2f} μm"
            )
            log(
                f"  {block_id} 步 {step_index}/{len(steps)} ｜ 命令 {commanded:+.1f} μm ｜ "
                f"实测 {measured_text} ｜ 状态 {status}",
                status=True,
            )
            if status == mcl.OPEN_STEP_ABNORMAL_JUMP:
                log(f"  {block_id} 出现异常步，中止本块（本块全部数据已保留）", status=True)
                break

        stats = mcl.summarize_open_block(rows)
        stats.update(
            {
                "block_id": block_id,
                "axis": axis,
                "mode": mode,
                "level_um": float(level_um),
                "vision_axis": vision_axis,
                "sign": float(gains.sign[axis]),
                "out_dir": str(out_dir),
                "raw_dir": str(raw_run_dir / block_id),
                "raw_retained": bool(keep_all_raw) and not bool(
                    config.MICRO_LOOP_DELETE_VIDEO_FILES_AFTER_GROUP
                ),
            }
        )
        mean_text = (
            "无"
            if stats["mean_increment_um"] is None
            else f"{stats['mean_increment_um']:+.2f} μm"
        )
        ratio_text = (
            "无"
            if stats["response_ratio"] is None
            else f"{stats['response_ratio']:.0%}"
        )
        log(
            f"块结束 {block_id}：{stats['verdict']} ｜ 有效 {stats['valid_steps']}/"
            f"{stats['steps']} ｜ 零响应 {stats['zero_response_steps']} ｜ 部分 "
            f"{stats['partial_response_steps']} ｜ 方向错 {stats['direction_error_steps']} ｜ "
            f"实测均值 {mean_text}（响应比 {ratio_text}）",
            status=True,
        )
        return stats

    # -------------------------------------------------------------------------
    # 单个目标的闭环
    # -------------------------------------------------------------------------

    def run_target(
        meter: mcl.GroupVisionMeter,
        *,
        axis: str,
        amplitude_um: int,
        direction: int,
        gains: mcl.ProbeGains,
        sigma_um: float,
        block_id: str,
        axis_drift: dict[str, Any],
    ) -> dict[str, Any]:
        """
        对一个 (轴, 幅值, 方向) 目标反复闭环纠偏，直到终止。

        输入：本组视觉测量器、轴、幅值、方向、探针测出的符号、本组测量噪声。
        输出：该目标的汇总行（同时写入 iterations/frames CSV 与 summary.json）。

        目标位置 = 参考片段的稳定位置 + direction · amplitude，也就是
        "让棋盘格在图像上移动 +A 或 −A μm"，机器人不需要知道任何绝对位置。
        """

        # 符号与视觉轴**只能**来自探针，不许在别处再写一份。
        # 之前 Y 轴固定读视觉 x 的缺陷就是因为这里各自抄了一遍映射。
        # 注意这里只取 vision_axis：sign 的全部作用在 axis_measured_um 内部
        # 就完成了（调用方拿到的 measured 已在机器人轴坐标系），本函数里
        # 再留一个 sign 局部量只会诱使下一处代码又乘一遍。
        vision_axis = str(gains.vision_axis[axis])
        slug = mcl.direction_slug(direction)
        target_key = f"{axis}_{int(amplitude_um):03d}um_{slug}"
        target_um = float(direction) * float(amplitude_um)
        group_dir = mcl.target_dir(run_dir, axis, amplitude_um)
        iterations_path = group_dir / f"{slug}_iterations.csv"
        frames_path = group_dir / f"{slug}_frames.csv"
        evidence_dir = group_dir / "evidence"

        iterations: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        errors_um: list[float] = []
        valid_flags: list[bool] = []
        last_measured: float | None = None
        peak_error_um = 0.0
        cumulative_command_um = 0.0
        sign_flip_streak = 0
        lost_streak = 0
        expected_frames_total = 0
        actual_frames_total = 0
        smallest_command_um = math.nan
        smallest_command_motion_um = math.nan
        # 迭代预算是**每个目标总量**，不随"收敛验证重开"而重置——
        # 否则最多 3 次重开会把单个目标的最坏时长翻三倍。
        next_iteration = 1
        termination = mcl.TERMINAL_STILL_SHRINKING
        evidence_written: set[str] = set()
        converged_verified: bool | None = None
        verify_delta_um = math.nan
        verify_attempts = 0

        def persist() -> None:
            """把已完成的数据落盘。任何中止路径都先调它。"""

            mcl.write_rows_csv(iterations_path, iterations, mcl.ITERATION_COLUMNS)
            mcl.write_rows_csv(frames_path, frame_rows, mcl.FRAME_COLUMNS)

        def track_drift(move: dict[str, Any]) -> str:
            """闭环侧对共用轴向漂移守卫的薄封装（本目标的轴与基准在此绑定）。"""

            return axis_drift_note(move, axis=axis, drift=axis_drift)

        def should_verify(state: str) -> bool:
            """这个终止状态是否值得再花一段片段做零命令验证。"""

            if not bool(config.MICRO_LOOP_VERIFY_EFFORT):
                return False
            if state not in (mcl.TERMINAL_CONVERGED, mcl.TERMINAL_CONVERGED_NOISE_LIMITED):
                return False
            if last_measured is None:
                return False
            if verify_attempts >= int(config.MICRO_LOOP_VERIFY_MAX_ATTEMPTS):
                return False
            # 迭代预算用完了就不再验证：验证后若失败也没有余量重收，
            # 白花一段片段还落个"未通过"的标签。
            return next_iteration <= int(config.MICRO_LOOP_MAX_ITER)

        def verify_convergence() -> bool | None:
            """
            零命令复测：不发任何命令，再录一段静止片段，看位置是否真的还在那里。

            返回 True（通过）、False（不通过，位置自己跑掉了）、
            None（片段不可用，无从判断）。

            这是唯一能把"真不动点"和"运气好落进去"分开的动作，成本只有一段片段。
            判据是 |零命令复测 − 残差| ≤ 2σ：σ 是本组实测的一次测量不确定度，
            所以只有"差得比测量本身能分辨的还多"才算没通过。
            """

            nonlocal converged_verified, verify_delta_um, verify_attempts
            nonlocal last_measured, termination, expected_frames_total, actual_frames_total
            nonlocal peak_error_um, note

            verify_attempts += 1
            residual_um, _snr = mcl.convergence_quality(errors_um, valid_flags, noise_um=sigma_um)
            saved, _unused = capture_window(
                f"{target_key}_verify{verify_attempts:02d}", command=None
            )
            expected_frames_total += mcl.expected_frames(
                int(saved["record_start_ns"]), int(saved["record_stop_ns"]), fps
            )
            actual_frames_total += int(saved["frame_count"])

            verify_measured: float | None = None
            n_tail = 0
            sigma_tail = math.nan
            if not bool(saved.get("truncated")):
                measurement, clip, paths = read_window(
                    saved["stem"], meter,
                    vision_axis=vision_axis,
                    iteration_id=f"{target_key}_verify{verify_attempts:02d}",
                    prev_measured_um=last_measured,
                    vision_to_axis_sign=float(gains.sign[axis]),
                )
                frame_rows.extend(measurement.frame_rows)
                n_tail = measurement.n_tail
                sigma_tail = measurement.sigma_tail_um
                verify_measured = mcl.GroupVisionMeter.axis_measured_um(
                    measurement, axis, gains
                )
                release(clip, paths, block_id=block_id)
            else:
                truncated_paths = mcl.clip_temp_paths(temp_dir, saved["stem"])
                release(
                    None,
                    [truncated_paths["raw"], truncated_paths["meta"], truncated_paths["frames"]],
                    block_id=block_id,
                )

            observation = (
                "零命令复测片段不可用"
                if verify_measured is None
                else f"零命令复测 {verify_measured:+.2f} μm，与残差 {residual_um:+.2f} μm 相差 "
                f"{abs(verify_measured - residual_um):.2f} μm"
            )

            if verify_measured is None:
                converged_verified = False
                note = f"收敛验证失败：{observation}"
                log(f"[状态] {axis} {target_um:+g} μm 收敛验证：片段不可用。")
                record_verification_row(verify_measured, residual_um, n_tail, sigma_tail, note)
                return None

            verify_delta_um = abs(verify_measured - residual_um)
            threshold_um = 2.0 * float(sigma_um)
            converged_verified = verify_delta_um <= threshold_um
            log(
                f"[状态] {axis} {target_um:+g} μm 收敛验证：{observation}"
                f"（阈值 {threshold_um:.2f} μm）｜ {'通过' if converged_verified else '不通过，重开闭环'}",
                status=True,
            )
            note = f"收敛验证{'通过' if converged_verified else '未通过'}：{observation}"
            record_verification_row(verify_measured, residual_um, n_tail, sigma_tail, note)
            if converged_verified:
                return True

            # 验证不通过：位置在没有任何命令的情况下变了。
            # 把新位置作为起点继续闭环；迭代预算不重置，所以不会拖长总时长。
            errors_um.append(target_um - verify_measured)
            valid_flags.append(True)
            peak_error_um = max(peak_error_um, abs(target_um - verify_measured))
            last_measured = verify_measured

            # 必须把状态退回"继续迭代"，否则外层已经判过的 CONVERGED 会留在原地，
            # 让下面那次迭代的重算失去意义。
            termination = mcl.TERMINAL_STILL_SHRINKING
            return False

        def record_verification_row(
            verify_measured: float | None,
            residual_um: float,
            n_tail: int,
            sigma_tail: float,
            row_note: str,
        ) -> None:
            """
            把零命令验证也写成一行迭代记录（command_um = 0）。

            这样 *_iterations.csv 里的序列才是完整的：
            `0 → +13 → −2 → +12 → … → 0(验证)`。验证这一段本身就是数据——
            它给出"最后一条命令之后，位置在没有任何输入时自己漂了多少"。
            """

            iterations.append(
                {
                    "iteration_id": f"{target_key}_verify{verify_attempts:02d}",
                    "axis": axis,
                    "direction": int(direction),
                    "target_amplitude_um": int(amplitude_um),
                    "target_um": target_um,
                    "measured_position_um": "" if verify_measured is None else verify_measured,
                    "error_before_um": target_um - residual_um,
                    "command_um": 0.0,
                    "achieved_robot_um": 0.0,
                    "actual_visual_displacement_um": (
                        "" if verify_measured is None else verify_measured - residual_um
                    ),
                    "error_after_um": (
                        "" if verify_measured is None else target_um - verify_measured
                    ),
                    "robot_tcp_before": "",
                    "robot_tcp_after": "",
                    "transverse_um": "",
                    "g_obs": "",
                    "n_valid_frames": 0,
                    "n_tail_frames": n_tail,
                    "sigma_tail_um": sigma_tail,
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "step_status": "VERIFY_NO_COMMAND",
                    "termination_state": termination,
                    "note": row_note,
                }
            )
            persist()

        while next_iteration <= int(config.MICRO_LOOP_MAX_ITER):
            iteration_index = next_iteration
            next_iteration += 1
            iteration_id = f"{target_key}_i{iteration_index:02d}"
            previous_measured = last_measured
            error_before = target_um - (0.0 if previous_measured is None else previous_measured)
            # 刻意**不传 sign**：error_before = target − last_measured，两端都
            # 已经是 axis_measured_um 归一过的机器人轴坐标，这里再乘一次 sign
            # 会在 sign=−1 的相机安装下把方向翻反。sign 的唯一用武之地是
            # 视觉测量入口，见 micro_closed_loop.command_for_error 的说明。
            command_um = mcl.command_for_error(error_before)
            below_deadband = command_um == 0.0

            log(
                f"{axis} {target_um:+g} μm ｜ 迭代 {iteration_index}/"
                f"{config.MICRO_LOOP_MAX_ITER} ｜ 位置 "
                f"{(0.0 if previous_measured is None else previous_measured):+.2f} μm ｜ 误差 "
                f"{error_before:+.2f} μm ｜ 指令 "
                f"{'死区不发' if below_deadband else f'{command_um:+.2f} μm'}"
            )

            saved, move = capture_window(
                f"{target_key}_{slug}",
                command=(
                    None
                    if below_deadband
                    else {"action": "MOVE", "axis": axis, "delta_um": float(command_um)}
                ),
            )
            move = move or {}
            expected_frames_total += mcl.expected_frames(
                int(saved["record_start_ns"]), int(saved["record_stop_ns"]), fps
            )
            actual_frames_total += int(saved["frame_count"])

            measured: float | None = None
            displacement_um: float | None = None
            n_tail = 0
            sigma_tail = math.nan
            note = ""

            if bool(saved.get("truncated")):
                note = "相机缓冲写满，本段被截断，本轮测量不可用"
                truncated_paths = mcl.clip_temp_paths(temp_dir, saved["stem"])
                release(
                    None,
                    [truncated_paths["raw"], truncated_paths["meta"], truncated_paths["frames"]],
                    block_id=block_id,
                )
            else:
                measurement, clip, paths = read_window(
                    saved["stem"], meter,
                    vision_axis=vision_axis,
                    iteration_id=iteration_id,
                    prev_measured_um=previous_measured,
                    vision_to_axis_sign=float(gains.sign[axis]),
                )
                frame_rows.extend(measurement.frame_rows)
                n_tail = measurement.n_tail
                sigma_tail = measurement.sigma_tail_um
                # 唯一的换算点：按探针结果挑视觉轴并归一符号。
                # 直接读 measurement.measured_um 会漏掉符号——视觉轴反向时
                # 误差会每步翻倍，而 CSV 里看起来完全正常。
                measured = mcl.GroupVisionMeter.axis_measured_um(measurement, axis, gains)
                displacement_um, g_obs = mcl.measured_delta_and_gain(
                    previous_measured_um=previous_measured,
                    measured_um=measured,
                    command_um=command_um,
                )
                if measured is None:
                    note = (
                        f"本轮测量不可用：尾窗 {measurement.n_tail} 帧，标准差 "
                        f"{measurement.sigma_tail_um:.2f} μm，合格率 "
                        f"{measurement.accept_rate:.1%}"
                        + (
                            f"，拒绝原因：{measurement.rejection_summary}"
                            if measurement.rejection_summary
                            else ""
                        )
                    )

                # 证据帧：每个目标最多 3 张全分辨率 PNG（绝不 JPEG）。
                if int(config.MICRO_LOOP_EVIDENCE_MAX) >= 1 and "01" not in evidence_written:
                    index = mcl.last_accepted_frame_index(measurement)
                    if index is not None:
                        mcl.save_evidence_png(clip, index, evidence_dir / "01_start_stable.png")
                        evidence_written.add("01")
                if int(config.MICRO_LOOP_EVIDENCE_MAX) >= 2:
                    index = mcl.peak_frame_index(
                        measurement,
                        # 证据帧的坐标是**视觉**坐标，所以轴名要给视觉轴，
                        # 零点也从同一视觉轴取。给机器人轴名会让 Y 的峰值帧
                        # 按视觉 x 的偏离来挑，挑出来的根本不是过冲那一帧。
                        axis=vision_axis,
                        reference_um=meter.ref_us[vision_axis],
                        target_um=target_um,
                    )
                    if index is not None:
                        mcl.save_evidence_png(
                            clip, index, evidence_dir / "02_max_offset_or_overshoot.png"
                        )
                        evidence_written.add("02")
                if int(config.MICRO_LOOP_EVIDENCE_MAX) >= 3:
                    index = mcl.last_accepted_frame_index(measurement)
                    if index is not None:
                        mcl.save_evidence_png(clip, index, evidence_dir / "03_final.png")
                        evidence_written.add("03")
                release(clip, paths, block_id=block_id)

            achieved_robot_um = float(move.get("achieved_um", math.nan))
            # 每次修正之后都跑一次全局漂移守卫（不是每段一次）。
            # 非空返回即"需要中止"；超警戒线但未越限的提示由守卫自己打日志。
            drift_abort_note = track_drift(move) if move else ""
            # g_obs 已在读到本轮测量后、覆盖 last_measured 之前由纯函数算好。
            # 两端位置与命令都在机器人测试轴坐标系里，不再乘探针 sign。
            if measured is None:
                g_obs = math.nan

            if measured is None:
                lost_streak += 1
                errors_um.append(math.nan)
                valid_flags.append(False)
                error_after = math.nan
                step_status = mcl.STEP_MEASUREMENT_LOST
            else:
                lost_streak = 0
                error_after = target_um - measured
                errors_um.append(error_after)
                valid_flags.append(True)
                peak_error_um = max(peak_error_um, abs(error_after))
                last_measured = measured
                if below_deadband:
                    step_status = mcl.STEP_NO_MOTION
                    if not note:
                        note = (
                            f"指令落在 {config.MICRO_LOOP_MIN_COMMAND_UM:g} μm 死区内，"
                            "未发送运动（仍照常测量与判收敛）"
                        )
                else:
                    step_status = str(move.get("command_status", mcl.STEP_VALID))
                if not below_deadband and (
                    not math.isfinite(smallest_command_um)
                    or abs(command_um) < abs(smallest_command_um)
                ):
                    smallest_command_um = abs(command_um)
                    smallest_command_motion_um = achieved_robot_um

            # 运行期联锁：探针通过之后仍可能发散。
            if below_deadband:
                sign_flip_streak = 0
            else:
                cumulative_command_um += abs(command_um)
                # 判据本身放在 mcl.sign_flip_detected 里，好让它能被直接喂数
                # 测试："sign=−1 的正常安装不得被判成方向异常"这条要求，
                # 埋在三千行主流程里是测不到的。
                flipped = mcl.sign_flip_detected(
                    command_um=command_um,
                    achieved_robot_um=achieved_robot_um,
                    measured_delta_um=displacement_um,
                    g_obs=g_obs,
                )
                sign_flip_streak = sign_flip_streak + 1 if flipped else 0

            termination = mcl.evaluate_termination(
                errors_um,
                valid_flags,
                tol_um=float(config.MICRO_LOOP_POSITION_TOL_UM),
                noise_um=sigma_um,
            )
            if sign_flip_streak >= int(config.MICRO_LOOP_SIGN_FLIP_ABORT_COUNT):
                termination = mcl.TERMINAL_ABORTED
                note = (
                    f"连续 {sign_flip_streak} 次指令与实测方向相反，疑似轴向符号写反，"
                    "已中止本目标"
                )
            elif drift_abort_note:
                termination = mcl.TERMINAL_ABORTED
                note = f"{drift_abort_note}，已中止本目标"
            elif cumulative_command_um > float(config.MICRO_LOOP_CUM_COMMAND_ABORT_UM):
                termination = mcl.TERMINAL_ABORTED
                note = (
                    f"本组累计指令 {cumulative_command_um:.0f} μm 超过上限，"
                    "疑似棘轮式走位，已中止本目标"
                )
            elif lost_streak >= 2:
                termination = mcl.TERMINAL_MEASUREMENT_LOST
                note = "连续两轮测量丢失，已中止本目标"
            elif (
                iteration_index >= int(config.MICRO_LOOP_MAX_ITER)
                and termination == mcl.TERMINAL_STILL_SHRINKING
            ):
                termination = mcl.TERMINAL_MAX_ITER

            # 5 μm 档最危险的错误不是"没收敛"，而是**完全没动却判成功**。
            # 当有效容差已经和幅值同量级时（tol_eff ≥ A），"误差落在容差内"
            # 几乎不构成证据：机器人一步不走，误差恰好就是幅值本身。
            # 所以 CONVERGED 还必须额外通过方向性运动确认——确实朝目标方向
            # 累计走过了幅值的一个可观比例。确认不了就报 MEASUREMENT_NOISE_LIMITED，
            # 那是"这个幅值下视觉分不出来"这一条真实结论，不是假成功也不是失败。
            if termination == mcl.TERMINAL_CONVERGED:
                confirmed, why = mcl.directional_motion_confirmed(
                    direction=int(direction),
                    amplitude_um=float(amplitude_um),
                    final_measured_um=measured,
                    sigma_um=sigma_um,
                )
                if not confirmed:
                    termination = mcl.TERMINAL_MEASUREMENT_NOISE_LIMITED
                    note = (note + "；" if note else "") + (
                        f"未获得朝目标方向的运动证据（{why}），"
                        "判为噪声受限而不是收敛"
                    )

            iterations.append(
                {
                    "iteration_id": iteration_id,
                    "axis": axis,
                    "direction": int(direction),
                    "target_amplitude_um": int(amplitude_um),
                    "target_um": target_um,
                    "measured_position_um": "" if measured is None else measured,
                    "error_before_um": error_before,
                    "command_um": command_um,
                    "achieved_robot_um": achieved_robot_um,
                    "actual_visual_displacement_um": (
                        "" if displacement_um is None else displacement_um
                    ),
                    "error_after_um": error_after,
                    "robot_tcp_before": mcl._format_pose(move.get("tcp_before")),
                    "robot_tcp_after": mcl._format_pose(move.get("tcp_after")),
                    "transverse_um": move.get("transverse_um", ""),
                    "g_obs": g_obs,
                    "n_valid_frames": sum(
                        1 for row in frame_rows if row.get("iteration_id") == iteration_id
                        and row.get("accepted") is True
                    ),
                    "n_tail_frames": n_tail,
                    "sigma_tail_um": sigma_tail,
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "step_status": step_status,
                    "termination_state": termination,
                    "note": note,
                }
            )
            persist()

            if termination != mcl.TERMINAL_STILL_SHRINKING:
                # 收敛时先做零命令验证，再决定是否真的结束。
                # 验证不通过说明刚才的"收敛"是运气——位置在没有任何命令的情况下变了，
                # 这时不结束，回到迭代阶段重收一次（迭代预算不重置，见 next_iteration）。
                if not should_verify(termination):
                    break
                verified = verify_convergence()
                if verified is not True:
                    continue
                break

        # 预算可能是在"收敛验证没过、回炉重收"之后耗尽的，那时 termination 被退回
        # STILL_SHRINKING。它不是终止状态，不能带着它写进汇总。
        if termination == mcl.TERMINAL_STILL_SHRINKING:
            termination = mcl.TERMINAL_MAX_ITER

        return {
            "converged_verified": converged_verified,
            "verify_delta_um": verify_delta_um,
            "verify_attempts": verify_attempts,
            "target_key": target_key,
            "axis": axis,
            "amplitude_um": int(amplitude_um),
            "direction": int(direction),
            "target_um": target_um,
            "iterations": iterations,
            "errors_um": errors_um,
            "valid_flags": valid_flags,
            "termination": termination,
            "peak_error_um": peak_error_um,
            "smallest_command_um": smallest_command_um,
            "smallest_command_motion_um": smallest_command_motion_um,
            "cumulative_command_um": cumulative_command_um,
            "final_measured": last_measured,
            "frame_rows": frame_rows,
            "expected_frames": expected_frames_total,
            "actual_frames": actual_frames_total,
            "note": note,
        }

    def drift_um_per_min(result: dict[str, Any]) -> float | str:
        """
        估算"没有任何命令时，位置自己漂移的速率"（μm/min）。

        只用 command_um == 0 的那些行（死区不发与零命令验证）：
        那些行里机器人一步没走，视觉却测到了位移，测到多少就是自由漂移多少——
        这是这个数唯一干净的来源。命令非零的行混不进来，因为它们的位移是要求的。

        只有一段零命令行时无法算速率（需要时间跨度），返回空字符串，
        而不是硬凑一个用单点算出来的数。
        """

        total_um = 0.0
        moments: list[datetime] = []
        for row in result.get("iterations", []):
            if float(row.get("command_um", 1.0) or 0.0) != 0.0:
                continue
            displacement = row.get("actual_visual_displacement_um")
            if displacement in (None, ""):
                continue
            try:
                value = float(displacement)
                moment = datetime.fromisoformat(str(row["timestamp"]))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            total_um += abs(value)
            moments.append(moment)

        if len(moments) < 2:
            return ""
        span_min = (max(moments) - min(moments)).total_seconds() / 60.0
        if span_min <= 0.0:
            return ""
        return total_um / span_min

    def verification_note(result: dict[str, Any]) -> str:
        """
        收敛验证的结论，当作汇总行的备注。

        这是 CONVERGED 那一行最要紧的旁注：没有它，"已收敛"与"碰巧落在容差里"
        在 CSV 里长得一模一样。
        """

        verified = result.get("converged_verified")
        if verified is None:
            return ""
        if verified:
            return (
                f"零命令复测确认是真不动点（与残差相差 {result['verify_delta_um']:.2f} μm）"
            )
        return (
            f"零命令复测与残差相差 {result['verify_delta_um']:.2f} μm，"
            f"超出 2σ，共验证 {result['verify_attempts']} 次仍未通过"
        )

    def summarise(result: dict[str, Any], *, sigma_um: float) -> dict[str, Any]:
        """把一个目标的闭环结果压成 master_summary.csv 的一行。"""

        errors = result["errors_um"]
        valid = result["valid_flags"]
        converge_count = int(config.MICRO_LOOP_CONVERGE_COUNT)
        recent = [float(v) for v, ok in zip(errors, valid) if ok][-converge_count:]
        residual_um, snr = mcl.convergence_quality(errors, valid, noise_um=sigma_um)

        termination = str(result["termination"])
        # "收敛了但残差与零在统计上不可分"是一个诚实且重要的标签，不能混进 CONVERGED。
        if termination == mcl.TERMINAL_CONVERGED and math.isfinite(snr) and snr < 2.0:
            termination = mcl.TERMINAL_CONVERGED_NOISE_LIMITED

        frames = result["frame_rows"]
        detected = sum(1 for row in frames if row.get("checker_is_valid") is True)
        amplitudes = [
            abs(float(v)) for v, ok in zip(errors, valid) if ok and math.isfinite(float(v))
        ]
        tail = amplitudes[-int(config.MICRO_LOOP_CYCLE_WINDOW):]

        # 最终残差必须是**相对于目标**的，不是"最终位置的绝对值"。
        # 上一版写的是 abs(final_measured)，那是位置而不是误差：目标在 −50 μm
        # 时它报 50 μm 的"误差"，在 +50 μm 收敛良好时也报 50——整列数全是错的，
        # 而且错得很有规律（正好等于幅值），最容易被当成"没收敛"。
        final_residual = mcl.closed_loop_final_residual_um(
            float(result["target_um"]), result["final_measured"]
        )
        return {
            "axis": result["axis"],
            "target_amplitude_um": result["amplitude_um"],
            "target_direction": result["direction"],
            "iteration_count": len(result["iterations"]),
            "termination_state": termination,
            "final_mean_error_um": (float(np.median(recent)) if recent else ""),
            "closed_loop_final_residual_um": (
                "" if final_residual is None else final_residual
            ),
            "final_position_um": (
                "" if result["final_measured"] is None else float(result["final_measured"])
            ),
            "final_error_peak_to_peak_um": (max(tail) - min(tail)) if len(tail) >= 2 else "",
            "max_overshoot_um": result["peak_error_um"],
            "smallest_command_issued_um": result["smallest_command_um"],
            "corresponding_actual_motion_um": result["smallest_command_motion_um"],
            "actual_frames": result["actual_frames"],
            "expected_frames": result["expected_frames"],
            "frame_ratio": (
                result["actual_frames"] / result["expected_frames"]
                if result["expected_frames"]
                else ""
            ),
            "valid_chessboard_frames": detected,
            "chessboard_detection_rate": (detected / len(frames)) if frames else 0.0,
            "initial_error_um": (errors[0] if errors else ""),
            "residual_snr": snr,
            "est_sigma_um": sigma_um,
            "limit_cycle_amplitude_um": (
                (max(tail) - min(tail))
                if termination == mcl.TERMINAL_LIMIT_CYCLE and len(tail) >= 2
                else ""
            ),
            "plateau_um": (
                float(np.median(tail))
                if termination == mcl.TERMINAL_STALLED and tail
                else ""
            ),
            "min_effective_motion_limit_um": (
                abs(residual_um)
                if termination
                in (mcl.TERMINAL_STALLED, mcl.TERMINAL_CONVERGED, mcl.TERMINAL_CONVERGED_NOISE_LIMITED)
                and math.isfinite(residual_um)
                else ""
            ),
            "drift_um_per_min": drift_um_per_min(result),
            "ref_cross_group_delta_um": result.get("ref_cross_group_delta_um", ""),
            "converged_verified": (
                "" if result.get("converged_verified") is None
                else bool(result["converged_verified"])
            ),
            "axis_reference_um": result.get("axis_reference_um", 0.0),
            "temp_bytes_peak": temp_bytes_peak,
            "trash_bytes": mcl.directory_bytes(trash_dir),
            "note": str(result.get("note", "")) or verification_note(result),
        }

    # -------------------------------------------------------------------------
    # 最终快速判读报告
    # -------------------------------------------------------------------------

    # 把开环判读译成人能直接读的一句话。刻意不用"最小分辨率"这种词：
    # 本轮只测了 5/20/50 三个档位，能说的只是"这一档稳不稳"。
    OPEN_VERDICT_TEXT = {
        mcl.OPEN_VERDICT_STABLE: "稳定有效",
        mcl.OPEN_VERDICT_PARTIAL: "部分有效",
        mcl.OPEN_VERDICT_NO_RESPONSE: "无法可靠动作",
        mcl.OPEN_VERDICT_UNRELIABLE: "不稳定，无法可靠判定",
    }
    # 整体判读取 seq 与 alt 里更悲观的那一档：闭环将来会频繁换向，
    # 若 alt 明显更差而整体只看 seq，就会把一个真实缺陷盖掉。
    VERDICT_PESSIMISM = {
        mcl.OPEN_VERDICT_STABLE: 0,
        mcl.OPEN_VERDICT_PARTIAL: 1,
        mcl.OPEN_VERDICT_UNRELIABLE: 2,
        mcl.OPEN_VERDICT_NO_RESPONSE: 3,
    }

    def write_final_report(
        static_summary: dict[str, Any],
        open_result: dict[str, Any],
        closed_rows: list[dict[str, Any]],
        skipped: list[dict[str, Any]],
    ) -> Path:
        """
        把本轮三个核心问题压成一份 JSON + 一份纯文本。

        输入：静止基线汇总、开环阶段结果、闭环各目标汇总行、因预算或异常未跑的目标。
        输出：JSON 报告路径（纯文本报告写在同目录同名 .txt）。

        只回答用户列出的那几个问题：静止噪声底是多少；X/Y 的 5/20/50 μm
        在连续同向与频繁换向下各是什么表现；闭环最终残差多少、有没有 STALLED /
        LIMIT_CYCLE / 噪声受限。**不**输出"UR10 最小分辨率 = XX μm"——
        本轮只测三个档位，那个数推不出来。
        """

        blocks = {str(item["block_id"]): item for item in open_result.get("blocks", [])}

        def open_cell(axis: str, level: int, mode: str) -> dict[str, Any]:
            """取某个 (轴, 档位, 模式) 的开环判读；没跑到就明确写 NOT_RUN。"""

            block = blocks.get(mcl.open_block_id(axis, mode, level))
            if block is None:
                return {"status": "NOT_RUN"}
            return {
                "status": "RUN",
                "verdict": block["verdict"],
                "verdict_text": OPEN_VERDICT_TEXT.get(
                    str(block["verdict"]), str(block["verdict"])
                ),
                "steps": block["steps"],
                "measured_steps": block["measured_steps"],
                "valid_steps": block["valid_steps"],
                "zero_response_steps": block["zero_response_steps"],
                "partial_response_steps": block["partial_response_steps"],
                "direction_error_steps": block["direction_error_steps"],
                "lost_steps": block["lost_steps"],
                "mean_increment_um": block["mean_increment_um"],
                "mean_abs_increment_um": block["mean_abs_increment_um"],
                "response_ratio": block["response_ratio"],
                "net_displacement_um": block["net_displacement_um"],
            }

        def closed_cell(axis: str, level: int, direction: int) -> dict[str, Any]:
            """取某个 (轴, 档位, 方向) 的闭环汇总；没跑到就明确写 NOT_RUN。"""

            for row in closed_rows:
                if (
                    str(row["axis"]) == axis
                    and int(row["target_amplitude_um"]) == int(level)
                    and int(row["target_direction"]) == int(direction)
                ):
                    return {
                        "status": "RUN",
                        "termination_state": row["termination_state"],
                        "iteration_count": row["iteration_count"],
                        "final_residual_um": row["closed_loop_final_residual_um"],
                        "final_position_um": row["final_position_um"],
                        "residual_snr": row["residual_snr"],
                        "smallest_command_issued_um": row["smallest_command_issued_um"],
                        "corresponding_actual_motion_um": row[
                            "corresponding_actual_motion_um"
                        ],
                        "min_effective_motion_limit_um": row[
                            "min_effective_motion_limit_um"
                        ],
                        "converged_verified": row["converged_verified"],
                    }
            return {"status": "NOT_RUN"}

        report: dict[str, Any] = {
            "kind": "MICRO_CLOSED_LOOP_QUICK_REPORT",
            "run_id": run_id,
            "levels_um": [int(value) for value in config.MICRO_LOOP_AMPLITUDES_UM],
            "order_note": "档位顺序固定 5 → 20 → 50 μm，轴顺序 X → Y，不做自适应搜索",
            "scope_note": (
                "本轮只测 5/20/50 μm 三个档位的开环微动与视觉闭环能力，"
                "不是从 5 到 200 μm 的最小分辨率标定曲线，因此报告里不给出"
                "“UR10 最小分辨率 = XX μm”这种结论。"
            ),
            "static": {
                "static_sigma_um": static_summary.get("static_sigma_um"),
                "static_rms_um": static_summary.get("static_rms_um"),
                "static_peak_to_peak_um": static_summary.get("static_peak_to_peak_um"),
                "analyze_frames": static_summary.get("analyze_frames"),
                "chessboard_detection_rate": static_summary.get(
                    "chessboard_detection_rate"
                ),
                "raw_archive": static_summary.get("raw_archive"),
                "raw_retained": static_summary.get("raw_retained"),
            },
            "raw_root": str(raw_run_dir),
            "raw_retained_all_blocks": bool(keep_all_raw) and not bool(
                config.MICRO_LOOP_DELETE_VIDEO_FILES_ON_EXIT
            ),
            # 时长只作信息记录：stopped_early 恒为 False（时长不再中止实验），
            # 保留这个字段是为了让读报告的人一眼看见"没有因为时间跳过任何目标"。
            "timing": {
                "time_notice_s": float(config.MICRO_LOOP_TIME_NOTICE_S),
                "windows_done": int(watch.done),
                "windows_planned_normal": int(watch.total_windows),
                "elapsed_s": float(watch.elapsed_s),
                "mean_window_s": (
                    None if watch.done == 0 else float(watch.mean_window_s)
                ),
                "stopped_early": False,
                "notice_raised": bool(watch.noticed),
                "reason": str(watch.reason),
            },
            "open_loop": {},
            "closed_loop": {},
            "answers": {},
            "not_run_targets": skipped,
        }

        for axis in config.MICRO_LOOP_AXES:
            open_axis: dict[str, Any] = {}
            closed_axis: dict[str, Any] = {}
            for level in config.MICRO_LOOP_AMPLITUDES_UM:
                level = int(level)
                cells = {
                    mode: open_cell(axis, level, mode)
                    for mode in config.MICRO_LOOP_OPEN_MODES
                }
                open_axis[f"{level:03d}um"] = cells
                closed_axis[f"{level:03d}um"] = {
                    "positive": closed_cell(axis, level, 1),
                    "negative": closed_cell(axis, level, -1),
                }

                # 整体判读：跑到的模式里挑最悲观的。
                run_verdicts = [
                    str(cell["verdict"]) for cell in cells.values() if cell["status"] == "RUN"
                ]
                if not run_verdicts:
                    overall = None
                else:
                    overall = max(run_verdicts, key=lambda v: VERDICT_PESSIMISM.get(v, 2))
                plus = closed_axis[f"{level:03d}um"]["positive"]
                report["answers"][f"{axis}_{level}um"] = {
                    "open_seq": cells.get(mcl.OPEN_MODE_SEQ, {}).get("verdict_text"),
                    "open_alt": cells.get(mcl.OPEN_MODE_ALT, {}).get("verdict_text"),
                    "open_overall": (
                        None if overall is None else OPEN_VERDICT_TEXT.get(overall, overall)
                    ),
                    "closed_termination_positive": plus.get("termination_state"),
                    "closed_final_residual_um_positive": plus.get("final_residual_um"),
                    "has_stalled": any(
                        state.get("termination_state") == mcl.TERMINAL_STALLED
                        for state in closed_axis[f"{level:03d}um"].values()
                    ),
                    "has_limit_cycle": any(
                        state.get("termination_state") == mcl.TERMINAL_LIMIT_CYCLE
                        for state in closed_axis[f"{level:03d}um"].values()
                    ),
                    # 两个"噪声受限"含义不同，但都是"视觉分不出这么小的位移"，
                    # 汇总时并在一起报，具体是哪一个在各自的 summary.json 里。
                    "has_noise_limited": any(
                        state.get("termination_state")
                        in (
                            mcl.TERMINAL_CONVERGED_NOISE_LIMITED,
                            mcl.TERMINAL_MEASUREMENT_NOISE_LIMITED,
                        )
                        for state in closed_axis[f"{level:03d}um"].values()
                    ),
                }
            report["open_loop"][axis] = open_axis
            report["closed_loop"][axis] = closed_axis

        report["min_reliable_level_um"] = open_result.get("min_reliable_level_um", {})
        # 开环块应当正好是 2 轴 × 2 模式 × 3 档 = 12 块。少一块只可能是被
        # 真实故障（异常跳跃、测量连续丢失等）打断，不再是"时间到"——
        # 时长已经不中止实验了。据实报出来，别让缺口的块悄悄消失。
        expected_open_blocks = 2 * 2 * len(config.MICRO_LOOP_AMPLITUDES_UM)
        if len(open_result.get("blocks", [])) < expected_open_blocks:
            report["open_loop_note"] = (
                f"开环阶段只跑了 {len(open_result.get('blocks', []))}"
                f"/{expected_open_blocks} 块，未跑到的块在 not_run_targets 里；"
                "原因见 experiment_log.txt 中该块的收尾状态。"
            )

        json_path = run_dir / "final_report.json"
        mcl.write_json(json_path, report)

        # 纯文本版：GPT 端要的是能直接读的几行，不是又一份嵌套 JSON。
        lines = [
            "UR10 微动开环 + 视觉闭环快速判读",
            f"运行目录：{run_dir}",
            f"原始 RAW 根目录：{raw_run_dir}（组内暂存：{keep_all_raw}；组末自动删视频）",
            "",
            f"静止基线：sigma {report['static']['static_sigma_um']} μm ｜ "
            f"RMS {report['static']['static_rms_um']} μm ｜ 峰峰值 "
            f"{report['static']['static_peak_to_peak_um']} μm",
            "",
            "开环（给固定命令，实际走了多少）：",
        ]
        for axis in config.MICRO_LOOP_AXES:
            for level in config.MICRO_LOOP_AMPLITUDES_UM:
                cells = report["open_loop"][axis][f"{int(level):03d}um"]
                parts = []
                for mode in config.MICRO_LOOP_OPEN_MODES:
                    cell = cells.get(mode, {})
                    if cell.get("status") != "RUN":
                        parts.append(f"{mode} 未跑")
                        continue
                    mean = cell["mean_increment_um"]
                    ratio = cell["response_ratio"]
                    parts.append(
                        f"{mode} {cell['verdict_text']}（有效 {cell['valid_steps']}/"
                        f"{cell['steps']}，均值 "
                        f"{'—' if mean is None else f'{mean:+.2f}'} μm，响应比 "
                        f"{'—' if ratio is None else f'{ratio:.0%}'}）"
                    )
                lines.append(f"  {axis} {int(level)} μm： " + " ｜ ".join(parts))
        lines.append("")
        lines.append("闭环（反复纠偏能否逼近目标）：")
        for axis in config.MICRO_LOOP_AXES:
            for level in config.MICRO_LOOP_AMPLITUDES_UM:
                key = f"{axis}_{int(level)}um"
                answer = report["answers"][key]
                lines.append(
                    f"  {axis} {int(level)} μm： 终止 {answer['closed_termination_positive']} ｜ "
                    f"最终残差 {answer['closed_final_residual_um_positive']} μm ｜ "
                    f"STALLED {answer['has_stalled']} ｜ LIMIT_CYCLE "
                    f"{answer['has_limit_cycle']} ｜ 噪声受限 {answer['has_noise_limited']}"
                )
        lines.append("")
        if skipped:
            lines.append(f"未跑到的目标 {len(skipped)} 个：" + "、".join(
                f"{item['axis']}{int(item['amplitude_um']):+d}μm" for item in skipped
            ))
        else:
            lines.append("所有目标均已执行。")
        lines.append(
            "注意：本轮只有 5/20/50 μm 三个档位，以上判读是"
            "“这一档稳不稳”，不是最小分辨率的精确值。"
        )
        text_path = run_dir / "final_report.txt"
        text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for line in lines:
            log(line, status=True)
        return json_path

    # -------------------------------------------------------------------------
    # 当前多目标视觉闭环（旧 5/20/50、alt、seq 状态机不再从本模式调用）
    # -------------------------------------------------------------------------

    class _TargetVisionLost(RuntimeError):
        def __init__(
            self,
            message: str,
            *,
            command_um: float | None = None,
            move: dict[str, Any] | None = None,
        ) -> None:
            super().__init__(message)
            self.command_um = command_um
            self.move = dict(move or {})

    def _pose_text(pose: Any) -> str:
        if pose is None:
            return ""
        try:
            return ";".join(f"{float(value):.9f}" for value in pose)
        except (TypeError, ValueError):
            return ""

    def _position_from_stem(
        stem: str,
        meter: mcl.GroupVisionMeter,
        gains: mcl.ProbeGains,
        *,
        iteration_id: str,
    ) -> tuple[mcl.RobotXYMeasurement, mcl.ClipMeasurement, mcl.RawClip, list[Path]]:
        clip: mcl.RawClip | None = None
        paths_dict = mcl.clip_temp_paths(temp_dir, stem)
        paths = [paths_dict["raw"], paths_dict["meta"], paths_dict["frames"]]
        try:
            measurement, clip, paths = read_window(
                stem,
                meter,
                vision_axis="x",
                iteration_id=iteration_id,
                tail_frames=int(config.MICRO_LOOP_MEASURE_FRAMES),
                tail_window_s=float(config.MICRO_LOOP_TAIL_WINDOW_S),
            )
            robot_xy = mcl.robot_xy_from_measurement(measurement, gains)
            return robot_xy, measurement, clip, paths
        except BaseException:
            release(clip, paths, keep=False)
            raise

    def _persist_target_rows(rows: list[dict[str, Any]]) -> None:
        mcl.write_rows_csv(
            run_dir / "iterations.csv", rows, mcl.TARGET_ITERATION_COLUMNS
        )

    def _run_target_iteration(
        meter: mcl.GroupVisionMeter,
        gains: mcl.ProbeGains,
        *,
        axis: str,
        target_index: int,
        target_nominal_um: float,
        target_absolute_um: float,
        iteration: int,
        baseline_noise_um: float,
    ) -> tuple[dict[str, Any], mcl.RawClip, list[Path], mcl.ClipMeasurement]:
        """一个活动窗口内完成 before 快照、command、after，并返回待清理文件。"""

        label = f"{axis}_T{target_index:02d}_I{iteration:02d}"
        watch.observe(label)
        stem = open_window(label)
        _batch_safe_sleep(
            float(config.MICRO_LOOP_PRE_SECONDS),
            error_queue,
            stop_event,
            stop_request_path,
        )

        before_saved = snapshot_active_window(label)
        before_clip: mcl.RawClip | None = None
        before_paths: list[Path] = []
        before_measurement: mcl.ClipMeasurement | None = None
        before_xy: mcl.RobotXYMeasurement | None = None
        before_error: BaseException | None = None
        try:
            before_xy, before_measurement, before_clip, before_paths = _position_from_stem(
                before_saved["stem"], meter, gains, iteration_id=f"{label}_before"
            )
            if not before_xy.is_usable:
                raise _TargetVisionLost(
                    f"运动前视觉测量不合格：{before_xy.reason}", command_um=0.0
                )
        except BaseException as exc:
            before_error = exc
        finally:
            if before_paths:
                release(before_clip, before_paths, block_id=f"{label}_snapshot", keep=True)
                cleanup_raw_group(
                    f"{label}_snapshot", f"{label} 运动前快照", quiet=True
                )

        def close_aborted_window(description: str) -> None:
            """命令前门禁失败时也封口并删除活动 RAW，避免下一轮撞上未关闭窗口。"""

            try:
                close_window()
            finally:
                aborted_paths = mcl.clip_temp_paths(temp_dir, stem)
                release(
                    None,
                    [aborted_paths["raw"], aborted_paths["meta"], aborted_paths["frames"]],
                    block_id=label,
                    keep=True,
                )
                cleanup_raw_group(label, description)

        if before_error is not None:
            close_aborted_window(f"{label} 未执行命令的失败窗口")
            raise before_error

        worker_error = _pop_worker_error(error_queue)
        if worker_error:
            stop_event.set()
            raise RuntimeError(worker_error)
        if _external_stop_requested(stop_request_path) or stop_event.is_set():
            stop_event.set()
            raise RuntimeError("发送微动命令前收到用户停止请求，未发送本轮运动。")

        assert before_xy is not None and before_xy.x_um is not None and before_xy.y_um is not None
        before = {"X": float(before_xy.x_um), "Y": float(before_xy.y_um)}
        if max(abs(before["X"]), abs(before["Y"])) > float(
            config.MICRO_TARGET_LOCAL_RANGE_UM
        ):
            close_aborted_window(f"{label} 局部范围门禁窗口")
            raise RuntimeError(
                f"{label} 视觉位置已超出局部实验范围 ±"
                f"{float(config.MICRO_TARGET_LOCAL_RANGE_UM):g} μm，拒绝继续运动。"
            )

        error_before = float(target_absolute_um) - before[axis]
        if abs(error_before) > float(config.MICRO_TARGET_MAX_COMMAND_UM):
            close_aborted_window(f"{label} 单步命令门禁窗口")
            raise RuntimeError(
                f"{label} 剩余误差 {error_before:+.2f} μm 超过单步安全上限 "
                f"{float(config.MICRO_TARGET_MAX_COMMAND_UM):g} μm。"
            )
        command_um = (
            error_before
            if abs(error_before) >= float(config.MICRO_LOOP_MIN_COMMAND_UM)
            else 0.0
        )

        move: dict[str, Any] = {}
        if command_um != 0.0:
            robot_commands.put(
                {"action": "MOVE", "axis": axis, "delta_um": command_um}
            )
            move = inbox.wait(
                "robot",
                "MOVE_DONE",
                error_queue,
                stop_event,
                stop_request_path,
                timeout_s=30.0,
            )
        else:
            _batch_safe_sleep(
                float(config.MICRO_LOOP_MOTION_SETTLE_SECONDS),
                error_queue,
                stop_event,
                stop_request_path,
            )
        _batch_safe_sleep(
            float(config.MICRO_LOOP_POST_SECONDS),
            error_queue,
            stop_event,
            stop_request_path,
        )
        saved = close_window()
        saved["stem"] = stem

        after_xy: mcl.RobotXYMeasurement | None = None
        after_measurement: mcl.ClipMeasurement | None = None
        after_clip: mcl.RawClip | None = None
        after_paths: list[Path] = []
        after_xy, after_measurement, after_clip, after_paths = _position_from_stem(
            stem, meter, gains, iteration_id=f"{label}_after"
        )
        if not after_xy.is_usable:
            release(after_clip, after_paths, block_id=label, keep=True)
            cleanup_raw_group(label, f"{label} 命令窗口")
            raise _TargetVisionLost(
                f"运动后视觉测量不合格：{after_xy.reason}",
                command_um=command_um,
                move=move,
            )

        assert after_xy.x_um is not None and after_xy.y_um is not None
        after = {"X": float(after_xy.x_um), "Y": float(after_xy.y_um)}
        if max(abs(after["X"]), abs(after["Y"])) > float(
            config.MICRO_TARGET_LOCAL_RANGE_UM
        ):
            release(after_clip, after_paths, block_id=label, keep=True)
            cleanup_raw_group(label, f"{label} 命令窗口")
            raise RuntimeError(
                f"{label} 运动后视觉位置超出局部实验范围 ±"
                f"{float(config.MICRO_TARGET_LOCAL_RANGE_UM):g} μm。"
            )

        other_axis = "Y" if axis == "X" else "X"
        vision_delta = after[axis] - before[axis]
        error_after = float(target_absolute_um) - after[axis]
        same_direction: bool | str = ""
        if command_um != 0.0:
            same_direction = bool(
                vision_delta != 0.0
                and (vision_delta > 0.0) == (command_um > 0.0)
            )
        achieved = float(move.get("achieved_um", 0.0)) if move else 0.0
        row: dict[str, Any] = {
            "axis": axis,
            "target_index": int(target_index),
            "target_nominal_um": float(target_nominal_um),
            "target_absolute_um": float(target_absolute_um),
            "iteration": int(iteration),
            "position_before_um": before[axis],
            "error_before_um": error_before,
            "command_um": command_um,
            "vision_delta_um": vision_delta,
            "position_after_um": after[axis],
            "error_after_um": error_after,
            "orthogonal_axis_before_um": before[other_axis],
            "orthogonal_axis_after_um": after[other_axis],
            "orthogonal_drift_um": after[other_axis] - before[other_axis],
            "command_and_motion_same_direction": same_direction,
            "rtde_before": _pose_text(move.get("tcp_before")),
            "rtde_after": _pose_text(move.get("tcp_after")),
            "rtde_delta_um": achieved,
            "baseline_noise_reference_um": baseline_noise_um,
            "stop_reason": "",
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "before_valid_frames": int(before_xy.n_tail),
            "after_valid_frames": int(after_xy.n_tail),
            "before_sigma_um": max(before_xy.sigma_x_um, before_xy.sigma_y_um),
            "after_sigma_um": max(after_xy.sigma_x_um, after_xy.sigma_y_um),
            "command_start_ns": move.get("command_start_ns", ""),
            "command_end_ns": move.get("command_end_ns", ""),
            "note": (
                "RTDE状态仅作旁证；微米级方向与终止判据只使用视觉。"
                + (f" RTDE备注：{move.get('note')}" if move.get("note") else "")
            ),
        }
        return row, after_clip, after_paths, after_measurement

    def _target_summary(
        rows: list[dict[str, Any]], *, axis: str, target_index: int, stop_reason: str
    ) -> dict[str, Any]:
        errors = [float(row["error_after_um"]) for row in rows]
        clear_gate = max(
            1.0,
            2.0 * float(rows[-1]["baseline_noise_reference_um"]),
        )
        clear_commands = [
            abs(float(row["command_um"]))
            for row in rows
            if row["command_and_motion_same_direction"] is True
            and abs(float(row["vision_delta_um"])) > clear_gate
        ]
        recent = errors[-5:]
        return {
            "axis": axis,
            "target_index": int(target_index),
            "target_nominal_um": float(rows[0]["target_nominal_um"]),
            "target_absolute_um": float(rows[0]["target_absolute_um"]),
            "initial_error_um": float(rows[0]["error_before_um"]),
            "iterations": len(rows),
            "final_error_um": errors[-1],
            "recent_error_min_um": min(recent),
            "recent_error_max_um": max(recent),
            "empirical_smallest_clear_same_direction_command_um": (
                min(clear_commands) if clear_commands else None
            ),
            "stop_reason": stop_reason,
        }

    def _write_multi_target_plots(
        rows: list[dict[str, Any]], summaries: list[dict[str, Any]]
    ) -> list[str]:
        rows = [
            row for row in rows
            if row.get("error_after_um") not in (None, "")
            and row.get("vision_delta_um") not in (None, "")
        ]
        if not rows:
            return []
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[str] = []

        fig, axes_plot = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
        for plot_axis, axis in zip(axes_plot, ("X", "Y")):
            for summary in [item for item in summaries if item["axis"] == axis]:
                subset = [
                    row for row in rows
                    if row["axis"] == axis and int(row["target_index"]) == int(summary["target_index"])
                ]
                plot_axis.plot(
                    [int(row["iteration"]) for row in subset],
                    [float(row["error_after_um"]) for row in subset],
                    marker="o",
                    label=f"T{summary['target_index']} {summary['target_absolute_um']:+.0f} μm",
                )
            plot_axis.axhspan(-3.0, 3.0, color="green", alpha=0.10)
            plot_axis.axhline(0.0, color="black", linewidth=0.8)
            plot_axis.set_ylabel(f"{axis} error / μm")
            plot_axis.grid(alpha=0.25)
            plot_axis.legend(ncol=3, fontsize=8)
        axes_plot[-1].set_xlabel("iteration")
        fig.tight_layout()
        path = plot_dir / "01_error_convergence.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(str(path))

        fig, ax = plt.subplots(figsize=(9, 6))
        styles = {("X", 1): ("tab:blue", "o"), ("X", -1): ("tab:cyan", "s"),
                  ("Y", 1): ("tab:orange", "^"), ("Y", -1): ("tab:red", "v")}
        for (axis, sign), (color, marker) in styles.items():
            subset = [
                row for row in rows
                if row["axis"] == axis and float(row["command_um"]) * sign > 0.0
            ]
            ax.scatter(
                [abs(float(row["command_um"])) for row in subset],
                [abs(float(row["vision_delta_um"])) for row in subset],
                c=color,
                marker=marker,
                label=f"{axis} {'+' if sign > 0 else '-'}",
                alpha=0.8,
            )
        max_value = max(
            [abs(float(row["command_um"])) for row in rows]
            + [abs(float(row["vision_delta_um"])) for row in rows]
            + [1.0]
        )
        ax.plot([0, max_value], [0, max_value], "k--", linewidth=0.8, label="y=x")
        ax.set_xlabel("|command| / μm")
        ax.set_ylabel("|vision measured motion| / μm")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = plot_dir / "02_command_vs_vision_motion.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(str(path))

        labels = [f"{item['axis']}T{item['target_index']}" for item in summaries]
        centers = [float(item["final_error_um"]) for item in summaries]
        lower = [center - float(item["recent_error_min_um"]) for center, item in zip(centers, summaries)]
        upper = [float(item["recent_error_max_um"]) - center for center, item in zip(centers, summaries)]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.errorbar(range(len(labels)), centers, yerr=[lower, upper], fmt="o", capsize=4)
        ax.axhspan(-3.0, 3.0, color="green", alpha=0.10)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(labels)), labels, rotation=45)
        ax.set_ylabel("final error and recent range / μm")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = plot_dir / "03_final_error_bands.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(str(path))
        return outputs

    def run_multi_target_phase(
        gains: mcl.ProbeGains, static_summary: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        matrix = mcl.probe_gain_matrix(gains)
        inverse = mcl.inverse_probe_matrix(gains)
        condition = float(np.linalg.cond(matrix))
        baseline_noise = float(static_summary.get("static_sigma_um") or 0.0)
        probe_signal = float(config.MICRO_LOOP_PROBE_UM) * float(
            min(np.linalg.norm(matrix[:, 0]), np.linalg.norm(matrix[:, 1]))
        )
        if probe_signal <= max(20.0, 10.0 * baseline_noise):
            raise RuntimeError(
                f"探针最小二维响应 {probe_signal:.1f} μm 未明显高于静止噪声 "
                f"{baseline_noise:.2f} μm，拒绝开始正式微动。"
            )
        transform_info = {
            "gain_matrix_visual_rows_robot_columns": matrix.tolist(),
            "inverse_visual_to_robot": inverse.tolist(),
            "determinant": float(np.linalg.det(matrix)),
            "condition_number": condition,
            "baseline_noise_reference_um": baseline_noise,
            "probe_min_signal_um": probe_signal,
        }
        mcl.write_json(run_dir / "visual_robot_transform.json", transform_info)
        log(
            f"完整 2×2 视觉→机器人变换通过：det={transform_info['determinant']:.4f}，"
            f"条件数={condition:.2f}，探针最小响应/静止噪声="
            f"{probe_signal / max(baseline_noise, 1e-9):.1f}",
            status=True,
        )

        robot_commands.put({"action": "STABILIZE"})
        inbox.wait(
            "robot", "STABILIZED", error_queue, stop_event, stop_request_path,
            timeout_s=30.0,
        )
        origin_saved, _ = capture_window(
            "experiment_origin",
            command=None,
            pre_seconds=float(config.MICRO_LOOP_REFERENCE_SECONDS),
            post_seconds=0.0,
        )
        origin_paths = mcl.clip_temp_paths(temp_dir, origin_saved["stem"])
        origin_clip = mcl.open_raw_clip(origin_paths["raw"])
        meter = mcl.GroupVisionMeter(label="multi_target_origin")
        try:
            primed = meter.prime(
                origin_clip,
                every_n=every_n,
                frames=int(config.MICRO_LOOP_REFERENCE_FRAMES),
            )
            mcl.write_json(
                run_dir / "experiment_origin.json",
                {"kind": "MICRO_MULTI_TARGET_ORIGIN", **primed, **transform_info},
            )
        finally:
            release(
                origin_clip,
                [origin_paths["raw"], origin_paths["meta"], origin_paths["frames"]],
                block_id="02_experiment_origin",
                keep=True,
            )
            cleanup_raw_group("02_experiment_origin", "本次实验视觉原点")
        log("本次实验视觉坐标原点已建立，X/Y 全部目标共用该原点。", status=True)

        summaries: list[dict[str, Any]] = []
        completed_axes: set[str] = set()
        for target in mcl.build_multi_target_sequence():
            axis = str(target["axis"])
            target_index = int(target["target_index"])
            target_nominal = float(target["target_nominal_um"])
            target_absolute = float(target["target_absolute_um"])
            if axis not in completed_axes:
                log(f"=== {axis} 轴多目标视觉闭环开始 ===", status=True)
                completed_axes.add(axis)
            target_rows: list[dict[str, Any]] = []
            errors: list[float] = []
            commands: list[float] = []
            motions: list[float] = []
            vision_loss_count = 0
            reverse_count = 0
            stop_reason = mcl.TARGET_MAX_ITER_REACHED
            log(
                f"{axis} Target {target_index}/6：相邻目标 {target_nominal:+.0f} μm，"
                f"绝对目标 {target_absolute:+.0f} μm",
                status=True,
            )

            for iteration in range(1, int(config.MICRO_TARGET_MAX_ITER) + 1):
                after_clip: mcl.RawClip | None = None
                after_paths: list[Path] = []
                after_measurement: mcl.ClipMeasurement | None = None
                try:
                    row, after_clip, after_paths, after_measurement = _run_target_iteration(
                        meter,
                        gains,
                        axis=axis,
                        target_index=target_index,
                        target_nominal_um=target_nominal,
                        target_absolute_um=target_absolute,
                        iteration=iteration,
                        baseline_noise_um=baseline_noise,
                    )
                    vision_loss_count = 0
                    error = float(row["error_after_um"])
                    command = float(row["command_um"])
                    motion = float(row["vision_delta_um"])
                    errors.append(error)
                    commands.append(command)
                    motions.append(motion)

                    reverse_gate = max(3.0, 2.0 * baseline_noise)
                    is_large_reverse = (
                        abs(command) > float(config.MICRO_TARGET_LARGE_REVERSE_COMMAND_UM)
                        and abs(motion) > reverse_gate
                        and command * motion < 0.0
                    )
                    reverse_count = reverse_count + 1 if is_large_reverse else 0
                    if reverse_count >= int(config.MICRO_TARGET_LARGE_REVERSE_COUNT):
                        raise RuntimeError(
                            f"{axis} Target {target_index} 连续 {reverse_count} 次大命令"
                            "产生明确反向视觉位移，已按安全异常终止。"
                        )
                    if command != 0.0 and abs(motion) > max(
                        500.0, 3.0 * abs(command) + 6.0 * baseline_noise
                    ):
                        raise RuntimeError(
                            f"{axis} Target {target_index} 视觉实测位移 {motion:+.1f} μm "
                            f"远超命令 {command:+.1f} μm，已按异常过冲终止。"
                        )

                    detected = mcl.classify_multi_target_stop(
                        errors,
                        commands,
                        motions,
                        baseline_noise_um=baseline_noise,
                    )
                    if detected is not None:
                        stop_reason = detected
                        row["stop_reason"] = detected
                    target_rows.append(row)
                    master_rows.append(row)
                    _persist_target_rows(master_rows)
                    log(
                        f"{axis} T{target_index}/6 ｜ 迭代 {iteration}/"
                        f"{int(config.MICRO_TARGET_MAX_ITER)} ｜ 误差 "
                        f"{float(row['error_before_um']):+.2f}→{error:+.2f} μm ｜ "
                        f"命令 {command:+.2f} μm ｜ 视觉实测 {motion:+.2f} μm",
                        status=True,
                    )
                    if detected is not None:
                        evidence_index = (
                            mcl.last_accepted_frame_index(after_measurement)
                            if after_measurement is not None
                            else None
                        )
                        if evidence_index is not None and after_clip is not None:
                            mcl.save_evidence_png(
                                after_clip,
                                evidence_index,
                                run_dir / axis / f"target_{target_index:02d}" / "final_evidence.png",
                            )
                        break
                except _TargetVisionLost as exc:
                    vision_loss_count += 1
                    loss_row = {column: "" for column in mcl.TARGET_ITERATION_COLUMNS}
                    loss_row.update(
                        {
                            "axis": axis,
                            "target_index": target_index,
                            "target_nominal_um": target_nominal,
                            "target_absolute_um": target_absolute,
                            "iteration": iteration,
                            "command_um": (
                                "" if exc.command_um is None else float(exc.command_um)
                            ),
                            "rtde_before": _pose_text(exc.move.get("tcp_before")),
                            "rtde_after": _pose_text(exc.move.get("tcp_after")),
                            "rtde_delta_um": exc.move.get("achieved_um", ""),
                            "baseline_noise_reference_um": baseline_noise,
                            "stop_reason": (
                                "VISION_ABORT"
                                if vision_loss_count >= int(config.MICRO_TARGET_VISION_LOSS_COUNT)
                                else "VISION_RETRY"
                            ),
                            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                            "note": str(exc),
                        }
                    )
                    master_rows.append(loss_row)
                    _persist_target_rows(master_rows)
                    log(
                        f"{axis} T{target_index}/6 第 {iteration} 次视觉测量不合格：{exc}；"
                        f"连续 {vision_loss_count}/{int(config.MICRO_TARGET_VISION_LOSS_COUNT)} 次。",
                        status=True,
                    )
                    if vision_loss_count >= int(config.MICRO_TARGET_VISION_LOSS_COUNT):
                        raise RuntimeError(
                            f"{axis} Target {target_index} 连续视觉测量失败，"
                            "为避免盲目运动已安全终止。"
                        ) from exc
                finally:
                    if after_paths:
                        block_id = f"{axis}_T{target_index:02d}_I{iteration:02d}"
                        release(after_clip, after_paths, block_id=block_id, keep=True)
                        cleanup_raw_group(block_id, f"{axis} T{target_index} 第 {iteration} 次迭代")

            if not target_rows:
                raise RuntimeError(f"{axis} Target {target_index} 没有获得任何有效迭代。")
            summary = _target_summary(
                target_rows, axis=axis, target_index=target_index, stop_reason=stop_reason
            )
            summaries.append(summary)
            target_dir = run_dir / axis / f"target_{target_index:02d}"
            mcl.write_json(target_dir / "summary.json", summary)
            recent_clear = summary["empirical_smallest_clear_same_direction_command_um"]
            log(
                f"[{axis} Target {target_index}] 目标 {target_absolute:+.0f} μm ｜ "
                f"迭代 {summary['iterations']} 次 ｜ 最终误差 "
                f"{summary['final_error_um']:+.2f} μm ｜ 最近范围 "
                f"{summary['recent_error_min_um']:+.2f}~{summary['recent_error_max_um']:+.2f} μm ｜ "
                f"最小明确同向响应命令 "
                f"{'无法确定' if recent_clear is None else f'{float(recent_clear):.2f} μm'} ｜ "
                f"状态 {stop_reason}",
                status=True,
            )

            if target_index == len(config.MICRO_TARGET_ABSOLUTE_UM):
                robot_commands.put({"action": "RETURN_START"})
                returned = inbox.wait(
                    "robot", "RETURNED", error_queue, stop_event, stop_request_path,
                    timeout_s=config.ROBOT_MOTION_TIMEOUT_S * 2,
                )
                log(
                    f"{axis} 轴目标序列结束，已回到实验初始位姿附近（RTDE偏差 "
                    f"{float(returned.get('drift_mm', math.nan)):.4f} mm）。",
                    status=True,
                )

        mcl.write_json(run_dir / "target_summaries.json", summaries)
        return master_rows, summaries, transform_info

    def write_multi_target_report(
        rows: list[dict[str, Any]],
        summaries: list[dict[str, Any]],
        transform_info: dict[str, Any],
        static_summary: dict[str, Any],
    ) -> None:
        plots = _write_multi_target_plots(rows, summaries)
        axis_reports: dict[str, Any] = {}
        text_lines = ["XY 多目标视觉闭环逼近实验", ""]
        for axis in ("X", "Y"):
            axis_rows = [
                row for row in rows
                if row["axis"] == axis
                and row.get("vision_delta_um") not in (None, "")
            ]
            axis_summaries = [item for item in summaries if item["axis"] == axis]
            clear = [
                abs(float(row["command_um"]))
                for row in axis_rows
                if row["command_and_motion_same_direction"] is True
                and abs(float(row["vision_delta_um"]))
                > max(1.0, 2.0 * float(row["baseline_noise_reference_um"]))
            ]
            final_errors = [abs(float(item["final_error_um"])) for item in axis_summaries]
            empirical = min(clear) if len(clear) >= 3 else None
            axis_reports[axis] = {
                "empirical_smallest_clear_same_direction_command_um": empirical,
                "final_abs_error_median_um": (
                    float(np.median(final_errors)) if final_errors else None
                ),
                "final_abs_error_max_um": max(final_errors) if final_errors else None,
                "target_stop_reasons": [item["stop_reason"] for item in axis_summaries],
                "conclusion": (
                    f"本轮多目标数据中，约 {empirical:.1f} μm 及以上曾出现明确同向响应；"
                    "该值是经验观测下限，不是严格最小运动单元。"
                    if empirical is not None
                    else "当前数据不能确定单一最小运动阈值。"
                ),
            }
            text_lines.append(f"{axis}轴：{axis_reports[axis]['conclusion']}")
            if final_errors:
                text_lines.append(
                    f"  多目标最终 |误差| 中位数 {np.median(final_errors):.2f} μm，"
                    f"最大 {max(final_errors):.2f} μm。"
                )
        text_lines.extend(
            [
                "",
                "说明：3 μm 稳定带接近视觉观测下限，不等于宣称机器人真实精度为 3 μm。",
                "RTDE actual_TCP_pose 仅用于安全、工作空间和旁证，不参与微米级方向终止。",
            ]
        )
        report = {
            "kind": "MICRO_MULTI_TARGET_FINAL_REPORT",
            "targets": summaries,
            "axes": axis_reports,
            "static_baseline": static_summary,
            "transform": transform_info,
            "plots": plots,
            "iteration_csv": str(run_dir / "iterations.csv"),
        }
        mcl.write_json(run_dir / "final_report.json", report)
        (run_dir / "final_report.txt").write_text(
            "\n".join(text_lines) + "\n", encoding="utf-8"
        )
        for line in text_lines:
            if line:
                log(line, status=True)

    # -------------------------------------------------------------------------
    # 主流程
    # -------------------------------------------------------------------------

    try:
        # 机器人先起、相机后起：CB3 的 RTDE 握手不要和相机初始化抢时间。
        robot.start()
        robot_ready = inbox.wait(
            "robot", "READY", error_queue, stop_event, stop_request_path,
            timeout_s=config.WORKER_READY_TIMEOUT_S,
        )
        log(
            f"UR10 就绪，初始 TCP 位姿 ({robot_ready['start_pose'][0]:.6f}, "
            f"{robot_ready['start_pose'][1]:.6f}, {robot_ready['start_pose'][2]:.6f}) m",
            status=True,
        )

        camera.start()
        camera_ready = inbox.wait(
            "camera", "READY", error_queue, stop_event, stop_request_path,
            timeout_s=config.WORKER_READY_TIMEOUT_S,
        )
        fps = float(camera_ready["actual_camera_fps"])
        capacity_frames = int(math.ceil(float(config.MICRO_LOOP_MAX_WINDOW_SECONDS) * fps)) + 8
        log(
            f"工业相机就绪：{camera_ready['full_width']}×{camera_ready['full_height']} ｜ "
            f"{fps:.3f} fps ｜ 缓冲可容纳 {capacity_frames} 帧"
            f"（约 {mcl.format_gib(capacity_frames * int(camera_ready['full_width']) * int(camera_ready['full_height']))}）",
            status=True,
        )
        log(f"磁盘剩余 {mcl.free_gb(config.OUTPUT_ROOT):.2f} GiB", status=True)

        # 当前唯一主动路径：通信 → 相机 → 探针 → 5 s 静止基线 → 单一视觉原点
        # → X 六目标 → 回原点 → Y 六目标 → 回原点 → 汇总。
        gains = run_probe()
        static_summary = run_static_baseline()
        rows, summaries, transform_info = run_multi_target_phase(gains, static_summary)
        write_multi_target_report(rows, summaries, transform_info, static_summary)
        log(f"全部实验完成：多目标视觉闭环 12/12 个目标。", status=True)
        log(f"结果目录：{run_dir}", status=True)
        log(
            f"RAW 运行目录：{raw_run_dir}（每次迭代视频已立即删除，任务收尾再兜底清理）",
            status=True,
        )
    finally:
        stop_event.set()
        for process in (robot, camera):
            if process.pid is not None:
                _join_or_terminate(process)
        mcl.write_rows_csv(
            run_dir / "iterations.csv", master_rows, mcl.TARGET_ITERATION_COLUMNS
        )
        # 循环全部结束之后才清回收目录，绝不中途清（那里可能有别人正在用的文件）。
        try:
            released = mcl.sweep_directory(trash_dir)
            log(f"临时文件回收目录已清空，释放 {mcl.format_gib(released)}")
        except OSError as exc:
            log(f"回收目录清理失败（不影响实验数据）：{exc}")
        try:
            released = mcl.sweep_directory(temp_dir)
            if released:
                log(f"活动临时目录已兜底清空，释放 {mcl.format_gib(released)}")
        except OSError as exc:
            log(f"活动临时目录兜底清理失败：{exc}")
        if bool(config.MICRO_LOOP_DELETE_VIDEO_FILES_ON_EXIT):
            released, failed = mcl.delete_video_payloads(raw_run_dir)
            if failed:
                log(
                    f"本次任务视频载荷已自动清理 {mcl.format_gib(released)}，"
                    f"但有 {len(failed)} 个文件仍被占用；下次运行前请关闭占用程序后删除："
                    + "，".join(str(path) for path in failed[:3])
                )
            else:
                log(
                    f"本次任务的大体积 RAW/视频文件已自动删除，释放 "
                    f"{mcl.format_gib(released)}；CSV/JSON/证据图保留。",
                    status=True,
                )
        for leftover in (temp_dir, trash_dir):
            try:
                if leftover.exists() and not any(leftover.iterdir()):
                    leftover.rmdir()
            except OSError:
                pass
        _clear_stop_request_file(stop_request_path)

    return run_dir


def run_boundary_check_mode(arguments: argparse.Namespace) -> Path:
    """Run the low-speed, no-flash, operator-observed envelope route."""

    from batch_plan import compute_plan_envelope, enabled_plan, read_plan
    from camera import boundary_camera_preview_worker
    from robot import boundary_check_robot_worker

    if arguments.batch_plan_file is None:
        raise ValueError("boundary_check 必须提供 --batch-plan-file。")
    if not arguments.ui_confirmed:
        from robot import require_operator_confirmation

        require_operator_confirmation("计算并检查视野边界")
    plan = enabled_plan(read_plan(arguments.batch_plan_file))
    envelope = compute_plan_envelope(plan)
    if not envelope:
        raise ValueError("当前启用计划没有运动轨迹，无法执行边界检查。")
    stop_request_path = arguments.stop_request_path
    if stop_request_path is not None and not stop_request_path.is_absolute():
        stop_request_path = (config.PROJECT_DIR / stop_request_path).resolve()
    _clear_stop_request_file(stop_request_path)
    output_dir = config.OUTPUT_ROOT / f"boundary_check_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "envelope.json").write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    context = mp.get_context("spawn")
    error_queue = context.Queue(maxsize=config.ERROR_QUEUE_MAXSIZE)
    status_queue = context.Queue(maxsize=32)
    stop_event = context.Event()
    camera = context.Process(
        target=boundary_camera_preview_worker,
        args=(error_queue, status_queue, stop_event),
        name="boundary-camera-preview",
    )
    robot = context.Process(
        target=boundary_check_robot_worker,
        args=(error_queue, status_queue, envelope, stop_event),
        name="boundary-robot",
    )
    inbox = _StatusInbox(status_queue)
    camera.start()
    try:
        inbox.wait(
            "camera", "READY", error_queue, stop_event, stop_request_path,
            timeout_s=config.WORKER_READY_TIMEOUT_S,
        )
        robot.start()
        ready = inbox.wait(
            "robot", "READY", error_queue, stop_event, stop_request_path,
            timeout_s=config.WORKER_READY_TIMEOUT_S,
        )
        print(f"[状态] 边界检查A点：{ready['start_pose']}", flush=True)
        finished = inbox.wait(
            "robot", "FINISHED", error_queue, stop_event, stop_request_path,
            timeout_s=config.ROBOT_MOTION_TIMEOUT_S * max(1, len(envelope)),
        )
        (output_dir / "result.json").write_text(
            json.dumps({"status": "COMPLETED", **finished}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("[状态] 视野边界检查完成，已回到A点；不会自动开始正式实验", flush=True)
    finally:
        stop_event.set()
        for process in (robot, camera):
            if process.pid is not None:
                _join_or_terminate(process)
        _clear_stop_request_file(stop_request_path)
    return output_dir


def run_batch_vision_offline_mode(arguments: argparse.Namespace) -> Path:
    """Explicitly process a finished batch; this mode never opens camera or UR."""

    from camera import write_batch_segment_vision

    batch_dir = arguments.batch_dir
    if batch_dir is None:
        candidates = sorted(
            (
                path
                for path in config.OUTPUT_ROOT.iterdir()
                if path.is_dir() and (path / "segments.csv").exists()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("outputs 中没有包含 segments.csv 的批次目录。")
        batch_dir = candidates[0]
    if not batch_dir.is_absolute():
        batch_dir = (config.PROJECT_DIR / batch_dir).resolve()
    segments_path = batch_dir / "segments.csv"
    if not segments_path.exists():
        raise FileNotFoundError(f"批次目录缺少 segments.csv：{segments_path}")
    with segments_path.open("r", encoding="utf-8-sig", newline="") as file:
        segments = list(csv.DictReader(file))

    processed_segments = 0
    skipped_segments = 0
    try:
        for segment in segments:
            if segment.get("status") != "COMPLETED":
                continue
            video_path = Path(segment["output_video"])
            if not video_path.is_absolute():
                video_path = batch_dir / video_path
            elif not video_path.exists():
                # Allow an intact batch folder to be copied to another computer.
                video_path = batch_dir / video_path.name
            vision_path = video_path.with_name(video_path.name.replace("_HIK.avi", "_VISION.txt"))
            timestamp_path = video_path.with_name(
                video_path.name.replace("_HIK.avi", "_FRAME_TIMESTAMPS.csv")
            )
            if vision_path.exists():
                print(f"[离线识别] 已存在，跳过且不覆盖：{vision_path.name}", flush=True)
                skipped_segments += 1
                continue
            segment_base = video_path.name.removesuffix("_HIK.avi")
            count = write_batch_segment_vision(
                video_path, timestamp_path, vision_path, segment_base
            )
            meta_path = video_path.with_name(video_path.name.replace("_HIK.avi", "_META.json"))
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["vision_processing"] = {
                    "mode": "explicit_batch_vision_offline",
                    "processed_at": datetime.now().isoformat(timespec="seconds"),
                    "frame_count": count,
                    "timestamp_sidecar": str(timestamp_path),
                    "uses_original_hardware_frame_id_and_host_ns": True,
                }
                meta["vision_processing_status"] = "COMPLETED"
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            processed_segments += 1
            print(
                f"[离线识别] {processed_segments} 段完成：{vision_path.name}，{count} 帧",
                flush=True,
            )
    except Exception:
        batch_meta_path = batch_dir / "batch_META.json"
        if batch_meta_path.exists():
            batch_meta = json.loads(batch_meta_path.read_text(encoding="utf-8"))
            batch_meta["vision_processing_status"] = "FAILED"
            batch_meta_path.write_text(
                json.dumps(batch_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        raise

    batch_meta_path = batch_dir / "batch_META.json"
    if batch_meta_path.exists():
        batch_meta = json.loads(batch_meta_path.read_text(encoding="utf-8"))
        batch_meta["vision_processing_status"] = "COMPLETED"
        batch_meta["vision_processed_at"] = datetime.now().isoformat(timespec="seconds")
        batch_meta["vision_processed_segments"] = processed_segments
        batch_meta["vision_skipped_existing_segments"] = skipped_segments
        batch_meta_path.write_text(
            json.dumps(batch_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"[状态] 批次离线视觉完成：新生成 {processed_segments} 段，跳过 {skipped_segments} 段",
        flush=True,
    )
    return batch_dir


def run_experiment_mode() -> Path:
    """
    用户选择 experiment 后进入这里，这是完整预实验的总流程。

    输入：
    - config.py 中的视觉来源、机器人轨迹、安全开关、记录时长；
    - 操作者在终端中的最后确认。

    输出：
    - experiment_时间戳 文件夹；
    - run_log.txt，其中混合保存 META、EVENT、VISION、ROBOT、ERROR 等记录。

    实验作用：
    - 相机进程负责取图和视觉计算；
    - 机器人进程负责连接 UR、执行轨迹和记录状态；
    - 写日志进程负责把两边数据按到达顺序落盘；
    - 主进程负责等 ready、发 start、等运动完成、保留后记录时间并统一收尾。
    只有这个模式使用多进程；单模块测试保持单进程，便于读代码和定位报错。
    """

    # 本段选择 Windows 友好的子进程启动方式。
    # 输入：三个 worker 函数；输出：独立 Python 子进程。
    # 可以把 spawn 理解为“重新启动一个 Python，再让它执行指定函数”。
    context = mp.get_context("spawn")

    # 本段建立两条跨进程消息通道。
    # record_queue 输出到 run_log.txt；error_queue 输出到主程序异常处理。
    # 实验数据走 record_queue，故障信息走 error_queue，二者分开能让主程序更快响应错误。
    record_queue = context.Queue(maxsize=config.RECORD_QUEUE_MAXSIZE)
    error_queue = context.Queue(maxsize=config.ERROR_QUEUE_MAXSIZE)

    # 本段建立完整实验的流程信号。
    # 输入/输出都是跨进程 Event：某一方置位，其他进程就能读到。
    # start_event：主进程通知相机和机器人同时进入正式阶段。
    # stop_event：主进程或异常路径通知所有子进程尽快收尾。
    # camera_ready / robot_ready：子进程告诉主进程“我已经准备好”。
    # motion_done：机器人告诉主进程“运动阶段已经结束”。
    start_event = context.Event()
    stop_event = context.Event()
    camera_ready = context.Event()
    robot_ready = context.Event()
    motion_done = context.Event()

    # 本段创建本次实验的输出目录。
    # 输入：当前日期时间；输出：experiment_时间戳 文件夹和其中的 run_log.txt 路径。
    # 实验作用：每次正式实验单独存档，避免日志互相覆盖。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUT_ROOT / f"experiment_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run_log.txt"

    # 本段加载完整实验真正需要的相机和机器人入口。
    # 延迟导入让普通帮助、配置检查或分析模式不会被硬件依赖提前卡住。
    from camera import camera_worker
    from robot import (
        build_trajectory,
        require_operator_confirmation,
        robot_worker,
        validate_trajectory,
    )

    # 本段在启动硬件子进程前做轨迹安全检查。
    # 输入：config.py 中生成的轨迹；输出：允许继续或直接抛错。
    # 实验作用：示例位姿、越界轨迹或过长线段会在这里停止，不让它进入真机流程。
    validate_trajectory(build_trajectory(), for_real_robot=True)

    # 本段定义三个长期运行的子进程。
    # 输入：队列和事件信号；输出：三个还未 start 的 Process 对象。
    # 实验作用：把写文件、取图识别、机器人控制分开，避免某一类工作阻塞另一类工作。
    writer_process = context.Process(
        target=record_writer_worker,
        args=(record_queue, log_path),
        name="record-writer",
    )
    camera_process = context.Process(
        target=camera_worker,
        args=(
            record_queue,
            error_queue,
            start_event,
            stop_event,
            camera_ready,
        ),
        name="camera-worker",
    )
    robot_process = context.Process(
        target=robot_worker,
        args=(
            record_queue,
            error_queue,
            start_event,
            stop_event,
            robot_ready,
            motion_done,
        ),
        name="robot-worker",
    )
    processes = [camera_process, robot_process]

    # 本段先启动写日志进程，并写入本次实验的 META。
    # 输入：config.py 中关键实验参数；输出：run_log.txt 第一类说明性记录。
    # 实验作用：后续即使只拿到日志文件，也能知道当时用的是哪种视觉来源、轨迹和坐标配置。
    writer_process.start()
    meta = {
        "kind": "META",
        "host_ns": time.perf_counter_ns(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "experiment",
        "vision_source": config.VISION_SOURCE,
        "vision_method": config.VISION_METHOD,
        "expected_vision_fps": config.EXPECTED_VISION_FPS,
        "save_point_coordinates": config.SAVE_POINT_COORDINATES,
        "spatial_coordinates_enabled": config.SPATIAL_COORDINATES_ENABLED,
        "spatial_coordinate_mode": config.SPATIAL_COORDINATE_MODE,
        "trajectory_type": config.TRAJECTORY_TYPE,
        "control_mode": config.CONTROL_MODE,
        "executor_mode": config.EXECUTOR_MODE,
        "pre_record_seconds": config.PRE_RECORD_SECONDS,
        "post_record_seconds": config.POST_RECORD_SECONDS,
        "robot_record_hz": config.ROBOT_RECORD_HZ,
    }
    record_queue.put(meta, timeout=1.0)

    try:
        # 本段启动相机和机器人，但暂不开始正式运动。
        # 输入：两个子进程；输出：camera_ready 和 robot_ready 最终都被置位。
        # 实验作用：确保相机能取图、机器人能连接并通过起点/轨迹检查后，再进入人工确认。
        camera_process.start()
        robot_process.start()
        _wait_ready(camera_ready, robot_ready, error_queue, stop_event)
        _record_event(record_queue, "all_workers_ready")

        # 本段是正式开始前的最后人工闸门。
        # 输入：操作者在终端输入的确认文本；输出：start_event 置位。
        # 实验作用：只有人确认现场安全后，相机和机器人才同时进入正式实验阶段。
        require_operator_confirmation("相机 + UR10 正式振动预实验")
        _record_event(record_queue, "experiment_started")
        start_event.set()

        # 本段等待机器人完成运动阶段。
        # 输入：robot_worker 置位的 motion_done；输出：进入 post 后记录阶段或因错误停止。
        # 实验作用：主程序不插手每个运动采样点，只监督运动阶段是否结束。
        _wait_motion_done(motion_done, error_queue, stop_event)
        print(
            f"[主程序] 运动阶段完成，继续记录 {config.POST_RECORD_SECONDS:.2f} s 残余振动。"
        )

        # 本段保持相机和机器人状态记录继续运行一小段时间。
        # 输入：POST_RECORD_SECONDS；输出：run_log.txt 中的后记录段。
        # 实验作用：观察机械臂运动结束后的残余振动如何衰减。
        post_deadline = time.perf_counter() + config.POST_RECORD_SECONDS
        while time.perf_counter() < post_deadline:
            worker_error = _pop_worker_error(error_queue)
            if worker_error:
                raise RuntimeError(worker_error)
            if stop_event.is_set():
                raise RuntimeError("后记录期间某个子进程异常停止。")
            time.sleep(0.05)

        _record_event(record_queue, "experiment_finished")
        stop_event.set()
    except Exception as exc:
        # 本段是完整实验的统一失败出口。
        # 输入：ready、运动、后记录、写队列等任意阶段抛出的异常。
        # 输出：stop_event 置位，并尽量把 ERROR 记录写进 run_log.txt。
        # 实验作用：出现问题时优先让硬件相关子进程收尾，日志里也留下失败原因。
        stop_event.set()
        try:
            record_queue.put(
                {
                    "kind": "ERROR",
                    "host_ns": time.perf_counter_ns(),
                    "message": f"{type(exc).__name__}: {exc}",
                },
                timeout=1.0,
            )
        except queue.Full:
            pass
        raise
    finally:
        # 本段是完整实验的统一收尾出口。
        # 输入：当前仍存在的子进程、队列和 stop_event；输出：子进程退出，队列关闭。
        # 实验作用：无论成功还是失败，都尽量避免留下后台取图、机器人连接或写文件进程。
        stop_event.set()

        # 只回收已经成功 start 的进程；pid 非空说明这个 Process 确实启动过。
        for process in processes:
            if process.pid is not None:
                _join_or_terminate(process)

        # 所有生产者停止后再发 None，保证队列中排在它前面的记录先全部落盘。
        record_queue.put(None, timeout=2.0)
        _join_or_terminate(writer_process)

        record_queue.close()
        error_queue.close()

    print(f"[主程序] 正式实验完成，原始记录：{log_path}")
    print("[主程序] 把 RUN_MODE 改为 analyze，即可生成时域、频域和摘要结果。")
    return run_dir


# =============================================================================
# 3. 命令行与总分流：程序启动后最先经过的用户选择入口
# =============================================================================

def _parse_arguments() -> argparse.Namespace:
    """
    读取用户在终端里临时补充的运行选项。

    输入：终端命令，例如 python main.py --mode robot_dry_run。
    输出：arguments 对象，其中包含 mode、analysis_file 等字段。

    实验作用：命令行参数是临时覆盖，不会改写 config.py 文件本身。
    例如你可以临时跑一次 robot_dry_run，而不用把 RUN_MODE 来回改动。
    """

    # 本段创建命令行解析器。输入是用户敲的参数；输出是可读的 Python 字段。
    # argparse 还会自动生成 --help 页面，并帮忙检查 mode 是否在允许列表中。
    parser = argparse.ArgumentParser(
        description="UR10 末端振动预实验：视觉、轨迹、真机、完整实验与离线分析。"
    )

    # 本段给用户一个临时改模式的入口。
    # 输入：--mode 后面的字符串；输出：arguments.mode。
    # 实验作用：同一份 config.py 不变，也能临时切到视觉测试、干跑或分析模式。
    parser.add_argument(
        "--mode",
        choices=sorted(config.VALID_RUN_MODES),
        default=None,
        help="临时覆盖 config.py 的 RUN_MODE。",
    )

    # 本段给 analyze 模式一个临时指定日志文件的入口。
    # 输入：--analysis-file 后面的路径；输出：arguments.analysis_file。
    # 实验作用：可以复查某一次指定实验，而不是总分析 outputs 里最新的一次。
    parser.add_argument(
        "--analysis-file",
        type=Path,
        default=None,
        help="analyze 模式下临时指定 run_log.txt 或 vision_results.txt。",
    )

    parser.add_argument("--speed-mm-s", type=float, default=config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S)
    parser.add_argument(
        "--one-way-time-s",
        type=float,
        default=None,
        help=(
            "A 到最远点的单程时间；省略时按轨迹使用缩短后的 X/D 或已验证的 L 默认值。"
        ),
    )
    parser.add_argument("--total-time-s", type=float, default=config.ROBOT_EXPERIMENT_DEFAULT_TOTAL_TIME_S)
    parser.add_argument("--angle-deg", type=float, default=config.ROBOT_EXPERIMENT_DEFAULT_ANGLE_DEG)
    parser.add_argument("--x-direction", choices=["+X", "-X"], default="+X")
    parser.add_argument("--y-direction", choices=["+Y", "-Y"], default="+Y")
    parser.add_argument("--x-speed-mm-s", type=float, default=config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S)
    parser.add_argument(
        "--x-one-way-time-s",
        type=float,
        default=config.ROBOT_EXPERIMENT_DEFAULT_X_ONE_WAY_TIME_S,
    )
    parser.add_argument("--y-speed-mm-s", type=float, default=config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S)
    parser.add_argument(
        "--y-one-way-time-s",
        type=float,
        default=config.ROBOT_EXPERIMENT_DEFAULT_Y_ONE_WAY_TIME_S,
    )
    parser.add_argument("--blend-mm", type=float, default=config.ROBOT_RELATIVE_BLEND_MM)
    parser.add_argument(
        "--batch-plan-file",
        type=Path,
        default=None,
        help="batch_experiment / boundary_check 使用的已编辑计划 JSON。",
    )
    parser.add_argument(
        "--batch-manual-label",
        default="",
        help="批次基础名称末尾的可选人工标签。",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="batch_vision_offline 要处理的已完成批次目录；省略时取最新批次。",
    )
    parser.add_argument(
        "--ui-confirmed",
        action="store_true",
        help="启动器已完成索尼录像、路径方向、无人区和急停确认。",
    )
    parser.add_argument(
        "--stop-request-path",
        type=Path,
        default=None,
        help="启动器为本次运动实验写入的 run 级停止请求文件。",
    )

    # 本段真正读取终端参数，并把字符串转换成 Python 可用的对象。
    return parser.parse_args()


def main() -> Path:
    """
    用户启动程序后真正执行的总流程。

    输入：
    - config.py 中的默认配置；
    - 可选命令行参数，例如 --mode 或 --analysis-file。

    输出：
    - 被选中模式创建的输出目录 Path。

    实验作用：
    - 它是全项目唯一的总分流点；
    - 它先确认“这次到底要跑什么”，再做配置检查，最后只进入一个模式入口；
    - 视觉、机器人、分析的具体工作都不在这里完成，而是在对应模块里完成。
    """

    # 本段读取用户启动程序时附加的临时选择。
    # 没有命令行参数时，arguments 中相关字段就是 None。
    arguments = _parse_arguments()

    # 本段决定最终模式。命令行 --mode 优先级更高，但不会修改 config.py。
    # 输入：arguments.mode 和 config.RUN_MODE；输出：selected_mode。
    selected_mode = arguments.mode or config.RUN_MODE

    # 本段是所有模式共同经过的启动前检查。
    # 输入：最终模式和 config.py 中的全部相关参数；输出：继续运行或抛出清晰错误。
    # 实验作用：在接触相机、机器人或分析文件前，先拦住明显错误配置。
    config.validate_config(selected_mode)

    print(f"[主程序] 当前模式：{selected_mode}")

    # 本段是用户模式到代码入口的一对一映射。
    # 输入：selected_mode；输出：调用对应 run_*_mode，并返回该模式输出目录。
    # 实验作用：保证一次运行只做一类事情，不会同时误跑视觉测试和真机实验。
    if selected_mode == "vision_test":
        return run_vision_test_mode()

    if selected_mode == "vision_capture":
        return run_vision_capture_mode()

    if selected_mode == "vision_offline":
        return run_vision_offline_mode()

    if selected_mode == "offline_static_test":
        return run_offline_static_test_mode(arguments)

    if selected_mode == "robot_dry_run":
        return run_robot_dry_run_mode()

    if selected_mode == "robot_connection_test":
        return run_robot_connection_test_mode()

    if selected_mode == "robot_test":
        return run_robot_test_mode()

    if selected_mode == "experiment":
        return run_experiment_mode()

    if selected_mode in RELATIVE_MOTION_MODES:
        return run_relative_motion_experiment_mode(selected_mode, arguments)

    if selected_mode == "batch_experiment":
        return run_batch_experiment_mode(arguments)

    if selected_mode == "micro_closed_loop":
        return run_micro_closed_loop_mode(arguments)

    if selected_mode == "batch_vision_offline":
        return run_batch_vision_offline_mode(arguments)

    if selected_mode == "boundary_check":
        return run_boundary_check_mode(arguments)

    if selected_mode == "analyze":
        return run_analysis_mode(arguments.analysis_file)

    # 本段是未来维护用的兜底检查。
    # 如果以后新增模式时只改了 config.py、忘了改 main.py，会在这里得到明确错误。
    raise RuntimeError(f"RUN_MODE={selected_mode!r} 通过了配置检查，但 main.py 没有对应分支。")


if __name__ == "__main__":
    # 本段只在“用户直接运行 main.py”时触发。
    # Windows 子进程会重新导入 main.py；这层保护防止子进程再次启动整套实验。
    mp.freeze_support()
    main()
