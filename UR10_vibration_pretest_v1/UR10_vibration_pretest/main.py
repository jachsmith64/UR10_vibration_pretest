"""
UR10 末端振动预实验的唯一总入口。

main.py 只负责“根据模式叫谁工作”和“完整实验的多进程协调”，不在这里实现视觉或轨迹算法。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import config


# =============================================================================
# 1. 通用记录进程
# =============================================================================

def record_writer_worker(record_queue: Any, output_path: Path) -> None:
    """
    独占打开正式实验 TXT，逐条写入相机、UR、事件和错误记录。

    多个生产者只往队列放字典，不直接碰文件，因此不会出现两行内容互相穿插。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", buffering=1) as file:
        while True:
            record = record_queue.get()
            if record is None:
                break

            # 每行是一条完整 JSON；扩展名保留 txt，方便记事本查看，也便于程序可靠恢复字段。
            line = json.dumps(record, ensure_ascii=False, allow_nan=True)
            file.write(line + "\n")


def _record_event(record_queue: Any, name: str, **extra: Any) -> None:
    """主程序产生的时间节点也进入同一队列，确保分析时使用完全相同的 host_ns 时基。"""

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
    """非阻塞读取一条子进程错误；没有错误时立刻返回，不拖慢主调度循环。"""

    try:
        return str(error_queue.get_nowait())
    except queue.Empty:
        return None


def _wait_ready(
    camera_ready: Any,
    robot_ready: Any,
    error_queue: Any,
    stop_event: Any,
) -> None:
    """等待相机和机器人都通过各自实际初始化，而不是仅仅等进程被创建。"""

    deadline = time.perf_counter() + config.WORKER_READY_TIMEOUT_S
    while not (camera_ready.is_set() and robot_ready.is_set()):
        worker_error = _pop_worker_error(error_queue)
        if worker_error:
            raise RuntimeError(worker_error)

        if stop_event.is_set():
            raise RuntimeError("某个子进程在 ready 前要求停止，但没有返回更多错误信息。")

        if time.perf_counter() >= deadline:
            missing = []
            if not camera_ready.is_set():
                missing.append("相机")
            if not robot_ready.is_set():
                missing.append("机器人")
            raise TimeoutError(f"等待 {'、'.join(missing)} ready 超时。")

        time.sleep(0.05)


def _wait_motion_done(
    motion_done: Any,
    error_queue: Any,
    stop_event: Any,
) -> None:
    """正式运动期间持续检查子进程错误，避免只盯着 motion_done 一直卡住。"""

    deadline = (
        time.perf_counter()
        + config.PRE_RECORD_SECONDS
        + config.ROBOT_MOTION_TIMEOUT_S
        + 5.0
    )
    while not motion_done.is_set():
        worker_error = _pop_worker_error(error_queue)
        if worker_error:
            raise RuntimeError(worker_error)

        if stop_event.is_set():
            raise RuntimeError("实验在运动完成前收到停止信号。")

        if time.perf_counter() >= deadline:
            raise TimeoutError("主程序等待 motion_done 超时。")

        time.sleep(0.05)


def _join_or_terminate(process: mp.Process) -> None:
    """
    先给子进程充分时间执行 finally 释放相机/RTDE，只有失去响应时才终止。

    terminate 是最后清理手段，因此正常流程不会靠它结束硬件连接。
    """

    process.join(timeout=config.WORKER_JOIN_TIMEOUT_S)
    if process.is_alive():
        print(f"[主程序警告] {process.name} 未按时退出，将终止该子进程。")
        process.terminate()
        process.join(timeout=2.0)


# =============================================================================
# 2. 五种模式的高层入口
# =============================================================================

def run_vision_test_mode() -> Path:
    """只在这里延迟导入 camera.py，使其他模式不必加载 OpenCV 和 MVS 适配代码。"""

    from camera import run_vision_test

    return run_vision_test()


def run_robot_dry_run_mode() -> Path:
    """dry-run 不创建 URRobot，也不导入 ur_rtde。"""

    from robot import run_robot_dry_run

    return run_robot_dry_run()


def run_robot_test_mode() -> Path:
    """机器人单机测试由 robot.py 自己执行双重运动开关检查。"""

    from robot import run_robot_test

    return run_robot_test()


def run_analysis_mode(analysis_file: Path | None = None) -> Path:
    """分析模块完全离线，可处理刚完成或历史保存的记录。"""

    from analyze import run_analysis

    return run_analysis(analysis_file)


def run_experiment_mode() -> Path:
    """
    启动记录、相机和机器人三个子进程，并负责 ready、开始、完成和停止信号。

    只有这个模式使用多进程；单模块测试保持单进程，报错堆栈更容易理解。
    """

    # Windows 使用 spawn 创建子进程，目标函数必须位于模块顶层，且所有对象必须可序列化。
    context = mp.get_context("spawn")
    record_queue = context.Queue(maxsize=config.RECORD_QUEUE_MAXSIZE)
    error_queue = context.Queue(maxsize=config.ERROR_QUEUE_MAXSIZE)

    start_event = context.Event()
    stop_event = context.Event()
    camera_ready = context.Event()
    robot_ready = context.Event()
    motion_done = context.Event()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUT_ROOT / f"experiment_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run_log.txt"

    # 延迟导入不仅减少启动依赖，也让 main.py 顶部永远不会因缺 MVS 或 ur_rtde 而报错。
    from camera import camera_worker
    from robot import (
        build_trajectory,
        require_operator_confirmation,
        robot_worker,
        validate_trajectory,
    )

    # 在创建任何硬件进程前先执行纯软件安全检查；示例位姿未确认时会在这里立即停止。
    validate_trajectory(build_trajectory(), for_real_robot=True)

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

    writer_process.start()
    meta = {
        "kind": "META",
        "host_ns": time.perf_counter_ns(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "experiment",
        "vision_source": config.VISION_SOURCE,
        "vision_method": config.VISION_METHOD,
        "trajectory_type": config.TRAJECTORY_TYPE,
        "control_mode": config.CONTROL_MODE,
        "executor_mode": config.EXECUTOR_MODE,
        "pre_record_seconds": config.PRE_RECORD_SECONDS,
        "post_record_seconds": config.POST_RECORD_SECONDS,
        "robot_record_hz": config.ROBOT_RECORD_HZ,
    }
    record_queue.put(meta, timeout=1.0)

    try:
        camera_process.start()
        robot_process.start()
        _wait_ready(camera_ready, robot_ready, error_queue, stop_event)
        _record_event(record_queue, "all_workers_ready")

        # 到这里才说明相机已成功取帧、UR 已连接、轨迹和起点已通过检查。
        require_operator_confirmation("相机 + UR10 正式振动预实验")
        _record_event(record_queue, "experiment_started")
        start_event.set()

        _wait_motion_done(motion_done, error_queue, stop_event)
        print(
            f"[主程序] 运动阶段完成，继续记录 {config.POST_RECORD_SECONDS:.2f} s 残余振动。"
        )

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
        stop_event.set()

        # 只 join 已经成功 start 的进程；Python 的 pid 非空可作为安全判断。
        for process in processes:
            if process.pid is not None:
                _join_or_terminate(process)

        # 所有生产者都停止后再发 None，保证队列里排在它前面的记录会全部落盘。
        record_queue.put(None, timeout=2.0)
        _join_or_terminate(writer_process)

        record_queue.close()
        error_queue.close()

    print(f"[主程序] 正式实验完成，原始记录：{log_path}")
    print("[主程序] 把 RUN_MODE 改为 analyze，即可生成时域、频域和摘要结果。")
    return run_dir


# =============================================================================
# 3. 命令行与总分流
# =============================================================================

def _parse_arguments() -> argparse.Namespace:
    """
    命令行参数是可选快捷方式，不会取代 config.py。

    例如 python main.py --mode robot_dry_run 可临时测试轨迹而不必来回编辑 RUN_MODE。
    """

    parser = argparse.ArgumentParser(
        description="UR10 末端振动预实验：视觉、轨迹、真机、完整实验与离线分析。"
    )
    parser.add_argument(
        "--mode",
        choices=sorted(config.VALID_RUN_MODES),
        default=None,
        help="临时覆盖 config.py 的 RUN_MODE。",
    )
    parser.add_argument(
        "--analysis-file",
        type=Path,
        default=None,
        help="analyze 模式下临时指定 run_log.txt 或 vision_results.txt。",
    )
    return parser.parse_args()


def main() -> Path:
    """验证配置后只进入一个分支，并在完成后立即返回。"""

    arguments = _parse_arguments()
    selected_mode = arguments.mode or config.RUN_MODE
    config.validate_config(selected_mode)

    print(f"[主程序] 当前模式：{selected_mode}")

    if selected_mode == "vision_test":
        return run_vision_test_mode()

    if selected_mode == "robot_dry_run":
        return run_robot_dry_run_mode()

    if selected_mode == "robot_test":
        return run_robot_test_mode()

    if selected_mode == "experiment":
        return run_experiment_mode()

    if selected_mode == "analyze":
        return run_analysis_mode(arguments.analysis_file)

    # validate_config 已经检查过，这行理论上不会到达；保留它便于未来新增模式时发现漏分支。
    raise RuntimeError(f"RUN_MODE={selected_mode!r} 通过了配置检查，但 main.py 没有对应分支。")


if __name__ == "__main__":
    # Windows 子进程会重新导入 main.py；这层保护防止它再次启动整套相机和机器人进程。
    mp.freeze_support()
    main()
