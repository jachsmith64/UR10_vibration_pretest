"""
UR10 末端振动预实验的唯一总入口。

main.py 只负责“根据模式叫谁工作”和“完整实验的多进程协调”，不在这里实现视觉或轨迹算法。

给刚读这个项目的你：
1. 先看 config.py，那里决定“这次运行什么模式”和“使用哪些参数”。
2. 再看本文件的 main()，它会根据 RUN_MODE 把任务分发给 camera.py、robot.py 或 analyze.py。
3. 如果只是离线看图片，程序会走 run_vision_test_mode()。
4. 如果只是检查轨迹，不连接机器人，程序会走 run_robot_dry_run_mode()。
5. 只有 experiment 模式才会同时启动相机进程、机器人进程和记录进程。

把本文件想象成“调度员”：
- 它不负责识别图像中的圆点；
- 它不负责计算机器人轨迹细节；
- 它不负责做频谱分析；
- 它只负责在正确的时间叫正确的模块开始/停止。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import time
from datetime import datetime
from multiprocessing.process import BaseProcess
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

    参数说明：
    - record_queue：主程序、相机进程、机器人进程都会把记录字典放到这里。
    - output_path：最终要写入的日志文件路径。

    结束方式：
    - 队列中收到 None 时，表示所有生产者都已经停止，可以安全关闭文件。
    """

    # 把传入路径再次转成 Path，是为了兼容调用方传字符串或 Path。
    output_path = Path(output_path)

    # 日志文件所在文件夹可能还不存在，所以先创建父目录。
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # buffering=1 表示行缓冲：每写完一行就尽快刷新，降低异常退出时丢日志的概率。
    with output_path.open("w", encoding="utf-8", buffering=1) as file:
        while True:
            # 这里会阻塞等待新记录；没有新记录时 writer 进程不会忙等占 CPU。
            record = record_queue.get()

            # None 是约定好的“停止信号”，不是一条真实实验数据。
            if record is None:
                break

            # 每行是一条完整 JSON；扩展名保留 txt，方便记事本查看，也便于程序可靠恢复字段。
            line = json.dumps(record, ensure_ascii=False, allow_nan=True)
            file.write(line + "\n")


def _record_event(record_queue: Any, name: str, **extra: Any) -> None:
    """
    把主程序产生的重要时间点写进记录队列。

    这些事件不是相机数据，也不是机器人状态，而是实验流程节点，例如：
    - all_workers_ready：相机和机器人都准备好了；
    - experiment_started：操作者确认后，正式开始；
    - experiment_finished：完整实验流程结束。

    extra 可以附加任意额外字段，例如轨迹类型。
    """

    # host_ns 使用 perf_counter_ns，是同一台电脑上的高精度单调时钟。
    # 它适合做相机记录、机器人记录和事件之间的相对对时。
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
    非阻塞读取一条子进程错误；没有错误时立刻返回。

    为什么不用 error_queue.get()：
    - get() 会一直等，主程序可能因此卡住；
    - get_nowait() 没有错误就立即抛 queue.Empty；
    - 我们捕获 queue.Empty 并返回 None，让主循环可以继续检查超时和停止信号。
    """

    try:
        # 只要有一条错误，主程序就应尽快停止整套实验。
        return str(error_queue.get_nowait())
    except queue.Empty:
        # 没有错误是正常情况，不需要报警。
        return None


def _wait_ready(
    camera_ready: Any,
    robot_ready: Any,
    error_queue: Any,
    stop_event: Any,
) -> None:
    """
    等待相机和机器人都通过各自实际初始化，而不是仅仅等进程被创建。

    输入信号：
    - camera_ready：相机进程成功打开图像来源并处理过第一帧后置位。
    - robot_ready：机器人进程完成轨迹检查、连接和起点检查后置位。
    - error_queue：任一子进程失败时会写错误。
    - stop_event：主程序或子进程要求停止时会置位。

    这个函数只等待，不连接相机、不连接机器人、不发运动命令。
    """

    # 设置一个绝对截止时间，避免某个子进程永远不 ready 时主程序无限等待。
    deadline = time.perf_counter() + config.WORKER_READY_TIMEOUT_S

    # 两个 ready 都亮起才算真正准备好。
    while not (camera_ready.is_set() and robot_ready.is_set()):
        # 优先检查子进程是否已经报告错误。
        worker_error = _pop_worker_error(error_queue)
        if worker_error:
            raise RuntimeError(worker_error)

        # 如果 stop_event 已经亮起，说明某处要求停止，但没有提供更具体错误。
        if stop_event.is_set():
            raise RuntimeError("某个子进程在 ready 前要求停止，但没有返回更多错误信息。")

        # 超时后给出到底是相机没 ready 还是机器人没 ready，便于定位问题。
        if time.perf_counter() >= deadline:
            missing = []
            if not camera_ready.is_set():
                missing.append("相机")
            if not robot_ready.is_set():
                missing.append("机器人")
            raise TimeoutError(f"等待 {'、'.join(missing)} ready 超时。")

        # 每 0.05 秒检查一次，既足够快，也不会让 CPU 忙等。
        time.sleep(0.05)


def _wait_motion_done(
    motion_done: Any,
    error_queue: Any,
    stop_event: Any,
) -> None:
    """
    正式运动期间等待机器人报告 motion_done，同时持续检查错误和超时。

    motion_done 是机器人子进程发出的“运动完成”信号。
    但只等 motion_done 不够安全：如果机器人进程异常退出而没来得及置位，
    主程序就会一直卡住。因此这里同时检查 error_queue、stop_event 和 deadline。
    """

    # 预记录时间 + 预计最大运动时间 + 5 秒余量，构成主程序等待上限。
    deadline = (
        time.perf_counter()
        + config.PRE_RECORD_SECONDS
        + config.ROBOT_MOTION_TIMEOUT_S
        + 5.0
    )

    # motion_done 未置位时一直循环检查。
    while not motion_done.is_set():
        # 子进程明确报告错误时，立刻把错误抛给主程序上层处理。
        worker_error = _pop_worker_error(error_queue)
        if worker_error:
            raise RuntimeError(worker_error)

        # stop_event 通常代表其他进程或异常路径要求停止。
        if stop_event.is_set():
            raise RuntimeError("实验在运动完成前收到停止信号。")

        # 超时说明运动完成信号迟迟没有回来，不能继续盲等。
        if time.perf_counter() >= deadline:
            raise TimeoutError("主程序等待 motion_done 超时。")

        # 小睡一会儿，避免循环占满 CPU。
        time.sleep(0.05)


def _join_or_terminate(process: BaseProcess) -> None:
    """
    先给子进程充分时间执行 finally 释放相机/RTDE，只有失去响应时才终止。

    terminate 是最后清理手段，因此正常流程不会靠它结束硬件连接。

    这段代码只处理进程生命周期，不知道子进程里面具体是相机还是机器人。
    """

    # 先给子进程一个正常退出的时间窗口，让它执行自己的 finally。
    process.join(timeout=config.WORKER_JOIN_TIMEOUT_S)

    # 如果超时后还活着，说明它没有按预期退出，只能作为最后手段终止。
    if process.is_alive():
        print(f"[主程序警告] {process.name} 未按时退出，将终止该子进程。")
        process.terminate()

        # terminate 后再 join 一次，尽量把进程资源回收干净。
        process.join(timeout=2.0)


# =============================================================================
# 2. 五种模式的高层入口
# =============================================================================

def run_vision_test_mode() -> Path:
    """
    启动视觉离线/单模块测试模式。

    这个函数的作用很小：只负责把任务转交给 camera.py 的 run_vision_test()。
    延迟导入的好处是：如果你只想看帮助或跑 dry-run，不会因为相机相关依赖出问题而失败。
    """

    # import 放在函数内部，只有真正进入 vision_test 时才加载 camera.py。
    from camera import run_vision_test

    # 真正的图像读取、识别、保存结果都在 camera.py 里完成。
    return run_vision_test()


def run_robot_dry_run_mode() -> Path:
    """
    启动机器人轨迹 dry-run。

    dry-run 只检查轨迹数字、生成报告和示意图。
    它不会创建 URRobot，不会导入 ur_rtde，也不会连接任何机器人 IP。
    """

    # 只有进入该模式才导入 robot.py，避免其他模式加载不必要代码。
    from robot import run_robot_dry_run

    # 具体轨迹生成和报告保存由 robot.py 完成。
    return run_robot_dry_run()


def run_robot_test_mode() -> Path:
    """
    启动机器人单机测试模式。

    注意：这个模式可能连接 UR。
    是否允许运动由 config.ROBOT_TEST_ALLOW_MOTION 和其他安全开关共同决定。
    """

    # 延迟导入，只有明确选择 robot_test 才触碰 robot.py 的真机相关入口。
    from robot import run_robot_test

    return run_robot_test()


def run_analysis_mode(analysis_file: Path | None = None) -> Path:
    """
    启动离线分析模式。

    analysis_file 可来自命令行 --analysis-file。
    如果不传，analyze.py 会按自己的规则在 outputs 里寻找最新记录。
    """

    # analyze.py 不连接相机或机器人，只读取已有日志并生成图表。
    from analyze import run_analysis

    return run_analysis(analysis_file)


def run_experiment_mode() -> Path:
    """
    启动记录、相机和机器人三个子进程，并负责 ready、开始、完成和停止信号。

    只有这个模式使用多进程；单模块测试保持单进程，报错堆栈更容易理解。
    """

    # Windows 使用 spawn 创建子进程，目标函数必须位于模块顶层，且所有对象必须可序列化。
    # 可以把 spawn 理解为“重新启动一个 Python，再让它执行指定函数”。
    context = mp.get_context("spawn")

    # record_queue：相机和机器人把数据放进来，record_writer_worker 从里面取出并写入文件。
    # error_queue：子进程遇到异常时，把错误文字发回主程序。
    record_queue = context.Queue(maxsize=config.RECORD_QUEUE_MAXSIZE)
    error_queue = context.Queue(maxsize=config.ERROR_QUEUE_MAXSIZE)

    # Event 是跨进程共享的“开关灯”：
    # start_event 亮起：相机/机器人开始正式记录或运动；
    # stop_event 亮起：所有子进程应尽快退出；
    # camera_ready / robot_ready 亮起：对应子进程已准备好；
    # motion_done 亮起：机器人运动阶段已完成。
    start_event = context.Event()
    stop_event = context.Event()
    camera_ready = context.Event()
    robot_ready = context.Event()
    motion_done = context.Event()

    # 每次正式实验单独建一个文件夹，避免日志互相覆盖。
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

    # 三个子进程分工：
    # 1. writer_process 专门写文件；
    # 2. camera_process 专门取图并识别；
    # 3. robot_process 专门读取/控制 UR。
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

    # 先启动写日志进程，确保后续 META、VISION、ROBOT 记录有地方落盘。
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
        # 相机和机器人进程启动后，主程序不会立即开始运动。
        # 它会先等两个进程都报告 ready，并持续监听 error_queue。
        camera_process.start()
        robot_process.start()
        _wait_ready(camera_ready, robot_ready, error_queue, stop_event)
        _record_event(record_queue, "all_workers_ready")

        # 到这里才说明相机已成功取帧、UR 已连接、轨迹和起点已通过检查。
        require_operator_confirmation("相机 + UR10 正式振动预实验")
        _record_event(record_queue, "experiment_started")
        start_event.set()

        # 运动期间主程序不直接控制每个采样点，只负责等待完成并监控错误。
        _wait_motion_done(motion_done, error_queue, stop_event)
        print(
            f"[主程序] 运动阶段完成，继续记录 {config.POST_RECORD_SECONDS:.2f} s 残余振动。"
        )

        # 运动结束后继续记录一小段时间，用来观察残余振动衰减。
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
        # 任意步骤出错，都先发 stop_event 要求子进程收尾，再把错误写入日志。
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
        # finally 无论成功或失败都会执行，负责关闭子进程和队列。
        # 这比让程序崩溃后留下一堆后台进程更安全。
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

    # argparse 是 Python 标准库的命令行参数解析工具。
    # 它会自动生成 --help 页面，也会帮我们检查 mode 是否在允许列表中。
    parser = argparse.ArgumentParser(
        description="UR10 末端振动预实验：视觉、轨迹、真机、完整实验与离线分析。"
    )

    # --mode 是临时覆盖，不会修改 config.py 文件本身。
    parser.add_argument(
        "--mode",
        choices=sorted(config.VALID_RUN_MODES),
        default=None,
        help="临时覆盖 config.py 的 RUN_MODE。",
    )

    # --analysis-file 只对 analyze 模式有意义，用于指定要分析的日志文件。
    parser.add_argument(
        "--analysis-file",
        type=Path,
        default=None,
        help="analyze 模式下临时指定 run_log.txt 或 vision_results.txt。",
    )

    # parse_args() 会读取终端命令行，并返回一个带属性的对象。
    return parser.parse_args()


def main() -> Path:
    """
    程序真正的总入口。

    它的流程非常固定：
    1. 读取命令行参数；
    2. 决定最终运行模式；
    3. 调用 config.validate_config() 做启动前检查；
    4. 根据模式进入唯一一个分支；
    5. 返回该模式创建的输出目录。
    """

    # 读取命令行，例如 --mode robot_dry_run。
    arguments = _parse_arguments()

    # 如果命令行指定了 --mode，就优先用命令行；否则使用 config.py 里的 RUN_MODE。
    selected_mode = arguments.mode or config.RUN_MODE

    # 任何模式启动前都先做配置检查，尽早发现拼写错误和明显危险参数。
    config.validate_config(selected_mode)

    print(f"[主程序] 当前模式：{selected_mode}")

    # 下面是清晰的一对一分流：一个模式只进入一个函数。
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
