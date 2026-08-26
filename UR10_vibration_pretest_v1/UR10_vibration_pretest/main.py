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

    if selected_mode == "robot_dry_run":
        return run_robot_dry_run_mode()

    if selected_mode == "robot_test":
        return run_robot_test_mode()

    if selected_mode == "experiment":
        return run_experiment_mode()

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
