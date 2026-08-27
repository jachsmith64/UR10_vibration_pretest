"""
UR10 预实验启动面板。

启动器只负责启动一个 main.py 子进程、显示输出和发送停止请求。
机器人/相机的多进程协调全部留在 main.py 内部完成。
"""

from __future__ import annotations

import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    Button,
    Entry,
    Frame,
    Label,
    LabelFrame,
    OptionMenu,
    StringVar,
    Tk,
    messagebox,
)
from tkinter.scrolledtext import ScrolledText
from typing import Any

import config


PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
OUTPUT_ROOT = PROJECT_DIR / "outputs"
CAPTURE_STOP_REQUEST_PATH = OUTPUT_ROOT / "vision_capture_stop.request"
MOTION_MODES = {"x_line_experiment", "xy_line_experiment", "xy_l_experiment"}


class LauncherApp:
    """管理按钮、子进程和输出窗口。"""

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[Any] = queue.Queue()
        self.current_mode: str | None = None
        self.current_stop_request_path: Path | None = None
        self.close_after_process = False
        self.robot_connection_verified = False
        self.motion_buttons: list[Button] = []

        root.title("UR10 预实验启动面板")
        root.geometry("1120x780")

        self.status = Label(root, text="空闲。请先检测机械臂通信。", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(10, 4))

        self._build_hardware_section(root)
        self._build_motion_sections(root)
        self._build_log_section(root)

        self.root.after(100, self.drain_output_queue)
        self.root.after(350, self.show_usage_guide)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_hardware_section(self, root: Tk) -> None:
        section = LabelFrame(root, text="A. 硬件检查")
        section.pack(fill="x", padx=10, pady=4)

        self.connection_button = Button(
            section,
            text="1. 检测机械臂通信（不会运动）",
            command=lambda: self.start_mode("robot_connection_test"),
            width=26,
        )
        self.connection_button.pack(side=LEFT, padx=6, pady=6)

        self.vision_button = Button(
            section,
            text="相机视觉测试",
            command=lambda: self.start_mode("vision_test"),
            width=16,
        )
        self.vision_button.pack(side=LEFT, padx=6, pady=6)

        self.capture_button = Button(
            section,
            text="高速采集",
            command=lambda: self.start_mode("vision_capture"),
            width=16,
        )
        self.capture_button.pack(side=LEFT, padx=6, pady=6)

        self.offline_button = Button(
            section,
            text="离线识别",
            command=lambda: self.start_mode("vision_offline"),
            width=16,
        )
        self.offline_button.pack(side=LEFT, padx=6, pady=6)

        self.dry_run_button = Button(
            section,
            text="轨迹 dry-run",
            command=lambda: self.start_mode("robot_dry_run"),
            width=16,
        )
        self.dry_run_button.pack(side=LEFT, padx=6, pady=6)

    def _add_labeled_entry(
        self,
        parent: Frame,
        label: str,
        default: float,
        width: int = 8,
    ) -> Entry:
        Label(parent, text=label).pack(side=LEFT, padx=(6, 2))
        entry = Entry(parent, width=width)
        entry.insert(0, str(default))
        entry.pack(side=LEFT, padx=(0, 6))
        return entry

    def _add_direction(self, parent: Frame, label: str, values: tuple[str, str]) -> StringVar:
        variable = StringVar(value=values[0])
        Label(parent, text=label).pack(side=LEFT, padx=(6, 2))
        OptionMenu(parent, variable, *values).pack(side=LEFT, padx=(0, 6))
        return variable

    def _build_motion_sections(self, root: Tk) -> None:
        x_section = LabelFrame(root, text="B. X直线实验")
        x_section.pack(fill="x", padx=10, pady=4)
        self.x_speed_entry = self._add_labeled_entry(
            x_section,
            "速度 mm/s",
            config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S,
        )
        self.x_total_time_entry = self._add_labeled_entry(
            x_section,
            "总名义时长 s",
            config.ROBOT_EXPERIMENT_DEFAULT_TOTAL_TIME_S,
        )
        self.x_direction = self._add_direction(x_section, "X方向", ("+X", "-X"))
        self.x_motion_button = Button(
            x_section,
            text="开始X直线实验",
            command=self.start_x_line_experiment,
            width=18,
            state="disabled",
        )
        self.x_motion_button.pack(side=RIGHT, padx=6, pady=6)
        self.motion_buttons.append(self.x_motion_button)

        xy_section = LabelFrame(root, text="C. X-Y倾斜直线实验")
        xy_section.pack(fill="x", padx=10, pady=4)
        self.xy_speed_entry = self._add_labeled_entry(
            xy_section,
            "速度 mm/s",
            config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S,
        )
        self.xy_total_time_entry = self._add_labeled_entry(
            xy_section,
            "总名义时长 s",
            config.ROBOT_EXPERIMENT_DEFAULT_TOTAL_TIME_S,
        )
        self.xy_angle_entry = self._add_labeled_entry(
            xy_section,
            "角度 deg",
            config.ROBOT_EXPERIMENT_DEFAULT_ANGLE_DEG,
        )
        self.xy_x_direction = self._add_direction(xy_section, "X方向", ("+X", "-X"))
        self.xy_y_direction = self._add_direction(xy_section, "Y方向", ("+Y", "-Y"))
        self.xy_motion_button = Button(
            xy_section,
            text="开始X-Y倾斜直线实验",
            command=self.start_xy_line_experiment,
            width=22,
            state="disabled",
        )
        self.xy_motion_button.pack(side=RIGHT, padx=6, pady=6)
        self.motion_buttons.append(self.xy_motion_button)

        l_section = LabelFrame(root, text="D. X-Y平面L折线实验")
        l_section.pack(fill="x", padx=10, pady=4)
        self.l_x_speed_entry = self._add_labeled_entry(
            l_section,
            "X速度 mm/s",
            config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S,
        )
        self.l_x_time_entry = self._add_labeled_entry(
            l_section,
            "X单程 s",
            config.ROBOT_EXPERIMENT_DEFAULT_X_ONE_WAY_TIME_S,
        )
        self.l_y_speed_entry = self._add_labeled_entry(
            l_section,
            "Y速度 mm/s",
            config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S,
        )
        self.l_y_time_entry = self._add_labeled_entry(
            l_section,
            "Y单程 s",
            config.ROBOT_EXPERIMENT_DEFAULT_Y_ONE_WAY_TIME_S,
        )
        self.l_x_direction = self._add_direction(l_section, "X方向", ("+X", "-X"))
        self.l_y_direction = self._add_direction(l_section, "Y方向", ("+Y", "-Y"))
        self.l_blend_entry = self._add_labeled_entry(
            l_section,
            "blend mm",
            config.ROBOT_RELATIVE_BLEND_MM,
        )
        self.l_motion_button = Button(
            l_section,
            text="开始L折线实验",
            command=self.start_xy_l_experiment,
            width=18,
            state="disabled",
        )
        self.l_motion_button.pack(side=RIGHT, padx=6, pady=6)
        self.motion_buttons.append(self.l_motion_button)

    def _build_log_section(self, root: Tk) -> None:
        section = LabelFrame(root, text="E. 日志和停止")
        section.pack(fill=BOTH, expand=True, padx=10, pady=(4, 10))

        button_row = Frame(section)
        button_row.pack(fill="x", padx=6, pady=4)

        self.analyze_button = Button(
            button_row,
            text="分析最新结果",
            command=lambda: self.start_mode("analyze"),
            width=16,
        )
        self.analyze_button.pack(side=LEFT, padx=(0, 8))

        self.outputs_button = Button(
            button_row,
            text="打开 outputs",
            command=self.open_outputs,
            width=16,
        )
        self.outputs_button.pack(side=LEFT, padx=(0, 8))

        self.guide_button = Button(
            button_row,
            text="操作说明",
            command=self.show_usage_guide,
            width=16,
        )
        self.guide_button.pack(side=LEFT, padx=(0, 8))

        self.stop_button = Button(
            button_row,
            text="停止当前任务",
            command=self.stop_process,
            width=16,
            state="disabled",
        )
        self.stop_button.pack(side=RIGHT)

        self.log = ScrolledText(section, wrap="word", font=("Consolas", 10))
        self.log.pack(fill=BOTH, expand=True, padx=6, pady=(4, 6))
        self.append_log(f"项目目录：{PROJECT_DIR}\n")
        self.append_log(f"使用解释器：{PYTHON_EXE}\n\n")

    def append_log(self, text: str) -> None:
        self.log.insert(END, text)
        self.log.see(END)

    def set_running_state(self, running: bool, mode: str | None = None) -> None:
        state = "disabled" if running else "normal"
        for button in (
            self.connection_button,
            self.vision_button,
            self.capture_button,
            self.offline_button,
            self.dry_run_button,
            self.analyze_button,
        ):
            button.configure(state=state)

        motion_state = (
            "disabled"
            if running or not self.robot_connection_verified
            else "normal"
        )
        for button in self.motion_buttons:
            button.configure(state=motion_state)

        self.stop_button.configure(state="normal" if running else "disabled")
        if running and mode is not None:
            self.status.configure(text=f"正在运行：main.py --mode {mode}")
        elif self.robot_connection_verified:
            self.status.configure(text="机械臂通信检测通过，可以进行运动实验。")
        else:
            self.status.configure(text="空闲。请先检测机械臂通信。")

    def start_mode(
        self,
        mode: str,
        extra_args: list[str] | None = None,
        stop_request_path: Path | None = None,
    ) -> None:
        if self.process is not None and self.process.poll() is None:
            self.append_log("\n已有任务正在运行，请先停止或等待结束。\n")
            return

        command = [str(PYTHON_EXE), "-u", "main.py", "--mode", mode]
        if extra_args:
            command.extend(extra_args)

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLCONFIGDIR"] = str(OUTPUT_ROOT / "matplotlib-cache")

        if mode == "vision_capture" and CAPTURE_STOP_REQUEST_PATH.exists():
            CAPTURE_STOP_REQUEST_PATH.unlink()
        if stop_request_path is not None and stop_request_path.exists():
            stop_request_path.unlink()

        self.append_log("\n" + "=" * 72 + "\n")
        self.append_log(f"启动命令：{' '.join(command)}\n\n")

        self.process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        self.current_mode = mode
        self.current_stop_request_path = stop_request_path
        self.close_after_process = False
        self.set_running_state(True, mode)

        thread = threading.Thread(
            target=self.read_process_output,
            args=(self.process, mode),
            daemon=True,
        )
        thread.start()

    def read_process_output(self, process: subprocess.Popen[str], mode: str) -> None:
        if process.stdout is None:
            return

        for line in process.stdout:
            self.output_queue.put(line)

        return_code = process.wait()
        self.output_queue.put(f"\n任务结束，退出码：{return_code}\n")
        self.output_queue.put(("PROCESS_DONE", mode, return_code, return_code == 0))

    def drain_output_queue(self) -> None:
        try:
            while True:
                message = self.output_queue.get_nowait()
                if isinstance(message, tuple) and message[:1] == ("PROCESS_DONE",):
                    _, finished_mode, return_code, completed = message
                    self.process = None
                    self.current_mode = None
                    self.current_stop_request_path = None
                    if finished_mode == "robot_connection_test":
                        self.robot_connection_verified = return_code == 0
                    self.set_running_state(False)
                    if finished_mode == "robot_connection_test":
                        if completed:
                            self.status.configure(text="机械臂通信检测通过，可以进行运动实验。")
                        else:
                            self.status.configure(text="机械臂通信检测失败，运动实验保持锁定。")
                    if self.close_after_process:
                        self.root.destroy()
                        return
                else:
                    text = str(message)
                    self.append_log(text)
                    if text.startswith("[状态]"):
                        self.status.configure(text=text.strip())
        except queue.Empty:
            pass

        self.root.after(100, self.drain_output_queue)

    def _read_float(self, entry: Entry, name: str) -> float:
        try:
            value = float(entry.get().strip())
        except ValueError as exc:
            raise ValueError(f"{name} 必须是数字。") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须是有限正数。")
        return value

    def _validate_speed(self, value: float, name: str) -> None:
        if not (
            config.ROBOT_EXPERIMENT_MIN_SPEED_MM_S
            <= value
            <= config.ROBOT_EXPERIMENT_MAX_SPEED_MM_S
        ):
            raise ValueError(
                f"{name} 必须在 {config.ROBOT_EXPERIMENT_MIN_SPEED_MM_S} 到 "
                f"{config.ROBOT_EXPERIMENT_MAX_SPEED_MM_S} mm/s 之间。"
            )

    def _make_stop_request_path(self, mode: str) -> Path:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.perf_counter_ns()}"
        return OUTPUT_ROOT / f"{mode}_{stamp}.stop.request"

    def _confirm_motion(self, title: str, summary: str) -> bool:
        return messagebox.askyesno(
            title,
            summary
            + "\n\n请确认：索尼相机已经开始录像；机械臂周围无人；方向和位移正确；急停可立即触及。",
        )

    def show_usage_guide(self) -> None:
        """给第一次使用启动器的人显示简短操作顺序。"""

        messagebox.showinfo(
            "操作说明",
            "1. 先点击“检测机械臂通信（不会运动）”。通过后才会解锁三个运动实验。\n\n"
            "2. 做运动实验前，先手动开始索尼相机录像，再填写速度、时间、方向等参数。\n\n"
            "3. 点击对应实验按钮后，确认弹窗中的安全项。工业相机会开始保存 RAW，随后按提示打开并关闭手机手电筒。\n\n"
            "4. 检测到手电筒并等待画面恢复后，机械臂才会开始一次往返运动；结束后工业相机会继续记录 1 秒并自动封口保存。\n\n"
            "5. 需要停止时点“停止当前任务”，请等待窗口提示安全收尾完成；真实危险以示教器急停/安全停止为准。\n\n"
            "6. 实验结束后再手动停止索尼相机录像，结果在 outputs 目录中查看。",
        )

    def _start_motion(self, mode: str, args: list[str], summary: str) -> None:
        if not self.robot_connection_verified:
            messagebox.showerror("未完成通信检测", "请先成功运行“检测机械臂通信”。")
            return
        if not self._confirm_motion("运动实验确认", summary):
            return
        stop_request_path = self._make_stop_request_path(mode)
        self.start_mode(
            mode,
            args + ["--ui-confirmed", "--stop-request-path", str(stop_request_path)],
            stop_request_path,
        )

    def start_x_line_experiment(self) -> None:
        try:
            speed = self._read_float(self.x_speed_entry, "X直线速度")
            total_time = self._read_float(self.x_total_time_entry, "X直线总名义时长")
            self._validate_speed(speed, "X直线速度")
            distance_mm = speed * total_time / 2.0
            if distance_mm > config.ROBOT_EXPERIMENT_MAX_SEGMENT_MM:
                raise ValueError("X直线单程距离超过安全上限。")
            summary = (
                f"X直线：{self.x_direction.get()}，速度 {speed:.3f} mm/s，"
                f"总名义时长 {total_time:.3f} s，单程 {distance_mm:.3f} mm。"
            )
            self._start_motion(
                "x_line_experiment",
                [
                    "--speed-mm-s",
                    str(speed),
                    "--total-time-s",
                    str(total_time),
                    "--x-direction",
                    self.x_direction.get(),
                ],
                summary,
            )
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))

    def start_xy_line_experiment(self) -> None:
        try:
            speed = self._read_float(self.xy_speed_entry, "X-Y直线速度")
            total_time = self._read_float(self.xy_total_time_entry, "X-Y直线总名义时长")
            angle = self._read_float(self.xy_angle_entry, "X-Y直线角度")
            self._validate_speed(speed, "X-Y直线速度")
            if angle > config.ROBOT_EXPERIMENT_MAX_ANGLE_DEG:
                raise ValueError("X-Y直线角度超过安全上限。")
            distance_mm = speed * total_time / 2.0
            if distance_mm > config.ROBOT_EXPERIMENT_MAX_SEGMENT_MM:
                raise ValueError("X-Y直线单程距离超过安全上限。")
            dx = distance_mm * math.cos(math.radians(angle))
            dy = distance_mm * math.sin(math.radians(angle))
            summary = (
                f"X-Y倾斜直线：{self.xy_x_direction.get()}/{self.xy_y_direction.get()}，"
                f"速度 {speed:.3f} mm/s，单程 {distance_mm:.3f} mm，"
                f"dx {dx:.3f} mm，dy {dy:.3f} mm。"
            )
            self._start_motion(
                "xy_line_experiment",
                [
                    "--speed-mm-s",
                    str(speed),
                    "--total-time-s",
                    str(total_time),
                    "--angle-deg",
                    str(angle),
                    "--x-direction",
                    self.xy_x_direction.get(),
                    "--y-direction",
                    self.xy_y_direction.get(),
                ],
                summary,
            )
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))

    def start_xy_l_experiment(self) -> None:
        try:
            x_speed = self._read_float(self.l_x_speed_entry, "L折线X速度")
            x_time = self._read_float(self.l_x_time_entry, "L折线X单程时间")
            y_speed = self._read_float(self.l_y_speed_entry, "L折线Y速度")
            y_time = self._read_float(self.l_y_time_entry, "L折线Y单程时间")
            blend = float(self.l_blend_entry.get().strip())
            if not math.isfinite(blend) or blend < 0:
                raise ValueError("blend 必须是有限非负数。")
            self._validate_speed(x_speed, "L折线X速度")
            self._validate_speed(y_speed, "L折线Y速度")
            dx = x_speed * x_time
            dy = y_speed * y_time
            if max(dx, dy) > config.ROBOT_EXPERIMENT_MAX_SEGMENT_MM:
                raise ValueError("L折线单段距离超过安全上限。")
            if blend > 0 and blend >= 0.5 * min(dx, dy):
                raise ValueError("blend 必须小于较短相邻线段的一半。")
            summary = (
                f"L折线：{self.l_x_direction.get()}/{self.l_y_direction.get()}，"
                f"X段 {dx:.3f} mm，Y段 {dy:.3f} mm，"
                f"总名义时长 {2.0 * (x_time + y_time):.3f} s。"
            )
            self._start_motion(
                "xy_l_experiment",
                [
                    "--x-speed-mm-s",
                    str(x_speed),
                    "--x-one-way-time-s",
                    str(x_time),
                    "--y-speed-mm-s",
                    str(y_speed),
                    "--y-one-way-time-s",
                    str(y_time),
                    "--x-direction",
                    self.l_x_direction.get(),
                    "--y-direction",
                    self.l_y_direction.get(),
                    "--blend-mm",
                    str(blend),
                ],
                summary,
            )
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))

    def stop_process(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.process = None
            self.set_running_state(False)
            return

        self.append_log("\n正在请求停止当前任务...\n")
        if self.current_mode == "vision_capture":
            try:
                OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
                CAPTURE_STOP_REQUEST_PATH.write_text("stop\n", encoding="utf-8")
                self.append_log("已通知高速采集自行退出并保存 RAW，请等待任务结束。\n")
                self.stop_button.configure(state="disabled")
            except OSError as exc:
                self.append_log(f"写入高速采集停止请求失败：{exc}\n")
            return

        if self.current_mode in MOTION_MODES:
            try:
                if self.current_stop_request_path is None:
                    self.current_stop_request_path = self._make_stop_request_path(self.current_mode)
                self.current_stop_request_path.write_text("stop\n", encoding="utf-8")
                self.append_log("已通知 main.py 安全停止：stopL、相机 RAW 封口、子进程退出。\n")
                self.stop_button.configure(state="disabled")
            except OSError as exc:
                self.append_log(f"写入运动实验停止请求失败：{exc}\n")
            return

        self.process.terminate()

    def open_outputs(self) -> None:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        os.startfile(OUTPUT_ROOT)

    def on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if self.current_mode == "vision_capture" or self.current_mode in MOTION_MODES:
                self.close_after_process = True
                self.stop_process()
                self.status.configure(text="正在安全停止，完成后关闭窗口。")
                return
            self.process.terminate()
        self.root.destroy()


def main() -> None:
    root = Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
