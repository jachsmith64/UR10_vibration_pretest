"""
UR10 预实验启动面板。

这个小窗口给刚开始调试项目时使用：
- 点击按钮启动 main.py 的安全模式；
- 在窗口里查看终端输出；
- 需要停止相机视觉测试时，点击“停止当前任务”。

安全边界：
- 本界面不提供 robot_test 和 experiment 按钮；
- 相机测试按钮只运行 vision_test；
- 机器人相关按钮只运行 robot_dry_run，不连接 UR、不发送运动指令。
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, Button, Frame, Label, Tk
from tkinter.scrolledtext import ScrolledText


PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
OUTPUT_ROOT = PROJECT_DIR / "outputs"


class LauncherApp:
    """管理按钮、子进程和输出窗口。"""

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()

        root.title("UR10 预实验启动面板")
        root.geometry("980x620")

        self.status = Label(
            root,
            text="空闲。推荐先运行“相机视觉测试”。",
            anchor="w",
        )
        self.status.pack(fill="x", padx=10, pady=(10, 4))

        button_row = Frame(root)
        button_row.pack(fill="x", padx=10, pady=4)

        self.vision_button = Button(
            button_row,
            text="相机视觉测试",
            command=lambda: self.start_mode("vision_test"),
            width=16,
        )
        self.vision_button.pack(side=LEFT, padx=(0, 8))

        self.dry_run_button = Button(
            button_row,
            text="轨迹 dry-run",
            command=lambda: self.start_mode("robot_dry_run"),
            width=16,
        )
        self.dry_run_button.pack(side=LEFT, padx=(0, 8))

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

        self.stop_button = Button(
            button_row,
            text="停止当前任务",
            command=self.stop_process,
            width=16,
            state="disabled",
        )
        self.stop_button.pack(side=RIGHT)

        self.log = ScrolledText(root, wrap="word", font=("Consolas", 10))
        self.log.pack(fill=BOTH, expand=True, padx=10, pady=(4, 10))
        self.append_log(f"项目目录：{PROJECT_DIR}\n")
        self.append_log(f"使用解释器：{PYTHON_EXE}\n\n")

        self.root.after(100, self.drain_output_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def append_log(self, text: str) -> None:
        """把文本追加到窗口底部。"""

        self.log.insert(END, text)
        self.log.see(END)

    def set_running_state(self, running: bool, mode: str | None = None) -> None:
        """根据是否有任务运行，启用或禁用按钮。"""

        state = "disabled" if running else "normal"
        self.vision_button.configure(state=state)
        self.dry_run_button.configure(state=state)
        self.analyze_button.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")

        if running and mode is not None:
            self.status.configure(text=f"正在运行：main.py --mode {mode}")
        else:
            self.status.configure(text="空闲。")

    def start_mode(self, mode: str) -> None:
        """启动 main.py 的指定安全模式。"""

        if self.process is not None and self.process.poll() is None:
            self.append_log("\n已有任务正在运行，请先停止或等待结束。\n")
            return

        command = [str(PYTHON_EXE), "main.py", "--mode", mode]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["MPLCONFIGDIR"] = str(OUTPUT_ROOT / "matplotlib-cache")

        self.append_log("\n" + "=" * 72 + "\n")
        self.append_log(f"启动命令：{' '.join(command)}\n")
        self.append_log("相机预览窗口中可按 q 或 Esc 结束 vision_test。\n\n")

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
        self.set_running_state(True, mode)

        thread = threading.Thread(target=self.read_process_output, daemon=True)
        thread.start()

    def read_process_output(self) -> None:
        """后台读取子进程输出，避免界面卡住。"""

        process = self.process
        if process is None or process.stdout is None:
            return

        for line in process.stdout:
            self.output_queue.put(line)

        return_code = process.wait()
        self.output_queue.put(f"\n任务结束，退出码：{return_code}\n")
        self.output_queue.put("__PROCESS_DONE__")

    def drain_output_queue(self) -> None:
        """把后台线程读到的输出搬到 Tkinter 文本框。"""

        try:
            while True:
                message = self.output_queue.get_nowait()
                if message == "__PROCESS_DONE__":
                    self.process = None
                    self.set_running_state(False)
                else:
                    self.append_log(message)
        except queue.Empty:
            pass

        self.root.after(100, self.drain_output_queue)

    def stop_process(self) -> None:
        """停止当前子进程。"""

        if self.process is None or self.process.poll() is not None:
            self.process = None
            self.set_running_state(False)
            return

        self.append_log("\n正在请求停止当前任务...\n")
        self.process.terminate()

    def open_outputs(self) -> None:
        """打开 outputs 文件夹。"""

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        os.startfile(OUTPUT_ROOT)

    def on_close(self) -> None:
        """关闭窗口时先尝试停止正在运行的任务。"""

        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()


def main() -> None:
    """创建并运行 Tkinter 窗口。"""

    root = Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
