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
    filedialog,
    messagebox,
    simpledialog,
)
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk import Scrollbar, Treeview
from typing import Any

import config
from batch_plan import (
    PRESETS,
    TRAJECTORY_LABELS,
    build_default_plan,
    compute_plan_envelope,
    normalize_plan,
    normalize_plan_row,
    preset_parameters,
    recommended_time_bounds,
    write_plan,
)


PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
OUTPUT_ROOT = PROJECT_DIR / "outputs"
CAPTURE_STOP_REQUEST_PATH = OUTPUT_ROOT / "vision_capture_stop.request"
MOTION_MODES = {
    "x_line_experiment",
    "xy_line_experiment",
    "xy_l_experiment",
    "batch_experiment",
    "boundary_check",
}
USAGE_GUIDE_TEXT = (
    "1. 先点击“检测机械臂通信（不会运动）”。通过后才会解锁三个运动实验。\n\n"
    "2. 做运动实验前，先手动开始索尼相机录像，再填写速度、时间、方向等参数。\n\n"
    "3. 点击对应实验按钮后，确认弹窗中的安全项。工业相机会开始保存 RAW，随后按提示打开并关闭手机手电筒。\n\n"
    "4. 检测到手电筒并等待画面恢复后，机械臂才会开始一次往返运动；结束后工业相机会继续记录 1 秒并自动封口保存。\n\n"
    "5. 需要停止时点“停止当前任务”，请等待窗口提示安全收尾完成；真实危险以示教器急停/安全停止为准。\n\n"
    "6. 完整批量实验只做连续录像和时间/机器人记录，不在实机采集阶段运行视觉识别。\n\n"
    "7. 实验结束后手动停止索尼相机录像；之后可点‘批次录像离线识别’，选择批次目录再生成各段 VISION。"
)


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
        self.batch_plan = build_default_plan()

        root.title("UR10 预实验启动面板")
        root.geometry("1500x980")

        self.status = Label(root, text="空闲。请先检测机械臂通信。", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(10, 4))

        self._build_hardware_section(root)
        self._build_motion_sections(root)
        self._build_batch_section(root)
        self._build_log_section(root)

        self.root.after(100, self.drain_output_queue)
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

        self.batch_offline_button = Button(
            section,
            text="批次录像离线识别",
            command=self.start_batch_vision_offline,
            width=18,
        )
        self.batch_offline_button.pack(side=LEFT, padx=6, pady=6)

    def start_batch_vision_offline(self) -> None:
        selected = filedialog.askdirectory(
            title="选择包含 segments.csv 的已完成批次目录",
            initialdir=str(OUTPUT_ROOT),
        )
        if not selected:
            return
        self.start_mode("batch_vision_offline", ["--batch-dir", selected])

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

    def _add_preset_buttons(
        self,
        parent: Frame,
        speed_entry: Entry,
        time_entry: Entry,
        trajectory_type: str,
    ) -> None:
        Label(parent, text="预设").pack(side=LEFT, padx=(6, 2))
        for preset_id in PRESETS:
            values = preset_parameters(preset_id, trajectory_type)
            Button(
                parent,
                text=preset_id,
                width=4,
                command=lambda v=values: self._apply_preset(speed_entry, time_entry, v),
            ).pack(side=LEFT, padx=1)

    @staticmethod
    def _apply_preset(speed_entry: Entry, time_entry: Entry, values: dict[str, float]) -> None:
        for entry, value in (
            (speed_entry, values["speed_mm_s"]),
            (time_entry, values["one_way_time_s"]),
        ):
            entry.delete(0, END)
            entry.insert(0, f"{value:.9g}")
            entry.event_generate("<KeyRelease>")

    def _bind_nominal_distance(
        self,
        speed_entry: Entry,
        time_entry: Entry,
        output: Label,
        *,
        trajectory_type: str,
        l_shape: bool = False,
    ) -> None:
        def update(_event: Any = None) -> None:
            try:
                speed = float(speed_entry.get())
                duration = float(time_entry.get())
                if not (math.isfinite(speed) and math.isfinite(duration)):
                    raise ValueError
                if l_shape:
                    text = (
                        f"名义单程总长 {speed * duration:.1f} mm；"
                        f"两段各 {speed * duration / 2.0:.1f} mm"
                    )
                else:
                    text = f"名义单程距离 {speed * duration:.1f} mm"
                lower, upper = recommended_time_bounds(trajectory_type)
                if not lower - 1e-7 <= duration <= upper + 1e-7:
                    output.configure(
                        text=(
                            text
                            + f"  ⚠ 单程时间不在该轨迹建议的 {lower:.1f}～{upper:.1f} s"
                        ),
                        fg="#c00000",
                    )
                else:
                    output.configure(text=text, fg="#006000")
            except (TypeError, ValueError):
                output.configure(text="请输入有效速度和单程时间", fg="#c00000")

        speed_entry.bind("<KeyRelease>", update)
        time_entry.bind("<KeyRelease>", update)
        update()

    def _build_motion_sections(self, root: Tk) -> None:
        x_section = LabelFrame(root, text="B. X直线实验")
        x_section.pack(fill="x", padx=10, pady=4)
        self.x_speed_entry = self._add_labeled_entry(
            x_section,
            "速度 mm/s",
            config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S,
        )
        self.x_one_way_time_entry = self._add_labeled_entry(
            x_section,
            "单程时间 s",
            config.ROBOT_EXPERIMENT_DEFAULT_X_LINE_ONE_WAY_TIME_S,
        )
        self._add_preset_buttons(
            x_section, self.x_speed_entry, self.x_one_way_time_entry, "x_line"
        )
        self.x_direction = self._add_direction(x_section, "X方向", ("+X", "-X"))
        self.x_distance_label = Label(x_section, text="")
        self.x_distance_label.pack(side=LEFT, padx=6)
        self._bind_nominal_distance(
            self.x_speed_entry,
            self.x_one_way_time_entry,
            self.x_distance_label,
            trajectory_type="x_line",
        )
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
        self.xy_one_way_time_entry = self._add_labeled_entry(
            xy_section,
            "单程时间 s",
            config.ROBOT_EXPERIMENT_DEFAULT_XY_LINE_ONE_WAY_TIME_S,
        )
        self._add_preset_buttons(
            xy_section, self.xy_speed_entry, self.xy_one_way_time_entry, "xy_line"
        )
        self.xy_angle_entry = self._add_labeled_entry(
            xy_section,
            "角度 deg",
            config.ROBOT_EXPERIMENT_DEFAULT_ANGLE_DEG,
        )
        self.xy_x_direction = self._add_direction(xy_section, "X方向", ("+X", "-X"))
        self.xy_y_direction = self._add_direction(xy_section, "Y方向", ("+Y", "-Y"))
        self.xy_distance_label = Label(xy_section, text="")
        self.xy_distance_label.pack(side=LEFT, padx=6)
        self._bind_nominal_distance(
            self.xy_speed_entry,
            self.xy_one_way_time_entry,
            self.xy_distance_label,
            trajectory_type="xy_line",
        )
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
        self.l_speed_entry = self._add_labeled_entry(
            l_section,
            "两段共同速度 mm/s",
            config.ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S,
        )
        self.l_one_way_time_entry = self._add_labeled_entry(
            l_section,
            "A→B→C单程总时间 s",
            config.ROBOT_EXPERIMENT_DEFAULT_L_ONE_WAY_TIME_S,
        )
        self._add_preset_buttons(
            l_section, self.l_speed_entry, self.l_one_way_time_entry, "l_shape"
        )
        self.l_x_direction = self._add_direction(l_section, "X方向", ("+X", "-X"))
        self.l_y_direction = self._add_direction(l_section, "Y方向", ("+Y", "-Y"))
        self.l_blend_entry = self._add_labeled_entry(
            l_section,
            "blend mm",
            config.ROBOT_RELATIVE_BLEND_MM,
        )
        self.l_distance_label = Label(l_section, text="")
        self.l_distance_label.pack(side=LEFT, padx=6)
        self._bind_nominal_distance(
            self.l_speed_entry,
            self.l_one_way_time_entry,
            self.l_distance_label,
            trajectory_type="l_shape",
            l_shape=True,
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

    def _build_batch_section(self, root: Tk) -> None:
        section = LabelFrame(root, text="E. 完整批量实验计划（双击或选中后编辑）")
        section.pack(fill="x", padx=10, pady=4)
        controls = Frame(section)
        controls.pack(fill="x", padx=6, pady=4)
        Button(
            controls,
            text="加载默认完整实验计划",
            command=self.load_default_batch_plan,
            width=24,
        ).pack(side=LEFT, padx=(0, 6))
        Button(
            controls, text="编辑选中行", command=self.edit_selected_plan_row, width=14
        ).pack(side=LEFT, padx=3)
        Button(
            controls, text="启用/禁用选中行", command=self.toggle_selected_plan_row, width=17
        ).pack(side=LEFT, padx=3)
        Button(
            controls, text="恢复默认计划", command=self.load_default_batch_plan, width=14
        ).pack(side=LEFT, padx=3)
        Label(controls, text="批次 manual-label").pack(side=LEFT, padx=(15, 3))
        self.batch_label_entry = Entry(controls, width=18)
        self.batch_label_entry.pack(side=LEFT, padx=3)
        self.boundary_button = Button(
            controls,
            text="计算并检查视野边界",
            command=self.start_boundary_check,
            width=22,
            state="disabled",
        )
        self.boundary_button.pack(side=RIGHT, padx=3)
        self.batch_button = Button(
            controls,
            text="开始完整批量实验",
            command=self.start_batch_experiment,
            width=20,
            state="disabled",
        )
        self.batch_button.pack(side=RIGHT, padx=3)
        self.motion_buttons.extend((self.batch_button, self.boundary_button))

        columns = (
            "execution_order",
            "trajectory_type",
            "preset_id",
            "condition_id",
            "speed_mm_s",
            "one_way_time_s",
            "nominal_one_way_distance_mm",
            "repeat_index",
            "manual_label",
            "enabled",
            "time_warning",
        )
        table_frame = Frame(section)
        table_frame.pack(fill="x", padx=6, pady=(0, 6))
        self.plan_tree = Treeview(table_frame, columns=columns, show="headings", height=9)
        headings = {
            "execution_order": "顺序",
            "trajectory_type": "轨迹",
            "preset_id": "预设",
            "condition_id": "条件ID",
            "speed_mm_s": "速度 mm/s",
            "one_way_time_s": "单程 s",
            "nominal_one_way_distance_mm": "名义单程 mm",
            "repeat_index": "重复",
            "manual_label": "manual_label",
            "enabled": "启用",
            "time_warning": "时间检查",
        }
        widths = (55, 110, 65, 95, 85, 75, 110, 55, 170, 55, 125)
        for column, width in zip(columns, widths):
            self.plan_tree.heading(column, text=headings[column])
            self.plan_tree.column(column, width=width, anchor="center", stretch=True)
        plan_scroll = Scrollbar(table_frame, orient="vertical", command=self.plan_tree.yview)
        self.plan_tree.configure(yscrollcommand=plan_scroll.set)
        self.plan_tree.pack(side=LEFT, fill="x", expand=True)
        plan_scroll.pack(side=RIGHT, fill="y")
        self.plan_tree.tag_configure("time_warning", foreground="#c00000")
        self.plan_tree.bind("<Double-1>", lambda _event: self.edit_selected_plan_row())
        self.refresh_plan_tree()

    def refresh_plan_tree(self) -> None:
        for item in self.plan_tree.get_children():
            self.plan_tree.delete(item)
        for row in normalize_plan(self.batch_plan):
            if row["trajectory_type"] == "static":
                lower = upper = 5.0
            else:
                lower, upper = recommended_time_bounds(row["trajectory_type"])
            duration_warning = (
                row["trajectory_type"] != "static"
                and row["preset_id"] == "CUSTOM"
                and not lower - 1e-7 <= float(row["one_way_time_s"]) <= upper + 1e-7
            )
            self.plan_tree.insert(
                "",
                END,
                iid=str(row["execution_order"]),
                values=(
                    row["execution_order"],
                    TRAJECTORY_LABELS[row["trajectory_type"]],
                    row["preset_id"],
                    row["condition_id"],
                    f"{row['speed_mm_s']:g}",
                    f"{row['one_way_time_s']:.1f}",
                    f"{row['nominal_one_way_distance_mm']:.1f}",
                    f"R{row['repeat_index']:02d}",
                    row["manual_label"],
                    "是" if row["enabled"] else "否",
                    f"⚠ 不在{lower:.1f}～{upper:.1f}s" if duration_warning else "正常",
                ),
                tags=("time_warning",) if duration_warning else (),
            )

    def load_default_batch_plan(self) -> None:
        self.batch_plan = build_default_plan()
        self.refresh_plan_tree()

    def _selected_plan_index(self) -> int:
        selection = self.plan_tree.selection()
        if not selection:
            raise ValueError("请先在计划表中选择一行。")
        order = int(selection[0])
        return next(
            index
            for index, row in enumerate(self.batch_plan)
            if int(row["execution_order"]) == order
        )

    def toggle_selected_plan_row(self) -> None:
        try:
            index = self._selected_plan_index()
            self.batch_plan[index]["enabled"] = not bool(self.batch_plan[index]["enabled"])
            self.refresh_plan_tree()
            self.plan_tree.selection_set(str(self.batch_plan[index]["execution_order"]))
        except ValueError as exc:
            messagebox.showerror("计划表", str(exc))

    def edit_selected_plan_row(self) -> None:
        try:
            index = self._selected_plan_index()
            row = dict(self.batch_plan[index])
            if row["trajectory_type"] == "static":
                label = simpledialog.askstring(
                    "编辑静止基线", "manual_label：", initialvalue=row["manual_label"], parent=self.root
                )
                if label is None:
                    return
                row["manual_label"] = label
            else:
                speed = simpledialog.askfloat(
                    "编辑计划行",
                    "speed_mm_s：",
                    initialvalue=float(row["speed_mm_s"]),
                    minvalue=config.ROBOT_EXPERIMENT_MIN_SPEED_MM_S,
                    maxvalue=config.ROBOT_EXPERIMENT_MAX_SPEED_MM_S,
                    parent=self.root,
                )
                if speed is None:
                    return
                duration = simpledialog.askfloat(
                    "编辑计划行",
                    "one_way_time_s：",
                    initialvalue=float(row["one_way_time_s"]),
                    minvalue=0.001,
                    maxvalue=config.ROBOT_EXPERIMENT_MAX_TOTAL_TIME_S,
                    parent=self.root,
                )
                if duration is None:
                    return
                label = simpledialog.askstring(
                    "编辑计划行", "manual_label：", initialvalue=row["manual_label"], parent=self.root
                )
                if label is None:
                    return
                row.update(speed_mm_s=speed, one_way_time_s=duration, manual_label=label)
                lower, upper = recommended_time_bounds(row["trajectory_type"])
                if not lower - 1e-7 <= duration <= upper + 1e-7:
                    messagebox.showwarning(
                        "单程时间警告",
                        f"该轨迹 one_way_time_s 建议范围为 {lower:.1f}～{upper:.1f} s；"
                        "程序保留你的输入，不会静默修改。",
                    )
            self.batch_plan[index] = normalize_plan_row(row)
            self.refresh_plan_tree()
            self.plan_tree.selection_set(str(row["execution_order"]))
        except ValueError as exc:
            messagebox.showerror("计划参数错误", str(exc))

    def _plan_with_current_directions(self) -> list[dict[str, Any]]:
        rows = []
        for source in self.batch_plan:
            row = dict(source)
            if row["trajectory_type"] == "x_line":
                row.update(x_direction=self.x_direction.get(), y_direction="+Y")
            elif row["trajectory_type"] == "xy_line":
                row.update(
                    x_direction=self.xy_x_direction.get(),
                    y_direction=self.xy_y_direction.get(),
                    angle_deg=float(self.xy_angle_entry.get()),
                )
            elif row["trajectory_type"] == "l_shape":
                row.update(x_direction=self.l_x_direction.get(), y_direction=self.l_y_direction.get())
            rows.append(row)
        return normalize_plan(rows)

    def _save_current_plan(self, purpose: str) -> Path:
        plan_dir = OUTPUT_ROOT / "_batch_plans"
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.perf_counter_ns()}"
        return write_plan(plan_dir / f"{purpose}_{stamp}.json", self._plan_with_current_directions())

    def _validate_enabled_plan(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enabled = [row for row in plan if row["enabled"]]
        if not enabled:
            raise ValueError("计划中没有启用行。")
        for row in enabled:
            if row["trajectory_type"] == "static":
                continue
            self._validate_speed(float(row["speed_mm_s"]), f"第 {row['execution_order']} 行速度")
            if (
                float(row["nominal_one_way_distance_mm"])
                > config.ROBOT_EXPERIMENT_MAX_SEGMENT_MM + 1e-9
            ):
                raise ValueError(
                    f"第 {row['execution_order']} 行名义单程距离超过 "
                    f"{config.ROBOT_EXPERIMENT_MAX_SEGMENT_MM:g} mm 最大位移限制。"
                )
            if row["trajectory_type"] == "l_shape":
                leg_mm = float(row["nominal_one_way_distance_mm"]) / 2.0
                if float(config.ROBOT_RELATIVE_BLEND_MM) >= 0.5 * leg_mm:
                    raise ValueError(
                        f"第 {row['execution_order']} 行 L 折线每段仅 {leg_mm:.3f} mm，"
                        f"1 mm blend 必须小于相邻短边允许值 {0.5 * leg_mm:.3f} mm；禁止运行。"
                    )
        return enabled

    def start_batch_experiment(self) -> None:
        if not self.robot_connection_verified:
            messagebox.showerror("未完成通信检测", "请先成功运行‘检测机械臂通信’。")
            return
        try:
            plan = self._plan_with_current_directions()
            enabled = self._validate_enabled_plan(plan)
            if not self._confirm_motion(
                "完整批量实验确认",
                f"将执行 {len(enabled)} 个启用分段；相机和 UR 各只初始化一次；"
                "手电筒仅在整批开始时检测一次。",
            ):
                return
            plan_path = self._save_current_plan("batch")
            stop_path = self._make_stop_request_path("batch_experiment")
            self.start_mode(
                "batch_experiment",
                [
                    "--batch-plan-file", str(plan_path),
                    "--batch-manual-label", self.batch_label_entry.get().strip(),
                    "--ui-confirmed", "--stop-request-path", str(stop_path),
                ],
                stop_path,
            )
        except ValueError as exc:
            messagebox.showerror("批量计划错误", str(exc))

    def start_boundary_check(self) -> None:
        if not self.robot_connection_verified:
            messagebox.showerror("未完成通信检测", "请先成功运行‘检测机械臂通信’。")
            return
        robot = None
        try:
            plan = self._plan_with_current_directions()
            self._validate_enabled_plan(plan)
            envelope = compute_plan_envelope(plan)
            if not envelope:
                raise ValueError("当前启用计划没有运动轨迹。")
            from robot import URRobot

            robot = URRobot()
            robot.connect(require_control=False)
            a = robot.current_tcp_pose()
            lines = [f"当前 A 点：{[round(value, 6) for value in a]}"]
            total_distance_mm = 0.0
            dwell_count = 0
            for trajectory_type in ("x_line", "xy_line", "l_shape"):
                if trajectory_type not in envelope:
                    continue
                item = envelope[trajectory_type]
                far = list(a)
                for axis in range(3):
                    far[axis] += item["farthest_offset_mm"][axis] / 1000.0
                lines.append(
                    f"{TRAJECTORY_LABELS[trajectory_type]} 最远点："
                    f"{[round(value, 6) for value in far]}；"
                    f"相对位移 "
                    f"{[round(value, 1) for value in item['farthest_offset_mm']]} mm"
                )
                if trajectory_type == "l_shape":
                    total_distance_mm += 2.0 * (
                        abs(item["dx_mm"]) + abs(item["dy_mm"])
                    )
                    dwell_count += 2
                else:
                    total_distance_mm += 2.0 * math.hypot(item["dx_mm"], item["dy_mm"])
                    dwell_count += 1
            estimated_s = (
                total_distance_mm / float(config.BOUNDARY_CHECK_SPEED_MM_S)
                + dwell_count * float(config.BOUNDARY_CHECK_DWELL_SECONDS)
            )
            lines.extend(
                [
                    "预计路线：A→X最大点→A；A→D最大点→A；A→L-B→L-C→B→A。",
                    f"检查速度 10 mm/s，加速度 0.05 m/s²，预计至少 {estimated_s:.1f} s。",
                    "此功能只显示实时画面，由操作者人工判断视野；完成后不会自动开始实验。",
                ]
            )
            if not messagebox.askyesno(
                "确认视野边界检查", "\n\n".join(lines) + "\n\n确认工作区无人且急停可触及后继续？"
            ):
                return
            plan_path = self._save_current_plan("boundary")
            stop_path = self._make_stop_request_path("boundary_check")
            self.start_mode(
                "boundary_check",
                ["--batch-plan-file", str(plan_path), "--ui-confirmed",
                 "--stop-request-path", str(stop_path)],
                stop_path,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("边界检查准备失败", str(exc))
        finally:
            if robot is not None:
                robot.disconnect()

    def _build_log_section(self, root: Tk) -> None:
        section = LabelFrame(root, text="F. 日志和停止")
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
        self.show_usage_guide()
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
            self.batch_offline_button,
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
        """把简短操作顺序写入日志，不打断当前操作。"""

        self.append_log("操作说明\n" + "=" * 72 + "\n")
        self.append_log(USAGE_GUIDE_TEXT + "\n")
        self.append_log("=" * 72 + "\n\n")

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
            one_way_time = self._read_float(self.x_one_way_time_entry, "X直线单程时间")
            self._validate_speed(speed, "X直线速度")
            distance_mm = speed * one_way_time
            if distance_mm > config.ROBOT_EXPERIMENT_MAX_SEGMENT_MM:
                raise ValueError("X直线单程距离超过安全上限。")
            summary = (
                f"X直线：{self.x_direction.get()}，速度 {speed:.1f} mm/s，"
                f"单程时间 {one_way_time:.1f} s，单程 {distance_mm:.1f} mm。"
            )
            self._start_motion(
                "x_line_experiment",
                [
                    "--speed-mm-s",
                    str(speed),
                    "--one-way-time-s",
                    str(one_way_time),
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
            one_way_time = self._read_float(self.xy_one_way_time_entry, "X-Y直线单程时间")
            angle = self._read_float(self.xy_angle_entry, "X-Y直线角度")
            self._validate_speed(speed, "X-Y直线速度")
            if angle > config.ROBOT_EXPERIMENT_MAX_ANGLE_DEG:
                raise ValueError("X-Y直线角度超过安全上限。")
            distance_mm = speed * one_way_time
            if distance_mm > config.ROBOT_EXPERIMENT_MAX_SEGMENT_MM:
                raise ValueError("X-Y直线单程距离超过安全上限。")
            dx = distance_mm * math.cos(math.radians(angle))
            dy = distance_mm * math.sin(math.radians(angle))
            summary = (
                f"X-Y倾斜直线：{self.xy_x_direction.get()}/{self.xy_y_direction.get()}，"
                f"速度 {speed:.1f} mm/s，单程 {distance_mm:.1f} mm，"
                f"dx {dx:.1f} mm，dy {dy:.1f} mm。"
            )
            self._start_motion(
                "xy_line_experiment",
                [
                    "--speed-mm-s",
                    str(speed),
                    "--one-way-time-s",
                    str(one_way_time),
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
            speed = self._read_float(self.l_speed_entry, "L折线共同速度")
            one_way_time = self._read_float(self.l_one_way_time_entry, "L折线单程总时间")
            blend = float(self.l_blend_entry.get().strip())
            if not math.isfinite(blend) or blend < 0:
                raise ValueError("blend 必须是有限非负数。")
            self._validate_speed(speed, "L折线共同速度")
            dx = speed * one_way_time / 2.0
            dy = speed * one_way_time / 2.0
            if max(dx, dy) > config.ROBOT_EXPERIMENT_MAX_SEGMENT_MM:
                raise ValueError("L折线单段距离超过安全上限。")
            if blend > 0 and blend >= 0.5 * min(dx, dy):
                raise ValueError("blend 必须小于较短相邻线段的一半。")
            summary = (
                f"L折线：{self.l_x_direction.get()}/{self.l_y_direction.get()}，"
                f"X段 {dx:.1f} mm，Y段 {dy:.1f} mm，"
                f"A→B→C 单程总时间 {one_way_time:.1f} s，"
                f"完整往返名义时间 {2.0 * one_way_time:.1f} s。"
            )
            self._start_motion(
                "xy_l_experiment",
                [
                    "--speed-mm-s",
                    str(speed),
                    "--one-way-time-s",
                    str(one_way_time),
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
                self.append_log("已通知 main.py 安全停止：stopL、当前相机文件封口、子进程退出。\n")
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
