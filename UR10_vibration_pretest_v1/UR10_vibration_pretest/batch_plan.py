"""UR10 批量实验的纯计算层。

本模块不导入相机 SDK 或 ur_rtde，也不连接任何硬件。UI、主进程和离线测试
共用这里的预设、计划、条件命名、文件命名、边界和时间线定义。
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PRESETS: dict[str, dict[str, float]] = {
    "P01": {"speed_mm_s": 1.0, "one_way_time_s": 9.0},
    "P05": {"speed_mm_s": 5.0, "one_way_time_s": 8.0},
    "P10": {"speed_mm_s": 10.0, "one_way_time_s": 7.0},
    "P20": {"speed_mm_s": 20.0, "one_way_time_s": 5.0},
}
PRESET_ORDER = tuple(PRESETS)
# L 保持实机验证通过的时间。X 和 D 的每个运动轴，均缩到同预设 L 单轴的 2/3：
# L 单轴 = v*t_L/2；目标轴位移 = v*t_L/3。
# 因此 X 的 t=t_L/3；45° D 的 t=t_L*sqrt(2)/3。
TRAJECTORY_TIME_SCALE = {
    "x_line": 1.0 / 3.0,
    "xy_line": math.sqrt(2.0) / 3.0,
    "l_shape": 1.0,
}
TRAJECTORY_CODES = {"static": "ST05", "x_line": "X", "xy_line": "D", "l_shape": "L"}
TRAJECTORY_LABELS = {
    "static": "静止基线",
    "x_line": "X直线",
    "xy_line": "XY倾斜直线",
    "l_shape": "L折线",
}
PLAN_COLUMNS = (
    "execution_order",
    "trajectory_type",
    "preset_id",
    "speed_mm_s",
    "one_way_time_s",
    "nominal_one_way_distance_mm",
    "repeat_index",
    "manual_label",
    "enabled",
)
SEGMENTS_COLUMNS = (
    "batch_id",
    "execution_order",
    "condition_id",
    "trajectory_type",
    "preset_id",
    "repeat_index",
    "speed_mm_s",
    "one_way_time_s",
    "nominal_one_way_distance_mm",
    "actual_start_pose",
    "actual_end_pose",
    "system_start_time",
    "system_end_time",
    "batch_elapsed_start_s",
    "batch_elapsed_end_s",
    "first_frame_id",
    "last_frame_id",
    "flash_system_time",
    "status",
    "output_video",
    "manual_label",
)


def _finite_positive(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} 必须是有限正数。")
    return number


def _number_token(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return f"{int(round(value)):02d}"
    text = f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return text


def preset_parameters(preset_id: str, trajectory_type: str) -> dict[str, float]:
    """返回某轨迹专用预设；PRESETS 中的时间是已验证 L 形基准时间。"""

    if preset_id not in PRESETS:
        raise ValueError(f"未知预设 {preset_id!r}。")
    if trajectory_type not in TRAJECTORY_TIME_SCALE:
        raise ValueError(f"未知轨迹 {trajectory_type!r}。")
    base = PRESETS[preset_id]
    exact_time = float(base["one_way_time_s"]) * float(
        TRAJECTORY_TIME_SCALE[trajectory_type]
    )
    # X/D 为了现场填写方便，统一向下保留 0.1 s；绝不因四舍五入超过 2/3 目标包络。
    one_way_time_s = (
        exact_time
        if trajectory_type == "l_shape"
        else int((exact_time + 1e-9) * 10.0) / 10.0
    )
    return {
        "speed_mm_s": float(base["speed_mm_s"]),
        "one_way_time_s": one_way_time_s,
    }


def recommended_time_bounds(trajectory_type: str) -> tuple[float, float]:
    durations = [
        preset_parameters(preset_id, trajectory_type)["one_way_time_s"]
        for preset_id in PRESET_ORDER
    ]
    return min(durations), max(durations)


def condition_id_for(
    speed_mm_s: float,
    one_way_time_s: float,
    trajectory_type: str = "l_shape",
) -> tuple[str, str]:
    """返回 (condition_id, preset_id)；自定义数值绝不沿用错误的 P 编号。"""

    speed = _finite_positive("speed_mm_s", speed_mm_s)
    duration = _finite_positive("one_way_time_s", one_way_time_s)
    for preset_id in PRESET_ORDER:
        values = preset_parameters(preset_id, trajectory_type)
        if math.isclose(speed, values["speed_mm_s"], abs_tol=1e-7) and math.isclose(
            duration, values["one_way_time_s"], rel_tol=1e-8, abs_tol=1e-7
        ):
            return preset_id, preset_id
    return f"V{_number_token(speed)}_T{_number_token(duration)}", "CUSTOM"


def trajectory_geometry(
    trajectory_type: str,
    speed_mm_s: float,
    one_way_time_s: float,
    *,
    angle_deg: float = 45.0,
) -> dict[str, float]:
    """按统一 one_way_time 定义计算 A→最远点的相对几何。"""

    if trajectory_type == "static":
        return {
            "nominal_one_way_distance_mm": 0.0,
            "dx_mm": 0.0,
            "dy_mm": 0.0,
            "segment_time_s": float(one_way_time_s),
        }
    speed = _finite_positive("speed_mm_s", speed_mm_s)
    duration = _finite_positive("one_way_time_s", one_way_time_s)
    distance = speed * duration
    if trajectory_type == "x_line":
        return {
            "nominal_one_way_distance_mm": distance,
            "dx_mm": distance,
            "dy_mm": 0.0,
            "segment_time_s": duration,
        }
    if trajectory_type == "xy_line":
        angle = math.radians(float(angle_deg))
        return {
            "nominal_one_way_distance_mm": distance,
            "dx_mm": distance * math.cos(angle),
            "dy_mm": distance * math.sin(angle),
            "segment_time_s": duration,
        }
    if trajectory_type == "l_shape":
        leg_time = duration / 2.0
        leg_distance = speed * leg_time
        return {
            "nominal_one_way_distance_mm": distance,
            "dx_mm": leg_distance,
            "dy_mm": leg_distance,
            "segment_time_s": leg_time,
            "x_leg_time_s": leg_time,
            "y_leg_time_s": leg_time,
            "x_leg_distance_mm": leg_distance,
            "y_leg_distance_mm": leg_distance,
        }
    raise ValueError(f"未知 trajectory_type={trajectory_type!r}。")


def normalize_plan_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    trajectory_type = str(normalized["trajectory_type"])
    if trajectory_type == "static":
        normalized.update(
            {
                "preset_id": "ST05",
                "condition_id": "ST05",
                "speed_mm_s": 0.0,
                "one_way_time_s": 5.0,
                "nominal_one_way_distance_mm": 0.0,
            }
        )
    else:
        speed = _finite_positive("speed_mm_s", normalized["speed_mm_s"])
        duration = _finite_positive("one_way_time_s", normalized["one_way_time_s"])
        condition_id, preset_id = condition_id_for(speed, duration, trajectory_type)
        geometry = trajectory_geometry(trajectory_type, speed, duration)
        normalized.update(
            {
                "speed_mm_s": speed,
                "one_way_time_s": duration,
                "preset_id": preset_id,
                "condition_id": condition_id,
                "nominal_one_way_distance_mm": geometry["nominal_one_way_distance_mm"],
            }
        )
    normalized["execution_order"] = int(normalized["execution_order"])
    normalized["repeat_index"] = int(normalized.get("repeat_index", 1))
    normalized["manual_label"] = str(normalized.get("manual_label", ""))
    normalized["enabled"] = bool(normalized.get("enabled", True))
    normalized.setdefault("angle_deg", 45.0)
    normalized.setdefault("x_direction", "+X")
    normalized.setdefault("y_direction", "+Y")
    return normalized


def build_default_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "execution_order": 1,
            "trajectory_type": "static",
            "preset_id": "ST05",
            "speed_mm_s": 0.0,
            "one_way_time_s": 5.0,
            "repeat_index": 1,
            "manual_label": "",
            "enabled": True,
        }
    ]
    order = 2
    for trajectory_type, repeat_count in (("x_line", 3), ("xy_line", 2), ("l_shape", 2)):
        for repeat_index in range(1, repeat_count + 1):
            for preset_id in PRESET_ORDER:
                values = preset_parameters(preset_id, trajectory_type)
                rows.append(
                    {
                        "execution_order": order,
                        "trajectory_type": trajectory_type,
                        "preset_id": preset_id,
                        "speed_mm_s": values["speed_mm_s"],
                        "one_way_time_s": values["one_way_time_s"],
                        "repeat_index": repeat_index,
                        "manual_label": "",
                        "enabled": True,
                    }
                )
                order += 1
    return [normalize_plan_row(row) for row in rows]


def normalize_plan(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalize_plan_row(row) for row in rows]
    orders = [row["execution_order"] for row in normalized]
    if len(orders) != len(set(orders)):
        raise ValueError("execution_order 不能重复。")
    return sorted(normalized, key=lambda row: row["execution_order"])


def enabled_plan(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in normalize_plan(rows) if row["enabled"]]


def compute_plan_envelope(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in enabled_plan(rows):
        trajectory_type = row["trajectory_type"]
        if trajectory_type == "static":
            continue
        geometry = trajectory_geometry(
            trajectory_type,
            row["speed_mm_s"],
            row["one_way_time_s"],
            angle_deg=float(row.get("angle_deg", 45.0)),
        )
        sign_x = 1.0 if row.get("x_direction", "+X") == "+X" else -1.0
        sign_y = 1.0 if row.get("y_direction", "+Y") == "+Y" else -1.0
        dx = sign_x * geometry["dx_mm"]
        dy = sign_y * geometry["dy_mm"]
        candidate = {
            "trajectory_type": trajectory_type,
            "execution_order": row["execution_order"],
            "condition_id": row["condition_id"],
            "nominal_one_way_distance_mm": geometry["nominal_one_way_distance_mm"],
            "dx_mm": dx,
            "dy_mm": dy,
            "point_b_offset_mm": [dx, 0.0, 0.0] if trajectory_type == "l_shape" else [dx, dy, 0.0],
            "farthest_offset_mm": [dx, dy, 0.0],
        }
        previous = result.get(trajectory_type)
        if previous is None or math.hypot(dx, dy) > math.hypot(
            previous["dx_mm"], previous["dy_mm"]
        ):
            result[trajectory_type] = candidate
    return result


def simulate_batch_schedule(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成离线名义时间线；相邻运动之间保留运动后 1 s 和运动前 1 s。"""

    schedule: list[dict[str, Any]] = []
    cursor = 0.0
    previous_was_motion = False
    for row in enabled_plan(rows):
        if row["trajectory_type"] == "static":
            record_start = cursor
            record_end = record_start + 5.0
            schedule.append(
                {
                    **row,
                    "record_start_s": record_start,
                    "motion_start_s": None,
                    "motion_end_s": None,
                    "record_end_s": record_end,
                }
            )
            cursor = record_end
            previous_was_motion = False
            continue

        record_start = cursor
        motion_start = record_start + 1.0
        motion_end = motion_start + 2.0 * float(row["one_way_time_s"])
        record_end = motion_end + 1.0
        schedule.append(
            {
                **row,
                "record_start_s": record_start,
                "motion_start_s": motion_start,
                "motion_end_s": motion_end,
                "record_end_s": record_end,
            }
        )
        if previous_was_motion:
            previous = next(item for item in reversed(schedule[:-1]) if item["motion_end_s"] is not None)
            gap = motion_start - float(previous["motion_end_s"])
            if gap < 2.0 - 1e-9:
                raise AssertionError(f"相邻运动静止间隔只有 {gap:.3f} s。")
        cursor = record_end
        previous_was_motion = True
    return schedule


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", str(label).strip())
    return cleaned.strip("-_")[:48]


def _existing_batch_sequence(output_root: Path) -> int:
    highest = 0
    pattern = re.compile(r"^\d{8}_\d{6}_B(\d+)(?:_|$)")
    if output_root.exists():
        for child in output_root.iterdir():
            match = pattern.match(child.name)
            if match:
                highest = max(highest, int(match.group(1)))
    counter_path = output_root / ".batch_sequence.json"
    if counter_path.exists():
        try:
            highest = max(highest, int(json.loads(counter_path.read_text(encoding="utf-8"))["last_sequence"]))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    return highest


def allocate_batch(output_root: Path, manual_label: str = "", now: datetime | None = None) -> tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    sequence = _existing_batch_sequence(output_root) + 1
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    label = sanitize_label(manual_label)
    batch_id = f"{stamp}_B{sequence:02d}" + (f"_{label}" if label else "")
    batch_dir = output_root / batch_id
    while batch_dir.exists():
        sequence += 1
        batch_id = f"{stamp}_B{sequence:02d}" + (f"_{label}" if label else "")
        batch_dir = output_root / batch_id
    batch_dir.mkdir(parents=False, exist_ok=False)
    counter_path = output_root / ".batch_sequence.json"
    temp_path = output_root / ".batch_sequence.json.tmp"
    temp_path.write_text(json.dumps({"last_sequence": sequence}, indent=2), encoding="utf-8")
    temp_path.replace(counter_path)
    return batch_id, batch_dir


def segment_stem(batch_id: str, row: dict[str, Any]) -> str:
    normalized = normalize_plan_row(row)
    if normalized["trajectory_type"] == "static":
        return f"{batch_id}_ST05"
    code = TRAJECTORY_CODES[normalized["trajectory_type"]]
    return (
        f"{batch_id}_{code}_{normalized['condition_id']}_"
        f"R{normalized['repeat_index']:02d}"
    )


def segment_output_paths(batch_dir: Path, batch_id: str, row: dict[str, Any]) -> dict[str, Path]:
    stem = segment_stem(batch_id, row)
    return {
        "video": batch_dir / f"{stem}_HIK.avi",
        "robot": batch_dir / f"{stem}_UR10.txt",
        "vision": batch_dir / f"{stem}_VISION.txt",
        "runlog": batch_dir / f"{stem}_RUNLOG.txt",
        "meta": batch_dir / f"{stem}_META.json",
    }


def assert_outputs_available(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("拒绝覆盖已有输出：" + ", ".join(existing))


def write_plan(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    normalized = normalize_plan(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_plan(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("批量计划 JSON 顶层必须是列表。")
    return normalize_plan(data)


def write_segments_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SEGMENTS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key in ("actual_start_pose", "actual_end_pose"):
                if isinstance(output.get(key), (list, dict)):
                    output[key] = json.dumps(output[key], ensure_ascii=False)
            writer.writerow(output)
    return path


def status_after_failure(rows: Iterable[dict[str, Any]], failed_execution_order: int) -> list[dict[str, Any]]:
    result = []
    for row in normalize_plan(rows):
        if row["execution_order"] < failed_execution_order:
            status = "COMPLETED"
        elif row["execution_order"] == failed_execution_order:
            status = "FAILED"
        else:
            status = "ABORTED"
        result.append({**row, "status": status})
    return result
