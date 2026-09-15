"""
数据记录 —— 完整时序的统一落盘格式 + 统计量计算。

时序列（16 个数值列，单位写死在列名里）：
    t        物理时间            [s]
    along_mm 名义沿程位置        [mm]
    y_d      标称 Y 目标         [m]
    y_sim    TCP 实际 Y          [m]
    e_um     偏差 y_sim − y_d    [µm]   （与预实验数据同口径）
    v_m_s    误差速度 de/dt      [m/s]  （SFC 公式的 SI 口径）
    Fd_N     等效扰动力          [N]    （扰动支路，控制器不读）
    F_spring_N         = −K·e                [N]
    F_linear_damping_N = −B0·v               [N]
    F_shear_N          = −μ|v|^(n−1)·v       [N]
    F_SFC_N            = 三项之和            [N]
    tau_1..6  τ_SFC = J_tcp^T·F_SFC          [N·m]

标量运行参数（case_id / mode / K / B0 / n / μ / r / v_ref / Fd 的 SHA-256）不逐行重复，
放在文件头的 `# key=value` 注释块里，可直接被 load_trace 读回。

落盘时按 TRACE_STRIDE 抽稀（默认 5 → 200 Hz），远高于关心的 45 Hz 振动带。
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

import config

TAU_COLS = [f"tau_{i + 1}" for i in range(6)]
TRACE_COLUMNS = (["t", "along_mm", "y_d", "y_sim", "e_um",
                  "e_raw_um", "b_um", "e_vib_um",
                  "v_m_s", "Fd_N",
                  "F_spring_N", "F_linear_damping_N", "F_shear_N", "F_SFC_N"]
                 + TAU_COLS)

# 单位说明，写进文件头，避免下游猜
UNITS = {
    "t": "s", "along_mm": "mm", "y_d": "m", "y_sim": "m", "e_um": "um",
    "e_raw_um": "um", "b_um": "um", "e_vib_um": "um",
    "v_m_s": "m/s", "Fd_N": "N", "F_spring_N": "N", "F_linear_damping_N": "N",
    "F_shear_N": "N", "F_SFC_N": "N",
    **{c: "N*m" for c in TAU_COLS},
}

# e_raw/b/e_vib 这三列【不是仿真量】，而是预实验参考序列在同一时间索引上的回放：
#   e_raw_um = 预实验原始横向误差（含慢偏移）
#   b_um     = 构造的理想慢偏移基线（见 slowoffset.py）
#   e_vib_um = e_raw − b，即真正被重建为 Fd 的净振动
# 它们只随模板时间轴变化，与 mode / K / B0 / n 无关，写进每组时序里是为了
# 让任一组都能直接做「原始 vs 去慢偏移 vs 仿真残差」的同轴对照，无需再拼文件。
REFERENCE_COLS = ("e_raw_um", "b_um", "e_vib_um")


def write_trace(path: Path, data: dict[str, np.ndarray],
                meta: dict[str, Any], stride: int = config.TRACE_STRIDE) -> Path:
    """写完整时序（抽稀）。data 的键必须是 TRACE_COLUMNS 的子集，缺失列报错。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [c for c in TRACE_COLUMNS if c not in data]
    if missing:
        raise ValueError(f"{path.name}: 缺列 {missing}")

    n = len(np.asarray(data["t"]))
    sl = slice(None, None, max(1, int(stride)))
    cols = {c: np.asarray(data[c], dtype=float)[sl] for c in TRACE_COLUMNS}

    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# trace n_full={n} stride={stride} n_written={cols['t'].size}\n")
        for key in sorted(meta):
            val = meta[key]
            if isinstance(val, float):
                fh.write(f"# {key}={val:.10g}\n")
            else:
                fh.write(f"# {key}={val}\n")
        fh.write("# units " + " ".join(f"{c}={UNITS[c]}" for c in TRACE_COLUMNS) + "\n")
        fh.write(",".join(TRACE_COLUMNS) + "\n")
        arr = np.column_stack([cols[c] for c in TRACE_COLUMNS])
        for row in arr:
            fh.write(",".join(f"{x:.6g}" for x in row) + "\n")
    return path


def load_trace(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """读回 write_trace 写出的文件，返回 (文件头元数据, 各列数组)。

    刻意手写解析：文件头是整块 `#` 注释，而 np.genfromtxt(names=True) 会把
    「第一行（去注释后为空）」当成列名，得到 1 个空列名并报 17 列错位。
    """
    path = Path(path)
    meta: dict[str, Any] = {}
    names: list[str] | None = None
    body: list[str] = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                s = line[1:].strip()
                if s.startswith("trace ") or s.startswith("units "):
                    continue
                if "=" in s:
                    k, v = s.split("=", 1)
                    try:
                        meta[k.strip()] = float(v)
                    except ValueError:
                        meta[k.strip()] = v
                continue
            if names is None:
                names = [c.strip() for c in line.strip().split(",")]
                continue
            if line.strip():
                body.append(line.rstrip("\r\n"))

    if names is None or not body:
        raise ValueError(f"{path.name}: 没有数据行")

    arr = np.loadtxt(io.StringIO("\n".join(body)), delimiter=",", ndmin=2)
    if arr.shape[1] != len(names):
        raise ValueError(f"{path.name}: 数据列数 {arr.shape[1]} 与列名数 {len(names)} 不符")
    return meta, {n: arr[:, i].astype(float) for i, n in enumerate(names)}


def save_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return path


# 兼容旧调用名（analysis.py 里用过）
save_summary = save_json
