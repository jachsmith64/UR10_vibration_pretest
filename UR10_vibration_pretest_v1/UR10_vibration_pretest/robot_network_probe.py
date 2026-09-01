"""UR 控制器纯只读网络探针。

这个脚本只完成两类操作：
1. 连接 Dashboard 29999 端口，发送固定的只读查询命令；
2. 连接 RTDE 30004 端口，确认 TCP 服务可达后立即断开。

它不导入 ur_rtde，不创建控制接口，也不包含上电、松闸、启动程序、
回零、move、speed、servo、stop 等命令。
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from typing import Any


DASHBOARD_PORT = 29999
RTDE_PORT = 30004
READ_ONLY_DASHBOARD_COMMANDS = (
    ("polyscope_version", "PolyscopeVersion"),
    ("robot_mode", "robotmode"),
    ("safety_status", "safetystatus"),
)


def _read_line(connection: socket.socket) -> str:
    data = bytearray()
    while len(data) < 8192:
        chunk = connection.recv(1024)
        if not chunk:
            break
        data.extend(chunk)
        if b"\n" in chunk:
            break
    return data.decode("utf-8", errors="replace").strip()


def query_dashboard(host: str, timeout_s: float) -> dict[str, str]:
    """只发送 READ_ONLY_DASHBOARD_COMMANDS 中列出的状态查询。"""

    result: dict[str, str] = {}
    with socket.create_connection((host, DASHBOARD_PORT), timeout=timeout_s) as connection:
        connection.settimeout(timeout_s)
        result["greeting"] = _read_line(connection)
        for key, command in READ_ONLY_DASHBOARD_COMMANDS:
            connection.sendall((command + "\n").encode("ascii"))
            result[key] = _read_line(connection)
    return result


def check_tcp_port(host: str, port: int, timeout_s: float) -> float:
    """建立 TCP 连接后立即断开，返回握手耗时（毫秒）。"""

    started = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout_s):
        pass
    return (time.perf_counter() - started) * 1000.0


def run_probe(host: str, timeout_s: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "host": host,
        "read_only": True,
        "dashboard_port": DASHBOARD_PORT,
        "rtde_port": RTDE_PORT,
        "dashboard_ok": False,
        "rtde_port_ok": False,
    }

    try:
        started = time.perf_counter()
        result["dashboard"] = query_dashboard(host, timeout_s)
        result["dashboard_latency_ms"] = round(
            (time.perf_counter() - started) * 1000.0,
            3,
        )
        result["dashboard_ok"] = True
    except Exception as exc:
        result["dashboard_error"] = f"{type(exc).__name__}: {exc}"

    try:
        result["rtde_connect_latency_ms"] = round(
            check_tcp_port(host, RTDE_PORT, timeout_s),
            3,
        )
        result["rtde_port_ok"] = True
    except Exception as exc:
        result["rtde_error"] = f"{type(exc).__name__}: {exc}"

    result["success"] = bool(result["dashboard_ok"] and result["rtde_port_ok"])
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读检查 UR Dashboard 与 RTDE 网络连通性，不发送运动命令。"
    )
    parser.add_argument("--host", required=True, help="示教器网络页显示的机械臂 IP。")
    parser.add_argument("--timeout-s", type=float, default=3.0, help="单次连接/读取超时。")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    if arguments.timeout_s <= 0:
        raise ValueError("--timeout-s 必须大于 0。")

    print("[安全] 纯只读探针：不会上电、松闸、启动程序、回零或发送运动/停止命令。")
    result = run_probe(arguments.host, float(arguments.timeout_s))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
