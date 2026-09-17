"""闭环机器人启动联锁的无硬件回归测试。"""

from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import main
import micro_closed_loop as mcl
import robot as robot_module


class _FakeRobot:
    """只实现启动与收尾接口；刻意不提供 read_dashboard_status。"""

    def __init__(self, safety_status: str) -> None:
        self.dashboard_info = {"safety_status": safety_status}
        self.connected_with_control = False
        self.stopped = False
        self.disconnected = False

    def connect(self, *, require_control: bool) -> None:
        self.connected_with_control = require_control

    def read_state(self) -> dict[str, object]:
        return {
            "host_ns": 1,
            "robot_timestamp_s": 1.0,
            "robot_mode": 7,
            "safety_mode": 1,
            "actual_tcp_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            "actual_tcp_speed": [0.0] * 6,
        }

    def current_tcp_pose(self) -> list[float]:
        return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]

    def stop_motion(self) -> None:
        self.stopped = True

    def disconnect(self) -> None:
        self.disconnected = True


class MicroClosedLoopRobotStartupTests(unittest.TestCase):
    def _run_worker(self, fake_robot: _FakeRobot):
        error_queue: queue.Queue[str] = queue.Queue()
        command_queue: queue.Queue[dict[str, str]] = queue.Queue()
        status_queue: queue.Queue[dict[str, object]] = queue.Queue()
        stop_event = threading.Event()
        command_queue.put({"action": "SHUTDOWN"})

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        with patch.object(robot_module, "URRobot", return_value=fake_robot):
            mcl.micro_closed_loop_robot_worker(
                error_queue,
                command_queue,
                status_queue,
                stop_event,
                Path(temp_dir.name),
            )
        return error_queue, status_queue, stop_event

    def test_connect_dashboard_info_reaches_ready_and_inbox_wait(self) -> None:
        fake_robot = _FakeRobot("Safetystatus: NORMAL")

        error_queue, status_queue, stop_event = self._run_worker(fake_robot)
        ready = main._StatusInbox(status_queue).wait(
            "robot",
            "READY",
            error_queue,
            stop_event,
            None,
            timeout_s=0.2,
        )

        self.assertEqual(ready["start_pose"], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        self.assertTrue(fake_robot.connected_with_control)
        self.assertTrue(fake_robot.stopped)
        self.assertTrue(fake_robot.disconnected)

    def test_non_normal_dashboard_status_still_blocks_ready(self) -> None:
        fake_robot = _FakeRobot("Safetystatus: PROTECTIVE_STOP")

        error_queue, status_queue, stop_event = self._run_worker(fake_robot)

        self.assertTrue(stop_event.is_set())
        self.assertTrue(status_queue.empty())
        self.assertIn("安全状态不是 NORMAL", error_queue.get_nowait())

    def test_idle_monitor_stops_when_measured_tcp_leaves_fixed_envelope(self) -> None:
        class OutsideRobot(_FakeRobot):
            def __init__(self) -> None:
                super().__init__("Safetystatus: NORMAL")
                self.read_count = 0

            def read_state(self) -> dict[str, object]:
                self.read_count += 1
                state = super().read_state()
                if self.read_count >= 2:
                    state["actual_tcp_pose"] = [0.300001, 0.2, 0.3, 0.0, 0.0, 0.0]
                return state

        fake_robot = OutsideRobot()
        error_queue: queue.Queue[str] = queue.Queue()
        command_queue: queue.Queue[dict[str, str]] = queue.Queue()
        status_queue: queue.Queue[dict[str, object]] = queue.Queue()
        stop_event = threading.Event()

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        with patch.object(robot_module, "URRobot", return_value=fake_robot):
            mcl.micro_closed_loop_robot_worker(
                error_queue,
                command_queue,
                status_queue,
                stop_event,
                Path(temp_dir.name),
            )

        self.assertTrue(stop_event.is_set())
        self.assertTrue(fake_robot.stopped)
        self.assertTrue(fake_robot.disconnected)
        self.assertIn("超过固定安全半径", error_queue.get_nowait())


if __name__ == "__main__":
    unittest.main()
