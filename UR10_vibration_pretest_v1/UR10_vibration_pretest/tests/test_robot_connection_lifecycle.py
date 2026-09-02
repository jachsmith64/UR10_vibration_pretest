from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import config
import robot as robot_module


class RobotConnectionLifecycleTests(unittest.TestCase):
    def test_control_is_created_before_receive_and_script_is_stopped(self) -> None:
        events: list[str] = []

        class Control:
            def __init__(self, host: str, frequency: float) -> None:
                events.append("control_created")

            def isConnected(self) -> bool:
                return True

            def stopScript(self) -> None:
                events.append("control_script_stopped")

            def disconnect(self) -> None:
                events.append("control_disconnected")

        class Receive:
            def __init__(self, host: str, frequency: float) -> None:
                events.append("receive_created")

            def isConnected(self) -> bool:
                return True

            def disconnect(self) -> None:
                events.append("receive_disconnected")

        modules = {
            "rtde_control": types.SimpleNamespace(RTDEControlInterface=Control),
            "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=Receive),
        }
        robot = robot_module.URRobot()
        with (
            patch.dict(sys.modules, modules),
            patch.object(robot_module, "read_dashboard_information", return_value={}),
            patch.object(config, "ROBOT_RTDE_CONNECT_ATTEMPTS", 1),
        ):
            robot.connect(require_control=True)
            robot.disconnect()

        self.assertEqual(events[:2], ["control_created", "receive_created"])
        self.assertLess(events.index("control_script_stopped"), events.index("receive_disconnected"))
        self.assertLess(events.index("receive_disconnected"), events.index("control_disconnected"))

    def test_caught_connection_failure_is_cleaned_then_retried(self) -> None:
        events: list[str] = []
        receive_attempt = 0

        class Control:
            def __init__(self, host: str, frequency: float) -> None:
                events.append("control_created")

            def isConnected(self) -> bool:
                return True

            def stopScript(self) -> None:
                events.append("control_script_stopped")

            def disconnect(self) -> None:
                events.append("control_disconnected")

        class Receive:
            def __init__(self, host: str, frequency: float) -> None:
                nonlocal receive_attempt
                receive_attempt += 1
                events.append(f"receive_attempt_{receive_attempt}")
                if receive_attempt == 1:
                    raise RuntimeError("temporary EOF")

            def isConnected(self) -> bool:
                return True

            def disconnect(self) -> None:
                events.append("receive_disconnected")

        modules = {
            "rtde_control": types.SimpleNamespace(RTDEControlInterface=Control),
            "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=Receive),
        }
        robot = robot_module.URRobot()
        with (
            patch.dict(sys.modules, modules),
            patch.object(robot_module, "read_dashboard_information", return_value={}),
            patch.object(config, "ROBOT_RTDE_CONNECT_ATTEMPTS", 2),
            patch.object(config, "ROBOT_RTDE_CONNECT_RETRY_DELAY_S", 0.0),
        ):
            robot.connect(require_control=True)
            robot.disconnect()

        self.assertEqual(receive_attempt, 2)
        self.assertIn("control_script_stopped", events)


if __name__ == "__main__":
    unittest.main()
