"""机械臂断电静止模式的纯离线测试；不打开相机、不接触机器人。"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import main
import micro_closed_loop as mcl
import offline_static_test as offline


def _summary(sigma: float, rms: float, peak: float) -> dict[str, float]:
    return {
        "static_sigma_um": sigma,
        "static_rms_um": rms,
        "static_peak_to_peak_um": peak,
        "chessboard_detection_rate": 1.0,
    }


class OfflineStaticComparisonTests(unittest.TestCase):
    def test_three_trials_are_aggregated_by_median(self) -> None:
        result = offline.aggregate_trials(
            [_summary(1.0, 2.0, 8.0), _summary(2.0, 4.0, 12.0), _summary(1.5, 3.0, 10.0)]
        )
        self.assertEqual(result["static_sigma_um"], 1.5)
        self.assertEqual(result["static_rms_um"], 3.0)
        self.assertEqual(result["static_peak_to_peak_um"], 10.0)
        self.assertEqual(result["static_rms_um_min"], 2.0)
        self.assertEqual(result["static_rms_um_max"], 4.0)

    def test_powered_state_clearly_higher_is_reported(self) -> None:
        result = offline.compare_power_states(
            _summary(1.0, 2.0, 5.0), _summary(2.0, 4.0, 10.0)
        )
        self.assertEqual(result["verdict"], "POWERED_STATE_HIGHER")

    def test_similar_baselines_do_not_blame_the_robot(self) -> None:
        result = offline.compare_power_states(
            _summary(1.0, 2.0, 5.0), _summary(1.1, 2.2, 5.5)
        )
        self.assertEqual(result["verdict"], "SIMILAR_BASELINES")

    def test_missing_powered_baseline_is_explicit(self) -> None:
        result = offline.compare_power_states(_summary(1.0, 2.0, 5.0), None)
        self.assertEqual(result["verdict"], "NO_POWERED_REFERENCE")

    def test_latest_powered_static_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "micro_motion_20260101_000000" / "static" / "static_summary.json"
            newer = root / "micro_motion_20260102_000000" / "static" / "static_summary.json"
            mcl.write_json(older, _summary(1.0, 2.0, 5.0))
            mcl.write_json(newer, _summary(2.0, 3.0, 6.0))
            older.touch()
            newer.touch()
            path, payload = offline.find_latest_powered_static(root)
            self.assertEqual(path, newer)
            self.assertEqual(payload["static_rms_um"], 3.0)


class OfflineStaticIsolationTests(unittest.TestCase):
    def test_mode_is_valid_even_when_robot_host_is_blank(self) -> None:
        with patch.object(config, "ROBOT_HOST", ""):
            config.validate_config("offline_static_test")

    def test_defaults_are_three_matching_five_second_trials(self) -> None:
        self.assertEqual(config.OFFLINE_STATIC_REPEATS, 3)
        self.assertEqual(config.OFFLINE_STATIC_SECONDS, config.MICRO_LOOP_STATIC_SECONDS)

    def test_module_has_no_robot_import(self) -> None:
        source = Path(offline.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import robot", source)
        self.assertNotIn("from robot", source)

    def test_main_route_only_forwards_stop_path(self) -> None:
        expected = Path("X:/offline.stop")
        returned = Path("X:/result")
        arguments = argparse.Namespace(stop_request_path=expected)
        with patch.object(offline, "run_offline_static_test", return_value=returned) as call:
            self.assertEqual(main.run_offline_static_test_mode(arguments), returned)
        call.assert_called_once_with(expected)


if __name__ == "__main__":
    unittest.main()

