from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from batch_plan import (
    PRESETS,
    allocate_batch,
    assert_outputs_available,
    build_default_plan,
    compute_plan_envelope,
    normalize_plan_row,
    preset_parameters,
    segment_output_paths,
    segment_stem,
    simulate_batch_schedule,
    status_after_failure,
    trajectory_geometry,
)


class BatchPlanTests(unittest.TestCase):
    def test_default_plan_has_29_rows_in_expected_round_order(self) -> None:
        plan = build_default_plan()
        self.assertEqual(len(plan), 29)
        self.assertEqual(plan[0]["trajectory_type"], "static")
        self.assertEqual(
            [(row["trajectory_type"], row["preset_id"], row["repeat_index"]) for row in plan[1:5]],
            [("x_line", "P01", 1), ("x_line", "P05", 1), ("x_line", "P10", 1), ("x_line", "P20", 1)],
        )
        self.assertEqual(sum(row["trajectory_type"] == "x_line" for row in plan), 12)
        self.assertEqual(sum(row["trajectory_type"] == "xy_line" for row in plan), 8)
        self.assertEqual(sum(row["trajectory_type"] == "l_shape" for row in plan), 8)

    def test_preset_distances(self) -> None:
        expected_x = {
            "P01": 3.0,
            "P05": 13.0,
            "P10": 23.0,
            "P20": 32.0,
        }
        for preset_id in PRESETS:
            values = preset_parameters(preset_id, "x_line")
            geometry = trajectory_geometry("x_line", **values)
            self.assertAlmostEqual(
                geometry["nominal_one_way_distance_mm"], expected_x[preset_id]
            )

    def test_x_and_diagonal_axes_do_not_exceed_two_thirds_of_l_axis(self) -> None:
        for preset_id in PRESETS:
            l_values = preset_parameters(preset_id, "l_shape")
            x_values = preset_parameters(preset_id, "x_line")
            d_values = preset_parameters(preset_id, "xy_line")
            l_geometry = trajectory_geometry("l_shape", **l_values)
            x_geometry = trajectory_geometry("x_line", **x_values)
            d_geometry = trajectory_geometry("xy_line", **d_values)
            target_axis = l_geometry["dx_mm"] * 2.0 / 3.0
            self.assertLessEqual(x_geometry["dx_mm"], target_axis + 1e-9)
            self.assertLessEqual(d_geometry["dx_mm"], target_axis + 1e-9)
            self.assertLessEqual(d_geometry["dy_mm"], target_axis + 1e-9)
            # P20 每 0.1 s 对应 2 mm，向下量化最多允许损失一个时间步。
            self.assertLess(target_axis - x_geometry["dx_mm"], 2.0)
            self.assertLess(target_axis - d_geometry["dx_mm"], 2.0)

    def test_l_shape_splits_one_way_time_evenly(self) -> None:
        geometry = trajectory_geometry("l_shape", 20.0, 5.0)
        self.assertAlmostEqual(geometry["x_leg_time_s"], 2.5)
        self.assertAlmostEqual(geometry["y_leg_time_s"], 2.5)
        self.assertAlmostEqual(geometry["x_leg_distance_mm"], 50.0)
        self.assertAlmostEqual(geometry["y_leg_distance_mm"], 50.0)

    def test_default_envelope(self) -> None:
        envelope = compute_plan_envelope(build_default_plan())
        self.assertAlmostEqual(envelope["x_line"]["dx_mm"], 32.0)
        expected_diagonal_axis = 20.0 * 2.3 / 2**0.5
        self.assertAlmostEqual(envelope["xy_line"]["dx_mm"], expected_diagonal_axis)
        self.assertAlmostEqual(envelope["xy_line"]["dy_mm"], expected_diagonal_axis)
        self.assertAlmostEqual(envelope["l_shape"]["dx_mm"], 50.0)
        self.assertAlmostEqual(envelope["l_shape"]["dy_mm"], 50.0)

    def test_custom_condition_changes_envelope_and_filename(self) -> None:
        row = normalize_plan_row(
            {
                "execution_order": 1,
                "trajectory_type": "x_line",
                "speed_mm_s": 15.0,
                "one_way_time_s": 6.0,
                "repeat_index": 1,
                "manual_label": "custom",
                "enabled": True,
            }
        )
        self.assertEqual(row["condition_id"], "V15_T06")
        self.assertEqual(row["preset_id"], "CUSTOM")
        self.assertAlmostEqual(compute_plan_envelope([row])["x_line"]["dx_mm"], 90.0)
        self.assertEqual(segment_stem("20260901_143210_B01", row), "20260901_143210_B01_X_V15_T06_R01")

    def test_two_second_inter_motion_analysis_window(self) -> None:
        schedule = [row for row in simulate_batch_schedule(build_default_plan()) if row["motion_start_s"] is not None]
        for previous, current in zip(schedule, schedule[1:]):
            self.assertAlmostEqual(current["motion_start_s"] - previous["motion_end_s"], 2.0)

    def test_abort_marks_remaining_rows(self) -> None:
        statuses = status_after_failure(build_default_plan(), 3)
        self.assertEqual(statuses[0]["status"], "COMPLETED")
        self.assertEqual(statuses[2]["status"], "FAILED")
        self.assertTrue(all(row["status"] == "ABORTED" for row in statuses[3:]))

    def test_output_files_are_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch_id, batch_dir = allocate_batch(root, now=datetime(2026, 9, 1, 14, 32, 10))
            paths = segment_output_paths(batch_dir, batch_id, build_default_plan()[1])
            paths["video"].write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                assert_outputs_available(paths)

    def test_batch_sequence_persists_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = allocate_batch(root, "test", datetime(2026, 9, 1, 14, 32, 10))
            second, _ = allocate_batch(root, "test", datetime(2026, 9, 1, 14, 33, 10))
            self.assertIn("_B01_test", first)
            self.assertIn("_B02_test", second)


if __name__ == "__main__":
    unittest.main()
