import math
import unittest

import config
from robot import build_relative_motion_trajectory


class RelativeTrajectorySemanticsTests(unittest.TestCase):
    """Pure trajectory tests: importing/running these never creates URRobot."""

    def setUp(self) -> None:
        self.a = [0.1, -0.2, 0.3, 1.0, 2.0, 3.0]
        self.common = {
            "speed_mm_s": 20.0,
            "one_way_time_s": 5.0,
            "x_direction": "+X",
            "y_direction": "+Y",
            "angle_deg": 45.0,
            "blend_mm": 1.0,
            "acceleration_m_s2": 0.10,
        }

    def test_x_is_a_to_b_then_a(self) -> None:
        trajectory, computed = build_relative_motion_trajectory("x_line", self.a, self.common)
        self.assertEqual([point.name for point in trajectory], ["A", "B", "A_return"])
        self.assertAlmostEqual(trajectory[1].pose[0] - self.a[0], 0.100)
        self.assertEqual(trajectory[-1].pose, self.a)
        self.assertAlmostEqual(computed["nominal_one_way_distance_mm"], 100.0)

    def test_diagonal_uses_path_speed(self) -> None:
        trajectory, _ = build_relative_motion_trajectory("xy_line", self.a, self.common)
        component = 0.100 / math.sqrt(2.0)
        self.assertAlmostEqual(trajectory[1].pose[0] - self.a[0], component)
        self.assertAlmostEqual(trajectory[1].pose[1] - self.a[1], component)
        self.assertEqual(trajectory[-1].pose, self.a)

    def test_l_splits_time_and_stops_at_c_and_final_a(self) -> None:
        trajectory, computed = build_relative_motion_trajectory("l_shape", self.a, self.common)
        self.assertEqual(
            [point.name for point in trajectory],
            ["A", "B", "C", "B_return", "A_return"],
        )
        self.assertAlmostEqual(trajectory[1].pose[0] - self.a[0], 0.050)
        self.assertAlmostEqual(trajectory[2].pose[1] - self.a[1], 0.050)
        self.assertAlmostEqual(computed["x_leg_time_s"], 2.5)
        self.assertAlmostEqual(computed["y_leg_time_s"], 2.5)
        self.assertAlmostEqual(trajectory[1].blend_radius_m, 0.001)
        self.assertEqual(trajectory[2].blend_radius_m, 0.0)
        self.assertEqual(trajectory[-1].blend_radius_m, 0.0)
        self.assertEqual(computed["corner_mode"], "BL01")

    def test_too_short_l_segment_is_rejected_before_hardware(self) -> None:
        parameters = dict(self.common, speed_mm_s=1.0, one_way_time_s=1.0)
        with self.assertRaisesRegex(ValueError, "交融半径"):
            build_relative_motion_trajectory("l_shape", self.a, parameters)

    def test_formal_acceleration_default_is_point_one(self) -> None:
        self.assertAlmostEqual(config.ROBOT_EXPERIMENT_ACCELERATION_M_S2, 0.10)

    def test_mode_specific_defaults_use_reduced_x_and_diagonal_times(self) -> None:
        base = {
            "speed_mm_s": 1.0,
            "x_direction": "+X",
            "y_direction": "+Y",
            "angle_deg": 45.0,
            "blend_mm": 1.0,
            "acceleration_m_s2": 0.10,
        }
        _, x = build_relative_motion_trajectory("x_line", self.a, base)
        _, diagonal = build_relative_motion_trajectory("xy_line", self.a, base)
        _, l_shape = build_relative_motion_trajectory("l_shape", self.a, base)
        self.assertAlmostEqual(x["one_way_time_s"], 3.0)
        self.assertAlmostEqual(diagonal["one_way_time_s"], 4.2)
        self.assertAlmostEqual(l_shape["one_way_time_s"], 9.0)
        self.assertAlmostEqual(x["dx_m"] * 1000.0, 3.0)
        self.assertAlmostEqual(diagonal["dx_m"] * 1000.0, 4.2 / math.sqrt(2.0))
        self.assertAlmostEqual(diagonal["dy_m"] * 1000.0, 4.2 / math.sqrt(2.0))
        self.assertAlmostEqual(l_shape["dx_m"] * 1000.0 * 2.0 / 3.0, 3.0)


if __name__ == "__main__":
    unittest.main()
