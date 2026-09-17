"""一键 XY 微动闭环能力测试的纯计算单测。

这些用例全部不碰相机 SDK、不碰 ur_rtde、不碰真机，只测离线可重算的那一层：
目标表、控制律、终止判据、噪声估计、目录分配、配置校验。

最要紧的两条是"反例"用例——它们不是覆盖率，而是判据能不能用的前提：
  1. 早期误差还很大但在单调收缩，不得误判成 LIMIT_CYCLE；
  2. 误差已低到噪声底时的换号，不得判成极限环。
删掉这两条，5 μm 档每一组都会被报成极限环，整轮实验的结论就废了。
"""

from __future__ import annotations

import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

import config
import micro_closed_loop as mcl


def _errors(values):
    """把误差序列补成 evaluate_termination 需要的 (errors_um, valid) 形式。"""

    return [float(value) for value in values], [True] * len(values)


class TargetTableTests(unittest.TestCase):
    def test_twelve_targets_in_the_required_order_and_signs(self) -> None:
        targets = mcl.build_targets()
        self.assertEqual(len(targets), 12)
        # 轴优先、幅值次之、方向最后：X 的 5/20/50 走完才轮到 Y。
        self.assertEqual(
            [(t["axis"], t["amplitude_um"], t["direction"]) for t in targets],
            [
                ("X", 5, 1), ("X", 5, -1),
                ("X", 20, 1), ("X", 20, -1),
                ("X", 50, 1), ("X", 50, -1),
                ("Y", 5, 1), ("Y", 5, -1),
                ("Y", 20, 1), ("Y", 20, -1),
                ("Y", 50, 1), ("Y", 50, -1),
            ],
        )
        # target_um 必须与 direction 同号，且绝对值等于幅值。
        for target in targets:
            self.assertEqual(target["target_um"], target["direction"] * target["amplitude_um"])
            self.assertEqual(
                target["target_id"].endswith("_pos") or target["target_id"].endswith("_neg"),
                True,
            )

    def test_target_ids_are_unique(self) -> None:
        ids = [t["target_id"] for t in mcl.build_targets()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_six_groups_each_holding_both_directions(self) -> None:
        groups = mcl.build_groups()
        self.assertEqual(len(groups), 6)
        for group in groups:
            self.assertEqual([t["direction"] for t in group["targets"]], [1, -1])
            for target in group["targets"]:
                self.assertEqual(target["axis"], group["axis"])
                self.assertEqual(target["amplitude_um"], group["amplitude_um"])


class CommandLawTests(unittest.TestCase):
    def test_proportional_and_sign(self) -> None:
        self.assertAlmostEqual(mcl.command_for_error(10.0, kp=1.0), 10.0)
        self.assertAlmostEqual(mcl.command_for_error(-10.0, kp=1.0), -10.0)

    def test_the_command_law_does_not_take_a_sign(self) -> None:
        # ★ 回归钉子：command_for_error **不接受** sign。
        #
        # 上一版这里有一条 `command_for_error(10.0, sign=-1.0) == -10.0` 的用例，
        # 它把"符号翻转"写进了控制律。但 error 已经来自 axis_measured_um 归一
        # 过的机器人轴坐标，控制律里再乘一次 sign 就是**用了两次符号**：在
        # sign=−1 的相机安装下（机器人 +Y 在图像上是 −y）命令被重新翻反，
        # 误差按 err_{k+1} = 2·err_k 翻倍（50 → 100 → 200 …），6 轮内机器人
        # 跑到半毫米之外，结束状态是 DIVERGING。
        #
        # 注意它**不会**被 SIGN_INVERTED 联锁抓到（双符号下指令与实测恒同号），
        # 所以症状看起来像"这台机械臂连 50 μm 都逼近不了"，而不是像代码错了。
        # 正是因为当时有一条测试"证明"它是对的行为，这个缺陷才活得下来。
        with self.assertRaises(TypeError):
            mcl.command_for_error(10.0, kp=1.0, sign=-1.0)

    def test_double_applied_sign_would_diverge_which_is_why_it_is_gone(self) -> None:
        # 把"上一版的错误控制律"按逐轮递推跑一遍，钉住它真实的后果：
        # 误差翻倍、结束于 DIVERGING，而不是收敛也不是 SIGN_INVERTED。
        # 这条用例的价值在于：以后有人若想"顺手把 sign 加回去"，会先看到
        # 它到底会造成什么，而不是看到一句抽象的警告。
        sign, target, max_iter = -1.0, 50.0, int(config.MICRO_LOOP_MAX_ITER)
        measured, errors = 0.0, []
        for _ in range(max_iter):
            error = target - measured
            errors.append(error)
            command = max(-100.0, min(100.0, sign * error))   # ← 错误的旧律
            measured += command
        self.assertEqual(errors, [50.0, 100.0, 200.0, 300.0, 400.0, 500.0])
        self.assertEqual(
            mcl.evaluate_termination(
                errors, [True] * len(errors),
                tol_um=float(config.MICRO_LOOP_POSITION_TOL_UM), noise_um=0.4,
            ),
            mcl.TERMINAL_DIVERGING,
        )

    def test_kp_scales_the_command(self) -> None:
        self.assertAlmostEqual(mcl.command_for_error(10.0, kp=0.5), 5.0)

    def test_clamped_at_max_correction(self) -> None:
        self.assertAlmostEqual(mcl.command_for_error(500.0, kp=1.0), 100.0)
        self.assertAlmostEqual(mcl.command_for_error(-500.0, kp=1.0), -100.0)

    def test_deadband_returns_exactly_zero(self) -> None:
        # 死区是必需的：robot.validate_trajectory 拒绝 length <= 1e-6 m，
        # 收敛时 Kp·error 可能只有 0.4 μm，硬发下去会让整轮在即将成功时抛错。
        self.assertEqual(mcl.command_for_error(1.0, kp=1.0), 0.0)
        self.assertEqual(mcl.command_for_error(-1.9, kp=1.0), 0.0)
        self.assertEqual(mcl.command_for_error(0.0, kp=1.0), 0.0)

    def test_deadband_boundary_is_inclusive(self) -> None:
        # 判据是 `abs(command) < MIN_COMMAND_UM`，所以恰好等于死区的命令**照发**。
        # 这是安全的一侧：死区必须严格大于 1.0 μm（validate_trajectory 拒绝
        # length <= 1e-6 m），边界取闭区间才不会在收敛瞬间踩到那条硬线。
        boundary = config.MICRO_LOOP_MIN_COMMAND_UM
        self.assertAlmostEqual(mcl.command_for_error(boundary, kp=1.0), boundary)
        just_under = boundary - 0.001
        self.assertEqual(mcl.command_for_error(just_under, kp=1.0), 0.0)

    def test_deadband_applies_after_clamping(self) -> None:
        # 限幅后的值才和死区比；限幅结果一定大于死区，所以大误差永远发得出去。
        self.assertNotEqual(mcl.command_for_error(1000.0, kp=1.0), 0.0)

    def test_non_finite_error_never_produces_a_command(self) -> None:
        # 测不到时不发命令，否则 NaN 会传进 validate_trajectory 直接抛错。
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.assertEqual(mcl.command_for_error(bad, kp=1.0), 0.0)


class EffectiveToleranceTests(unittest.TestCase):
    def test_noise_relative_widens_the_tolerance(self) -> None:
        # 容差 3 μm、测量噪声 4 μm 时，"误差 ≤ 3 μm"只是在筛噪声。
        self.assertAlmostEqual(mcl.effective_tolerance_um(3.0, 4.0), 8.0)

    def test_noise_relative_never_tightens_below_nominal(self) -> None:
        self.assertAlmostEqual(mcl.effective_tolerance_um(3.0, 0.1), 3.0)

    def test_invalid_noise_falls_back_to_nominal(self) -> None:
        for bad in (float("nan"), float("inf"), 0.0, -1.0):
            self.assertAlmostEqual(mcl.effective_tolerance_um(3.0, bad), 3.0)


class MedianOfTailTests(unittest.TestCase):
    def test_only_the_trailing_window_is_used(self) -> None:
        # 前 9 个值是 1000，最后 3 个是 1；窗口 0.3 s 只应看到后者。
        times = [i * 100_000_000 for i in range(12)]  # 每帧 0.1 s
        values = [1000.0] * 9 + [1.0, 1.0, 1.0]
        self.assertAlmostEqual(mcl.median_of_tail(values, times, 0.35), 1.0)

    def test_window_is_measured_in_time_not_frame_count(self) -> None:
        # 相机丢帧时"最后 N 帧"对应的真实时长会变，所以窗口按时间戳取。
        times = [0, 100_000_000, 200_000_000, 900_000_000, 1_000_000_000]
        values = [5.0, 5.0, 5.0, 99.0, 99.0]
        self.assertAlmostEqual(mcl.median_of_tail(values, times, 0.25), 99.0)

    def test_median_is_robust_to_a_single_bad_frame(self) -> None:
        times = [i * 10_000_000 for i in range(9)]
        values = [2.0, 2.0, 2.0, 2.0, 50.0, 2.0, 2.0, 2.0, 2.0]
        self.assertAlmostEqual(mcl.median_of_tail(values, times, 1.0), 2.0)

    def test_empty_or_mismatched_input_returns_none(self) -> None:
        self.assertIsNone(mcl.median_of_tail([], [], 0.2))
        self.assertIsNone(mcl.median_of_tail([1.0, 2.0], [1], 0.2))


class FrameBudgetTests(unittest.TestCase):
    def test_expected_frames_from_real_span_and_fps(self) -> None:
        start = 1_000_000_000
        self.assertEqual(mcl.expected_frames(start, start + 1_000_000_000, 132.23), 132)

    def test_degenerate_inputs_return_zero(self) -> None:
        self.assertEqual(mcl.expected_frames(0, 1_000_000_000, 0.0), 0)
        self.assertEqual(mcl.expected_frames(0, 1_000_000_000, -1.0), 0)
        self.assertEqual(mcl.expected_frames(1_000_000_000, 1_000_000_000, 132.23), 0)
        self.assertEqual(mcl.expected_frames(2_000_000_000, 1_000_000_000, 132.23), 0)

    def test_window_bytes_is_one_byte_per_pixel(self) -> None:
        # 全幅 Mono8：1936×1464×帧率×秒数。
        self.assertEqual(
            mcl.estimate_window_bytes(1936, 1464, 132.23, 6.0),
            int(round(1936 * 1464 * 132.23 * 6.0)),
        )
        # 最坏窗口必须落在内存预算内（本机 15.77 GiB 总量）。
        worst = mcl.estimate_window_bytes(
            config.MICRO_LOOP_FULL_FRAME_WIDTH,
            config.MICRO_LOOP_FULL_FRAME_HEIGHT,
            132.23,
            config.MICRO_LOOP_MAX_WINDOW_SECONDS,
        )
        self.assertLess(worst, 8 * 1024 ** 3)


class DiskTests(unittest.TestCase):
    def test_disk_free_bytes_works_before_the_directory_exists(self) -> None:
        # 磁盘门禁必须跑在创建任何目录之前，所以这里要能接受不存在的路径。
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "not_yet" / "deep" / "inside"
            self.assertFalse(missing.exists())
            self.assertGreater(mcl.disk_free_bytes(missing), 0)
            self.assertGreater(mcl.free_gb(missing), 0.0)

    def test_format_gib(self) -> None:
        self.assertEqual(mcl.format_gib(1024 ** 3), "1.00 GiB")


class RunDirectoryTests(unittest.TestCase):
    def test_allocate_creates_a_fresh_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            run_id, run_dir = mcl.allocate_closed_loop_run(root, datetime(2026, 9, 17, 12, 0, 0))
            self.assertEqual(run_id, "micro_motion_20260917_120000")
            self.assertTrue(run_dir.is_dir())

    def test_allocate_never_overwrites_an_existing_run(self) -> None:
        # 同一个时间戳重复分配时必须让路，绝不能把上一轮的 CSV 冲掉。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            stamp = datetime(2026, 9, 17, 12, 0, 0)
            first_id, first_dir = mcl.allocate_closed_loop_run(root, stamp)
            (first_dir / "master_summary.csv").write_text("keep me", encoding="utf-8")

            second_id, second_dir = mcl.allocate_closed_loop_run(root, stamp)
            third_id, third_dir = mcl.allocate_closed_loop_run(root, stamp)

            self.assertEqual(first_id, "micro_motion_20260917_120000")
            self.assertEqual(second_id, "micro_motion_20260917_120000_02")
            self.assertEqual(third_id, "micro_motion_20260917_120000_03")
            self.assertEqual(len({first_dir, second_dir, third_dir}), 3)
            self.assertEqual((first_dir / "master_summary.csv").read_text(encoding="utf-8"), "keep me")

    def test_target_dir_name_is_zero_padded(self) -> None:
        run_dir = Path("X:/run")
        self.assertEqual(mcl.target_dir(run_dir, "X", 5), run_dir / "X" / "005um")
        self.assertEqual(mcl.target_dir(run_dir, "Y", 50), run_dir / "Y" / "050um")

    def test_direction_slug(self) -> None:
        self.assertEqual(mcl.direction_slug(1), "positive")
        self.assertEqual(mcl.direction_slug(-1), "negative")


class CsvRoundTripTests(unittest.TestCase):
    def test_rows_survive_a_write_read_cycle(self) -> None:
        columns = ["iteration_id", "command_um", "note"]
        rows = [
            {"iteration_id": 1, "command_um": 13.5, "note": "中文也不能坏"},
            {"iteration_id": 2, "command_um": -2.0, "note": ""},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "iterations.csv"
            mcl.write_rows_csv(path, rows, columns)
            back = mcl.read_rows_csv(path)
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0]["iteration_id"], "1")
        self.assertEqual(back[0]["note"], "中文也不能坏")

    def test_unknown_keys_are_dropped_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            mcl.write_rows_csv(path, [{"a": 1, "extra": 2}], ["a"])
            back = mcl.read_rows_csv(path)
        self.assertEqual(list(back[0].keys()), ["a"])

    def test_overwrite_false_refuses_to_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            mcl.write_rows_csv(path, [{"a": 1}], ["a"])
            with self.assertRaises(FileExistsError):
                mcl.write_rows_csv(path, [{"a": 2}], ["a"], overwrite=False)


class SigmaEstimateTests(unittest.TestCase):
    def test_mad_sigma_matches_a_clean_series(self) -> None:
        # 单帧噪声用 MAD 估，比 std 更抗离群点。
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(mcl.mad_sigma(values), 1.4826)

    def test_mad_sigma_ignores_a_gross_outlier(self) -> None:
        values = [1.0, 1.0, 1.0, 1.0, 1000.0]
        self.assertLess(mcl.mad_sigma(values), 1.0)

    def test_measurement_sigma_uses_block_bootstrap_of_tail_medians(self) -> None:
        # 完全静止的序列 → 一次闭环测量的不确定度就是 0。
        self.assertAlmostEqual(
            mcl.estimate_measurement_sigma_um([3.0] * 40, 8), 0.0
        )
        # 有明显块状漂移时必须有非零估计（否则容差会被误判满足）。
        drifting = [float(i) for i in range(40)]
        self.assertGreater(mcl.estimate_measurement_sigma_um(drifting, 8), 0.0)

    def test_measurement_sigma_needs_enough_samples(self) -> None:
        self.assertTrue(math.isnan(mcl.estimate_measurement_sigma_um([1.0, 2.0], 8)))


class TerminationTests(unittest.TestCase):
    """终止判据全分支。tol=3 μm、噪声按各自用例给。"""

    def test_converged_after_three_consecutive_in_tolerance(self) -> None:
        errors, valid = _errors([10.0, 5.0, 2.0, -1.0, 1.5])
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=0.0),
            mcl.TERMINAL_CONVERGED,
        )

    def test_converged_requires_consecutive_not_scattered(self) -> None:
        # 中间那次超出容差就打断连击；序列也不够 2 个窗口，只能继续。
        errors, valid = _errors([10.0, 1.0, 9.0, 1.0, 1.0])
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=0.0),
            mcl.TERMINAL_STILL_SHRINKING,
        )

    def test_early_large_but_monotonically_shrinking_is_not_a_limit_cycle(self) -> None:
        """反例 1：误差还很大但一路在缩，绝不能报 LIMIT_CYCLE。

        这正是"增益正常、闭环正在工作"的典型前几轮。误判成极限环，
        整组实验的核心结论就反过来。
        """
        errors, valid = _errors([50.0, 45.0, 40.0, 36.0, 33.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0])
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=0.5),
            mcl.TERMINAL_STILL_SHRINKING,
        )

    def test_large_error_oscillating_without_progress_is_not_a_limit_cycle(self) -> None:
        """反例 1 的强化版：换号够多，但根本没走到路程的一半。

        这里 |e| 一直在 43–50 μm 之间来回，一次真实纠偏都没发生——
        那是探针符号错或命令没发出去，不是极限环。门 2 专门杀这个。
        """
        errors, valid = _errors([50.0, -48.0, 49.0, -47.0, 48.0, -46.0,
                                 47.0, -45.0, 46.0, -44.0, 45.0, -43.0])
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=1.0),
            mcl.TERMINAL_STILL_SHRINKING,
        )

    def test_sign_changes_below_the_noise_floor_are_not_a_limit_cycle(self) -> None:
        """反例 2：噪声 4 μm 时，±6 μm 的换号是噪声，不是极限环。

        不设这道门，任何收敛到噪声底的组都会被报成极限环。
        """
        errors, valid = _errors([50.0, 30.0, 4.0, -4.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0, 10.0, -10.0])
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=4.0),
            mcl.TERMINAL_STILL_SHRINKING,
        )

    def test_true_limit_cycle_is_detected(self) -> None:
        # 反复换号，且 6 步窗口的峰峰值与中位幅值都不再收缩。
        errors, valid = _errors([50.0, 30.0, 20.0, -19.0, 20.0, -19.0,
                                 20.0, -19.0, 20.0, -19.0, 20.0, -19.0])
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=1.0),
            mcl.TERMINAL_LIMIT_CYCLE,
        )

    def test_converging_oscillation_above_the_floor_is_not_a_limit_cycle(self) -> None:
        # 环路增益略大于 1：误差换号，但包络每步缩 0.85 → 6 步缩到 0.38。
        # 0.38 < PTP_KEEP(0.5) 且 < ABS_KEEP(0.7)，所以只有真极限环能存活。
        errors = [50.0 * ((-0.85) ** k) for k in range(12)]
        valid = [True] * 12
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=0.5),
            mcl.TERMINAL_STILL_SHRINKING,
        )

    def test_one_sided_plateau_is_stalled_not_limit_cycle(self) -> None:
        """5 μm 档的头号结果：控制器忽略了小于阈值的命令，误差冻结在一个常数上。

        这里一次号都不换，原始判据（换号+峰峰值不缩）永远不会触发，
        最后只会报 MAX_ITER —— 而 MAX_ITER 完全掩盖了"存在最小有效运动尺度"。
        """
        errors, valid = _errors([50.0, 47.0, 44.0, 41.0, 38.0, 35.0,
                                 33.0, 32.0, 32.1, 31.9, 32.05, 31.95])
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=0.5),
            mcl.TERMINAL_STALLED,
        )

    def test_plateau_below_the_tolerance_is_not_stalled(self) -> None:
        # 停在容差以内是收敛，不是"最小有效运动尺度"。
        errors, valid = _errors([5.0, 2.0, 1.0, 1.0, 1.0])
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=0.0),
            mcl.TERMINAL_CONVERGED,
        )

    def test_growing_error_is_diverging(self) -> None:
        errors, valid = _errors([5.0, 6.0, 8.0, 10.0, 13.0, 16.0,
                                 20.0, 25.0, 31.0, 38.0, 46.0, 55.0])
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=0.5),
            mcl.TERMINAL_DIVERGING,
        )

    def test_diverging_is_reported_even_when_no_progress_was_made(self) -> None:
        # 发散必须先于"无进展门"判定：发散序列恰好总是满足 min|recent| >= 0.5|e0|，
        # 排在门 2 之后就会被吞成 STILL_SHRINKING，而它是最该被报出来的状态。
        errors, valid = _errors([100.0, 150.0, 200.0, 260.0, 330.0, 420.0,
                                 530.0, 670.0, 850.0, 1080.0, 1370.0, 1740.0])
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=0.5),
            mcl.TERMINAL_DIVERGING,
        )

    def test_not_enough_samples_keeps_iterating(self) -> None:
        # 不足两个窗口时没有可比的两段，不下任何结论。
        errors, valid = _errors([50.0, 40.0, 30.0])
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=0.5),
            mcl.TERMINAL_STILL_SHRINKING,
        )

    def test_invalid_iterations_are_skipped_not_interpolated(self) -> None:
        # 中间几次测不到时，不能把无效点当成数据去比较；窗口只由合格点组成。
        errors, valid = _errors([50.0, 45.0, 40.0, 36.0, 33.0, 30.0,
                                 28.0, 26.0, 24.0, 22.0, 20.0, 18.0])
        valid[3] = False
        valid[7] = False
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=0.5),
            mcl.TERMINAL_STILL_SHRINKING,
        )

    def test_a_measurement_gap_blocks_the_comparison(self) -> None:
        # 两个窗口之间隔了太多无效迭代时，它们根本不是同一段过程。
        errors, valid = _errors([50.0, 45.0, 40.0, 36.0, 33.0, 30.0,
                                 28.0, 26.0, 24.0, 22.0, 20.0, 18.0])
        for index in (6, 7, 8, 9, 10):
            valid[index] = False
        # 剩下的合格点不足以构成两段可比窗口 → 继续。
        self.assertEqual(
            mcl.evaluate_termination(errors, valid, tol_um=3.0, noise_um=0.5),
            mcl.TERMINAL_STILL_SHRINKING,
        )

    def test_no_valid_iterations_at_all(self) -> None:
        self.assertEqual(
            mcl.evaluate_termination([1.0, 2.0], [False, False], tol_um=3.0, noise_um=0.5),
            mcl.TERMINAL_STILL_SHRINKING,
        )


class ConvergenceQualityTests(unittest.TestCase):
    def test_large_residual_against_small_noise_has_high_snr(self) -> None:
        errors, valid = _errors([10.0, 8.0, 7.0, 7.2, 6.9])
        residual, snr = mcl.convergence_quality(errors, valid, noise_um=0.5)
        self.assertAlmostEqual(residual, 7.0, places=6)
        self.assertGreater(snr, 2.0)

    def test_residual_indistinguishable_from_zero_is_low_snr(self) -> None:
        # 残差和零在统计上不可分 → 只能诚实记 CONVERGED_NOISE_LIMITED。
        errors, valid = _errors([3.0, 1.0, 0.1, -0.1, 0.2])
        residual, snr = mcl.convergence_quality(errors, valid, noise_um=4.0)
        self.assertLess(snr, 2.0)

    def test_zero_noise_with_zero_residual_does_not_divide_by_zero(self) -> None:
        errors, valid = _errors([1.0, 0.0, 0.0, 0.0])
        residual, snr = mcl.convergence_quality(errors, valid, noise_um=0.0)
        self.assertEqual(residual, 0.0)
        self.assertEqual(snr, 0.0)


class ProbeGateTests(unittest.TestCase):
    """探针各门禁逐一触发：构造假增益矩阵，确认每一种坏情况都被具名中止。"""

    def _baseline(self) -> dict[str, float]:
        return {"x": 0.0, "y": 0.0}

    def _forward(self, dx_x: float, dy_x: float, dx_y: float, dy_y: float):
        return {
            "X": {"x": dx_x, "y": dy_x},
            "Y": {"x": dx_y, "y": dy_y},
        }

    def _backward(self, dx_x: float, dy_x: float, dx_y: float, dy_y: float):
        return {
            "X": {"x": dx_x, "y": dy_x},
            "Y": {"x": dx_y, "y": dy_y},
        }

    def test_a_clean_probe_passes_and_sets_the_signs(self) -> None:
        gains = mcl.evaluate_probe_gains(
            self._baseline(),
            self._forward(1000.0, 0.0, 0.0, -1000.0),
            self._backward(0.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(gains.vision_axis["X"], "x")
        self.assertEqual(gains.vision_axis["Y"], "y")
        self.assertEqual(gains.sign["X"], 1.0)
        # 机器人 +Y 对应视觉 −y：符号必须由探针测出来，不能假设。
        self.assertEqual(gains.sign["Y"], -1.0)

    def test_no_visual_motion_aborts(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            mcl.evaluate_probe_gains(
                self._baseline(),
                self._forward(0.0, 0.0, 0.0, 1000.0),
                self._backward(0.0, 0.0, 0.0, 0.0),
            )
        self.assertIn("没有任何位移", str(ctx.exception))

    def test_gain_below_the_allowed_band_aborts(self) -> None:
        # 增益 0.1：20 次迭代也走不完 50 μm，闭环不可能成功。
        with self.assertRaises(RuntimeError) as ctx:
            mcl.evaluate_probe_gains(
                self._baseline(),
                self._forward(100.0, 0.0, 0.0, 1000.0),
                self._backward(0.0, 0.0, 0.0, 0.0),
            )
        self.assertIn("超出允许范围", str(ctx.exception))

    def test_gain_above_the_allowed_band_aborts(self) -> None:
        # 增益 2.5：过冲振荡风险。
        with self.assertRaises(RuntimeError) as ctx:
            mcl.evaluate_probe_gains(
                self._baseline(),
                self._forward(2500.0, 0.0, 0.0, 1000.0),
                self._backward(0.0, 0.0, 0.0, 0.0),
            )
        self.assertIn("超出允许范围", str(ctx.exception))

    def test_large_hysteresis_aborts(self) -> None:
        # 正向 1.0、反向 0.6 → 差 40%，远超 20%。回差这么大时闭环不是良定的。
        with self.assertRaises(RuntimeError) as ctx:
            mcl.evaluate_probe_gains(
                self._baseline(),
                self._forward(1000.0, 0.0, 0.0, 1000.0),
                self._backward(-400.0, 0.0, 0.0, -400.0),
            )
        self.assertIn("回差", str(ctx.exception))

    def test_both_robot_axes_on_one_vision_axis_aborts(self) -> None:
        # 像平面与机器人 XY 平面接近侧视：两个自由度不可分辨。
        with self.assertRaises(RuntimeError) as ctx:
            mcl.evaluate_probe_gains(
                self._baseline(),
                self._forward(1000.0, 0.0, 500.0, 0.0),
                self._backward(0.0, 0.0, 0.0, 0.0),
            )
        self.assertIn("不可分辨", str(ctx.exception))

    def test_cross_coupling_is_recorded_not_fatal(self) -> None:
        # 用户允许视觉上存在未补偿的横向位移，但操作者必须看得见 → 记 note。
        gains = mcl.evaluate_probe_gains(
            self._baseline(),
            self._forward(1000.0, 400.0, 0.0, 1000.0),
            self._backward(0.0, 0.0, 0.0, 0.0),
        )
        self.assertAlmostEqual(gains.cross_coupling["X"], 0.4)
        self.assertTrue(any("交叉耦合" in note for note in gains.notes))

    def test_probe_json_is_serialisable(self) -> None:
        gains = mcl.evaluate_probe_gains(
            self._baseline(),
            self._forward(1000.0, 0.0, 0.0, 1000.0),
            self._backward(0.0, 0.0, 0.0, 0.0),
        )
        payload = gains.to_json()
        self.assertEqual(payload["sign"]["X"], 1.0)
        self.assertIn("gain_matrix", payload)


class RobotSideAchievedTests(unittest.TestCase):
    def test_on_target_step_is_valid(self) -> None:
        status, _ = mcl.evaluate_achieved_robot_side(10.0, 10.0)
        self.assertEqual(status, mcl.STEP_VALID)

    def test_ignored_command_is_flagged_small(self) -> None:
        # 机器人没动：这就是最小有效运动尺度的直接证据。
        status, _ = mcl.evaluate_achieved_robot_side(10.0, 0.2)
        self.assertEqual(status, mcl.STEP_SMALL)

    def test_overshoot_is_flagged_large(self) -> None:
        status, _ = mcl.evaluate_achieved_robot_side(10.0, 40.0)
        self.assertEqual(status, mcl.STEP_LARGE)

    def test_wrong_direction_is_flagged_sign(self) -> None:
        status, note = mcl.evaluate_achieved_robot_side(10.0, -10.0)
        self.assertEqual(status, mcl.STEP_SIGN)
        self.assertIn("方向相反", note)

    def test_zero_command_is_trivially_valid(self) -> None:
        status, _ = mcl.evaluate_achieved_robot_side(0.0, 0.0)
        self.assertEqual(status, mcl.STEP_VALID)


class TempWorkspaceTests(unittest.TestCase):
    def test_workspace_lives_under_the_output_root(self) -> None:
        # 临时目录与运行目录必须同卷，os.replace 才是原子的、不跨卷。
        root = Path("X:/outputs")
        temp_dir, trash_dir = mcl.temp_workspace(root)
        self.assertEqual(temp_dir, root / mcl.TEMP_DIR_NAME)
        self.assertEqual(trash_dir, root / mcl.TRASH_DIR_NAME)
        self.assertEqual(temp_dir.parent, trash_dir.parent)

    def test_prepare_creates_both_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_dir, trash_dir = mcl.prepare_temp_workspace(Path(tmp))
            self.assertTrue(temp_dir.is_dir())
            self.assertTrue(trash_dir.is_dir())

    def test_drop_moves_files_into_trash_instead_of_unlinking(self) -> None:
        # Windows 上 np.memmap 的句柄不保证随 del 立即释放，
        # 直接 unlink 会抛 WinError 32，在 240 次迭代的循环里足以弄死整轮。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            temp_dir, trash_dir = mcl.prepare_temp_workspace(root)
            victim = temp_dir / "clip.raw"
            victim.write_bytes(b"\x00" * 16)
            mcl.drop_temp_files([victim], trash_dir)
            self.assertFalse(victim.exists())
            self.assertTrue((trash_dir / "clip.raw").exists())

    def test_drop_tolerates_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            temp_dir, trash_dir = mcl.prepare_temp_workspace(root)
            mcl.drop_temp_files([temp_dir / "never_existed.raw"], trash_dir)

    def test_sweep_removes_regular_files_and_reports_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _temp_dir, trash_dir = mcl.prepare_temp_workspace(root)
            (trash_dir / "a.raw").write_bytes(b"\x00" * 1024)
            (trash_dir / "b.csv").write_bytes(b"\x00" * 512)
            freed = mcl.sweep_directory(trash_dir)
            self.assertGreaterEqual(freed, 1536)
            self.assertEqual(mcl.directory_bytes(trash_dir), 0)

    def test_clip_temp_paths_are_namespaced_by_stem(self) -> None:
        paths = mcl.clip_temp_paths(Path("X:/t"), "run_X_005um_positive_007")
        self.assertEqual(paths["raw"].name, "run_X_005um_positive_007.raw")
        self.assertEqual(paths["frames"].name, "run_X_005um_positive_007_frames.csv")

    def test_clip_stem_is_unique_per_counter(self) -> None:
        stems = {mcl.next_clip_stem("run", "X_005um_positive", i) for i in range(5)}
        self.assertEqual(len(stems), 5)


class PoseFormatTests(unittest.TestCase):
    def test_pose_round_trips_as_one_csv_field(self) -> None:
        text = mcl._format_pose([0.1, -0.2, 0.3, 0.0, 1.5707, -3.14])
        self.assertEqual(len(text.split(";")), 6)
        self.assertIn("0.100000000", text)

    def test_missing_pose_is_empty_not_nan(self) -> None:
        self.assertEqual(mcl._format_pose(None), "")
        self.assertEqual(mcl._format_pose(["bad", "pose"]), "")


class ClipMeasurementTests(unittest.TestCase):
    def _measurement(self, **overrides):
        base = dict(
            raw_path=Path("X:/clip.raw"),
            frame_count=100,
            n_valid=95,
            measured_um=7.5,
            transverse_um=0.1,
            n_tail=20,
            sigma_tail_um=0.5,
            peak_axis_um=9.0,
            positions=[],
            times_ns=[],
            frame_rows=[],
            first_row_ns=0,
            last_row_ns=1,
        )
        base.update(overrides)
        return mcl.ClipMeasurement(**base)

    def test_usable_measurement(self) -> None:
        self.assertTrue(self._measurement().is_usable)

    def test_truncated_buffer_makes_it_unusable(self) -> None:
        # 缓冲写满被截断时，这一轮必须记测量丢失，绝不能拿半段数据发命令。
        self.assertFalse(self._measurement(truncated=True).is_usable)

    def test_too_few_tail_frames_is_unusable(self) -> None:
        self.assertFalse(self._measurement(n_tail=1, sigma_tail_um=0.1).is_usable)

    def test_noisy_tail_is_unusable(self) -> None:
        # 尾部还在动 → 这一轮测的不是停稳后的位置。
        self.assertFalse(self._measurement(sigma_tail_um=50.0).is_usable)

    def test_missing_or_non_finite_measurement_is_unusable(self) -> None:
        self.assertFalse(self._measurement(measured_um=None).is_usable)
        self.assertFalse(self._measurement(measured_um=float("nan")).is_usable)

    def test_rates_come_from_frame_rows(self) -> None:
        rows = [
            {"checker_is_valid": True, "accepted": True},
            {"checker_is_valid": True, "accepted": False},
            {"checker_is_valid": False, "accepted": False},
            {"checker_is_valid": True, "accepted": True},
        ]
        measurement = self._measurement(frame_rows=rows)
        self.assertAlmostEqual(measurement.detection_rate, 0.75)
        self.assertAlmostEqual(measurement.accept_rate, 0.5)

    def test_rates_are_zero_without_frames(self) -> None:
        measurement = self._measurement(frame_rows=[])
        self.assertEqual(measurement.detection_rate, 0.0)
        self.assertEqual(measurement.accept_rate, 0.0)

    def test_last_accepted_frame_index_skips_rejected_frames(self) -> None:
        rows = [
            {"frame_index": 0, "accepted": True},
            {"frame_index": 1, "accepted": True},
            {"frame_index": 2, "accepted": False},
            {"frame_index": 3, "accepted": False},
        ]
        measurement = self._measurement(frame_rows=rows)
        self.assertEqual(mcl.last_accepted_frame_index(measurement), 1)

    def test_last_accepted_frame_index_is_none_when_nothing_accepted(self) -> None:
        measurement = self._measurement(frame_rows=[{"frame_index": 0, "accepted": False}])
        self.assertIsNone(mcl.last_accepted_frame_index(measurement))


class PeakFrameTests(unittest.TestCase):
    def _measurement(self, positions_um):
        # peak_frame_index 读的是 frame_rows（CSV 列 checker_dx_mm，单位 mm），
        # 不是 positions；这里按真实落盘的形状构造。
        rows = [
            {
                "accepted": True,
                "frame_index": index,
                "checker_dx_mm": float(value) / 1000.0,
                "checker_dy_mm": 0.0,
            }
            for index, value in enumerate(positions_um)
        ]
        return mcl.ClipMeasurement(
            raw_path=Path("X:/clip.raw"),
            frame_count=len(rows),
            n_valid=len(rows),
            measured_um=0.0,
            transverse_um=0.0,
            n_tail=1,
            sigma_tail_um=0.0,
            peak_axis_um=0.0,
            positions=[],
            times_ns=[],
            frame_rows=rows,
            first_row_ns=0,
            last_row_ns=1,
        )

    def test_peak_is_measured_from_the_target_not_the_zero(self) -> None:
        # 目标 50 μm 时，停在 0 μm 的那帧离目标 50 μm，比 45 μm 那帧更远。
        # 若以参考零点为准，每一帧的"偏离"都指回起点，过冲证据帧就废了。
        measurement = self._measurement([0.0, 45.0, 50.0])
        self.assertEqual(
            mcl.peak_frame_index(measurement, axis="X", reference_um=0.0, target_um=50.0),
            0,
        )

    def test_peak_finds_the_largest_deviation_from_the_target(self) -> None:
        # 相对目标 50 μm：frame0 差 0、frame1 差 8（过冲）、frame2 差 10（不足）。
        measurement = self._measurement([50.0, 58.0, 40.0])
        self.assertEqual(
            mcl.peak_frame_index(measurement, axis="X", reference_um=0.0, target_um=50.0),
            2,
        )

    def test_rejected_frames_are_not_candidates(self) -> None:
        measurement = self._measurement([50.0, 90.0, 50.0])
        measurement.frame_rows[1]["accepted"] = False
        self.assertEqual(
            mcl.peak_frame_index(measurement, axis="X", reference_um=0.0, target_um=50.0),
            0,
        )

    def test_peak_uses_the_y_column_for_the_y_axis(self) -> None:
        measurement = self._measurement([0.0, 0.0, 0.0])
        for index, row in enumerate(measurement.frame_rows):
            row["checker_dy_mm"] = (index * 10.0) / 1000.0
        self.assertEqual(
            mcl.peak_frame_index(measurement, axis="Y", reference_um=0.0, target_um=0.0),
            2,
        )

    def test_peak_is_none_without_any_accepted_frame(self) -> None:
        measurement = self._measurement([1.0])
        measurement.frame_rows[0]["accepted"] = False
        self.assertIsNone(
            mcl.peak_frame_index(measurement, axis="X", reference_um=0.0, target_um=0.0)
        )


class ConfigValidationTests(unittest.TestCase):
    """直接调用 validate_config 的校验分支：坏配置必须在启动前被拒。"""

    def _validated_with(self, name: str, value: object) -> None:
        original = getattr(config, name)
        setattr(config, name, value)
        try:
            config.validate_config()
        finally:
            setattr(config, name, original)

    def test_the_shipped_config_is_valid(self) -> None:
        config.validate_config()

    def test_micro_closed_loop_is_a_registered_run_mode(self) -> None:
        self.assertIn("micro_closed_loop", config.VALID_RUN_MODES)

    def test_the_discarded_mode_is_gone(self) -> None:
        self.assertNotIn("micro_motion_experiment", config.VALID_RUN_MODES)

    def test_min_command_must_stay_above_one_micrometre(self) -> None:
        # 这条是硬约束：robot.validate_trajectory 拒绝 length <= 1e-6 m。
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_MIN_COMMAND_UM", 1.0)
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_MIN_COMMAND_UM", 0.5)

    def test_min_command_must_be_below_the_clamp(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_MIN_COMMAND_UM", 200.0)

    def test_amplitudes_must_ascend_and_be_unique(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_AMPLITUDES_UM", (50, 5, 20))
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_AMPLITUDES_UM", (5, 5, 50))
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_AMPLITUDES_UM", ())

    def test_axes_are_restricted_to_x_and_y(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_AXES", ("X", "Z"))

    def test_kp_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_KP", 0.0)

    def test_clamp_must_cover_the_largest_target(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_MAX_CORRECTION_UM", 10.0)

    def test_max_iter_must_cover_two_cycle_windows(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_MAX_ITER", 5)

    def test_cycle_window_floor(self) -> None:
        # 下界是 3 而不是更大：MICRO_LOOP_MAX_ITER 要求 >= 2×CYCLE_WINDOW，
        # 用户又把迭代上限定在 5~8，取 6 时 CYCLE_WINDOW 只能到 3。
        # 若把下界留在 4，两条约束直接互斥，配置无法通过校验。
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_CYCLE_WINDOW", 2)
        self.assertIsNone(self._validated_with("MICRO_LOOP_CYCLE_WINDOW", 3))

    def test_sign_change_threshold_must_fit_the_window(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_CYCLE_MIN_SIGN_CHANGES", 6)
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_CYCLE_MIN_SIGN_CHANGES", 0)

    def test_keep_ratios_must_be_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_CYCLE_PTP_KEEP", 0.0)
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_CYCLE_PTP_KEEP", 1.5)
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_CYCLE_ABS_KEEP", 0.0)

    def test_tolerance_mode_is_restricted(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_POSITION_TOL_MODE", "whatever")

    def test_disk_thresholds_must_be_ordered(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_MIN_FREE_GB_CONTINUE", 30.0)
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_TRASH_QUOTA_GB", 50.0)

    def test_probe_gain_band_must_be_ordered_and_effective(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_GAIN_MIN", 3.0)
        # 增益下限 × 探针位移必须能产生可测的位移。
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_PROBE_UM", 1.0)

    def test_hysteresis_ratio_must_be_a_probability(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_PROBE_HYSTERESIS_RATIO", 1.0)

    def test_worst_case_window_must_fit_the_buffer(self) -> None:
        # 窗口超时之和必须落在缓冲上限内，否则每段都要临时扩容或写满截断。
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_MAX_WINDOW_SECONDS", 1.0)

    def test_plausible_shift_must_exceed_the_iteration_jump_guard(self) -> None:
        with self.assertRaises(ValueError):
            self._validated_with("MICRO_LOOP_MAX_ITER_JUMP_UM", 5000.0)

    def test_validation_restores_the_shipped_config(self) -> None:
        # 上面的用例会临时改全局配置；确认收尾没有把模块配置改坏。
        self.assertEqual(config.MICRO_LOOP_MIN_COMMAND_UM, 2.0)
        self.assertEqual(tuple(config.MICRO_LOOP_AMPLITUDES_UM), (5, 20, 50))
        config.validate_config()


class OutputColumnTests(unittest.TestCase):
    def test_master_summary_has_the_required_columns(self) -> None:
        for column in (
            "axis",
            "target_amplitude_um",
            "target_direction",
            "iteration_count",
            "termination_state",
            "final_mean_error_um",
            "closed_loop_final_residual_um",
            "final_position_um",
            "final_error_peak_to_peak_um",
            "max_overshoot_um",
            "smallest_command_issued_um",
            "corresponding_actual_motion_um",
            "actual_frames",
            "expected_frames",
            "frame_ratio",
            "valid_chessboard_frames",
            "chessboard_detection_rate",
        ):
            self.assertIn(column, mcl.MASTER_COLUMNS)

    def test_iteration_columns_keep_the_robot_and_vision_side_apart(self) -> None:
        # achieved_robot_um 与 measured_um 并排，才有"机器人动了吗 / 视觉看到了吗"三分表。
        for column in ("command_um", "achieved_robot_um", "measured_position_um", "g_obs"):
            self.assertIn(column, mcl.ITERATION_COLUMNS)

    def test_frame_columns_carry_the_acceptance_evidence(self) -> None:
        for column in ("checker_is_valid", "accepted", "checker_found_by", "in_tail_window"):
            self.assertIn(column, mcl.FRAME_COLUMNS)


class ClosedLoopEntryGateTests(unittest.TestCase):
    """无硬件干跑：入口门禁必须逐个拦住，且**在创建任何目录之前**。

    这些用例不碰真机、不碰相机——每个门禁都排在设备连接之前，
    所以构造一个坏配置直接调用入口函数，就能验证它确实在启动前被拒。
    """

    def setUp(self) -> None:
        import argparse
        import shutil

        self._tmp = tempfile.mkdtemp(prefix="mcl_gate_")
        self._saved: dict[str, object] = {}
        self._patch("OUTPUT_ROOT", Path(self._tmp) / "outputs")
        self._shutil = shutil
        self.arguments = argparse.Namespace(ui_confirmed=True, stop_request_path=None)

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            setattr(config, name, value)
        self._shutil.rmtree(self._tmp, ignore_errors=True)

    def _patch(self, name: str, value: object) -> None:
        if name not in self._saved:
            self._saved[name] = getattr(config, name)
        setattr(config, name, value)

    def _assert_no_output_was_created(self) -> None:
        # 门禁失败时连输出根目录都不该被建出来，更不该有运行目录。
        root = Path(self._tmp) / "outputs"
        if root.exists():
            self.assertEqual(list(root.iterdir()), [])

    def test_locked_relative_motion_is_rejected(self) -> None:
        import main

        self._patch("ROBOT_RELATIVE_MOTION_ENABLED", False)
        with self.assertRaises(PermissionError) as ctx:
            main.run_micro_closed_loop_mode(self.arguments)
        self.assertIn("ROBOT_RELATIVE_MOTION_ENABLED", str(ctx.exception))
        self._assert_no_output_was_created()

    def test_sfc_control_mode_is_rejected(self) -> None:
        # 避免把开环运动误当成 SFC。
        import main

        self._patch("CONTROL_MODE", "sfc")
        with self.assertRaises(ValueError) as ctx:
            main.run_micro_closed_loop_mode(self.arguments)
        self.assertIn("SFC", str(ctx.exception))
        self._assert_no_output_was_created()

    def test_non_checkerboard_vision_is_rejected(self) -> None:
        import main

        self._patch("VISION_METHOD", "feature")
        with self.assertRaises(ValueError) as ctx:
            main.run_micro_closed_loop_mode(self.arguments)
        self.assertIn("VISION_METHOD", str(ctx.exception))
        self.assertIn("棋盘格", str(ctx.exception))
        self._assert_no_output_was_created()

    def test_insufficient_disk_is_rejected_before_any_directory_exists(self) -> None:
        """磁盘门禁必须跑在创建目录之前——空间不够时连目录都不该建出来。"""

        import main

        saved = mcl.free_gb
        mcl.free_gb = lambda path: 0.5  # noqa: ARG005 - 固定返回不足的剩余空间
        try:
            with self.assertRaises(RuntimeError) as ctx:
                main.run_micro_closed_loop_mode(self.arguments)
        finally:
            mcl.free_gb = saved
        self.assertIn("GiB", str(ctx.exception))
        # 这里是关键断言：如果实现改成先建目录再查磁盘，这一条会失败。
        self.assertFalse((Path(self._tmp) / "outputs").exists())

    def test_unconfirmed_run_asks_the_operator_before_anything(self) -> None:
        """没有 --ui-confirmed 时必须先要人确认，不能默默开跑。

        这里把确认函数换成一个哨兵异常，避免测试真的去读 stdin 卡住；
        断言的是"它被调用了"，而不是它内部怎么问。
        """
        import main
        import robot

        class _Asked(Exception):
            pass

        def _sentinel(description: str) -> None:
            raise _Asked(description)

        saved = robot.require_operator_confirmation
        robot.require_operator_confirmation = _sentinel
        try:
            self.arguments.ui_confirmed = False
            with self.assertRaises(_Asked) as ctx:
                main.run_micro_closed_loop_mode(self.arguments)
        finally:
            robot.require_operator_confirmation = saved
        self.assertIn("微动闭环", str(ctx.exception))
        self._assert_no_output_was_created()


class QuickScopeTests(unittest.TestCase):
    """用户第 15 条要求的验收清单：逐条对应一个用例。"""

    def _validated_with(self, name: str, value: object) -> None:
        """把某个配置项临时设成给定值跑一遍校验，结束后还原。"""

        original = getattr(config, name)
        setattr(config, name, value)
        try:
            config.validate_config()
        finally:
            setattr(config, name, original)

    def test_formal_levels_are_only_5_20_50(self) -> None:
        self.assertEqual(tuple(int(v) for v in config.MICRO_LOOP_AMPLITUDES_UM), (5, 20, 50))

    def test_level_order_is_ascending_and_fixed(self) -> None:
        levels = [int(v) for v in config.MICRO_LOOP_AMPLITUDES_UM]
        self.assertEqual(levels, sorted(levels))
        # 顺序是写死的常量，不随任何运行期状态变化。
        self.assertEqual(
            [int(v) for v in config.MICRO_LOOP_AMPLITUDES_UM], [5, 20, 50]
        )

    def test_two_axes_two_modes_three_levels_make_twelve_blocks(self) -> None:
        blocks = [
            mcl.open_block_id(axis, mode, level)
            for axis in config.MICRO_LOOP_AXES
            for mode in config.MICRO_LOOP_OPEN_MODES
            for level in config.MICRO_LOOP_AMPLITUDES_UM
        ]
        self.assertEqual(len(blocks), 12)
        self.assertEqual(len(set(blocks)), 12, "块名必须互不相同，否则会互相覆盖")

    def test_seq_mode_is_three_positive_then_three_negative(self) -> None:
        steps = mcl.open_loop_steps(axis="X", mode=mcl.OPEN_MODE_SEQ, level_um=5)
        self.assertEqual([s["direction"] for s in steps], [1, 1, 1, -1, -1, -1])
        self.assertEqual(
            [s["commanded_increment_um"] for s in steps], [5.0, 5.0, 5.0, -5.0, -5.0, -5.0]
        )

    def test_alt_mode_is_strict_alternation(self) -> None:
        steps = mcl.open_loop_steps(axis="Y", mode=mcl.OPEN_MODE_ALT, level_um=20)
        self.assertEqual([s["direction"] for s in steps], [1, -1, 1, -1, 1, -1])
        self.assertEqual(
            [s["commanded_increment_um"] for s in steps],
            [20.0, -20.0, 20.0, -20.0, 20.0, -20.0],
        )

    def test_both_modes_net_to_zero_so_blocks_need_no_homing(self) -> None:
        # 这是"整个开环阶段只建一次视觉零点"的前提；一旦不成立，
        # 块与块之间必须回位，链式相减会算出一个巨大的假增量。
        for axis in ("X", "Y"):
            for mode in (mcl.OPEN_MODE_SEQ, mcl.OPEN_MODE_ALT):
                steps = mcl.open_loop_steps(axis=axis, mode=mode, level_um=50)
                net = sum(float(s["commanded_increment_um"]) for s in steps)
                self.assertAlmostEqual(net, 0.0, places=9, msg=f"{axis}/{mode} 净位移不为零")

    def test_both_axes_run_both_modes_at_all_three_levels(self) -> None:
        for axis in ("X", "Y"):
            for mode in (mcl.OPEN_MODE_SEQ, mcl.OPEN_MODE_ALT):
                for level in (5, 20, 50):
                    steps = mcl.open_loop_steps(axis=axis, mode=mode, level_um=level)
                    self.assertEqual(len(steps), 6)
                    self.assertTrue(all(s["axis"] == axis for s in steps))
                    self.assertTrue(all(s["mode"] == mode for s in steps))
                    self.assertTrue(all(s["target_level_um"] == float(level) for s in steps))

    def test_closed_loop_iteration_cap_is_between_five_and_eight(self) -> None:
        self.assertGreaterEqual(int(config.MICRO_LOOP_MAX_ITER), 5)
        self.assertLessEqual(int(config.MICRO_LOOP_MAX_ITER), 8)

    def test_a_target_that_never_moves_is_never_confirmed_as_converged(self) -> None:
        """第 8 条的核心：5 μm 目标、实测≈0、容差比幅值还大时不得判 CONVERGED。"""

        confirmed, why = mcl.directional_motion_confirmed(
            direction=1,
            amplitude_um=5.0,
            final_measured_um=0.05,  # 机器人根本没动
            sigma_um=3.0,  # 有效容差 = 2σ = 6 μm > 5 μm
        )
        self.assertFalse(confirmed)
        self.assertIn("净位移", why)

    def test_a_genuine_arrival_still_counts_as_confirmed(self) -> None:
        # 反例必须成对存在，否则上面那条可以靠"永远返回 False"骗过去。
        confirmed, _ = mcl.directional_motion_confirmed(
            direction=1, amplitude_um=5.0, final_measured_um=4.7, sigma_um=0.6
        )
        self.assertTrue(confirmed)

    def test_negative_targets_are_judged_in_their_own_direction(self) -> None:
        """第 9 条：sign=−1 的方向不能被误判成"朝反方向运动"。

        目标在 −20 μm 时机器人合法地向 −X 走，未按 direction 归一化的实现
        会把 −19 μm 的合法位移判成"方向错误"。
        """

        confirmed, _ = mcl.directional_motion_confirmed(
            direction=-1, amplitude_um=20.0, final_measured_um=-19.0, sigma_um=0.8
        )
        self.assertTrue(confirmed)
        # 真的朝反方向走了才该失败。
        wrong_way, _ = mcl.directional_motion_confirmed(
            direction=-1, amplitude_um=20.0, final_measured_um=19.0, sigma_um=0.8
        )
        self.assertFalse(wrong_way)

    def test_oscillating_back_and_forth_is_not_directional_motion(self) -> None:
        # 净位移用绝对值之和会漏掉这一条：走了 10 μm 又回来，净位移是 0。
        confirmed, _ = mcl.directional_motion_confirmed(
            direction=1, amplitude_um=20.0, final_measured_um=0.0, sigma_um=0.8
        )
        self.assertFalse(confirmed)

    def test_missing_or_non_finite_final_position_is_never_confirmed(self) -> None:
        for bad in (None, float("nan"), float("inf")):
            confirmed, _ = mcl.directional_motion_confirmed(
                direction=1, amplitude_um=50.0, final_measured_um=bad, sigma_um=0.5
            )
            self.assertFalse(confirmed)

    def test_noise_limited_state_is_its_own_terminal_state(self) -> None:
        # 不能复用 CONVERGED，也不能复用 CONVERGED_NOISE_LIMITED：
        # 前者是假成功，后者的含义是"确实到位了但残差与零不可分"。
        self.assertEqual(mcl.TERMINAL_MEASUREMENT_NOISE_LIMITED, "MEASUREMENT_NOISE_LIMITED")
        self.assertNotIn(
            mcl.TERMINAL_MEASUREMENT_NOISE_LIMITED,
            (mcl.TERMINAL_CONVERGED, mcl.TERMINAL_CONVERGED_NOISE_LIMITED),
        )

    def test_directional_fraction_must_be_a_probability(self) -> None:
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                self._validated_with("MICRO_LOOP_MIN_DIRECTIONAL_FRACTION", bad)
        self.assertIsNone(self._validated_with("MICRO_LOOP_MIN_DIRECTIONAL_FRACTION", 0.5))


class VisionAxisMappingTests(unittest.TestCase):
    """第 9 条：Y 轴必须读探针给出的视觉轴，不能再出现"Y 实验固定读视觉 x"。"""

    @staticmethod
    def _measurement(
        vision_axis: str,
        measured_um: float | None,
        *,
        transverse_um: float | None = None,
        n_tail: int = 20,
        sigma: float = 0.4,
        truncated: bool = False,
    ) -> "mcl.ClipMeasurement":
        return mcl.ClipMeasurement(
            raw_path=Path("X:/nonexistent.raw"),
            frame_count=0,
            n_valid=0,
            measured_um=measured_um,
            transverse_um=transverse_um,
            n_tail=n_tail,
            sigma_tail_um=sigma,
            peak_axis_um=0.0,
            positions=[],
            times_ns=[],
            frame_rows=[],
            first_row_ns=0,
            last_row_ns=0,
            truncated=truncated,
            vision_axis=vision_axis,
        )

    @staticmethod
    def _gains(vision_x: str, vision_y: str, sign_x: float, sign_y: float) -> object:
        return mcl.ProbeGains(
            gain={"X": {vision_x: 1.0}, "Y": {vision_y: 1.0}},
            sign={"X": sign_x, "Y": sign_y},
            vision_axis={"X": vision_x, "Y": vision_y},
            hysteresis={"X": 0.0, "Y": 0.0},
        )

    def test_each_robot_axis_takes_the_vision_axis_the_probe_assigned(self) -> None:
        gains = self._gains("y", "x", 1.0, 1.0)
        # 视觉 x 与视觉 y 互换：这正是"Y 实验固定读视觉 x"那个缺陷的形状。
        on_vision_x = self._measurement("x", 111.0)
        on_vision_y = self._measurement("y", 222.0)
        self.assertEqual(
            mcl.GroupVisionMeter.axis_measured_um(on_vision_y, "X", gains), 222.0
        )
        self.assertEqual(
            mcl.GroupVisionMeter.axis_measured_um(on_vision_x, "Y", gains), 111.0
        )

    def test_a_measurement_from_the_wrong_vision_axis_is_rejected(self) -> None:
        # 测量本身记着自己是在哪条视觉轴上做的；与探针不一致时必须抛错，
        # 而不是悄悄按探针的轴去读一个不相干的数。
        gains = self._gains("x", "y", 1.0, 1.0)
        with self.assertRaises(ValueError):
            mcl.GroupVisionMeter.axis_measured_um(self._measurement("x", 5.0), "Y", gains)

    def test_a_missing_axis_mapping_is_a_loud_error(self) -> None:
        gains = self._gains("x", "x", 1.0, 1.0)
        del gains.vision_axis["Y"]
        with self.assertRaises(KeyError):
            mcl.GroupVisionMeter.axis_measured_um(self._measurement("x", 5.0), "Y", gains)

    def test_sign_minus_one_flips_the_reading(self) -> None:
        # 机器人 X 正向 = 视觉 x 负向时，视觉读到 +10 μm 表示机器人朝 −X 走了 10 μm。
        gains = self._gains("x", "y", -1.0, +1.0)
        self.assertEqual(
            mcl.GroupVisionMeter.axis_measured_um(self._measurement("x", 10.0), "X", gains),
            -10.0,
        )
        self.assertEqual(
            mcl.GroupVisionMeter.axis_measured_um(self._measurement("y", 10.0), "Y", gains),
            10.0,
        )

    def test_unusable_measurement_maps_to_none_not_zero(self) -> None:
        # 返回 0 会被上层当成"测到了位置 0"，从而算出一个假的位移。
        gains = self._gains("x", "y", 1.0, 1.0)
        for bad in (
            self._measurement("x", None),
            self._measurement("x", 10.0, truncated=True),
            self._measurement("x", 10.0, n_tail=0),
            self._measurement("x", float("nan")),
        ):
            self.assertIsNone(mcl.GroupVisionMeter.axis_measured_um(bad, "X", gains))

    # ---- 一次测量同时给两条轴（开环零点用） -----------------------------

    def test_one_measurement_yields_both_axes_without_any_second_pass(self) -> None:
        # 开环零点要同时锚定 X 与 Y。若为此把同一段片段扫两遍，识别开销翻倍；
        # 一次识别本来就同时算出了两条视觉轴的尾窗中位数。
        gains = self._gains("x", "y", 1.0, 1.0)
        on_vision_x = self._measurement("x", 111.0, transverse_um=222.0)
        self.assertEqual(
            mcl.GroupVisionMeter.axis_positions_um(on_vision_x, gains),
            {"X": 111.0, "Y": 222.0},
        )

    def test_both_axes_follow_the_swapped_probe_mapping(self) -> None:
        # 探针把 X→视觉 y、Y→视觉 x（这正是历史缺陷的形状）。主轴仍是
        # measured_um，另一条走 transverse_um，两条都不许张冠李戴。
        gains = self._gains("y", "x", 1.0, 1.0)
        measurement = self._measurement("y", 111.0, transverse_um=222.0)
        self.assertEqual(
            mcl.GroupVisionMeter.axis_positions_um(measurement, gains),
            {"X": 111.0, "Y": 222.0},
        )

    def test_each_axis_gets_its_own_sign(self) -> None:
        gains = self._gains("x", "y", -1.0, +1.0)
        measurement = self._measurement("x", 10.0, transverse_um=10.0)
        self.assertEqual(
            mcl.GroupVisionMeter.axis_positions_um(measurement, gains),
            {"X": -10.0, "Y": 10.0},
        )

    def test_two_axes_asking_for_one_vision_axis_is_a_loud_error(self) -> None:
        # 探针门禁本该拦住这种退化情形，但这里必须再拦一次——静默返回一个
        # 张冠李戴的位置比抛错危险得多。
        gains = self._gains("x", "x", 1.0, 1.0)
        with self.assertRaises(ValueError):
            mcl.GroupVisionMeter.axis_positions_um(
                self._measurement("x", 1.0, transverse_um=2.0), gains
            )

    def test_a_missing_mapping_is_a_loud_error_here_too(self) -> None:
        gains = self._gains("x", "y", 1.0, 1.0)
        del gains.vision_axis["Y"]
        with self.assertRaises(KeyError):
            mcl.GroupVisionMeter.axis_positions_um(
                self._measurement("x", 1.0, transverse_um=2.0), gains
            )

    def test_unusable_measurement_gives_none_for_every_axis(self) -> None:
        # 给 0.0 会被上层当成"零点在 0 μm"，于是第一次微动的位移里混进一个
        # 常数偏置——而且它看起来完全正常。
        gains = self._gains("x", "y", 1.0, 1.0)
        for bad in (
            self._measurement("x", None, transverse_um=222.0),
            self._measurement("x", 111.0, transverse_um=222.0, truncated=True),
            self._measurement("x", 111.0, transverse_um=222.0, n_tail=0),
            self._measurement("x", float("nan"), transverse_um=222.0),
        ):
            self.assertEqual(
                mcl.GroupVisionMeter.axis_positions_um(bad, gains),
                {"X": None, "Y": None},
            )

    def test_a_missing_transverse_axis_nulls_only_that_axis(self) -> None:
        # transverse_um 为 None（例如该视觉轴这一批帧没算出尾窗）时，主轴仍
        # 可用；不能因为一条轴缺数就把另一条也丢掉，更不能补 0。
        gains = self._gains("x", "y", 1.0, 1.0)
        positions = mcl.GroupVisionMeter.axis_positions_um(
            self._measurement("x", 111.0), gains
        )
        self.assertEqual(positions, {"X": 111.0, "Y": None})

    def test_axis_subset_is_honoured(self) -> None:
        gains = self._gains("x", "y", 1.0, 1.0)
        measurement = self._measurement("x", 111.0, transverse_um=222.0)
        self.assertEqual(
            mcl.GroupVisionMeter.axis_positions_um(measurement, gains, robot_axes=("Y",)),
            {"Y": 222.0},
        )


class RunPlanBudgetTests(unittest.TestCase):
    """第 14 条：时间与数据量估算要如实反映**真实的**实验规模。"""

    def test_run_plan_never_quotes_hours_for_a_normal_run(self) -> None:
        plan = mcl.estimate_run_plan()
        self.assertLess(float(plan["seconds_normal"]) / 60.0, 12.0)

    def test_budget_lines_are_minutes_not_hours(self) -> None:
        lines = mcl.format_budget_lines(mcl.estimate_run_plan())
        text = "\n".join(lines)
        self.assertNotIn("小时", text)
        self.assertIn("分钟", text)

    def test_normal_plan_is_cheaper_than_the_worst_case(self) -> None:
        plan = mcl.estimate_run_plan()
        self.assertLess(float(plan["seconds_normal"]), float(plan["seconds_worst"]))
        self.assertLessEqual(int(plan["windows_normal"]), int(plan["windows_worst"]))

    def test_worst_case_data_volume_is_reported_and_fits_the_free_disk(self) -> None:
        plan = mcl.estimate_run_plan()
        need_gb = plan["bytes_worst"] / (1024.0 ** 3)
        self.assertGreater(need_gb, 0.0)
        free_gb = mcl.free_gb(config.MICRO_LOOP_RAW_ROOT)
        if math.isfinite(free_gb):
            self.assertGreaterEqual(free_gb, need_gb)

    def test_static_baseline_raw_is_retained_by_policy(self) -> None:
        # 第 10 条：静止基线的完整 RAW 永久保留，且全局保留策略不允许静默全删。
        self.assertTrue(bool(config.MICRO_LOOP_KEEP_ALL_RAW))
        self.assertGreater(float(config.MICRO_LOOP_KEEP_ALL_RESERVE_GB), 0.0)
        # 保留目标卷与工程输出目录分开，避免几十 GB 二进制淹没 outputs/。
        self.assertNotEqual(
            Path(config.MICRO_LOOP_RAW_ROOT).resolve(),
            Path(config.OUTPUT_ROOT).resolve(),
        )

    def test_raw_blocks_are_named_per_the_agreed_scheme(self) -> None:
        self.assertEqual(mcl.open_block_id("X", "seq", 5), "X_open_seq_005um")
        self.assertEqual(mcl.open_block_id("Y", "alt", 50), "Y_open_alt_050um")
        self.assertEqual(mcl.closed_block_id("X", 20), "X_closed_020um")
        self.assertEqual(mcl.closed_block_id("Y", 5), "Y_closed_005um")

    def test_spanning_every_n_covers_the_whole_clip_not_just_the_tail(self) -> None:
        # 静止基线要的是"整段 5 s 抖了多少"；只看尾部会把前面几秒扔掉，
        # 噪声底被系统性低估。
        every_n = mcl.spanning_every_n(frame_count=660, target=64)
        self.assertGreaterEqual(every_n * 63, 660 * 0.9)


class SignChainTests(unittest.TestCase):
    """
    符号**整条链路**只能转换一次。

    链路：原始视觉坐标 → 探针的 vision_axis/sign → axis_measured_um →
          target → error → command → 机器人运动 → 方向判据。

    上一版的缺陷正是在这条链路的最后一段又乘了一次 sign：error 已是机器人
    轴坐标，控制律再乘一次，于是 sign=−1 的相机安装（机器人 +Y 在图像上是
    −y）下命令被翻反。这里逐段把链路接起来测，而不是只测单个函数。

    另一个刻意的设计：所有断言都写成"指令的方向 = (target − measured) 的方向"，
    而不是写成"指令 = target 本身"。前者是这条链路真正的不变量，后者在
    measured ≠ 0 时会给出错误结论。
    """

    @staticmethod
    def _gains(vision_axis: str, sign: float) -> object:
        return mcl.ProbeGains(
            gain={"X": {vision_axis: sign}, "Y": {vision_axis: sign}},
            sign={"X": sign, "Y": sign},
            vision_axis={"X": vision_axis, "Y": vision_axis},
            hysteresis={"X": 0.0, "Y": 0.0},
        )

    @staticmethod
    def _measurement(vision_axis: str, measured_um: float | None) -> "mcl.ClipMeasurement":
        return mcl.ClipMeasurement(
            raw_path=Path("X:/nonexistent.raw"),
            frame_count=0,
            n_valid=0,
            measured_um=measured_um,
            transverse_um=None,
            n_tail=20,
            sigma_tail_um=0.4,
            peak_axis_um=0.0,
            positions=[],
            times_ns=[],
            frame_rows=[],
            first_row_ns=0,
            last_row_ns=0,
            truncated=False,
            vision_axis=vision_axis,
        )

    def _chain(
        self, *, vision_axis: str, sign: float, vision_position_um: float,
        target_um: float,
    ) -> float:
        """跑完 视觉 → measured → error → command，返回机器人轴系下的指令。"""

        gains = self._gains(vision_axis, sign)
        measured = mcl.GroupVisionMeter.axis_measured_um(
            self._measurement(vision_axis, vision_position_um), "Y", gains
        )
        assert measured is not None
        return mcl.command_for_error(target_um - measured)

    # ---- sign = +1（相机顺装）------------------------------------------

    def test_sign_plus_one_positive_error_gives_a_positive_command(self) -> None:
        command = self._chain(
            vision_axis="y", sign=1.0, vision_position_um=0.0, target_um=20.0
        )
        self.assertAlmostEqual(command, 20.0)

    def test_sign_plus_one_negative_target_gives_a_negative_command(self) -> None:
        command = self._chain(
            vision_axis="y", sign=1.0, vision_position_um=0.0, target_um=-20.0
        )
        self.assertAlmostEqual(command, -20.0)

    # ---- sign = −1（机器人 +Y 在图像上是 −y）----------------------------

    def test_sign_minus_one_still_gives_a_positive_command_for_a_positive_target(
        self,
    ) -> None:
        # 机器人当前在原点，目标 +50：指令必须是 +50（机器人正向）。
        # 若控制律再乘一次 sign，这里会得到 −50 —— 闭环第一步就走反。
        command = self._chain(
            vision_axis="y", sign=-1.0, vision_position_um=0.0, target_um=50.0
        )
        self.assertAlmostEqual(command, 50.0)

    def test_sign_minus_one_still_gives_a_negative_command_for_a_negative_target(
        self,
    ) -> None:
        command = self._chain(
            vision_axis="y", sign=-1.0, vision_position_um=0.0, target_um=-50.0
        )
        self.assertAlmostEqual(command, -50.0)

    def test_sign_minus_one_measures_a_robot_positive_move_as_positive(self) -> None:
        # 机器人已经朝 +Y 走了 30 μm；视觉上看到的是 −30（因为装反了）。
        # 归一后 measured 必须是 +30，于是 error = 50 − 30 = +20，指令 +20。
        command = self._chain(
            vision_axis="y", sign=-1.0, vision_position_um=-30.0, target_um=50.0
        )
        self.assertAlmostEqual(command, 20.0)

    def test_sign_minus_one_command_points_toward_the_target_not_away(self) -> None:
        # 与上一条的差值恒为 (target − measured)，方向必须由这个差决定。
        for target, position in ((50.0, 0.0), (50.0, 30.0), (50.0, 80.0), (-50.0, -20.0)):
            with self.subTest(target=target, position=position):
                command = self._chain(
                    vision_axis="y", sign=-1.0,
                    vision_position_um=-position, target_um=target,
                )
                expected = target - position
                self.assertAlmostEqual(command, expected)
                self.assertEqual(command > 0.0, expected > 0.0)

    def test_the_vision_axis_the_probe_picked_is_the_one_that_is_used(self) -> None:
        # 视觉 x 与 y 上的数完全不同：选错轴会读出一个不相干的位移。
        gains = self._gains("y", -1.0)
        measurement = mcl.ClipMeasurement(
            raw_path=Path("X:/nonexistent.raw"), frame_count=0, n_valid=0,
            measured_um=-30.0, transverse_um=999.0, n_tail=20, sigma_tail_um=0.4,
            peak_axis_um=0.0, positions=[], times_ns=[], frame_rows=[],
            first_row_ns=0, last_row_ns=0, truncated=False, vision_axis="y",
        )
        self.assertAlmostEqual(
            mcl.GroupVisionMeter.axis_measured_um(measurement, "Y", gains), 30.0
        )


class SignFlipInterlockTests(unittest.TestCase):
    """第 6 条验收：「不允许因为 sign=−1 正常映射而触发 direction error」。"""

    def test_an_arm_that_obeys_the_command_is_not_a_direction_error(self) -> None:
        # 指令 +20，机器人走了 +20，视觉增量也是 +20 —— 每一段都已归一，
        # 判据看到的全是同号，不得报方向异常。
        for command, achieved, delta, g_obs in (
            (20.0, 20.0, 20.0, 1.0),
            (-20.0, -20.0, -20.0, 1.0),
            (5.0, 4.8, 4.8, 0.96),
            (-5.0, -4.8, -4.8, 0.96),
        ):
            with self.subTest(command=command):
                self.assertFalse(
                    mcl.sign_flip_detected(
                        command_um=command, achieved_robot_um=achieved,
                        measured_delta_um=delta, g_obs=g_obs,
                    )
                )

    def test_a_genuinely_inverted_axis_is_still_caught(self) -> None:
        # 反向的情形必须照旧抓到，否则这条联锁就白留了。
        self.assertTrue(
            mcl.sign_flip_detected(
                command_um=20.0, achieved_robot_um=-20.0,
                measured_delta_um=-20.0, g_obs=-1.0,
            )
        )
        self.assertTrue(
            mcl.sign_flip_detected(
                command_um=20.0, achieved_robot_um=-20.0,
                measured_delta_um=-20.0, g_obs=float("nan"),
            )
        )

    def test_a_tiny_opposite_blip_is_not_evidence(self) -> None:
        # 地板值以下的反向读数是编码器抖动，不是符号错误。
        self.assertFalse(
            mcl.sign_flip_detected(
                command_um=20.0, achieved_robot_um=-0.5,
                measured_delta_um=-0.5, g_obs=float("nan"),
            )
        )

    def test_a_small_command_negative_g_obs_is_not_evidence(self) -> None:
        # 小指令的 g_obs 基本是噪声，只有指令够大时负增益才算数。
        self.assertFalse(
            mcl.sign_flip_detected(
                command_um=2.0, achieved_robot_um=2.0,
                measured_delta_um=2.0, g_obs=-0.3,
            )
        )

    def test_no_measurement_this_round_is_not_evidence(self) -> None:
        self.assertFalse(
            mcl.sign_flip_detected(
                command_um=20.0, achieved_robot_um=-20.0,
                measured_delta_um=None, g_obs=float("nan"),
            )
        )


class ClosedTargetCountTests(unittest.TestCase):
    """第 2 条：闭环目标数是 12（2 轴 × 3 档 × 2 方向），不是 6。"""

    def test_the_formal_closed_loop_covers_twelve_targets(self) -> None:
        targets = mcl.build_targets()
        self.assertEqual(len(targets), 12)
        self.assertEqual(
            [(t["axis"], t["amplitude_um"], t["direction"]) for t in targets],
            [
                ("X", 5, 1), ("X", 5, -1), ("X", 20, 1), ("X", 20, -1),
                ("X", 50, 1), ("X", 50, -1),
                ("Y", 5, 1), ("Y", 5, -1), ("Y", 20, 1), ("Y", 20, -1),
                ("Y", 50, 1), ("Y", 50, -1),
            ],
        )

    def test_the_run_plan_uses_the_real_target_count(self) -> None:
        # 上一版预算里写的是 len(axes)*len(levels) = 6，漏了方向这一维，
        # 于是时长、RAW 数据量、磁盘检查全都只按一半的量在算。
        plan = mcl.estimate_run_plan()
        self.assertEqual(int(plan["closed_targets"]), len(mcl.build_targets()))
        self.assertEqual(int(plan["closed_targets"]), 12)
        self.assertEqual(
            int(plan["closed_steps_max"]),
            12 * int(config.MICRO_LOOP_MAX_ITER),
        )

    def test_the_planned_windows_cover_both_directions_of_every_group(self) -> None:
        plan = mcl.estimate_run_plan()
        # 闭环正常路径每目标 4 轮 + 1 段收敛验证 = 5 段，×12 = 60 段；
        # 若还是按 6 个目标算，这里只有 30 段。
        normal_closed = int(plan["parts_normal"]["closed_loop"]["windows"]) + int(
            plan["parts_normal"]["closed_verify"]["windows"]
        )
        self.assertEqual(normal_closed, 12 * 5)
        worst_closed = int(plan["parts_worst"]["closed_loop"]["windows"]) + int(
            plan["parts_worst"]["closed_verify"]["windows"]
        )
        self.assertEqual(worst_closed, 12 * (int(config.MICRO_LOOP_MAX_ITER) + 1))

    def test_group_references_are_still_six_not_twelve(self) -> None:
        # 组参考是按 (轴, 档) 建的，组内 +A/−A **共用**同一个零点——
        # 这样才能测到反向时的死区/回差。目标数是 12，参考段数仍是 6。
        plan = mcl.estimate_run_plan()
        self.assertEqual(int(plan["parts_normal"]["group_reference"]["windows"]), 6)


class NoTimeTruncationTests(unittest.TestCase):
    """第 3 条：时长**不是**中止条件。"""

    def test_the_plan_no_longer_reports_a_hard_cutoff(self) -> None:
        plan = mcl.estimate_run_plan()
        for removed in (
            "seconds_hard_ceiling",
            "seconds_expected_worst",
            "worst_over_budget",
            "time_budget_s",
            "time_budget_soft_s",
        ):
            with self.subTest(key=removed):
                self.assertNotIn(removed, plan)

    def test_the_plan_reports_a_notice_line_instead(self) -> None:
        plan = mcl.estimate_run_plan()
        self.assertEqual(
            float(plan["time_notice_s"]), float(config.MICRO_LOOP_TIME_NOTICE_S)
        )

    def test_the_budget_lines_say_the_notice_does_not_stop_anything(self) -> None:
        text = "\n".join(mcl.format_budget_lines(mcl.estimate_run_plan()))
        self.assertIn("不中止实验", text)
        self.assertIn("不跳过任何目标", text)
        # 上一版会打印"硬上限…超时即安全收尾"，那句必须消失。
        self.assertNotIn("安全收尾", text)

    def test_a_normal_run_is_allowed_to_take_eleven_to_twelve_minutes(self) -> None:
        # 操作者明确接受 10～12 分钟。这条用例是防止有人为了"压进 10 分钟"
        # 把必要的分析帧数或重复次数砍掉。
        plan = mcl.estimate_run_plan()
        minutes = float(plan["seconds_normal"]) / 60.0
        self.assertGreater(minutes, 10.0)
        self.assertLess(minutes, 12.5)

    def test_all_twelve_targets_fit_in_the_normal_plan_without_truncation(self) -> None:
        # 正常路径必须规划到全部 12 个闭环目标：一旦有人重新引入
        # "时间到就跳过剩余目标"，这个数会立刻小于 12。
        plan = mcl.estimate_run_plan()
        self.assertEqual(int(plan["closed_targets"]), 12)
        self.assertGreaterEqual(
            float(plan["seconds_normal"]),
            12 * float(plan["window_seconds_normal"]),
        )


class OpenBlockSummaryTests(unittest.TestCase):
    """开环块统计：这几个数才是"最小可靠微动"的唯一来源。"""

    @staticmethod
    def _row(commanded: float, measured: object, status: str) -> dict[str, object]:
        return {
            "block_id": "X_open_seq_005um",
            "axis": "X",
            "mode": "seq",
            "target_level_um": 5.0,
            "step_index": 1,
            "direction": 1 if commanded >= 0 else -1,
            "commanded_increment_um": commanded,
            "vision_axis": "x",
            "vision_before_um": 0.0,
            "vision_after_um": measured,
            "measured_increment_um": measured,
            "robot_tcp_before": "",
            "robot_tcp_after": "",
            "robot_achieved_um": float("nan"),
            "command_timestamp_ns": 0,
            "measurement_timestamp_ns": 0,
            "settling_time_s": float("nan"),
            "settle_ok": False,
            "status": status,
            "note": "",
        }

    def test_a_block_where_nothing_moved_is_no_response(self) -> None:
        rows = [
            self._row(5.0, 0.0, mcl.OPEN_STEP_ZERO),
            self._row(5.0, 0.0, mcl.OPEN_STEP_ZERO),
        ]
        stats = mcl.summarize_open_block(rows)
        self.assertEqual(stats["verdict"], mcl.OPEN_VERDICT_NO_RESPONSE)
        self.assertEqual(stats["zero_response_steps"], 2)

    def test_a_fully_responsive_block_is_stable(self) -> None:
        rows = [
            self._row(5.0, 4.9, mcl.OPEN_STEP_VALID),
            self._row(-5.0, -5.1, mcl.OPEN_STEP_VALID),
        ]
        stats = mcl.summarize_open_block(rows)
        self.assertEqual(stats["verdict"], mcl.OPEN_VERDICT_STABLE)
        self.assertAlmostEqual(float(stats["response_ratio"]), 1.0, places=1)

    def test_min_reliable_level_picks_the_smallest_stable_level(self) -> None:
        stats = [
            {"axis": "X", "mode": "seq", "level_um": 5.0, "verdict": mcl.OPEN_VERDICT_NO_RESPONSE},
            {"axis": "X", "mode": "seq", "level_um": 20.0, "verdict": mcl.OPEN_VERDICT_STABLE},
            {"axis": "X", "mode": "seq", "level_um": 50.0, "verdict": mcl.OPEN_VERDICT_STABLE},
        ]
        self.assertEqual(mcl.min_reliable_level(stats, mode="seq", axis="X"), 20.0)

    def test_min_reliable_level_is_none_when_nothing_is_stable(self) -> None:
        stats = [
            {"axis": "Y", "mode": "alt", "level_um": 5.0, "verdict": mcl.OPEN_VERDICT_PARTIAL},
        ]
        self.assertIsNone(mcl.min_reliable_level(stats, mode="alt", axis="Y"))

    def test_min_reliable_level_does_not_mix_axes_or_modes(self) -> None:
        stats = [
            {"axis": "X", "mode": "seq", "level_um": 50.0, "verdict": mcl.OPEN_VERDICT_STABLE},
            {"axis": "Y", "mode": "alt", "level_um": 5.0, "verdict": mcl.OPEN_VERDICT_STABLE},
        ]
        self.assertIsNone(mcl.min_reliable_level(stats, mode="seq", axis="Y"))
        self.assertEqual(mcl.min_reliable_level(stats, mode="seq", axis="X"), 50.0)

    def test_open_step_columns_cover_every_field_the_loop_writes(self) -> None:
        required = {
            "axis", "mode", "target_level_um", "step_index", "commanded_increment_um",
            "direction", "vision_before_um", "vision_after_um", "measured_increment_um",
            "robot_tcp_before", "robot_tcp_after", "command_timestamp_ns",
            "measurement_timestamp_ns", "settling_time_s", "status",
        }
        self.assertTrue(required.issubset(set(mcl.OPEN_STEP_COLUMNS)))



if __name__ == "__main__":
    unittest.main()
