from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import config
from camera import _resize_for_preview, _update_batch_frame_gap_stats


class BatchCameraStartupTests(unittest.TestCase):
    def test_recording_preview_can_be_reduced_without_changing_source(self) -> None:
        frame = np.zeros((100, 200), dtype=np.uint8)
        preview = _resize_for_preview(frame, max_width=50)
        self.assertEqual(preview.shape, (25, 50))
        self.assertEqual(frame.shape, (100, 200))

    def test_new_baseline_ignores_gap_before_first_formal_frame(self) -> None:
        previous, received, missing = _update_batch_frame_gap_stats(
            1021, None, 0, 0
        )
        self.assertEqual(previous, 1021)
        self.assertEqual(received, 1)
        self.assertEqual(missing, 0)

    def test_gap_aborts_only_when_it_exceeds_thirty_frames(self) -> None:
        with patch.object(config, "BATCH_MAX_CONSECUTIVE_MISSING_FRAMES", 30):
            previous, received, missing = _update_batch_frame_gap_stats(
                131, 100, 1, 0
            )
            self.assertEqual((previous, received, missing), (131, 2, 30))

        with patch.object(config, "BATCH_MAX_CONSECUTIVE_MISSING_FRAMES", 30):
            with self.assertRaisesRegex(
                RuntimeError,
                r"连续缺失 31 帧.*上一帧 100.*当前帧 132.*允许上限 30 帧",
            ):
                _update_batch_frame_gap_stats(132, 100, 1, 0)

    def test_cumulative_loss_guard_aborts_only_above_fifteen_percent(self) -> None:
        with patch.object(config, "BATCH_MAX_MISSING_RATIO", 0.15):
            # 17 missing + 100 received = 14.53%, so the batch continues.
            previous, received, missing = _update_batch_frame_gap_stats(
                101, 100, 99, 17
            )
            self.assertEqual((previous, received, missing), (101, 100, 17))

        with patch.object(config, "BATCH_MAX_MISSING_RATIO", 0.15):
            with self.assertRaisesRegex(RuntimeError, r"累计缺帧率"):
                # 18 missing + 100 received = 15.25%.
                _update_batch_frame_gap_stats(101, 100, 99, 18)


if __name__ == "__main__":
    unittest.main()
