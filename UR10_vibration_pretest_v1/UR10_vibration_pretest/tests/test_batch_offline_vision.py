import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from camera import write_batch_segment_vision


class _FakeVisionProcessor:
    def process_frame(self, packet):
        return (
            {"kind": "VISION", "frame_id": packet.frame_id, "host_ns": packet.host_ns},
            packet.frame,
        )


class BatchOfflineVisionTests(unittest.TestCase):
    def test_offline_video_restores_hardware_frame_ids_and_host_times(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "sample_HIK.avi"
            timestamps = root / "sample_FRAME_TIMESTAMPS.csv"
            vision = root / "sample_VISION.txt"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24), False
            )
            self.assertTrue(writer.isOpened())
            for value in (20, 40, 60):
                writer.write(np.full((24, 32), value, dtype=np.uint8))
            writer.release()
            with timestamps.open("w", encoding="utf-8-sig", newline="") as file:
                csv_writer = csv.DictWriter(
                    file,
                    fieldnames=(
                        "segment_frame_index",
                        "frame_id",
                        "host_ns",
                        "camera_timestamp_raw",
                    ),
                )
                csv_writer.writeheader()
                for index, frame_id in enumerate((501, 502, 503)):
                    csv_writer.writerow(
                        {
                            "segment_frame_index": index,
                            "frame_id": frame_id,
                            "host_ns": 9_000_000_000 + index,
                            "camera_timestamp_raw": 7000 + index,
                        }
                    )
            with patch("camera.VisionProcessor", _FakeVisionProcessor):
                count = write_batch_segment_vision(
                    video, timestamps, vision, "sample"
                )
            self.assertEqual(count, 3)
            lines = [json.loads(line) for line in vision.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["frame_id"] for row in lines[1:]], [501, 502, 503])
            self.assertEqual(
                [row["host_ns"] for row in lines[1:]],
                [9_000_000_000, 9_000_000_001, 9_000_000_002],
            )


if __name__ == "__main__":
    unittest.main()
