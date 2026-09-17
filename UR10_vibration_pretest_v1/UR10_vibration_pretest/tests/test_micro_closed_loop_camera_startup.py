"""闭环相机 READY 消息契约的无硬件回归测试。"""

from __future__ import annotations

import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import main
import micro_closed_loop as mcl


class _FakePacket:
    def __init__(self) -> None:
        self.frame = np.zeros((3, 4), dtype=np.uint8)
        self.host_ns = 1
        self.frame_id = 1
        self.camera_timestamp_raw = 1


class _FakeHikCameraSource:
    actual_camera_fps = 132.3

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def __iter__(self):
        return iter([_FakePacket()])


class _StreamingPacket(_FakePacket):
    def __init__(self, frame_id: int) -> None:
        super().__init__()
        self.host_ns = frame_id * 1_000_000
        self.frame_id = frame_id
        self.camera_timestamp_raw = frame_id


class _StreamingHikCameraSource(_FakeHikCameraSource):
    def __iter__(self):
        frame_id = 0
        while True:
            frame_id += 1
            time.sleep(0.001)
            yield _StreamingPacket(frame_id)


class MicroClosedLoopCameraStartupTests(unittest.TestCase):
    @staticmethod
    def _fake_modules(source_type):
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.destroyAllWindows = lambda: None
        fake_camera = types.ModuleType("camera")
        fake_camera.HikCameraSource = source_type
        fake_camera._resize_for_preview = lambda frame, **kwargs: frame
        return fake_cv2, fake_camera

    def test_ready_uses_actual_camera_fps_field_consumed_by_main(self) -> None:
        fake_cv2, fake_camera = self._fake_modules(_FakeHikCameraSource)

        error_queue: queue.Queue[str] = queue.Queue()
        command_queue: queue.Queue[dict[str, str]] = queue.Queue()
        status_queue: queue.Queue[dict[str, object]] = queue.Queue()
        worker_stop_event = threading.Event()
        command_queue.put({"action": "SHUTDOWN"})

        with (
            patch.dict(sys.modules, {"cv2": fake_cv2, "camera": fake_camera}),
            patch.object(mcl.config, "SHOW_PREVIEW", False),
            patch.object(mcl.config, "MICRO_LOOP_CAMERA_WARMUP_SECONDS", 0.0),
        ):
            mcl.micro_closed_loop_camera_worker(
                error_queue,
                command_queue,
                status_queue,
                worker_stop_event,
            )

        self.assertTrue(error_queue.empty())
        ready = main._StatusInbox(status_queue).wait(
            "camera",
            "READY",
            error_queue,
            threading.Event(),
            None,
            timeout_s=0.2,
        )
        self.assertAlmostEqual(float(ready["actual_camera_fps"]), 132.3)
        self.assertEqual(ready["full_width"], 4)
        self.assertEqual(ready["full_height"], 3)

    def test_open_and_close_window_writes_raw_csv_and_metadata(self) -> None:
        fake_cv2, fake_camera = self._fake_modules(_StreamingHikCameraSource)
        error_queue: queue.Queue[str] = queue.Queue()
        command_queue: queue.Queue[dict[str, object]] = queue.Queue()
        status_queue: queue.Queue[dict[str, object]] = queue.Queue()
        stop_event = threading.Event()
        inbox = main._StatusInbox(status_queue)

        with (
            patch.dict(sys.modules, {"cv2": fake_cv2, "camera": fake_camera}),
            patch.object(mcl.config, "SHOW_PREVIEW", False),
            patch.object(mcl.config, "MICRO_LOOP_CAMERA_WARMUP_SECONDS", 0.0),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            worker = threading.Thread(
                target=mcl.micro_closed_loop_camera_worker,
                args=(error_queue, command_queue, status_queue, stop_event),
                daemon=True,
            )
            worker.start()
            inbox.wait(
                "camera", "READY", error_queue, stop_event, None, timeout_s=0.5
            )

            command_queue.put(
                {
                    "action": "OPEN_WINDOW",
                    "temp_dir": temp_dir,
                    "stem": "window_contract",
                    "clip_label": "window contract",
                    "capacity_frames": 100,
                }
            )
            inbox.wait(
                "camera", "WINDOW_OPENED", error_queue, stop_event, None, timeout_s=0.5
            )
            time.sleep(0.01)
            command_queue.put({"action": "CLOSE_WINDOW"})
            saved = inbox.wait(
                "camera", "WINDOW_SAVED", error_queue, stop_event, None, timeout_s=0.5
            )

            command_queue.put({"action": "SHUTDOWN"})
            worker.join(timeout=0.5)
            self.assertFalse(worker.is_alive())
            self.assertTrue(error_queue.empty())
            self.assertGreater(int(saved["frame_count"]), 0)
            self.assertEqual(
                Path(str(saved["raw_path"])).stat().st_size,
                int(saved["frame_count"]) * 4 * 3,
            )
            frame_lines = Path(str(saved["frames_path"])).read_text(
                encoding="utf-8-sig"
            ).splitlines()
            self.assertEqual(len(frame_lines), int(saved["frame_count"]) + 1)
            self.assertTrue(Path(str(saved["meta_path"])).is_file())

    def test_snapshot_copies_before_frames_while_source_window_stays_open(self) -> None:
        fake_cv2, fake_camera = self._fake_modules(_StreamingHikCameraSource)
        error_queue: queue.Queue[str] = queue.Queue()
        command_queue: queue.Queue[dict[str, object]] = queue.Queue()
        status_queue: queue.Queue[dict[str, object]] = queue.Queue()
        stop_event = threading.Event()
        inbox = main._StatusInbox(status_queue)

        with (
            patch.dict(sys.modules, {"cv2": fake_cv2, "camera": fake_camera}),
            patch.object(mcl.config, "SHOW_PREVIEW", False),
            patch.object(mcl.config, "MICRO_LOOP_CAMERA_WARMUP_SECONDS", 0.0),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            worker = threading.Thread(
                target=mcl.micro_closed_loop_camera_worker,
                args=(error_queue, command_queue, status_queue, stop_event),
                daemon=True,
            )
            worker.start()
            inbox.wait("camera", "READY", error_queue, stop_event, None, timeout_s=0.5)
            command_queue.put(
                {
                    "action": "OPEN_WINDOW", "temp_dir": temp_dir,
                    "stem": "source", "clip_label": "source", "capacity_frames": 100,
                }
            )
            inbox.wait("camera", "WINDOW_OPENED", error_queue, stop_event, None, timeout_s=0.5)
            time.sleep(0.02)
            command_queue.put(
                {
                    "action": "SNAPSHOT_WINDOW", "temp_dir": temp_dir,
                    "stem": "before", "clip_label": "before", "frame_count": 4,
                }
            )
            snapshot = inbox.wait(
                "camera", "WINDOW_SNAPSHOT", error_queue, stop_event, None, timeout_s=0.5
            )
            self.assertEqual(int(snapshot["frame_count"]), 4)
            self.assertEqual(Path(str(snapshot["raw_path"])).stat().st_size, 4 * 4 * 3)

            command_queue.put({"action": "CLOSE_WINDOW"})
            saved = inbox.wait(
                "camera", "WINDOW_SAVED", error_queue, stop_event, None, timeout_s=0.5
            )
            self.assertGreaterEqual(int(saved["frame_count"]), 4)
            command_queue.put({"action": "SHUTDOWN"})
            worker.join(timeout=0.5)
            self.assertFalse(worker.is_alive())
            self.assertTrue(error_queue.empty())


if __name__ == "__main__":
    unittest.main()
