from __future__ import annotations

import queue
import threading
import unittest

from main import _wait_for_worker_error


class BatchErrorReportingTests(unittest.TestCase):
    def test_delayed_worker_error_is_not_masked_by_stop_event(self) -> None:
        errors: queue.Queue[str] = queue.Queue()
        timer = threading.Timer(0.02, errors.put, args=("camera detail",))
        timer.start()
        try:
            self.assertEqual(_wait_for_worker_error(errors, 0.2), "camera detail")
        finally:
            timer.cancel()

    def test_missing_worker_error_times_out_cleanly(self) -> None:
        errors: queue.Queue[str] = queue.Queue()
        self.assertIsNone(_wait_for_worker_error(errors, 0.01))


if __name__ == "__main__":
    unittest.main()
