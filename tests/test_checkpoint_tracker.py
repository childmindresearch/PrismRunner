"""Unit tests for Checkpoint Tracker."""

import os
import tempfile
import unittest

from prismrunner.tracker.checkpoint import CheckpointTracker


class TestCheckpointTracker(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.chk_file = os.path.join(self.temp_dir.name, "chk.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_checkpoint_lifecycle(self):
        tracker = CheckpointTracker(self.chk_file)
        self.assertFalse(tracker.is_completed("doc_1", "SchemaA", 0))

        tracker.mark_completed("doc_1", "SchemaA", 0, {"field": "value1"})
        self.assertTrue(tracker.is_completed("doc_1", "SchemaA", 0))
        self.assertFalse(tracker.is_completed("doc_1", "SchemaA", 1))

        tracker.save()
        self.assertTrue(os.path.exists(self.chk_file))

        # Reload tracker from disk
        tracker_reloaded = CheckpointTracker(self.chk_file)
        self.assertTrue(tracker_reloaded.is_completed("doc_1", "SchemaA", 0))
        cached = tracker_reloaded.get_cached_result("doc_1", "SchemaA", 0)
        self.assertEqual(cached, {"field": "value1"})


if __name__ == "__main__":
    unittest.main()
