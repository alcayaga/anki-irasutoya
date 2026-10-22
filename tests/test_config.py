"""Unit tests for add-on configuration management and persistence."""
import unittest

import config
from config import (
    CONFIG_KEY_FULL_DELAY,
    CONFIG_KEY_OVERWRITE,
    CONFIG_KEY_SOURCE,
    CONFIG_KEY_TARGET,
    CONFIG_KEY_THUMB_DELAY,
    get_config,
    update_user_preferences,
)


class TestConfig(unittest.TestCase):
    """Test suite for default configuration and user preference updates."""

    def setUp(self):
        """Reset in-memory config before each test."""
        config._memory_config.clear()

    def tearDown(self):
        """Reset in-memory config after each test."""
        config._memory_config.clear()

    def test_default_config(self):
        """Verify default configuration keys and default values."""
        cfg = get_config()
        self.assertIn(CONFIG_KEY_SOURCE, cfg)
        self.assertIn(CONFIG_KEY_TARGET, cfg)
        self.assertIn(CONFIG_KEY_OVERWRITE, cfg)
        self.assertIn(CONFIG_KEY_THUMB_DELAY, cfg)
        self.assertIn(CONFIG_KEY_FULL_DELAY, cfg)
        self.assertEqual(cfg[CONFIG_KEY_THUMB_DELAY], 50)
        self.assertEqual(cfg[CONFIG_KEY_FULL_DELAY], 300)

    def test_update_preferences(self):
        """Verify updating and persisting user field and overwrite preferences."""
        update_user_preferences("Vocab", "Picture", "overwrite")
        cfg = get_config()
        self.assertEqual(cfg[CONFIG_KEY_SOURCE], "Vocab")
        self.assertEqual(cfg[CONFIG_KEY_TARGET], "Picture")
        self.assertEqual(cfg[CONFIG_KEY_OVERWRITE], "overwrite")


if __name__ == "__main__":
    unittest.main()
