import unittest
from config import (
    get_config,
    save_config,
    update_user_preferences,
    CONFIG_KEY_SOURCE,
    CONFIG_KEY_TARGET,
    CONFIG_KEY_OVERWRITE,
    CONFIG_KEY_THUMB_DELAY,
    CONFIG_KEY_FULL_DELAY,
)


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = get_config()
        self.assertIn(CONFIG_KEY_SOURCE, cfg)
        self.assertIn(CONFIG_KEY_TARGET, cfg)
        self.assertIn(CONFIG_KEY_OVERWRITE, cfg)
        self.assertIn(CONFIG_KEY_THUMB_DELAY, cfg)
        self.assertIn(CONFIG_KEY_FULL_DELAY, cfg)
        self.assertEqual(cfg[CONFIG_KEY_THUMB_DELAY], 50)
        self.assertEqual(cfg[CONFIG_KEY_FULL_DELAY], 300)

    def test_update_preferences(self):
        update_user_preferences("Vocab", "Picture", "overwrite")
        cfg = get_config()
        self.assertEqual(cfg[CONFIG_KEY_SOURCE], "Vocab")
        self.assertEqual(cfg[CONFIG_KEY_TARGET], "Picture")
        self.assertEqual(cfg[CONFIG_KEY_OVERWRITE], "overwrite")


if __name__ == "__main__":
    unittest.main()
