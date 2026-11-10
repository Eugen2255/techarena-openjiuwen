import unittest

import mylib as m


class T(unittest.TestCase):
    def test_clamp_upper(self):
        self.assertEqual(m.clamp(5, 0, 3), 3)

    def test_clamp_keeps_value(self):
        self.assertEqual(m.clamp(5, 0, 3), 5)


if __name__ == '__main__':
    unittest.main()
