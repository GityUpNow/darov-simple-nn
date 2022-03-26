import unittest

from darov.components.activations import linear


class TestActivation(unittest.TestCase):

    def test_linear(self):
        test_data = [0.0, 0.5, 1.0]

        for data in test_data:
            self.assertEqual(data, linear(data))


if __name__ == '__main__':
    unittest.main()
