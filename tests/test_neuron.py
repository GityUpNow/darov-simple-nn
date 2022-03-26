import unittest

from darov.components.neuron import Neuron
import darov.components.activations as activations


class TestNeuron(unittest.TestCase):

    def test_linear(self):
        neuron = Neuron(
            [1.0, -0.5, 0.7],
            0.2,
            activations.linear,
        )
        input_values = [10.0, 1.0, -3.0]

        self.assertEqual(neuron.calculate(input_values), 7.6)


if __name__ == '__main__':
    unittest.main()
