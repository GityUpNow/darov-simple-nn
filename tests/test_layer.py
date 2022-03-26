import unittest

from darov.components.layer import Layer
from darov.components.neuron import Neuron
import darov.components.activations as activations


class TestLayer(unittest.TestCase):

    def test_layer_linear(self):
        input_values = [5.7, -9.0, 3.4]

        neuron1 = Neuron([0.5, 0.1, 0.2], 0.3, activations.linear)
        neuron2 = Neuron([0.3, -0.8, 0.6], 0.7, activations.linear)
        neuron3 = Neuron([0.9, -0.7, 1.0], -0.8, activations.linear)
        neuron4 = Neuron([-0.9, -0.6, 0.0], -0.3, activations.linear)

        layer = Layer([neuron1, neuron2, neuron3, neuron4])
        output = layer.calculate(input_values)

        expected_output = [2.93, 11.65, 14.03, -0.03]
        for (actual, expected) in zip(output, expected_output):
            self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
