import unittest

from darov.components.network import Network
from darov.components.neuron import Neuron
from darov.components.layer import Layer
import darov.components.activations as activations


class TestNetwork(unittest.TestCase):

    def test_linear(self):
        input_values = [3, -10, 0, 4, -2, -9, -5]

        network = Network([
            Layer([
                Neuron([0.1, -0.2, -0.1, -0.7, -0.6, -0.3, 0.8], -0.2,
                       activations.linear),
                Neuron([0.4, -0.5, 1.0, -1.0, 0.1, -0.1, 0.9], 0.3,
                       activations.linear)
            ]),
            Layer([
                Neuron([0.2, 0.0], 0.8, activations.linear),
                Neuron([0.5, -0.7], 0.0, activations.linear),
                Neuron([0.0, -0.4], 0.0, activations.linear),
                Neuron([1.0, -0.7], -0.5, activations.linear)
            ]),
            Layer([
                Neuron([-0.5, 0.3, 0.7, 0.3], 0.0, activations.linear),
                Neuron([0.2, -0.6, 0.7, 1.0], -0.9, activations.linear),
                Neuron([-0.6, -1.0, -0.1, 0.3], 0.1, activations.linear)
            ]),
            Layer([
                Neuron([-0.7, 0.1, 0.6], -1.0, activations.linear),
                Neuron([0.8, -0.9, 0.7], -0.5, activations.linear),
                Neuron([-0.6, 0.7, -1.0], 0.3, activations.linear)
            ])
        ])

        expected_output = [
            -1.7442000000000002, -0.11650000000000027, 0.44220000000000037
        ]
        output = network.calculate(input_values)
        for (actual, expected) in zip(output, expected_output):
            self.assertEqual(actual, expected)
