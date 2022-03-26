from .neuron import Neuron


class Layer:

    def __init__(self, neurons: list[Neuron]):
        """Layer in a neural network.

        Args:
            neurons (list[Neuron]): Neurons associated with this layer.
        """
        self.neurons = neurons

    def calculate(self, input_values: list[float]) -> list[float]:
        """Calculate output vector of layer.

        Args:
            input_values (list[float]): Input value from previous layer.

        Returns:
            list[float]: Output vector of layer.
        """
        return [neuron.calculate(input_values) for neuron in self.neurons]
