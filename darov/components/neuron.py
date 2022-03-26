from typing import Callable


class Neuron:
    """A simple Neuron.
    """

    def __init__(self, weights: list[float], bias: float,
                 activation_function: Callable[[float], float]):
        """Constructs a new neuron.

        Args:
            weights (list[float]): Weights of the connections to the previous layer.
            bias (float): Bias of the neuron.
            activation_function (Callable[[float], float]): Activation function of the neuron.
        """
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def calculate(self, input_values: list[float]) -> float:
        """Calculates the output of the neuron.

        Args:
            input_values (list[float]): Input values from the previous layer.

        Returns:
            float: Output of the neuron.

        Raises:
            ValueError: When input differs in length from weights.
        """
        intermediary = 0.0
        for (input_value, weight) in zip(input_values,
                                         self.weights,
                                         strict=True):
            intermediary += input_value * weight

        result = self.activation_function(intermediary)

        return result
