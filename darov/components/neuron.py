from typing import Callable


class Neuron:
    """A simple Neuron.
    """

    def __init__(self,
                 weights: list[float],
                 bias: float,
                 activation_function: Callable[[float], float],
                 min_value: float = 0.0,
                 max_value: float = 1.0):
        """Constructs a new neuron.

        Args:
            weights (list[float]): Weights of the connections to the previous layer.
            bias (float): Bias of the neuron.
            activation_function (Callable[[float], float]): Activation function of the neuron.
            min_value (float): Min value weight, bias or activation function can take. Inclusive.
            max_value (float): Max value weight, bias or activation function can take. Inclusive.

        Raises:
            ValueError: Weights or bias out of MIN AND MAX range.
        """
        self.min_value = min_value
        self.max_value = max_value

        for weight in weights:
            self.value_out_of_range(weight)
        self.weights = weights

        self.value_out_of_range(bias)
        self.bias = bias

        self.activation_function = activation_function

    def value_out_of_range(self, value):
        """Checks if the given value fits the enforced range.

        Args:
            value (float): Value to check.

        Raises:
            ValueError: When value is outside of allowed range.
        """
        if value < self.min_value or value > self.max_value:
            raise ValueError("Value out of range.")

    def calculate(self, input_values: list[float]) -> float:
        """Calculates the output of the neuron.

        Args:
            input_values (list[float]): Input values from the previous layer.

        Returns:
            float: Output of the neuron.

        Raises:
            ValueError: When input differs in length from weights.
            ValueError: When value is outside of allowed range.
        """
        intermediary = 0.0
        for (input_value, weight) in zip(input_values,
                                         self.weights,
                                         strict=True):
            self.value_out_of_range(input_value)
            intermediary += input_value * weight

        result = self.activation_function(intermediary)
        self.value_out_of_range(result)

        return result
