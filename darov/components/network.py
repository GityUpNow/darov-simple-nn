from .layer import Layer


class Network:
    """A complete ANN.

    This network only contains hidden and output layers, input is handled via calculate.
    """

    def __init__(self, layers: list[Layer]):
        """Construct a new ANN.

        Args:
            layers (list[Layer]): Hidden layers.
        """
        self.layers = layers

    def calculate(self, input_values: list[float]) -> list[float]:
        """Calculates the whole network for the give inputs.

        Args:
            input_values (list[float]): Input values to the network.

        Returns:
            list[float]: Output vector of the network.
        """
        current = input_values

        for layer in self.layers:
            current = layer.calculate(current)

        return current
