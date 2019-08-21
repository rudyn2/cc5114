from src.neural_network.activation_function.ActivationFunc import ActivationFunc


class Step(ActivationFunc):
    """
    Step activation function class.
    """

    def get_value(self, x: float):
        """
        Maps from x to f(x) where f(x) corresponds to the step function.
        :param x:                           Input of function.
        :return:                            Mapped value.
        """
        return 1 if x >= 0 else 0

    def get_derivative(self, x):
        """
        Maps from x to f(x) where f(x) corresponds to the derivative of step function.
        :param x:                           Input of function.
        :return:                            Mapped value.
        """
        return 0
