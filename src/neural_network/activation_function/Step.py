from src.neural_network.activation_function.ActivationFunc import ActivationFunc


class Step(ActivationFunc):

    def get_value(self, x):
        return 1 if x >= 0 else 0

    def get_derivative(self, x):
        return 0
