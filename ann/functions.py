import numpy as np
from typing import Callable


class ActivationFunction:
    def __init__(self,
                 function: Callable[[np.ndarray], np.ndarray],
                 derivative: Callable[[np.ndarray], np.ndarray],
                 name: str):
        self.function = function
        self.derivative = derivative
        self.name = name

    def apply_on_tensor(self, tensor: np.ndarray) -> np.ndarray:
        return self.function(tensor)

    def get_gradient_from_tensor(self, tensor: np.ndarray) -> np.ndarray:
        return self.derivative(tensor)

    def __call__(self, tensor: np.ndarray, gradient=False) -> np.ndarray:
        return self.apply_on_tensor(tensor) if not gradient else self.derivative(tensor)


LINEAR = ActivationFunction(
    np.vectorize(lambda x: x), 
    np.vectorize(lambda x: 1), 
    "LINEAR"
)
RELU = ActivationFunction(
    np.vectorize(lambda x: max(x, 0.0)),
    np.vectorize(lambda x: float(x > 0)),
    "RELU"
)
LEAKY_RELU = ActivationFunction(
    np.vectorize(lambda x: max(x, 0.1*x)),
    np.vectorize(lambda x: 1.0 if x >= 0 else 0.1),
    "LeakyRELU"    
)
SOFTPLUS = ActivationFunction(
    lambda x: np.log(1 + np.exp(x)),
    lambda x: 1 / (1 + np.exp(-x)),
    "SOFTPLUS"
)
TANH = ActivationFunction(
    lambda x: np.tanh(x),
    lambda x: 1 - np.tanh(x) ** 2,
    "TANH"
)
SIGMOID = ActivationFunction(
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2,
    "SIGMOID"
)


def softmax_processing_function(tensor: np.ndarray) -> np.ndarray:
    divisor = np.sum(np.exp(tensor))
    return np.vectorize(lambda x: np.exp(x) / divisor)(tensor)


SOFTMAX = ActivationFunction(softmax_processing_function, None, "SOFTMAX")


class LossFunction:
    def __init__(self, function: Callable[[np.ndarray, np.ndarray], np.ndarray], derivative: Callable[[np.ndarray, np.ndarray], np.ndarray], name: str):
        self.function = function
        self.derivative = derivative
        self.name = name

    def calculate_loss(self, obtained: np.ndarray, expected: np.ndarray) -> np.ndarray:
        return self.function(obtained, expected)

    def calculate_gradient(self, obtained: np.ndarray, expected: np.ndarray) -> np.ndarray:
        return self.derivative(obtained, expected)

    def __call__(self, obtained: np.ndarray, expected: np.ndarray, gradient = False) -> np.ndarray:
        return self.calculate_loss(obtained, expected) if not gradient else self.calculate_gradient(obtained, expected)


def multi_class_log_loss(obtained_val: float, expected_val: float) -> float:
    return -expected_val * np.log(obtained_val) if expected_val != 0 else 0


MSE = LossFunction(
    lambda obtained, expected: ((obtained - expected) ** 2) / 2,
    lambda obtained, expected: obtained - expected, 
    "MEAN SQUARED ERROR"
)
MAE = LossFunction(
    lambda obtained, expected: np.abs(obtained - expected),
    lambda obtained, expected: (obtained > expected) * 2 - 1,
    "MEAN ABSOLUTE ERROR"
)
MCCEL = LossFunction(
    np.vectorize(multi_class_log_loss),
    lambda obtained, expected: np.vectorize(multi_class_log_loss)(obtained, expected) - expected, 
    "MULTI CLASS CROSS ENTROPY LOSS"
)

if __name__ == "__main__":
    expected = np.array(((1, 0, 0),)).T
    obtained = np.array(((0.5, 0.3, 0.2),)).T
    print(MSE(obtained, expected))
