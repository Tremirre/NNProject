import numpy as np
import ann.functions as f


def to_batches(data, batch_size):
    batches = []
    k = 0
    while k + batch_size <= len(data):
        batches.append(data[k:k + batch_size])
        k += batch_size
    batches.append(data[k:])
    return batches


labeled_entry = tuple[np.ndarray, np.ndarray]


class Network:
    def __init__(self,
                 layers: tuple[int, ...],
                 learning_rate: float,
                 functions: tuple[f.ActivationFunction, ...],
                 loss_function: f.LossFunction):
        self.weight_sizes = [(layers[i + 1], layers[i]) for i in range(len(layers) - 1)]
        self.bias_sizes = [size for size in layers[1:]]
        self.weights = [np.random.standard_normal(size) for size in self.weight_sizes]
        self.biases = [np.random.standard_normal(size) for size in self.bias_sizes]
        self.functions = functions
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def calculate_all(self, provided_input: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        current_output = provided_input
        raw_outputs = []
        func_outputs = []
        for weight, bias, function in zip(self.weights, self.biases, self.functions):
            raw = weight @ current_output + bias.T
            raw_outputs.append(raw)
            current_output = function(raw)
            func_outputs.append(current_output)
        return raw_outputs, func_outputs

    def calculate(self, input_vector: np.ndarray):
        return self.calculate_all(input_vector)[1][-1]

    def get_error(self, output: np.array, expected: np.array) -> np.array:
        return np.sum(self.loss_function(output, expected))

    def calculate_gradients(self, raw_outputs, func_outputs, expected):
        loss_gradient = self.loss_function(func_outputs[-1], expected, gradient=True)
        gradients = [loss_gradient * self.get_derivative(raw_outputs, func_outputs, len(self.weights) - 1, expected)]
        for idx, weight in enumerate(self.weights[:0:-1]):
            gradients.append(
                (weight.T @ gradients[-1]) * self.get_derivative(raw_outputs, func_outputs, len(self.weights) - idx - 2, expected)
            )
        return gradients[::-1]

    def get_derivative(self, raw_outputs, func_outputs, layer_index, expected):
        if self.functions[layer_index] is f.SIGMOID:
            return func_outputs[layer_index] * (1 - func_outputs[layer_index])
        if self.functions[layer_index] is f.SOFTMAX:
            return func_outputs[layer_index] - np.array((expected,)).T
        return self.functions[layer_index](raw_outputs[layer_index], gradient=True)

    def update_gradients_and_deltas(self,
                                    input_vector: np.ndarray,
                                    expected: np.ndarray,
                                    delta_weights: list,
                                    delta_biases: list) -> None:
        raw_outputs, func_outputs = self.calculate_all(input_vector)
        gradients = self.calculate_gradients(raw_outputs, func_outputs, expected)
        for i in reversed(range(1, len(func_outputs))):
            delta_biases[i] += gradients[i]
            delta_weights[i] += np.outer(gradients[i], func_outputs[i - 1].T)
        delta_biases[0] += gradients[0]
        delta_weights[0] += np.outer(gradients[0], input_vector)

    def backpropagation(self, training_data: list[labeled_entry], validation_data: list[labeled_entry]) -> float:
        if len(training_data) == 0:
            return 0.0
        delta_weights = [0] * len(self.weights)
        delta_biases = [0] * len(self.weights)
        for entry, label in training_data:
            self.update_gradients_and_deltas(entry, label, delta_weights, delta_biases)

        for i, _ in enumerate(delta_weights):
            delta_weights[i] /= len(training_data)
            delta_biases[i] /= len(training_data)
            self.weights[i] -= self.learning_rate * delta_weights[i]
            self.biases[i] -= self.learning_rate * delta_biases[i].T
            self.weights[i] = np.clip(self.weights[i], -10, 10)
            self.biases[i] = np.clip(self.biases[i], -10, 10)

        return sum([self.get_error(self.calculate(entry), label) for entry, label in validation_data])

    def print_net(self):
        print("--------------------------------------------------", end='')
        for i, (weight_layer, bias_layer, func) in enumerate(zip(self.weights, self.biases, self.functions)):
            print(f"\nLayer {i + 1}")
            print("\tWeights:")
            for j, neuron_weights in enumerate(weight_layer):
                print(f"\t\tNeuron {j}:\t{[round(w, 2) for w in neuron_weights]}")
            print(f"\n\tBiases:\t{[round(b, 2) for b in bias_layer]}")
            print(f"\n\tApplied Function: {func.name}")
        print("--------------------------------------------------")


def main():
    data = [
        (np.array([1, 1]), np.array(0)),
        (np.array([0, 0]), np.array(0)),
        (np.array([1, 0]), np.array(1)),
        (np.array([0, 1]), np.array(1))
    ]
    np.random.seed(0)
    net = Network((2, 2, 1), 1, (f.RELU, f.RELU), loss_function=f.MSE)
    net.print_net()
    val = net.calculate(data[-1][0])
    net.backpropagation([data[-1]])
    net.print_net()
    net.calculate(data[-1][0])
    print(val)
    return
    for pair, label in data:
        print(pair, net.calculate(pair), label)

    for i in range(300):
        net.backpropagation(data)
    print('===========================')
    for pair, label in data:
        print(pair, net.calculate(pair), label)


if __name__ == "__main__":
    main()
