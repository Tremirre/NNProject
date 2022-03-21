import numpy as np
import functions as f


def to_batches(data, batch_size):
    batches = []
    k = 0
    while k + batch_size <= len(data):
        batches.append(data[k:k + batch_size])
        k += batch_size
    batches.append(data[k:])
    return batches


class Network:
    def __init__(self,
                 layers: tuple[int, ...],
                 learning_rate: float,
                 functions: tuple[f.ActivationFunction, ...],
                 loss_function: f.LossFunction):
        self.weight_sizes = [(layers[i + 1], layers[i]) for i in range(len(layers) - 1)]
        self.bias_sizes = [(1, size) for size in layers[1:]]
        self.weights = [np.random.standard_normal(size) for size in self.weight_sizes]
        self.biases = [np.random.standard_normal(size) for size in self.bias_sizes]
        self.functions = functions
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def calculate_all(self, provided_input):
        current_output = np.array([provided_input]).T
        outputs = []
        for weight, bias, function in zip(self.weights, self.biases, self.functions):
            current_output = function(np.matmul(weight, current_output) + bias.T)
            outputs.append(current_output)
        return outputs

    def calculate(self, a):
        return self.calculate_all(a)[-1]

    def get_error(self, output, expected):
        return self.loss_function(output, np.array((expected,)).T)

    def calculate_errors(self, outputs, cls):
        errors = [np.array([])] * len(self.weights)
        errors[-1] = self.get_error(outputs[-1], cls)
        for i in range(len(errors) - 2, -1, -1):
            errors[i] = np.dot(self.weights[i + 1].T, errors[i + 1])
        return errors

    def get_derivative(self, inputs, outputs, i, expected):
        if self.functions[i] is f.SIGMOID:
            return outputs[i] * (1 - outputs[i])
        if self.functions[i] is f.SOFTMAX:
            return outputs[i] - np.array((expected,)).T
        outputs_vector = outputs[i - 1] if i > 0 else np.array([inputs]).T
        return self.functions[i](np.matmul(self.weights[i], outputs_vector), gradient=True)

    def initialize_gradients_and_deltas(self):
        gradients = []
        deltas = []
        for w_size in self.weight_sizes:
            gradients.append(np.zeros(w_size))
        for b_size in self.bias_sizes:
            deltas.append(np.zeros(b_size))
        return gradients, deltas

    def update_gradients_and_deltas(self, input_vector, expected, gradients, deltas):
        outputs = self.calculate_all(input_vector)
        errors = self.calculate_errors(outputs, expected)

        for i in range(len(errors) - 1, -1, -1):
            derivative = self.get_derivative(input_vector, outputs, i, expected)
            temp = self.learning_rate * errors[i] * derivative
            gradients[i] += temp
            if i > 0:
                deltas[i] += temp * outputs[i - 1].T
            else:
                deltas[i] += temp * np.array(input_vector).T

    def backpropagation(self, data):
        if len(data) == 0:
            return
        # gradients, deltas = self.initialize_gradients_and_deltas()
        gradients = [0] * len(self.weights)
        deltas = [0] * len(self.weights)
        for pair, cls in data:
            self.update_gradients_and_deltas(pair, cls, gradients, deltas)

        for delta in deltas:
            delta /= len(data)
        for gradient in gradients:
            gradient /= len(data)

        for i in range(len(self.weights)):
            self.weights[i] += deltas[i]
            self.biases[i] += gradients[i].T
            self.weights[i] = np.clip(self.weights[i], -10, 10)
            self.biases[i] = np.clip(self.biases[i], -10, 10)

    def print_net(self):
        print("--------------------------------------------------", end='')
        for i, (weight_layer, bias_layer, func) in enumerate(zip(self.weights, self.biases, self.functions)):
            print(f"\nLayer {i + 1} ({func.name})")
            print("\tWeights:")
            for j, neuron_weights in enumerate(weight_layer):
                print(f"\t\tNeuron {j}:\t{[round(w, 2) for w in neuron_weights]}")
            print(f"\n\tBiases:\t{[round(b, 2) for b in bias_layer[0]]}")
        print("--------------------------------------------------")


def main():
    net = Network((2, 4, 3), 0.1, (f.LINEAR, f.SOFTMAX), loss_function=f.MCCEL)
    print([round(num[0], 4) for num in net.calculate([1, 1])])
    print([round(num[0], 4) for num in net.calculate([0, 1])])
    print([round(num[0], 4) for num in net.calculate([0, 0])])
    for i in range(20):
        net.backpropagation([((1, 1), (1, 0, 0)), ((0, 1), (0, 1, 0)), ((0, 0), (0, 0, 1))])
    print([round(num[0], 4) for num in net.calculate([1, 1])])
    print([round(num[0], 4) for num in net.calculate([0, 1])])
    print([round(num[0], 4) for num in net.calculate([0, 0])])


if __name__ == "__main__":
    main()
