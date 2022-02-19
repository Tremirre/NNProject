import numpy as np

FUNCTIONS = {
    "sig": lambda x: 1/(1 + np.exp(-x)),
    "lin": lambda x: x,
    "ReLU": np.vectorize(lambda x: max(x, 0)),
    "tanh": lambda x: np.tanh(x)/2 + 0.5
}

DERIVATIVES = {
    "lin": np.vectorize(lambda x: 1),
    "ReLU": np.vectorize(lambda x: int(x > 0)),
    "tanh": lambda x: (1 / np.cosh(x))**2 / 2
}


def to_batches(data, batch_size):
    batches = []
    k = 0
    while k + batch_size <= len(data):
        batches.append(data[k:k + batch_size])
        k += batch_size
    batches.append(data[k:])
    return batches


class Network:
    def __init__(self, layers, learningrate, functions=None):
        self.weightsizes = [(layers[i+1], layers[i]) for i in range(len(layers) - 1)]
        self.biassizes = [(1, size) for size in layers[1:]]
        self.weights = [np.random.standard_normal(size) for size in self.weightsizes]
        self.biases = [np.random.standard_normal(size) for size in self.biassizes]
        if functions:
            self.functions = functions
            while len(functions) < len(layers) - 1:
                self.functions.append("sig")
            self.functions[-1] = "sig"
        else:
            self.functions = ["sig" for i in self.weights]
        self.learningrate = learningrate

    def calculate_all(self, provided_input):
        current_output = np.array([provided_input]).T
        outputs = []
        for weight, bias, function in zip(self.weights, self.biases, self.functions):
            current_output = FUNCTIONS[function](np.matmul(weight, current_output) + bias.T)
            outputs.append(current_output)
        return outputs

    def calculate(self, a):
        return self.calculate_all(a)[-1][0][0]

    @staticmethod
    def get_error(output, cls):
        error = cls - output
        if abs(error) < 0.5:
            return np.array(error / 1.5)
        return np.array(error * 1.5)

    def calculate_errors(self, outputs, cls):
        errors = [np.array([])] * len(self.weights)
        errors[-1] = self.get_error(outputs[-1], cls)
        for i in range(len(errors) - 2, -1, -1):
            errors[i] = np.dot(self.weights[i+1].T, errors[i+1])
        return errors

    def get_derivative(self, pair, outputs, i):
        if self.functions[i] == "sig":
            return outputs[i] * (1 - outputs[i])
        elif self.functions[i] in DERIVATIVES:
            if i > 0:
                return DERIVATIVES[self.functions[i]](np.matmul(self.weights[i], outputs[i-1]))
            return DERIVATIVES[self.functions[i]](np.matmul(self.weights[0], np.array([pair]).T))
        else:
            raise Exception("Improper function: " + self.functions[i])

    def initialize_gradients_and_deltas(self):
        gradients = []
        deltas = []
        for w_size in self.weightsizes:
            gradients.append(np.zeros(w_size))
        for b_size in self.biassizes:
            deltas.append(np.zeros(b_size))
        return gradients, deltas

    def update_gradients_and_deltas(self, pair, cls, gradients, deltas):
        outputs = self.calculate_all(pair)
        errors = self.calculate_errors(outputs, cls)

        for i in range(len(errors) - 1, -1, -1):
            derivative = self.get_derivative(pair, outputs, i)
            temp = self.learningrate * errors[i] * derivative
            gradients[i] += temp
            if i > 0:
                deltas[i] += temp * outputs[i-1].T
            else:
                deltas[i] += temp * np.array(pair).T

    def backpropagation(self, data):
        if len(data) == 0:
            return
        #gradients, deltas = self.initialize_gradients_and_deltas()
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
        for i in range(len(self.weights)):
            print("\nLayer {}".format(i+1))
            print("Weights:", end='\t')
            for weight in self.weights[i][0]:
                print("{:.2f}".format(weight), end=' ; ')
            print("\nBiases:", end='\t\t')
            for bias in self.biases[i][0]:
                print("{:.2f}".format(bias), end=' ; ')
            print()
        print("--------------------------------------------------")
