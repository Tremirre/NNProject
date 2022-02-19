import random as rand
import numpy as np
from ann.neural import Network, to_batches


class EvolutionaryNetworkSystem:
    def __init__(self, pop_size, pmut, pcros, layers, learning_rate, functions=None):
        self.population = [Network(layers, learning_rate, functions) for _ in range(pop_size)]
        self.pmut = pmut
        self.pcros = pcros

    @staticmethod
    def mutate(Net, mutation_factor):
        for bias in Net.biases:
            for element in bias:
                element += np.random.normal(0)*mutation_factor
        for weight in Net.weights:
            for element in weight:
                element += np.random.normal(0)*mutation_factor

    @staticmethod
    def crossover(Net1, Net2):
        NewNet1 = Net1
        NewNet2 = Net2
        for i in range(len(Net1.biases)):
            if i % 2 == 0:
                NewNet1.biases[i] = Net1.biases[i]
                NewNet2.biases[i] = Net2.biases[i]
            else:
                NewNet1.biases[i] = Net2.biases[i]
                NewNet2.biases[i] = Net1.biases[i]

        for i in range(len(Net1.weights)):
            if i % 2 == 0:
                NewNet1.weights[i] = Net1.weights[i]
                NewNet2.weights[i] = Net2.weights[i]
            else:
                NewNet1.weights[i] = Net2.weights[i]
                NewNet2.weights[i] = Net1.weights[i]

        return NewNet1, NewNet2

    @staticmethod
    def getfitness(network, data):
        return sum([abs(network.get_error(network.calculate_all(point)[-1], cls)[0][0]) for point, cls in data])

    def tournament(self, data, size=4):
        newpopulation = []
        while len(newpopulation) < len(self.population):
            sample = rand.sample(self.population, size)
            best = sample[0]
            best_error = self.getfitness(best, data)
            for network in sample[1:]:
                current_error = self.getfitness(network, data)
                if current_error < best_error:
                    best = network
                    best_error = current_error
            newpopulation.append(best)
        return newpopulation

    def evolve(self, data, maxgen=5, mutation_factor=1/400):
        generation = 0
        best = self.getbestnetwork(data)

        while generation < maxgen:
            generation += 1

            for network in self.population:
                for i in range(1):
                    for batch in to_batches(data, 4):
                        network.backpropagation(batch)

            for network in self.population:
                if np.random.random() < self.pmut:
                    self.mutate(network, mutation_factor)

            for network in self.population:
                if np.random.random() < self.pcros:
                    self.crossover(network, rand.choice(self.population))

            for network in self.population:
                if np.random.random() < self.pcros:
                    self.crossover(network, rand.choice(self.population))

            self.population = self.tournament(data)
            current_best = self.getbestnetwork(data)
            if self.getfitness(current_best, data) < self.getfitness(best, data):
                best = current_best

        return best

    def getbestnetwork(self, data):
        best = self.population[0]
        best_error = self.getfitness(best, data)
        for network in self.population[1:]:
            current_error = self.getfitness(network, data)
            if current_error < best_error:
                best = network
                best_error = current_error
        return best

    def getaverageerror(self, data):
        avg = sum([self.getfitness(network, data) for network in self.population])
        return avg/len(self.population)

    def getbesterror(self, data):
        network = self.getbestnetwork(data)
        return self.getfitness(network, data)

    def geterrors(self, data):
        errors = []
        for network in self.population:
            errors.append(self.getfitness(network, data))
        return errors
