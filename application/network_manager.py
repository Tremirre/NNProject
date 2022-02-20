from ann.neural import Network, to_batches
from ann.ens import EvolutionaryNetworkSystem
import random as rand


class NetworkManager:
    def __init__(self, parent_app):
        self.redpoints = []
        self.greenpoints = []
        self.parent = parent_app
        self.network = Network([2, 8, 8, 4, 1], 0.045, ['lin', 'sig', 'lin', 'sig'])
        self.ens = EvolutionaryNetworkSystem(20, 0.6, 0.5, [2, 8, 8, 5, 1], 0.045, ['lin', 'sig', 'lin', 'sig'])
        self.progress_label = parent_app.window.CounterLabel

    def update_learning_rate(self, lr):
        self.network.learningrate = lr

    def reset_network(self):
        self.network = Network([2, 8, 8, 4, 1], 0.045, ['lin', 'sig', 'lin', 'sig'])
        self.ens = EvolutionaryNetworkSystem(20, 0.6, 0.5, [2, 8, 8, 5, 1], 0.045, ['ReLU', 'sig', 'ReLU', 'sig'])

    def clear_points(self):
        self.redpoints = []
        self.greenpoints = []

    def machine_learning(self):
        points = [([(point.x() - 350) / 700, (point.y() - 350) / 700], 0) for point in self.redpoints]
        points += [([(point.x() - 350) / 700, (point.y() - 350) / 700], 1) for point in self.greenpoints]
        best_network = self.network
        best_count = 0
        if len(points) > 0:
            for i in range(5000):
                correct_count = 0
                rand.shuffle(points)
                batches = to_batches(points, 4)
                for pair in points:
                    r = self.network.calculate(pair[0])
                    if pair[1] == 0:
                        if r < 0.5:
                            correct_count += 1
                    else:
                        if r >= 0.5:
                            correct_count += 1
                if correct_count == len(points):
                    break
                if correct_count > best_count:
                    best_count = correct_count
                    best_network = self.network
                for batch in batches:
                    self.network.backpropagation(batch)
        self.network = best_network

    def progress_report(self, e):
        self.progress_label.setText(str(e))

    def evolve_network(self):
        points = [([(point.x() - 350) / 700, (point.y() - 350) / 700], 0) for point in self.redpoints]
        points += [([(point.x() - 350) / 700, (point.y() - 350) / 700], 1) for point in self.greenpoints]
        if len(points) > 0:
            rand.shuffle(points)
            self.ens.evolve(points, 20)
        self.network = self.ens.get_best_network(points)
