from PyQt5.QtCore import *
from ann.neural import Network, to_batches
from ann.ens import EvolutionaryNetworkSystem
import random as rand


class NetworkManager:

    class MLWorker(QObject):
        finished = pyqtSignal()
        progress = pyqtSignal(int)

        def run(self, parent):
            points = [([(point.x() - 350) / 700, (point.y() - 350) / 700], 0) for point in parent.redpoints]
            points += [([(point.x() - 350) / 700, (point.y() - 350) / 700], 1) for point in parent.greenpoints]
            best_network = parent.network
            best_count = 0
            if len(points) > 0:
                for i in range(5000):
                    self.progress.emit(i + 1)
                    correct_count = 0
                    rand.shuffle(points)
                    batches = to_batches(points, 4)
                    for pair in points:
                        r = parent.network.calculate(pair[0])
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
                        best_network = parent.network
                    for batch in batches:
                        parent.network.backpropagation(batch)
            parent.network = best_network
            self.finished.emit()

    class ENSWorker(QObject):...

    def __init__(self, parent_app):
        self.redpoints = []
        self.greenpoints = []
        self.parent = parent_app
        self.network = Network([2, 8, 8, 4, 1], 0.045, ['lin', 'sig', 'lin', 'sig'])
        self.ens = EvolutionaryNetworkSystem(20, 0.6, 0.5, [2, 8, 8, 5, 1], 0.045, ['lin', 'sig', 'lin', 'sig'])
        self.progress_label = parent_app.window.CounterLabel
        self.thread = None
        self.worker = None
        self.thread_active = False

    def update_learning_rate(self, lr):
        self.network.learning_rate = lr
        print(self.thread)

    def reset_network(self):
        self.network = Network([2, 8, 8, 4, 1], 0.045, ['lin', 'sig', 'lin', 'sig'])
        self.ens = EvolutionaryNetworkSystem(20, 0.6, 0.5, [2, 8, 8, 5, 1], 0.045, ['ReLU', 'sig', 'ReLU', 'sig'])

    def clear_points(self):
        self.redpoints.clear()
        self.greenpoints.clear()

    def machine_learning(self):
        if not self.thread_active:
            self.thread_active = True
            self.thread = QThread()
            self.worker = self.MLWorker()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(lambda: self.worker.run(self))
            self.worker.finished.connect(self.terminate_thread)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.progressReport)
            self.thread.start()

    def terminate_thread(self):
        self.worker.deleteLater()
        self.parent.updateScreen()
        self.thread.quit()
        self.thread_active = False

    def progressReport(self, e):
        self.progress_label.setText(str(e))

    def evolve_network(self):
        points = [([(point.x() - 350) / 700, (point.y() - 350) / 700], 0) for point in self.redpoints]
        points += [([(point.x() - 350) / 700, (point.y() - 350) / 700], 1) for point in self.greenpoints]
        if len(points) > 0:
            rand.shuffle(points)
            self.ens.evolve(points, 20)
        self.network = self.ens.getbestnetwork(points)
