import sys
from threading import Thread, Lock
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from application.main_window import Window
from application.network_manager import NetworkManager
from application.painter import Painter
from application.settings import SettingsWindow


class App:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.settings_window = SettingsWindow(parent_app=self)
        self.window = Window(parent_app=self)
        self.netmanager = NetworkManager(parent_app=self)
        self.painter = Painter(self.window.image, QPixmap(700, 700))
        self.worker_busy = False

    def train_network(self):
        if not self.worker_busy:
            self.worker_busy = True
            worker_thread = Thread(target=self.train_and_paint)
            worker_thread.start()

    def update_learning_rate(self):
        lr = 0.001 + self.window.LearningSlider.value() * 0.01
        self.netmanager.update_learning_rate(lr)

    def reset_network(self):
        self.netmanager.reset_network()
        self.update_screen()

    def clear_map(self):
        self.painter.reset_image()
        self.netmanager.clear_points()

    def update_screen(self):
        self.painter.paint_scene(self.netmanager.network)
        self.painter.paint_points(self.netmanager.greenpoints, self.netmanager.redpoints)

    def run(self):
        self.window.show()
        sys.exit(self.app.exec_())

    def train_and_paint(self):
        if self.window.BackButton.isChecked():
            self.netmanager.machine_learning()
        else:
            self.netmanager.evolve_network()
        self.update_screen()
        self.worker_busy = False
