import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from application.window import Window
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

    def trainNetwork(self):
        if self.window.BackButton.isChecked():
            self.netmanager.machine_learning()
        else:
            self.netmanager.evolve_network()
            self.updateScreen()

    def updateLearningrate(self):
        lr = 0.001 + self.window.LearningSlider.value() * 0.01
        self.netmanager.update_learning_rate(lr)

    def resetNetwork(self):
        self.netmanager.reset_network()
        self.updateScreen()

    def clearMap(self):
        self.painter.reset_image()
        self.netmanager.clear_points()

    def updateScreen(self):
        self.painter.paint_scene(self.netmanager.network)
        self.painter.paint_points(self.netmanager.greenpoints, self.netmanager.redpoints)

    def run(self):
        self.window.show()
        sys.exit(self.app.exec_())
