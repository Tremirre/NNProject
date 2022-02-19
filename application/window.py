from PyQt5 import uic
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMainWindow


class Window(QMainWindow):
    def __init__(self, parent_app):
        super(Window, self).__init__()
        uic.loadUi("application/resource/Image.ui", self)
        self.setFixedSize(1000, 700)
        self.setWindowTitle("Neural Network Tester")
        self.setWindowIcon(QIcon("resource/icon.ico"))
        self.BackButton.setChecked(True)
        self.parent = parent_app
        self.connect_elements()

    def mousePressEvent(self, event):
        point = QPoint(event.x() - 300, event.y())
        if 300 <= event.x() <= 1000 and 0 <= event.y() <= 700:
            if event.button() == 1:
                self.parent.netmanager.redpoints.append(point)
                self.parent.painter.paint_red(point)
            elif event.button() == 4:
                print(self.parent.netmanager.network.calculate([(point.x() - 350) / 700, (point.y() - 350) / 700]))
                print(point.x(), point.y())
            else:
                self.parent.netmanager.greenpoints.append(point)
                self.parent.painter.paint_green(point)
        if self.parent.window.AutoButton.isChecked():
            self.parent.train_network()

    def connect_elements(self):
        self.ClearButton.clicked.connect(self.parent.clear_map)
        self.LearningSlider.sliderMoved.connect(self.parent.update_learning_rate)
        self.PredictButton.clicked.connect(self.parent.train_network)
        self.ResetButton.clicked.connect(self.parent.reset_network)
        self.CustomizeButton.clicked.connect(self.initialize_settings)

    def initialize_settings(self):
        self.parent.settings_window.import_network()
        self.parent.settings_window.show()
