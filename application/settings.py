from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget, QLabel, QSpinBox, QComboBox, QPushButton
from ann.neural import Network
from application.settings_painter import SettingsPainter


class MenuTitle:
    def __init__(self, parent_window, x, y):
        self.x = x
        self.y = y
        self.label1 = QLabel("Layer name: ", parent_window)
        self.label2 = QLabel("Layer size: ", parent_window)
        self.label3 = QLabel("Layer function: ", parent_window)
        self.setup_elements()

    def setup_elements(self):
        self.label1.move(self.x + 3, self.y)
        self.label2.move(self.x + 80, self.y)
        self.label3.move(self.x + 140, self.y)


class NeuralMenuSegment:
    def __init__(self, parent_window, x, y, text):
        self.x = x
        self.y = y
        self.text = text
        self.label = QLabel(text, parent_window)
        self.spin_box = QSpinBox(parent_window)
        self.spin_box.valueChanged.connect(parent_window.update_image)
        if self.text != "InputLayer":
            self.combo_box = QComboBox(parent_window)
            self.combo_box.addItems(["Sigmoid", "Linear", "ReLU", "Protected Tanh"])
            self.combo_box.currentIndexChanged.connect(parent_window.update_image)
        self.setup_elements()

    def setup_elements(self):
        self.label.move(self.x + 3, self.y)
        self.label.resize(73, 20)
        self.spin_box.move(self.x + 80, self.y)
        self.spin_box.resize(54, 20)
        self.spin_box.setMinimum(1)
        if self.text == "OutputLayer":
            self.spin_box.setMaximum(1)
            self.spin_box.setDisabled(True)
            self.combo_box.setCurrentText("Sigmoid")
            self.combo_box.setDisabled(True)
        if self.text == "InputLayer":
            self.spin_box.setMinimum(2)
            self.spin_box.setMaximum(2)
            self.spin_box.setDisabled(True)
        else:
            self.spin_box.setMaximum(10)
            self.combo_box.move(self.x + 140, self.y)
            self.combo_box.resize(100, 20)
            self.combo_box.show()
        self.label.show()
        self.spin_box.show()

    def shift_menu_segment(self, offset):
        self.y += offset
        self.setup_elements()

    def destroy(self):
        self.label.deleteLater()
        self.spin_box.deleteLater()
        if self.text != "InputLayer":
            self.combo_box.deleteLater()

    def edit_segment(self, value, boxtype):
        self.spin_box.setValue(value)
        self.combo_box.setCurrentText(boxtype)
        self.combo_box.show()


class NeuralMenu:
    def __init__(self, parent_window):
        self.sx = 500
        self.sy = 5
        self.parent = parent_window
        self.title = MenuTitle(self.parent, self.sx, self.sy)
        self.addButton = QPushButton("+", self.parent)
        self.minusButton = QPushButton("-", self.parent)
        self.addButton.resize(25, 25)
        self.minusButton.resize(25, 25)
        self.addButton.move(self.sx + 40, self.sy + 100)
        self.minusButton.move(self.sx + 175, self.sy + 100)
        self.menus = [NeuralMenuSegment(self.parent, self.sx, self.sy + 30, "InputLayer"),
                      NeuralMenuSegment(self.parent, self.sx, self.sy + 55, "OutputLayer")]
        self.addButton.clicked.connect(lambda: self.add_layer(True))
        self.minusButton.clicked.connect(lambda: self.remove_layer(True))
        self.minusButton.setDisabled(True)

    def move_elements(self, direction):
        self.menus[-1].shift_menu_segment(direction * 25)
        self.addButton.move(self.sx + 40, self.menus[-1].y + 45)
        self.minusButton.move(self.sx + 175, self.menus[-1].y + 45)

    def add_layer(self, vizupdate, hiddenlayer = None, function = None):
        if len(self.menus) < 10:
            self.move_elements(1)
            newlayer = NeuralMenuSegment(self.parent, self.sx, self.menus[-1].y - 25,
                                         "HiddenLayer {}".format(len(self.menus) - 1))
            if hiddenlayer and function:
                newlayer.edit_segment(hiddenlayer, function)
            self.menus.insert(-1, newlayer)
            self.minusButton.setDisabled(False)
            if len(self.menus) == 10:
                self.addButton.setDisabled(True)
            if vizupdate:
                self.parent.update_image()

    def remove_layer(self, vizupdate):
        if len(self.menus) > 2:
            self.move_elements(-1)
            self.menus[-2].destroy()
            self.menus.remove(self.menus[-2])
            self.addButton.setDisabled(False)
            if len(self.menus) == 2:
                self.minusButton.setDisabled(True)
            if vizupdate:
                self.parent.update_image()

    def reset_layers(self):
        self.minusButton.setDisabled(True)
        self.addButton.setDisabled(False)
        for layer in self.menus[1:-1]:
            layer.destroy()
            self.menus.remove(layer)
        self.menus[-1].shift_menu_segment(60 - self.menus[-1].y)


class SettingsWindow(QWidget):
    def __init__(self, parent_app):
        super(SettingsWindow, self).__init__()
        self.settingspainter = None
        self.parent = parent_app
        self.setFixedSize(768, 768)
        self.setWindowIcon(QIcon("resource/icon.ico"))
        self.setWindowTitle("Settings")
        self.image = QLabel(self)
        self.menu = NeuralMenu(self)
        self.OKButton = QPushButton("Ok", self)
        self.CancelButton = QPushButton("Cancel", self)
        self.CancelButton.clicked.connect(self.close)
        self.OKButton.move(520, 700)
        self.CancelButton.move(630, 700)
        self.OKButton.resize(90, 20)
        self.CancelButton.resize(90, 20)
        self.OKButton.clicked.connect(self.update_network)
        self.settingspainter = SettingsPainter(self.image, QPixmap(480, 768), self.menu.menus)

    def import_network(self):
        self.menu.reset_layers()
        network = self.parent.netmanager.network
        hiddenlayers = []
        functions = network.functions
        for layer in network.weights[:-1]:
            hiddenlayers.append(len(layer))
        for hiddenlayer, function in zip(hiddenlayers, functions):
            self.menu.add_layer(False, hiddenlayer, function)
        self.update_image()

    def update_network(self):
        convert = {
            "Sigmoid": "sig",
            "Linear": "lin",
            "ReLU": "ReLU",
            "Protected Tanh": "tanh"
        }
        newfunctions = []
        newlayers = []
        for layer in self.menu.menus:
            if len(newlayers):
                newfunctions.append(convert[layer.combo_box.currentText()])
            newlayers.append(layer.spin_box.value())
        self.parent.netmanager.network = Network(newlayers, 0.045, newfunctions)
        self.close()

    def update_image(self):
        if self.settingspainter:
            self.settingspainter.redraw_image(self.menu.menus)

    def update_buttons_state(self):
        ...