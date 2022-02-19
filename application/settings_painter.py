from PyQt5.QtCore import *
from PyQt5.QtGui import QPainter, QPen

MID_X = 240
MID_Y = 384


class Neuron:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 12

    def draw_neuron(self, qp):
        qp.drawEllipse(QPoint(self.x, self.y), self.radius, self.radius)


class Layer:
    def __init__(self, size, y, function):
        self.y = y
        self.size = size
        self.neurons = self.deploy_neurons()
        self.function = function

    def deploy_neurons(self):
        neurons = []
        if self.size % 2 == 0:
            start_x = MID_X - (self.size - 1) * 17
        else:
            start_x = MID_X - (self.size // 2) * 34
        for i in range(self.size):
            neurons.append(Neuron(start_x, self.y))
            start_x += 34
        return neurons

    def draw_layer(self, qp):
        for neuron in self.neurons:
            neuron.draw_neuron(qp)


class VizNetwork:
    def __init__(self, menu):
        self.layers = None
        self.import_network(menu)

    def import_network(self, menu):
        layers = []
        if len(menu) % 2 == 0:
            start_y = MID_Y - (len(menu) - 1) * 22
        else:
            start_y = MID_Y - (len(menu) // 2) * 48
        layers.append(Layer(menu[0].spin_box.value(), start_y, None))
        start_y += 48
        for layer in menu[1:]:
            layers.append(Layer(layer.spin_box.value(), start_y, layer.combo_box.currentText()))
            start_y += 48
        self.layers = layers

    def draw_network(self, scene):
        colors = {
            "Sigmoid": Qt.darkRed,
            "Linear": Qt.blue,
            "ReLU": Qt.darkCyan,
            "Protected Tanh": Qt.darkGreen
        }
        qp = QPainter()
        qp.begin(scene)
        qp.setBrush(Qt.white)
        for layer in self.layers:
            layer.draw_layer(qp)
        for i in range(len(self.layers) - 1):
            qp.setPen(QPen(colors[self.layers[i+1].function], 1, Qt.SolidLine))
            for preneuron in self.layers[i].neurons:
                for nextneuron in self.layers[i+1].neurons:
                    qp.drawLine(preneuron.x, preneuron.y, nextneuron.x, nextneuron.y)
        qp.end()


class SettingsPainter:
    def __init__(self, image, scene, menu):
        self.image = image
        self.scene = scene
        self.reset_image()
        self.viznetwork = VizNetwork(menu)
        self.viznetwork.draw_network(self.scene)
        self.image.setPixmap(self.scene)

    def reset_image(self):
        self.scene.fill(Qt.black)
        self.image.setPixmap(self.scene)

    def redraw_image(self, menu):
        self.reset_image()
        self.viznetwork = VizNetwork(menu)
        self.viznetwork.draw_network(self.scene)
        self.image.setPixmap(self.scene)
