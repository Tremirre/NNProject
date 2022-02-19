from PyQt5.QtCore import *
from PyQt5.QtGui import QPainter, QPen, QColor


class Painter:
    def __init__(self, image, scene):
        self.image = image
        self.scene = scene
        self.reset_image()
        self.resolution = 4

    def reset_image(self):
        self.scene.fill(Qt.white)
        self.image.setPixmap(self.scene)

    def paint_red(self, point):
        qp = QPainter()
        qp.begin(self.scene)
        qp.setBrush(Qt.darkRed)
        qp.drawEllipse(point, 3, 3)
        qp.end()
        self.image.setPixmap(self.scene)

    def paint_green(self, point):
        qp = QPainter()
        qp.begin(self.scene)
        qp.setBrush(Qt.darkGreen)
        qp.drawRoundedRect(point.x(), point.y(), 6, 6, 1, 1)
        qp.end()
        self.image.setPixmap(self.scene)

    def paint_points(self, greenpoints, redpoints):
        qp = QPainter()
        qp.begin(self.scene)
        qp.setBrush(Qt.darkGreen)
        for point in greenpoints:
            qp.drawRoundedRect(point.x(), point.y(), 6, 6, 1, 1)
        qp.setBrush(Qt.darkRed)
        for point in redpoints:
            qp.drawEllipse(point, 3, 3)
        qp.end()
        self.image.setPixmap(self.scene)

    def paint_scene(self, network):
        self.scene.fill(QColor(100, 255, 100))
        qp = QPainter()
        qp.begin(self.scene)
        qp.setPen(QPen(QColor(255, 100, 100), self.resolution, Qt.SolidLine))
        for x in range(-50, 710, self.resolution):
            for y in range(-50, 710, self.resolution):
                r = network.calculate([(x - 350) / 700, (y - 350) / 700])
                if r < 0.5:
                    qp.drawPoint(QPoint(x, y))
        qp.end()
