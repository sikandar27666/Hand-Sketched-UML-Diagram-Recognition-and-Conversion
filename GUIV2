import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QColorDialog, QInputDialog, QFileDialog
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QPoint
import xml.etree.ElementTree as ET

class DrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.drawing = False
        self.shape_type = None
        self.last_point = QPoint()
        self.current_color = QColor(Qt.black)  # Default color is black
        self.shapes = []  # List to store drawn shapes

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Diagram Maker")

        # Button to draw a line
        self.line_button = QPushButton("Line", self)
        self.line_button.setGeometry(10, 10, 60, 30)
        self.line_button.clicked.connect(lambda: self.set_shape("line"))

        # Button to draw a rectangle
        self.rect_button = QPushButton("Rectangle", self)
        self.rect_button.setGeometry(80, 10, 80, 30)
        self.rect_button.clicked.connect(lambda: self.set_shape("rectangle"))

        # Button to draw an oval
        self.oval_button = QPushButton("Oval", self)
        self.oval_button.setGeometry(170, 10, 60, 30)
        self.oval_button.clicked.connect(lambda: self.set_shape("oval"))

        # Button to add text
        self.text_button = QPushButton("Text", self)
        self.text_button.setGeometry(240, 10, 60, 30)
        self.text_button.clicked.connect(self.add_text)

        # Button to choose color
        self.color_button = QPushButton("Choose Color", self)
        self.color_button.setGeometry(310, 10, 100, 30)
        self.color_button.clicked.connect(self.choose_color)

        # Button to save as XML
        self.save_button = QPushButton("Save as XML", self)
        self.save_button.setGeometry(420, 10, 100, 30)
        self.save_button.clicked.connect(self.save_as_xml)

        # Button to load from XML
        self.load_button = QPushButton("Load from XML", self)
        self.load_button.setGeometry(530, 10, 100, 30)
        self.load_button.clicked.connect(self.load_from_xml)

        # Button to clear the board
        self.clear_button = QPushButton("Clear Board", self)
        self.clear_button.setGeometry(640, 10, 100, 30)
        self.clear_button.clicked.connect(self.clear_board)

        # Button to quit the application
        self.quit_button = QPushButton("Quit", self)
        self.quit_button.setGeometry(750, 10, 60, 30)
        self.quit_button.clicked.connect(self.quit_app)

    def set_shape(self, shape):
        self.shape_type = shape

    def choose_color(self):
        # Open a color dialog to select a color
        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color

    def add_text(self):
        # Open a dialog to input text
        text, ok = QInputDialog.getText(self, 'Input Text', 'Enter text:')
        if ok and text:
            # Store the text and its location to be drawn later
            self.shapes.append(("text", text, self.last_point, self.current_color))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing and self.shape_type:
            self.shapes.append((self.shape_type, self.last_point, event.pos(), self.current_color))
            self.update()  # Update the canvas

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)

        # Iterate through all stored shapes and draw them
        for shape in self.shapes:
            if shape[0] == "line":
                painter.setPen(QPen(shape[3], 2, Qt.SolidLine))
                painter.drawLine(shape[1], shape[2])
            elif shape[0] == "rectangle":
                painter.setPen(QPen(shape[3], 2, Qt.SolidLine))
                painter.drawRect(min(shape[1].x(), shape[2].x()), min(shape[1].y(), shape[2].y()),
                                 abs(shape[1].x() - shape[2].x()), abs(shape[1].y() - shape[2].y()))
            elif shape[0] == "oval":
                painter.setPen(QPen(shape[3], 2, Qt.SolidLine))
                painter.drawEllipse(min(shape[1].x(), shape[2].x()), min(shape[1].y(), shape[2].y()),
                                    abs(shape[1].x() - shape[2].x()), abs(shape[1].y() - shape[2].y()))
            elif shape[0] == "text":
                painter.setPen(QPen(shape[3], 2, Qt.SolidLine))
                painter.drawText(shape[2], shape[1])

    def save_as_xml(self):
        """Save all shapes to an XML file."""
        filename, _ = QFileDialog.getSaveFileName(self, "Save as XML", "", "XML Files (*.xml)")
        if filename:
            root = ET.Element("drawing")

            for shape in self.shapes:
                shape_elem = ET.SubElement(root, "shape")
                shape_elem.set("type", shape[0])

                if shape[0] == "text":
                    shape_elem.set("content", shape[1])
                    shape_elem.set("x", str(shape[2].x()))
                    shape_elem.set("y", str(shape[2].y()))
                else:
                    shape_elem.set("start_x", str(shape[1].x()))
                    shape_elem.set("start_y", str(shape[1].y()))
                    shape_elem.set("end_x", str(shape[2].x()))
                    shape_elem.set("end_y", str(shape[2].y()))

                shape_elem.set("color", shape[3].name())  # Save color

            tree = ET.ElementTree(root)
            tree.write(filename)

    def load_from_xml(self):
        """Load shapes from an XML file and reconstruct the drawing."""
        filename, _ = QFileDialog.getOpenFileName(self, "Open XML", "", "XML Files (*.xml)")
        if filename:
            tree = ET.parse(filename)
            root = tree.getroot()

            self.shapes.clear()

            for shape_elem in root.findall("shape"):
                shape_type = shape_elem.get("type")
                color = QColor(shape_elem.get("color"))

                if shape_type == "text":
                    content = shape_elem.get("content")
                    x = int(shape_elem.get("x"))
                    y = int(shape_elem.get("y"))
                    point = QPoint(x, y)
                    self.shapes.append((shape_type, content, point, color))
                else:
                    start_x = int(shape_elem.get("start_x"))
                    start_y = int(shape_elem.get("start_y"))
                    end_x = int(shape_elem.get("end_x"))
                    end_y = int(shape_elem.get("end_y"))
                    start_point = QPoint(start_x, start_y)
                    end_point = QPoint(end_x, end_y)
                    self.shapes.append((shape_type, start_point, end_point, color))

            self.update()  # Refresh the canvas with loaded shapes

    def clear_board(self):
        """Clear the canvas by removing all shapes."""
        self.shapes.clear()  # Clear the list of shapes
        self.update()  # Refresh the canvas

    def quit_app(self):
        """Quit the application."""
        self.close()  # Close the application window


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec_())
