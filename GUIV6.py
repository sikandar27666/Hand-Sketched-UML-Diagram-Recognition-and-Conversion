import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QToolBar, QColorDialog, QInputDialog, QFileDialog
from PyQt5.QtGui import QPainter, QColor, QPen, QImage
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
import xml.etree.ElementTree as ET


class Shape:
    """Class to represent a shape with a type, position, and color."""
    def __init__(self, shape_type, start_point, end_point, color, text=''):
        self.shape_type = shape_type
        self.start_point = start_point
        self.end_point = end_point
        self.color = color
        self.text = text  # For text shapes only

    def bounding_rect(self):
        """Get the bounding rectangle of the shape for selection."""
        if self.shape_type == "text":
            # Adjust for text bounding box
            return QRect(self.start_point, QSize(len(self.text) * 10, 20))  # Approximate size for text
        return QRect(self.start_point, self.end_point).normalized()


class DrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.drawing = False
        self.shape_type = None
        self.last_point = QPoint()
        self.current_color = QColor(Qt.black)  # Default color is black
        self.shapes = []  # List to store drawn shapes
        self.current_shape = None  # Currently drawn shape
        self.selected_shape = None  # Shape selected for editing
        self.edit_mode = False  # Flag for edit mode

    def initUI(self):
        self.setGeometry(100, 100, 900, 600)  # Increased width to 900
        self.setWindowTitle("Diagram Maker")

        # Create a toolbar
        self.toolbar_left = QToolBar("Shapes", self)
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar_left)

        # Create a right toolbar for actions
        self.toolbar_right = QToolBar("Actions", self)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar_right)

        # Add shape buttons to the left toolbar
        self.add_tool_button(self.toolbar_left, "Line", "line", self.set_shape)
        self.add_tool_button(self.toolbar_left, "Rectangle", "rectangle", self.set_shape)
        self.add_tool_button(self.toolbar_left, "Oval", "oval", self.set_shape)
        self.add_tool_button(self.toolbar_left, "Diamond", "diamond", self.set_shape)
        self.add_tool_button(self.toolbar_left, "Text", None, self.add_text)
        self.add_tool_button(self.toolbar_left, "Eraser", None, self.eraser)

        # Add action buttons to the right toolbar
        self.add_tool_button(self.toolbar_right, "Choose Color", None, self.choose_color)
        self.add_tool_button(self.toolbar_right, "Save as XML", None, self.save_as_xml)
        self.add_tool_button(self.toolbar_right, "Load from XML", None, self.load_from_xml)
        self.add_tool_button(self.toolbar_right, "Clear Board", None, self.clear_board)
        self.add_tool_button(self.toolbar_right, "Save as PNG", None, self.save_as_png)
        self.add_tool_button(self.toolbar_right, "Edit Mode", None, self.toggle_edit_mode, toggle=True)
        self.add_tool_button(self.toolbar_right, "Quit", None, self.quit_app)

    def add_tool_button(self, toolbar, text, shape_type, slot, toggle=False):
        """Add a button to the toolbar."""
        action = QAction(text, self)
        action.triggered.connect(lambda: slot(shape_type) if shape_type else slot())
        if toggle:
            action.setCheckable(True)  # Make it toggleable for edit mode
        toolbar.addAction(action)

    def set_shape(self, shape):
        self.shape_type = shape

    def toggle_edit_mode(self):
        """Toggle edit mode on or off."""
        self.edit_mode = not self.edit_mode
        self.sender().setChecked(self.edit_mode)  # Update the button state

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color

    def add_text(self):
        text, ok = QInputDialog.getText(self, 'Input Text', 'Enter text:')
        if ok and text:
            self.shapes.append(Shape("text", self.last_point, self.last_point, self.current_color, text))

    def eraser(self):
        """Remove a selected shape."""
        if self.selected_shape:
            self.shapes.remove(self.selected_shape)
            self.selected_shape = None
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.edit_mode:
                # Check if a shape is selected
                for shape in self.shapes:
                    if shape.bounding_rect().contains(event.pos()):
                        self.selected_shape = shape
                        break
            else:
                self.drawing = True
                self.last_point = event.pos()
                self.current_shape = None

    def mouseMoveEvent(self, event):
        if self.drawing and self.shape_type:
            self.current_shape = Shape(self.shape_type, self.last_point, event.pos(), self.current_color)
            self.update()
        elif self.edit_mode and self.selected_shape:
            # Move the selected shape
            dx = event.pos().x() - self.last_point.x()
            dy = event.pos().y() - self.last_point.y()
            self.selected_shape.start_point += QPoint(dx, dy)
            self.selected_shape.end_point += QPoint(dx, dy)
            self.last_point = event.pos()  # Update last point for dragging
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.edit_mode and self.selected_shape:
                self.selected_shape = None  # Deselect shape after moving
            elif self.current_shape:
                self.drawing = False
                self.shapes.append(self.current_shape)
                self.current_shape = None
                self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        # Draw all finalized shapes
        for shape in self.shapes:
            self.draw_shape(painter, shape)

        # Draw the current shape being drawn
        if self.current_shape:
            self.draw_shape(painter, self.current_shape)

    def draw_shape(self, painter, shape):
        painter.setPen(QPen(shape.color, 2, Qt.SolidLine))
        if shape.shape_type == "line":
            painter.drawLine(shape.start_point, shape.end_point)
        elif shape.shape_type == "rectangle":
            painter.drawRect(QRect(shape.start_point, shape.end_point).normalized())
        elif shape.shape_type == "oval":
            painter.drawEllipse(QRect(shape.start_point, shape.end_point).normalized())
        elif shape.shape_type == "text":
            painter.drawText(shape.start_point, shape.text)
        elif shape.shape_type == "diamond":
            self.draw_diamond(painter, shape)

    def draw_diamond(self, painter, shape):
        center = (shape.start_point + shape.end_point) / 2
        size = QPoint(abs(shape.end_point.x() - shape.start_point.x()) / 2,
                      abs(shape.end_point.y() - shape.start_point.y()) / 2)
        points = [
            QPoint(center.x(), center.y() - size.y()),
            QPoint(center.x() - size.x(), center.y()),
            QPoint(center.x(), center.y() + size.y()),
            QPoint(center.x() + size.x(), center.y())
        ]
        painter.drawPolygon(*points)

    def save_as_xml(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save as XML", "", "XML Files (*.xml)")
        if filename:
            root = ET.Element("drawing")

            for shape in self.shapes:
                shape_elem = ET.SubElement(root, "shape")
                shape_elem.set("type", shape.shape_type)

                if shape.shape_type == "text":
                    shape_elem.set("content", shape.text)
                    shape_elem.set("x", str(shape.start_point.x()))
                    shape_elem.set("y", str(shape.start_point.y()))
                else:
                    shape_elem.set("start_x", str(shape.start_point.x()))
                    shape_elem.set("start_y", str(shape.start_point.y()))
                    shape_elem.set("end_x", str(shape.end_point.x()))
                    shape_elem.set("end_y", str(shape.end_point.y()))

                shape_elem.set("color", shape.color.name())

            tree = ET.ElementTree(root)
            tree.write(filename)

    def load_from_xml(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open XML", "", "XML Files (*.xml)")
        if filename:
            tree = ET.parse(filename)
            root = tree.getroot()

            # Determine the maximum x and y coordinates in the XML
            max_x = 0
            max_y = 0
            for shape_elem in root.findall("shape"):
                if shape_elem.tag == "shape":
                    if shape_elem.get("type") != "text":
                        end_x = int(shape_elem.get("end_x", 0))
                        end_y = int(shape_elem.get("end_y", 0))
                        max_x = max(max_x, end_x)
                        max_y = max(max_y, end_y)

            # Define canvas dimensions
            canvas_width = self.width()
            canvas_height = self.height()

            # Calculate scaling factors
            scale_x = canvas_width / max_x if max_x > 0 else 1
            scale_y = canvas_height / max_y if max_y > 0 else 1
            scale = min(scale_x, scale_y)  # Use the smaller scale factor to maintain aspect ratio

            self.shapes.clear()

            for shape_elem in root.findall("shape"):
                shape_type = shape_elem.get("type")
                color = QColor(shape_elem.get("color", "#000000"))  # Default to black if color is missing

                if shape_type == "text":
                    content = shape_elem.get("content")
                    x = int(shape_elem.get("x", 0))  # Default to 0 if x is missing
                    y = int(shape_elem.get("y", 0))  # Default to 0 if y is missing
                    point = QPoint(int(x * scale), int(y * scale))  # Scale the position
                    self.shapes.append(Shape(shape_type, point, point, color, content))
                else:
                    start_x = int(shape_elem.get("start_x", 0))  # Default to 0 if start_x is missing
                    start_y = int(shape_elem.get("start_y", 0))  # Default to 0 if start_y is missing
                    end_x = int(shape_elem.get("end_x", 0))  # Default to 0 if end_x is missing
                    end_y = int(shape_elem.get("end_y", 0))  # Default to 0 if end_y is missing
                    start_point = QPoint(int(start_x * scale), int(start_y * scale))  # Scale the start point
                    end_point = QPoint(int(end_x * scale), int(end_y * scale))  # Scale the end point
                    self.shapes.append(Shape(shape_type, start_point, end_point, color))

            self.update()
    def clear_board(self):
        self.shapes.clear()
        self.update()

    def quit_app(self):
        self.close()

    def save_as_png(self):
        """Save the current drawing as a PNG file."""
        filename, _ = QFileDialog.getSaveFileName(self, "Save as PNG", "", "PNG Files (*.png)")
        if filename:
            # Create an image to draw on
            image = QImage(self.size(), QImage.Format_ARGB32)
            image.fill(Qt.white)  # Fill the image with white background

            painter = QPainter(image)
            # Draw all finalized shapes onto the image
            for shape in self.shapes:
                self.draw_shape(painter, shape)
            painter.end()

            # Save the image as PNG
            image.save(filename)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec_())
