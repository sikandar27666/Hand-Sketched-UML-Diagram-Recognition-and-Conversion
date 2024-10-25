import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
import os

def detect_lines_and_rectangles(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using the Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    line_segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            line_segments.append(((x1, y1), (x2, y2)))

    # Detect rectangles using contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            cv2.drawContours(image, [approx], 0, (255, 0, 0), 3)
            x, y, w, h = cv2.boundingRect(approx)
            rectangles.append((x, y, w, h))

    return image, line_segments, rectangles

def write_drawio_xml(rectangles, line_segments, output_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mxfile = Element("mxfile")
    diagram = SubElement(mxfile, "diagram", name="Page-1")
    root = SubElement(diagram, "mxGraphModel")

    root_cell = SubElement(root, "root")
    mx_cell = SubElement(root_cell, "mxCell", id="0")
    mx_cell1 = SubElement(root_cell, "mxCell", id="1", parent="0")

    for i, (x, y, w, h) in enumerate(rectangles):
        cell_id = str(i + 2)
        cell = SubElement(root_cell, "mxCell", id=cell_id, value="", style="rounded=0;whiteSpace=wrap;html=1;",
                          vertex="1", parent="1")
        geometry = SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), as_="geometry")

    for j, ((x1, y1), (x2, y2)) in enumerate(line_segments):
        edge_id = str(len(rectangles) + j + 2)
        edge = SubElement(root_cell, "mxCell", id=edge_id, edge="1", parent="1",
                          style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;")
        geometry = SubElement(edge, "mxGeometry", relative="1", as_="geometry")
        points = SubElement(geometry, "Array", as_="points")
        point = SubElement(points, "mxPoint", x=str((x1 + x2) // 2), y=str((y1 + y2) // 2))

    tree = ElementTree(mxfile)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

if __name__ == "__main__":
    image_path = '11.jpeg'
    output_dir = 'assets'
    output_path = os.path.join(output_dir, 'sequence_diagram.xml')

    # Detect lines and rectangles
    image, line_segments, rectangles = detect_lines_and_rectangles(image_path)

    # Write the detected diagram to XML
    write_drawio_xml(rectangles, line_segments, output_path)

    # Display the processed image
    cv2.imshow('Detected Lines and Rectangles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
