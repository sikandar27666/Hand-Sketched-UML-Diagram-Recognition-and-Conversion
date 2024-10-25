import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Function to detect shapes and lines
def detect_shapes_and_lines(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # edge detection
    edged = cv2.Canny(blurred, 50, 150)  # Edge detection

    #contour detection algorithm
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []

    # Detect shapes (rectangles, diamonds, circles)
    for contour in contours:
        shape = {}
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # Detect squares and rectangles (4 vertices)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            shape_type = "rectangle"
            shape["type"] = shape_type
            shape["position"] = {"x": x, "y": y, "width": w, "height": h}

        # Detect diamonds (rhombus)
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            if aspect_ratio < 0.95:  # More squished aspect ratios
                shape["type"] = "diamond"
                shape["position"] = {"x": x, "y": y, "width": w, "height": h}

        # Detect circles
        elif len(approx) > 8:
            shape["type"] = "ellipse"
            (x, y, r) = cv2.minEnclosingCircle(contour)
            shape["position"] = {"center_x": int(x), "center_y": int(y), "radius": int(r)}

        if "type" in shape:
            shapes.append(shape)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                shapes.append({"type": "line", "position": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}})

    return shapes


# Function to generate XML in a format Draw.io can interpret
def generate_xml(shapes):

    root = ET.Element("mxfile")
    diagram = ET.SubElement(root, "diagram", name="UMLDiagram")
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel")
    root_cell = ET.SubElement(mxGraphModel, "root")

    # Basic XML elements required by Draw.io
    ET.SubElement(root_cell, "mxCell", id="0")
    ET.SubElement(root_cell, "mxCell", id="1", parent="0")

    for i, shape in enumerate(shapes):
        shape_id = str(i + 2)  # Ensure unique IDs for each shape
        shape_element = ET.SubElement(root_cell, "mxCell", id=shape_id, parent="1", style="")

        if shape["type"] == "rectangle":
            style = "shape=rectangle"
            x = shape["position"]["x"]
            y = shape["position"]["y"]
            width = shape["position"]["width"]
            height = shape["position"]["height"]
            shape_element.set("style", style)
            ET.SubElement(shape_element, "mxGeometry", x=str(x), y=str(y), width=str(width), height=str(height),
                          as_="geometry")

        elif shape["type"] == "diamond":
            style = "shape=rhombus"
            x = shape["position"]["x"]
            y = shape["position"]["y"]
            width = shape["position"]["width"]
            height = shape["position"]["height"]
            shape_element.set("style", style)
            ET.SubElement(shape_element, "mxGeometry", x=str(x), y=str(y), width=str(width), height=str(height),
                          as_="geometry")

        elif shape["type"] == "ellipse":
            style = "shape=ellipse"
            center_x = shape["position"]["center_x"]
            center_y = shape["position"]["center_y"]
            radius = shape["position"]["radius"]
            shape_element.set("style", style)
            ET.SubElement(shape_element, "mxGeometry", x=str(center_x - radius), y=str(center_y - radius),
                          width=str(2 * radius), height=str(2 * radius), as_="geometry")

        elif shape["type"] == "line":
            style = "edgeStyle=orthogonalEdgeStyle;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;"
            x1 = shape["position"]["x1"]
            y1 = shape["position"]["y1"]
            x2 = shape["position"]["x2"]
            y2 = shape["position"]["y2"]
            shape_element.set("style", style)
            mxGeometry = ET.SubElement(shape_element, "mxGeometry", as_="geometry")
            mxGeometry.set("relative", "1")
            mxPoint = ET.SubElement(mxGeometry, "mxPoint", x=str(x1), y=str(y1))
            ET.SubElement(mxPoint, "mxPoint", x=str(x2), y=str(y2))

    tree = ET.ElementTree(root)
    return tree


# Main function
def process_image(image_path):
    # read the image file
    image = cv2.imread(image_path)
    shapes = detect_shapes_and_lines(image)

    # Generate XML in Draw.io-compatible format
    xml_tree = generate_xml(shapes)

    # Save XML to file
    xml_tree.write("output_drawio.xml")
    print("XML has been saved as output_drawio.xml")

    # Optional: Display shapes on the original image
    for shape in shapes:
        if shape["type"] == "rectangle":
            cv2.rectangle(image,
                          (shape["position"]["x"], shape["position"]["y"]),
                          (shape["position"]["x"] + shape["position"]["width"],
                           shape["position"]["y"] + shape["position"]["height"]),
                          (0, 255, 0), 2)
        elif shape["type"] == "diamond":
            cv2.rectangle(image,
                          (shape["position"]["x"], shape["position"]["y"]),
                          (shape["position"]["x"] + shape["position"]["width"],
                           shape["position"]["y"] + shape["position"]["height"]),
                          (255, 0, 255), 2)  # Magenta for diamond
        elif shape["type"] == "ellipse":
            cv2.circle(image,
                       (shape["position"]["center_x"], shape["position"]["center_y"]),
                       shape["position"]["radius"],
                       (255, 0, 0), 2)  # Blue for ellipse
        elif shape["type"] == "line":
            cv2.line(image,
                     (shape["position"]["x1"], shape["position"]["y1"]),
                     (shape["position"]["x2"], shape["position"]["y2"]),
                     (0, 0, 255), 2)  # Red for lines

    # Show the original image with shapes and lines overlaid
    cv2.imshow("Shapes and Lines Detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



image_path = "imgTest.jpeg"  # Change to your UML diagram image path
process_image(image_path)
