import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Function to preprocess the image (adaptive thresholding and edge detection)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    # Adaptive threshold to emphasize shapes
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# Function to detect shapes in an image
def detect_shapes(image):
    processed = preprocess_image(image)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []

    for contour in contours:
        shape = {}
        area = cv2.contourArea(contour)
        if area < 500:  # Adjust the area threshold to detect valid shapes
            continue

        # Polygon approximation with a more flexible epsilon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)  # More flexible approximation

        # Check for rectangles or squares
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            angle_check = check_angle(approx)

            if 0.9 <= aspect_ratio <= 1.1 and angle_check:  # Square
                shape["type"] = "square"
            elif angle_check:  # Rectangle
                shape["type"] = "rectangle"
            shape["position"] = {"x": x, "y": y, "width": w, "height": h}

        # Check for ellipses (exclude false circles based on circularity)
        elif len(approx) > 5 and is_circle(contour):
            ((x, y), r) = cv2.minEnclosingCircle(contour)
            shape["type"] = "ellipse"
            shape["position"] = {"center_x": int(x), "center_y": int(y), "radius": int(r)}

        if "type" in shape:
            shapes.append(shape)

    # Add line detection using Hough Transform
    lines = detect_lines(processed)
    for line in lines:
        shape = {"type": "line", "position": line}
        shapes.append(shape)

    return shapes

# Function to check if the contour is circular enough to be considered an ellipse/circle
def is_circle(contour):
    # Calculate the circularity of the contour
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return 0.7 <= circularity <= 1.2  # More strict circularity check to prevent false circles

# Function to check if a quadrilateral has 90-degree angles (rectangular)
def check_angle(approx):
    for i in range(4):
        p1 = approx[i][0]
        p2 = approx[(i+1) % 4][0]
        p3 = approx[(i+2) % 4][0]
        angle = calculate_angle(p1, p2, p3)
        if not (80 <= angle <= 100):  # 90-degree tolerance
            return False
    return True

# Helper function to calculate angle between three points
def calculate_angle(p1, p2, p3):
    vec1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    vec2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return np.degrees(angle)

# Function to detect lines using Hough Transform
def detect_lines(processed_image):
    lines = cv2.HoughLinesP(processed_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    line_positions = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                line_positions.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return line_positions

# Function to generate Draw.io compatible XML
def generate_drawio_xml(shapes):
    # Root element of the mxfile
    root = ET.Element("mxfile")
    diagram = ET.SubElement(root, "diagram", attrib={"name": "Page-1"})

    # mxGraphModel for the diagram
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", attrib={
        "dx": "1485", "dy": "2120", "grid": "1", "gridSize": "10", "guides": "1",
        "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1",
        "pageScale": "1", "pageWidth": "850", "pageHeight": "1100", "math": "0"
    })

    root_cell = ET.SubElement(mxGraphModel, "root")
    ET.SubElement(root_cell, "mxCell", id="0")
    ET.SubElement(root_cell, "mxCell", id="1", parent="0")

    for i, shape in enumerate(shapes):
        shape_id = str(i + 2)
        mxCell = ET.SubElement(root_cell, "mxCell", id=shape_id, parent="1", vertex="1")

        if shape["type"] == "rectangle" or shape["type"] == "square":
            style = "rounded=0;whiteSpace=wrap;html=1;" if shape[
                                                               "type"] == "rectangle" else "rounded=1;whiteSpace=wrap;html=1;"
            mxCell.set("style", style)
            mxGeometry = ET.SubElement(mxCell, "mxGeometry", as_="geometry")
            mxGeometry.set("x", str(shape["position"]["x"]))
            mxGeometry.set("y", str(shape["position"]["y"]))
            mxGeometry.set("width", str(shape["position"]["width"]))
            mxGeometry.set("height", str(shape["position"]["height"]))
            mxGeometry.set("as", "geometry")

        elif shape["type"] == "ellipse":
            mxCell.set("style", "ellipse;whiteSpace=wrap;html=1;")
            mxGeometry = ET.SubElement(mxCell, "mxGeometry", as_="geometry")
            mxGeometry.set("x", str(shape["position"]["center_x"] - shape["position"]["radius"]))
            mxGeometry.set("y", str(shape["position"]["center_y"] - shape["position"]["radius"]))
            mxGeometry.set("width", str(2 * shape["position"]["radius"]))
            mxGeometry.set("height", str(2 * shape["position"]["radius"]))
            mxGeometry.set("as", "geometry")

        elif shape["type"] == "line":
            mxCell.set("style", "edgeStyle=orthogonalEdgeStyle;orthogonalLoop=1;")
            mxGeometry = ET.SubElement(mxCell, "mxGeometry", as_="geometry")
            mxGeometry.set("x1", str(shape["position"]["x1"]))
            mxGeometry.set("y1", str(shape["position"]["y1"]))
            mxGeometry.set("x2", str(shape["position"]["x2"]))
            mxGeometry.set("y2", str(shape["position"]["y2"]))

    return ET.ElementTree(root)

# Main function to process the image and generate the XML
def process_image(image_path):
    image = cv2.imread(image_path)
    shapes = detect_shapes(image)

    # Generate Draw.io compatible XML
    xml_tree = generate_drawio_xml(shapes)

    # Save the XML file
    xml_tree.write("output_drawio.xml", encoding="UTF-8", xml_declaration=True)
    print("XML has been saved as output_drawio.xml")

    # Optional: Display shapes for verification
    for shape in shapes:
        if shape["type"] == "rectangle" or shape["type"] == "square":
            cv2.rectangle(image,
                          (shape["position"]["x"], shape["position"]["y"]),
                          (shape["position"]["x"] + shape["position"]["width"],
                           shape["position"]["y"] + shape["position"]["height"]),
                          (0, 255, 0), 2)
        elif shape["type"] == "ellipse":
            cv2.circle(image,
                       (shape["position"]["center_x"], shape["position"]["center_y"]),
                       shape["position"]["radius"], (255, 0, 0), 2)
        elif shape["type"] == "line":
            cv2.line(image,
                     (shape["position"]["x1"], shape["position"]["y1"]),
                     (shape["position"]["x2"], shape["position"]["y2"]),
                     (0, 0, 255), 2)

    cv2.imshow("Detected Shapes and Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the image path
image_path = "im6.jpg"
process_image(image_path)
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Function to preprocess the image (adaptive thresholding and edge detection)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    # Adaptive threshold to emphasize shapes
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# Function to detect shapes in an image
def detect_shapes(image):
    processed = preprocess_image(image)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []

    for contour in contours:
        shape = {}
        area = cv2.contourArea(contour)
        if area < 500:  # Adjust the area threshold to filter small noise
            continue

        # Polygon approximation with a flexible epsilon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)  # Flexible approximation

        if len(approx) == 4:  # Detecting rectangles, squares, and diamonds
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:  # Square
                shape["type"] = "square"
            else:
                shape["type"] = "rectangle"
            shape["position"] = {"x": x, "y": y, "width": w, "height": h}

        elif len(approx) > 5:  # Detecting ellipses/circles
            ((x, y), r) = cv2.minEnclosingCircle(contour)
            shape["type"] = "ellipse"
            shape["position"] = {"center_x": int(x), "center_y": int(y), "radius": int(r)}

        if "type" in shape:
            shapes.append(shape)

    return shapes

# Function to generate Draw.io compatible XML
import xml.etree.ElementTree as ET


def generate_drawio_xml(shapes):
    root = ET.Element("mxfile")
    diagram = ET.SubElement(root, "diagram", attrib={"name": "Page-1"})
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", attrib={
        "dx": "1485", "dy": "2120", "grid": "1", "gridSize": "10", "guides": "1",
        "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1",
        "pageScale": "1", "pageWidth": "850", "pageHeight": "1100", "math": "0"
    })
    root_cell = ET.SubElement(mxGraphModel, "root")
    ET.SubElement(root_cell, "mxCell", id="0")
    ET.SubElement(root_cell, "mxCell", id="1", parent="0")

    for i, shape in enumerate(shapes):
        shape_id = str(i + 2)
        mxCell = ET.SubElement(root_cell, "mxCell", id=shape_id, parent="1")

        if shape["type"] in ["rectangle", "square"]:
            mxCell.set("style", "rounded=0;whiteSpace=wrap;html=1;" if shape[
                                                                           "type"] == "rectangle" else "rounded=1;whiteSpace=wrap;html=1;")
            mxCell.set("vertex", "1")
            mxGeometry = ET.SubElement(mxCell, "mxGeometry", attrib={
                "x": str(shape["position"]["x"]),
                "y": str(shape["position"]["y"]),
                "width": str(shape["position"]["width"]),
                "height": str(shape["position"]["height"]),
                "as": "geometry"
            })
        elif shape["type"] == "ellipse":
            mxCell.set("style", "ellipse;whiteSpace=wrap;html=1;")
            mxCell.set("vertex", "1")
            mxGeometry = ET.SubElement(mxCell, "mxGeometry", attrib={
                "x": str(shape["position"]["center_x"] - shape["position"]["radius"]),
                "y": str(shape["position"]["center_y"] - shape["position"]["radius"]),
                "width": str(2 * shape["position"]["radius"]),
                "height": str(2 * shape["position"]["radius"]),
                "as": "geometry"
            })
        elif shape["type"] == "line":
            mxCell.set("style", "edgeStyle=orthogonalEdgeStyle;orthogonalLoop=1;")
            mxCell.set("edge", "1")
            mxGeometry = ET.SubElement(mxCell, "mxGeometry", attrib={"relative": "1", "as": "geometry"})

    return ET.ElementTree(root)


# Main function to process the image and generate the XML
def process_image(image_path):
    image = cv2.imread(image_path)
    shapes = detect_shapes(image)

    # Generate Draw.io compatible XML
    xml_tree = generate_drawio_xml(shapes)

    # Save the XML file
    xml_tree.write("output_drawio.xml", encoding="UTF-8", xml_declaration=True)
    print("XML has been saved as output_drawio.xml")

    # Optional: Display shapes for verification
    for shape in shapes:
        if shape["type"] == "rectangle" or shape["type"] == "square":
            cv2.rectangle(image,
                          (shape["position"]["x"], shape["position"]["y"]),
                          (shape["position"]["x"] + shape["position"]["width"],
                           shape["position"]["y"] + shape["position"]["height"]),
                          (0, 255, 0), 2)
        elif shape["type"] == "ellipse":
            cv2.circle(image,
                       (shape["position"]["center_x"], shape["position"]["center_y"]),
                       shape["position"]["radius"], (255, 0, 0), 2)

    cv2.imshow("Detected Shapes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the image path
image_path = "im6.jpg"
process_image(image_path)
