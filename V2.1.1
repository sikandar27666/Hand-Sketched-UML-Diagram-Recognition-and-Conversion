import cv2
import numpy as np
import xml.etree.ElementTree as ET


# Function to detect shapes
def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Blur the image to reduce noise
    edged = cv2.Canny(blurred, 30, 150)  # Edge detection with adjusted thresholds

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []

    for contour in contours:
        shape = {}
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # Detect squares and rectangles (4 vertices)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            # Use aspect ratio and additional check for pill shape (rounded rectangle)
            if 0.95 <= aspect_ratio <= 1.05:
                shape["type"] = "square"  # If aspect ratio ~1, it's a square
            else:
                shape["type"] = "rectangle"
            shape["position"] = {"x": x, "y": y, "width": w, "height": h}

        # Detect rhombus (diamonds)
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            if 0.75 <= aspect_ratio <= 0.95:  # More squished aspect ratios indicate diamonds
                shape["type"] = "diamond"
                shape["position"] = {"x": x, "y": y, "width": w, "height": h}

        # Detect circles and ellipses
        elif len(approx) > 8:
            shape["type"] = "ellipse"  # Treat circles as ellipses
            ((x, y), r) = cv2.minEnclosingCircle(contour)
            shape["position"] = {"center_x": int(x), "center_y": int(y), "radius": int(r)}

        # Detect lines using Hough Line Transform
        if len(approx) == 2:
            lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
            if lines is not None:
                shape["type"] = "line"
                shape["lines"] = lines

        # Detect arrowheads by looking for small triangles
        if len(approx) == 3:
            shape["type"] = "arrowhead"
            (x, y, w, h) = cv2.boundingRect(approx)
            shape["position"] = {"x": x, "y": y, "width": w, "height": h}

        # Only append if the shape has a 'type' key
        if "type" in shape:
            shapes.append(shape)

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

        if shape["type"] == "rectangle" or shape["type"] == "square":
            style = "shape=rectangle"
            x = shape["position"]["x"]
            y = shape["position"]["y"]
            width = shape["position"]["width"]
            height = shape["position"]["height"]
            shape_element.set("style", style)
            ET.SubElement(shape_element, "mxGeometry", x=str(x), y=str(y), width=str(width), height=str(height), as_="geometry")

        elif shape["type"] == "diamond":
            style = "shape=rhombus"
            x = shape["position"]["x"]
            y = shape["position"]["y"]
            width = shape["position"]["width"]
            height = shape["position"]["height"]
            shape_element.set("style", style)
            ET.SubElement(shape_element, "mxGeometry", x=str(x), y=str(y), width=str(width), height=str(height), as_="geometry")

        elif shape["type"] == "ellipse":
            style = "shape=ellipse"
            center_x = shape["position"]["center_x"]
            center_y = shape["position"]["center_y"]
            radius = shape["position"]["radius"]
            shape_element.set("style", style)
            ET.SubElement(shape_element, "mxGeometry", x=str(center_x - radius), y=str(center_y - radius),
                          width=str(2 * radius), height=str(2 * radius), as_="geometry")

        elif shape["type"] == "line":
            for line in shape["lines"]:
                for x1, y1, x2, y2 in line:
                    style = "shape=line"
                    shape_element.set("style", style)
                    ET.SubElement(shape_element, "mxGeometry", x=str(x1), y=str(y1), width=str(x2 - x1), height=str(y2 - y1), as_="geometry")

        elif shape["type"] == "arrowhead":
            style = "shape=triangle"
            x = shape["position"]["x"]
            y = shape["position"]["y"]
            width = shape["position"]["width"]
            height = shape["position"]["height"]
            shape_element.set("style", style)
            ET.SubElement(shape_element, "mxGeometry", x=str(x), y=str(y), width=str(width), height=str(height), as_="geometry")

    tree = ET.ElementTree(root)
    return tree


# Main function
def process_image(image_path):
    image = cv2.imread(image_path)
    shapes = detect_shapes(image)

    # Generate XML in Draw.io-compatible format
    xml_tree = generate_xml(shapes)

    # Save XML to file
    xml_tree.write("output_drawio.xml")
    print("XML has been saved as output_drawio.xml")

    # Optional: Display shapes on the original image
    for shape in shapes:
        if shape["type"] == "rectangle" or shape["type"] == "square":
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
            for line in shape["lines"]:
                for x1, y1, x2, y2 in line:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for line

    # Show the original image with shapes overlayed
    cv2.imshow("Shapes Detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Provide the image path
image_path = "C:/Users/MOON/Desktop/03.jpeg"  # Change to your UML diagram image path
process_image(image_path)
