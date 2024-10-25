import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Function to preprocess the image (Canny edge detection)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    edges = cv2.Canny(blurred, 50, 150)  # Use Canny edge detection
    return edges

# Function to detect lines using Hough Transform with stricter parameters
def detect_lines(image):
    processed = preprocess_image(image)
    lines = cv2.HoughLinesP(processed, 1, np.pi / 180, threshold=120, minLineLength=100, maxLineGap=10)
    detected_lines = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                detected_lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return detected_lines

# Function to detect shapes based on contours and their properties
def detect_shapes(image):
    processed = preprocess_image(image)
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []

    for contour in contours:
        shape = {}
        area = cv2.contourArea(contour)
        if area < 500:  # Filter small noise
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:  # Detect quadrilateral shapes
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            rect = cv2.minAreaRect(contour)
            angle = rect[-1]

            # Adjust for rotated rectangles (i.e., diamonds)
            if angle < -45:
                angle += 90

            # Diamond detection: rotated square-like shape with a diagonal aspect
            if 0.9 <= aspect_ratio <= 1.1 and 30 <= abs(angle) <= 60:
                shape["type"] = "diamond"
            # Rectangle detection: wider or taller than square
            elif aspect_ratio > 1.1 or aspect_ratio < 0.9 and abs(angle) < 30:
                shape["type"] = "rectangle"
            else:
                shape["type"] = "unknown"  # Fallback for quadrilaterals that don't match the conditions

            shape["position"] = {"x": x, "y": y, "width": w, "height": h}

        elif 5 <= len(approx) <= 7:  # Arrowhead detection
            shape["type"] = "arrow"
            shape["position"] = {"x": int(approx[0][0][0]), "y": int(approx[0][0][1])}

        if "type" in shape:
            shapes.append(shape)

    return shapes

# Function to combine line and shape detection
def detect_shapes_and_lines(image):
    lines = detect_lines(image)
    shapes = detect_shapes(image)

    # Filter out lines that are already part of detected shapes
    filtered_lines = []
    for line in lines:
        is_part_of_shape = False
        for shape in shapes:
            if shape["type"] in ["rectangle", "diamond"]:
                x1, y1 = shape["position"]["x"], shape["position"]["y"]
                x2, y2 = x1 + shape["position"]["width"], y1 + shape["position"]["height"]
                # Check if the line is within the bounding box of a shape
                if (min(line["x1"], line["x2"]) >= x1 and max(line["x1"], line["x2"]) <= x2 and
                        min(line["y1"], line["y2"]) >= y1 and max(line["y1"], line["y2"]) <= y2):
                    is_part_of_shape = True
                    break
        if not is_part_of_shape:
            filtered_lines.append(line)

    return shapes, filtered_lines

# Function to generate Draw.io compatible XML
# Function to generate Draw.io compatible XML
def generate_xml(shapes, lines):
    root = ET.Element("drawing")

    for shape in shapes:
        shape_elem = ET.SubElement(root, "shape")
        shape_elem.set("type", shape["type"])

        if shape["type"] in ["rectangle", "diamond"]:
            shape_elem.set("start_x", str(shape["position"]["x"]))
            shape_elem.set("start_y", str(shape["position"]["y"]))
            shape_elem.set("end_x", str(shape["position"]["x"] + shape["position"]["width"]))
            shape_elem.set("end_y", str(shape["position"]["y"] + shape["position"]["height"]))
            shape_elem.set("color", "#FF0000")  # Assuming red for shapes, can be adjusted

        elif shape["type"] == "arrow":
            shape_elem.set("start_x", str(shape["position"]["x"]))
            shape_elem.set("start_y", str(shape["position"]["y"]))
            shape_elem.set("end_x", str(shape["position"]["x"]))  # Arrow can have start and end points as the same
            shape_elem.set("end_y", str(shape["position"]["y"]))  # Adjust as needed
            shape_elem.set("color", "#0000FF")  # Assuming blue for arrows, can be adjusted

    for line in lines:
        line_elem = ET.SubElement(root, "shape")  # Use "shape" for lines as well
        line_elem.set("type", "line")
        line_elem.set("start_x", str(line["x1"]))
        line_elem.set("start_y", str(line["y1"]))
        line_elem.set("end_x", str(line["x2"]))
        line_elem.set("end_y", str(line["y2"]))
        line_elem.set("color", "#00FF00")  # Assuming green for lines, can be adjusted

    return ET.ElementTree(root)
# Main function to process the image and generate the XML
def process_image(image_path):
    image = cv2.imread(image_path)
    shapes, lines = detect_shapes_and_lines(image)

    # Generate compatible XML
    xml_tree = generate_xml(shapes, lines)
    xml_tree.write("output_drawio.xml", encoding="UTF-8", xml_declaration=True)
    print("XML has been saved as output_drawio.xml")

    # Optional: Display shapes and lines for verification
    for shape in shapes:
        color = (0, 255, 0)
        label = shape["type"].capitalize()

        if shape["type"] in ["rectangle", "diamond"]:
            color = (255, 0, 0)
            cv2.rectangle(image,
                          (shape["position"]["x"], shape["position"]["y"]),
                          (shape["position"]["x"] + shape["position"]["width"],
                           shape["position"]["y"] + shape["position"]["height"]),
                          color, 2)
        elif shape["type"] == "arrow":
            color = (0, 0, 255)
            cv2.putText(image, "Arrow",
                        (shape["position"]["x"], shape["position"]["y"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check if 'x' and 'y' are in the shape's position before trying to access them
        if "x" in shape["position"] and "y" in shape["position"]:
            text_x = shape["position"]["x"]
            text_y = shape["position"]["y"] - 15

            cv2.putText(image, label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    for line in lines:
        cv2.line(image,
                 (line["x1"], line["y1"]),
                 (line["x2"], line["y2"]),
                 (0, 255, 0), 2)

    cv2.imshow("Detected Shapes and Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the image path
image_path = "im.png"
process_image(image_path)
