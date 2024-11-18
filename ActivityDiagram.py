import cv2
import easyocr
from roboflow import Roboflow
import json
import re
from spellchecker import SpellChecker
import math
import uuid
def process_image(image_path, json_output_path): 
    # Initialize Roboflow for object detection
    rf = Roboflow(api_key="d0WzawaqJjGZhrlt4xkz")
    project = rf.workspace().project("uml-activity")
    model = project.version(5).model

    # Load the image
    image = cv2.imread(image_path)

    # Run Object Detection
    predictions = model.predict(image_path, confidence=40, overlap=40).json()

    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Initialize SpellChecker
    spell = SpellChecker()

    # Preprocess image for OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)

    # Detect text with EasyOCR
    ocr_results = reader.readtext(dilated)

    # Function to clean and correct OCR text
    def clean_and_correct_text(name):
        name = re.sub(r'[^A-Za-z0-9\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        words = name.split()
        corrected_words = [spell.correction(word) or word for word in words]
        return ' '.join(corrected_words)

    # Process Object Detection Predictions
    bounding_boxes = []
    for detection in predictions['predictions']:
        x = detection['x']
        y = detection['y']
        width = detection['width']
        height = detection['height']
        class_name = detection['class']
        confidence = detection['confidence']
        detection_id = detection["detection_id"]

        x1, y1 = x - width / 2, y - height / 2
        x2, y2 = x + width / 2, y + height / 2

        bounding_boxes.append({
            "class": class_name,
            "confidence": confidence,
            "detection_id": detection_id,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "name": ""
        })

    # Assign OCR-detected text to bounding boxes
    for (bbox, name, prob) in ocr_results:
        (top_left, _, bottom_right, _) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        corrected_text = clean_and_correct_text(name)
        for detection in bounding_boxes:
            x1, y1, x2, y2 = detection["bbox"]["x1"], detection["bbox"]["y1"], detection["bbox"]["x2"], detection["bbox"]["y2"]
            if top_left[0] > x1 and top_left[1] > y1 and bottom_right[0] < x2 and bottom_right[1] < y2:
                detection["name"] = corrected_text

    # Separate predictions into control flows and shapes
    control_flows, shapes = [], []
    for pred in bounding_boxes:
        if pred['class'] == 'control flow':
            control_flows.append(pred)
        elif pred['class'] in ['action', 'decision', 'fork', 'start', 'final']:
            shapes.append(pred)

    # Ensure there's a final node and sort shapes
    final_node = next((s for s in shapes if s['class'] == 'final'), None)
    if not final_node:
        shapes.append({
            'class': 'final',
            'detection_id': 'final-node',
            'bbox': {'x1': 0, 'y1': 1000, 'x2': 100, 'y2': 1100}
        })
    shapes.sort(key=lambda s: (s['bbox']['y1'], s['bbox']['x1']))

    # Function to calculate the center of a shape
    def calculate_center(bbox):
        return ((bbox['x1'] + bbox['x2']) / 2, (bbox['y1'] + bbox['y2']) / 2)

    # Create connections between shapes
    connections = set()
    for i in range(len(shapes) - 1):
        source = shapes[i]
        target = shapes[i + 1]
        if source['class'] == 'start' and target['class'] not in ['start', 'final']:
            connections.add(('control flow', source['detection_id'], target['detection_id']))
        if source['class'] in ['action', 'decision', 'fork']:
            connections.add(('control flow', source['detection_id'], target['detection_id']))
        if target['class'] == 'final' and source['class'] != 'final':
            connections.add(('control flow', source['detection_id'], target['detection_id']))

    # Add connections for control flows
    for control_flow in control_flows:
        control_flow_center = calculate_center(control_flow['bbox'])
        nearest_source, nearest_target = None, None
        min_source_distance, min_target_distance = float('inf'), float('inf')

        for shape in shapes:
            shape_center = calculate_center(shape['bbox'])
            distance = math.dist(control_flow_center, shape_center)
            if distance < min_source_distance:
                min_source_distance = distance
                nearest_source = shape
            if distance < min_target_distance and shape != nearest_source:
                min_target_distance = distance
                nearest_target = shape

        if nearest_source and nearest_target:
            connections.add(('control flow', nearest_source['detection_id'], nearest_target['detection_id']))

    # Convert set back to a list of dictionaries
    connections_list = [{'type': t, 'from': f, 'to': to} for t, f, to in connections]

    # Save final JSON output
    output_data = {"predictions": bounding_boxes, "connections": connections_list}
    with open(json_output_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Processed image and saved JSON to '{json_output_path}'")



class XMLWriter:
    def __init__(self, indent_string="\t"):
        self.lines = []
        self.indent_string = indent_string
        self.indentations = []

    def indent(self):
        self.indentations.append(self.indent_string)

    def outdent(self):
        if self.indentations:
            self.indentations.pop()

    def write_line(self, line=""):
        if line:
            self.lines.append("".join(self.indentations) + line)
        else:
            self.lines.append("")

    def get_data(self):
        return "\n".join(self.lines)


def convert_yolo_to_xmi(yolo_json_path, xmi_path):
    try:
        # Load the YOLO JSON file
        with open(yolo_json_path, 'r') as f:
            yolo_data = json.load(f)

        writer = XMLWriter()
        writer.write_line('<?xml version="1.0" encoding="UTF-8"?>')
        writer.write_line(
            '<xmi:XMI xmi:version="2.1" xmlns:uml="http://schema.omg.org/spec/UML/2.0" xmlns:xmi="http://schema.omg.org/spec/XMI/2.1">'
        )
        writer.indent()
        writer.write_line('<xmi:Documentation exporter="StarUML" exporterVersion="2.0"/>')
        writer.write_line('<uml:Model xmi:id="AAAAAAGSzpfoUdWBzfE=" xmi:type="uml:Model" name="RootModel">')
        writer.indent()

        # Define the Activity Diagram
        writer.write_line(
            '<packagedElement xmi:id="AAAAAAGSzpWYqNQV9lI=" name="Activity1" visibility="public" isReentrant="true" xmi:type="uml:Activity" isReadOnly="false" isSingleExecution="false">'
        )
        writer.indent()

        nodes = {}
        adjusted_positions = []

        # Create nodes with bounding box positions and dimensions
        for prediction in yolo_data.get("predictions", []):
            node_id = str(uuid.uuid4()).replace('-', '')  # Unique ID
            prediction_class = prediction["class"]

            # Determine node type based on class
            node_type = {
                "start": "uml:InitialNode",
                "final": "uml:ActivityFinalNode",
                "decision": "uml:DecisionNode",
                "fork": "uml:ForkNode",
                "join": "uml:JoinNode",
                "action": "uml:OpaqueAction"
            }.get(prediction_class, "")

            node_name = prediction.get("name", prediction_class.capitalize())
            nodes[prediction["detection_id"]] = node_id  # Store the ID for reference in edges

            # Calculate adjusted position to avoid overlap
            bounding_box = prediction.get("bbox", {})
            x, y, width, height = bounding_box.get("x", 0), bounding_box.get("y", 0), bounding_box.get("width", 0), bounding_box.get("height", 0)
            adjusted_x, adjusted_y = x, y
            for ax, ay, aw, ah in adjusted_positions:
                if x < ax + aw and x + width > ax and y < ay + ah and y + height > ay:
                    adjusted_x = ax + aw + 10  # Move right
                    adjusted_y = ay  # Align vertically with the previous element
                    break

            adjusted_positions.append((adjusted_x, adjusted_y, width, height))

            # Write node to XMI
            writer.write_line(
                f'<node xmi:id="{node_id}" name="{node_name}" visibility="public" xmi:type="{node_type}">'
            )
            writer.indent()

            # Add shape element with adjusted bounding box
            writer.write_line(
                f'<shape x="{adjusted_x}" y="{adjusted_y}" width="{width}" height="{height}" />'
            )

            writer.outdent()  # End node
            writer.write_line('</node>')

        # Create edges based on the connections list
        connections = yolo_data.get("connections", [])
        for conn in connections:
            from_id = nodes.get(conn["from"])
            to_id = nodes.get(conn["to"])
            edge_id = str(uuid.uuid4()).replace('-', '')
            if from_id and to_id:
                writer.write_line(
                    f'<edge xmi:id="{edge_id}" visibility="public" source="{from_id}" target="{to_id}" xmi:type="uml:ControlFlow"/>'
                )

        writer.outdent()  # End packagedElement (Activity)
        writer.write_line('</packagedElement>')
        writer.outdent()  # End Model
        writer.write_line('</uml:Model>')
        writer.outdent()  # End XMI
        writer.write_line('</xmi:XMI>')

        # Write the generated XMI to a file
        with open(xmi_path, 'w') as f:
            f.write(writer.get_data())

        print(f"Successfully converted to XMI: {xmi_path}")

    except FileNotFoundError:
        print(f"Error: The file '{yolo_json_path}' was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example Usage: Ensure JSON data is updated with connections before calling
try:
    with open('activity_connection.json', 'r') as json_file:
        data = json.load(json_file)

    # Add connections logic here (if not already in JSON)
    connections = data.get("connections", [])
    # Update or modify connections if needed
    data["connections"] = connections

    # Save updated JSON
    with open("activity_connection.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Connections have been updated in activity_connection.json with {len(connections)} connections.")
    convert_yolo_to_xmi('activity_connection.json', 'update.xmi')

except FileNotFoundError:
    print("activity_connection.json not found.")
except Exception as ex:
    print(f"An unexpected error occurred: {ex}")
