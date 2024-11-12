import cv2
import easyocr
from roboflow import Roboflow
import json
import re
from spellchecker import SpellChecker
import math
import uuid
# Initialize Roboflow for object detection
rf = Roboflow(api_key="d0WzawaqJjGZhrlt4xkz")
project = rf.workspace().project("uml-activity")
model = project.version(5).model

# Load the image
image_path = 'R.jpg'
image = cv2.imread(image_path)

# Run Object Detection
predictions = model.predict(image_path, confidence=40, overlap=40).json()

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)

# Initialize SpellChecker
spell = SpellChecker()

# Convert to grayscale for OCR
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use adaptive thresholding to make text more visible
adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Dilate to connect characters more effectively
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)

# Detect text with EasyOCR
ocr_results = reader.readtext(dilated)

# Define a function to clean and correct OCR text
def clean_and_correct_text(name):
    name = re.sub(r'[^A-Za-z0-9\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    words = name.split()
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    corrected_text = ' '.join(corrected_words)

    return corrected_text

# Prepare data to be written to JSON
bounding_boxes = []

# Process Object Detection Predictions
for detection in predictions['predictions']:
    x = detection['x']
    y = detection['y']
    width = detection['width']
    height = detection['height']
    class_name = detection['class']
    confidence = detection['confidence']
    detection_id = detection["detection_id"]

    x1 = x - width / 2
    y1 = y - height / 2
    x2 = x + width / 2
    y2 = y + height / 2

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, f"{class_name} ({confidence:.2f})", (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    bounding_boxes.append({
        "class": class_name,
        "confidence": confidence,
        "detection_id": detection_id,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "name": ""
    })

# Process EasyOCR Results
for (bbox, name, prob) in ocr_results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple([int(val) for val in top_left])
    bottom_right = tuple([int(val) for val in bottom_right])

    corrected_text = clean_and_correct_text(name)

    for detection in bounding_boxes:
        x1, y1, x2, y2 = detection["bbox"]["x1"], detection["bbox"]["y1"], detection["bbox"]["x2"], detection["bbox"]["y2"]

        if top_left[0] > x1 and top_left[1] > y1 and bottom_right[0] < x2 and bottom_right[1] < y2:
            if corrected_text:
                detection["name"] = corrected_text
            cv2.putText(image, corrected_text, (top_left[0] + 5, top_left[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

# Save the image
cv2.imwrite('activity.jpg', image)
print("Output image saved as 'activity.jpg'")

# Prepare final JSON structure
output_data = {"predictions": bounding_boxes}

with open("activity.json", "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print("Bounding boxes data with corrected text has been written to 'activity.json'")

# Load predictions from the JSON file
with open("activity.json") as json_file:
    data = json.load(json_file)

predictions = data['predictions']

# Separate predictions into control flows and other shapes
control_flows = []
shapes = []


# Classify predictions
for pred in predictions:
    if pred['class'] == 'control flow':
        control_flows.append(pred)
    elif pred['class'] in ['action', 'decision', 'fork', 'start', 'final']:
        shapes.append(pred)

# Sort shapes by their vertical position (Y-axis)
shapes.sort(key=lambda shape: (shape['bbox']['y1'], shape['bbox']['x1']))  # Sorting by Y first, then X for tie-breaking

# Ensure there's a final node at the end if it doesn't exist
final_node = next((shape for shape in shapes if shape['class'] == 'final'), None)
if not final_node:
    # Add a new final node at the end with a placeholder position
    final_node = {
        'class': 'final',
        'detection_id': 'final-node',
        'bbox': {'x1': 0, 'y1': 1000, 'x2': 100, 'y2': 1100}  # Add a large Y-coordinate to ensure it's at the end
    }
    shapes.append(final_node)

# Re-sort to place final node at the end
shapes.sort(key=lambda shape: (shape['bbox']['y1'], shape['bbox']['x1']))

# Now we will create connections based on the sorted nodes
connections = []

# Function to calculate the center of a shape
def calculate_center(bbox):
    return ((bbox['x1'] + bbox['x2']) / 2, (bbox['y1'] + bbox['y2']) / 2)

# Create connections between the nodes based on the sorted list
for i in range(len(shapes) - 1):
    source = shapes[i]
    target = shapes[i + 1]

    # Ensure that the start node connects to the first action/activity node
    if source['class'] == 'start' and target['class'] not in ['start', 'final']:
        connections.append({
            'type': 'control flow',
            'from': source['detection_id'],
            'to': target['detection_id'],
        })

    # For actions, decisions, forks, ensure correct flow
    if source['class'] in ['action', 'decision', 'fork']:
        connections.append({
            'type': 'control flow',
            'from': source['detection_id'],
            'to': target['detection_id'],
        })

    # If the target is the final node, ensure it's connected last
    if target['class'] == 'final' and source['class'] not in ['final']:
        connections.append({
            'type': 'control flow',
            'from': source['detection_id'],
            'to': target['detection_id'],
        })

# Add the connections for control flows as well
for control_flow in control_flows:
    # For each control flow, connect to nearest shapes
    control_flow_center = calculate_center(control_flow['bbox'])
    nearest_source = None
    nearest_target = None
    min_source_distance = float('inf')
    min_target_distance = float('inf')

    for shape in shapes:
        shape_center = calculate_center(shape['bbox'])
        distance = math.sqrt((control_flow_center[0] - shape_center[0]) ** 2 + (control_flow_center[1] - shape_center[1]) ** 2)

        if distance < min_source_distance:
            min_source_distance = distance
            nearest_source = shape
        if distance < min_target_distance and shape['detection_id'] != nearest_source['detection_id']:
            min_target_distance = distance
            nearest_target = shape

    if nearest_source and nearest_target:
        connections.append({
            'type': 'control flow',
            'from': nearest_source['detection_id'],
            'to': nearest_target['detection_id'],
        })

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
            '<xmi:XMI xmi:version="2.1" xmlns:uml="http://schema.omg.org/spec/UML/2.0" xmlns:xmi="http://schema.omg.org/spec/XMI/2.1">')
        writer.indent()
        writer.write_line('<xmi:Documentation exporter="StarUML" exporterVersion="2.0"/>')
        writer.write_line('<uml:Model xmi:id="AAAAAAGSzpfoUdWBzfE=" xmi:type="uml:Model" name="RootModel">')
        writer.indent()

        # Define the Activity Diagram
        writer.write_line(
            '<packagedElement xmi:id="AAAAAAFF+qBWK6M3Z8Y=" name="Model" visibility="public" xmi:type="uml:Model"/>')
        writer.write_line(
            '<packagedElement xmi:id="AAAAAAGSzpWYqNQV9lI=" name="Activity1" visibility="public" isReentrant="true" xmi:type="uml:Activity" isReadOnly="false" isSingleExecution="false">')
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

            # Calculate adjusted position to avoid overlap and fit the diagram
            bounding_box = prediction.get("bbox", {})
            x, y, width, height = bounding_box.get("x", 0), bounding_box.get("y", 0), bounding_box.get("width", 0), bounding_box.get("height", 0)

            # Adjust position to avoid overlaps
            adjusted_x, adjusted_y = x, y
            for ax, ay, aw, ah in adjusted_positions:
                if x < ax + aw and x + width > ax and y < ay + ah and y + height > ay:
                    adjusted_x = ax + aw + 10  # Move right
                    adjusted_y = ay  # Align vertically with the previous element
                    break

            adjusted_positions.append((adjusted_x, adjusted_y, width, height))

            # Write node to XMI
            writer.write_line(
                f'<node xmi:id="{node_id}" name="{node_name}" visibility="public" xmi:type="{node_type}">')
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
                    f'<edge xmi:id="{edge_id}" visibility="public" source="{from_id}" target="{to_id}" xmi:type="uml:ControlFlow"/>')

        writer.outdent()  # End packagedElement (Activity)
        writer.write_line('</packagedElement>')
        writer.outdent()  # End Model
        writer.write_line('</uml:Model>')
        writer.outdent()  # End XMI
        writer.write_line('</xmi:XMI>')

        # Write the generated XMI to a file
        with open(xmi_path, 'w') as f:
            f.write(writer.get_data())

    except FileNotFoundError:
        print(f"Error: The file '{yolo_json_path}' was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Update data with connections
data['connections'] = connections

# Write updated data to JSON
with open("activity_connection.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"Connections have been updated in activity_connection.json with {len(connections)} connections.")
convert_yolo_to_xmi('activity_connection.json', 'update.xmi')