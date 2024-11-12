from roboflow import Roboflow
import json
import cv2
import easyocr
from spellchecker import SpellChecker
import math
import uuid
import xml.sax.saxutils as xml_escape
# Initialize Roboflow
rf = Roboflow(api_key="f4MXdAqumbSttxl94Dil")
project = rf.workspace().project("class-diagram-9jrsx")
model = project.version(5).model

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Specify language and disable GPU if necessary

# Initialize the SpellChecker
spell = SpellChecker()

def save_bounding_boxes(image_path, confidence=60, overlap=30):
    # Create a new JSON structure for the bounding boxes
    bounding_boxes = {"prediction": []}
    try:
        # Load the image for OCR processing
        image = cv2.imread(image_path)
        # Run object detection to detect bounding boxes
        predictions = model.predict(image_path, confidence=confidence, overlap=overlap).json()

        # Iterate through all detections and add to the list
        for idx, bounding_box in enumerate(predictions['predictions'], start=1):  # Enumerate for unique IDs
            x = bounding_box['x']
            y = bounding_box['y']
            width = bounding_box['width']
            height = bounding_box['height']

            # Calculate corner points for the bounding box
            x1 = int(x - (width / 2))
            y1 = int(y - (height / 2))
            x2 = int(x + (width / 2))
            y2 = int(y + (height / 2))

            # Check if this detection is a class (based on the class name in object detection)
            if bounding_box['class'] == "class":  # Replace "class" with your actual class name if needed
                # Divide the bounding box into three sections (Name 20%, Attributes 50%, Operations 30%)
                name_section = (y1, int(y1 + height * 0.2))  # Name section: 20% of the height
                attribute_section = (int(y1 + height * 0.2), int(y1 + height * 0.7))  # Attribute section: 50% of the height
                operation_section = (int(y1 + height * 0.7), y2)  # Operation section: 30% of the height

                # Crop the detected areas for OCR
                name_crop = image[name_section[0]:name_section[1], x1:x2]
                attribute_crop = image[attribute_section[0]:attribute_section[1], x1:x2]
                operation_crop = image[operation_section[0]:operation_section[1], x1:x2]

                # Run OCR on the cropped regions
                name_text = " ".join(reader.readtext(name_crop, detail=0)) or None
                attribute_text = " ".join(reader.readtext(attribute_crop, detail=0)) or None
                operation_text = " ".join(reader.readtext(operation_crop, detail=0)) or None

                # Skip further processing if OCR returns None for any section
                if name_text is None:
                    name_text = "Name"  # Fallback if no OCR result
                if attribute_text is None:
                    attribute_text = "Attributes"  # Fallback if no OCR result
                if operation_text is None:
                    operation_text = "Operations"  # Fallback if no OCR result

                # Add the bounding box information to the list, including name, attributes, and operations
                bounding_boxes["prediction"].append({
                    "detection_id": bounding_box['detection_id'],
                    "class": bounding_box['class'],
                    "confidence": bounding_box['confidence'],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "name": name_text,  # Fallback text if OCR result is None
                    "attributes": attribute_text,  # Fallback text if OCR result is None
                    "operations": operation_text  # Fallback text if OCR result is None
                })
            else:
                # For non-class elements (like associations, generalizations), skip name, attributes, and operations
                bounding_boxes["prediction"].append({
                    "detection_id": bounding_box['detection_id'],
                    "class": bounding_box['class'],
                    "confidence": bounding_box['confidence'],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

        return bounding_boxes
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def euclidean_distance(box1, box2):
    # Calculate Euclidean distance between two bounding box centers
    center1 = ((box1['x1'] + box1['x2']) / 2, (box1['y1'] + box1['y2']) / 2)
    center2 = ((box2['x1'] + box2['x2']) / 2, (box2['y1'] + box2['y2']) / 2)
    return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def find_nearest_classes(rel_bbox, classes, count=2):
    distances = []
    for class_id, class_data in classes.items():
        class_bbox = class_data['bbox']
        distance = euclidean_distance(rel_bbox, class_bbox)
        distances.append((distance, class_id))

    # Sort and get closest 'count' classes
    distances.sort(key=lambda x: x[0])
    return [class_id for _, class_id in distances[:count]]


def process_yolo_predictions(yolo_json):
    predictions = yolo_json['prediction']
    classes, relationships, connections = {}, [], []

    # Class and Relationship Detection
    for prediction in predictions:
        bbox = {
            'x1': prediction['x1'], 'y1': prediction['y1'],
            'x2': prediction['x2'], 'y2': prediction['y2'],
            'confidence': prediction['confidence'], 'detection_id': prediction['detection_id']
        }
        class_name = prediction['class']

        # Categorize elements
        if class_name == "class":
            class_id = prediction['detection_id']
            classes[class_id] = {
                'bbox': bbox,
                'name': prediction['name'],
                'attributes': prediction['attributes'],
                'operations': prediction['operations']
            }
        elif class_name in ["association", "one-way-association", "aggregation", "composition", "generalization"]:
            relationships.append({'type': class_name, 'bbox': bbox})

    # Establish connections
    for relationship in relationships:
        rel_bbox = relationship['bbox']
        nearest_classes = find_nearest_classes(rel_bbox, classes)

        if relationship['type'] == 'association' and len(nearest_classes) >= 2:
            from_class, to_class = nearest_classes[:2]
            connections.append({'type': 'association', 'from': from_class, 'to': to_class})
        elif relationship['type'] in ['one-way-association', 'aggregation', 'composition'] and len(nearest_classes) > 1:
            from_class, to_class = nearest_classes[0], nearest_classes[1]
            connections.append({'type': relationship['type'], 'from': from_class, 'to': to_class})
        elif relationship['type'] == 'generalization' and len(nearest_classes) >= 2:
            from_class, to_class = nearest_classes[:2]
            connections.append({'type': 'generalization', 'from': from_class, 'to': to_class})

    return classes, relationships, connections


def save_updated_json(file_path, classes, relationships, connections):
    updated_json = {
        'classes': classes,
        'relationships': relationships,
        'connections': connections
    }

    with open(file_path, 'w') as json_file:
        json.dump(updated_json, json_file, indent=4)
    print(f"Updated JSON file created at: {file_path}")




# Helper function to escape XML special characters
def escape_xml(value):
    return xml_escape.escape(value, {"\"": "&quot;", "'": "&apos;"})
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

def convert_yolo_to_class_xmi(yolo_json_path, xmi_path):
    try:
        with open(yolo_json_path, 'r') as f:
            yolo_data = json.load(f)

        writer = XMLWriter()
        writer.write_line('<?xml version="1.0" encoding="UTF-8"?>')
        writer.write_line(
            '<xmi:XMI xmi:version="2.1" xmlns:uml="http://schema.omg.org/spec/UML/2.0" xmlns:xmi="http://schema.omg.org/spec/XMI/2.1">'
        )
        writer.indent()
        writer.write_line('<packagedElement xmi:id="{}" xmi:type="uml:Model">'.format(generate_guid()))
        writer.indent()
        writer.write_line('<name>ClassDiagram1</name>')

        # Dictionary to store class IDs by detection ID for later reference
        classes = {}
        for detection_id, class_data in yolo_data.get("classes", {}).items():
            class_id = generate_guid()
            class_name = escape_xml(class_data.get("name", "UnnamedClass"))
            classes[str(detection_id)] = class_id
            writer.write_line(
                '<packagedElement isAbstract="false" isActive="false" isFinalSpecialization="false" isLeaf="false" '
                'xmi:id="{}" xmi:type="uml:Class">'.format(class_id)
            )
            writer.indent()
            writer.write_line('<name>{}</name>'.format(class_name))
            writer.write_line('<visibility>public</visibility>')

            # Write attributes as UML properties
            attributes = class_data.get("attributes", "").split(',')
            for attr in attributes:
                attr_id = generate_guid()
                attr_name = escape_xml(attr.strip())
                if attr_name:
                    writer.write_line(
                        '<ownedAttribute xmi:id="{}" name="{}" visibility="public" xmi:type="uml:Property"/>'.format(
                            attr_id, attr_name
                        )
                    )

            # Write operations as UML operations
            operations = class_data.get("operations", "").split(',')
            for op in operations:
                op_id = generate_guid()
                op_name = escape_xml(op.strip())
                if op_name:
                    writer.write_line(
                        '<ownedOperation xmi:id="{}" name="{}" visibility="public" xmi:type="uml:Operation"/>'.format(
                            op_id, op_name
                        )
                    )

            writer.outdent()
            writer.write_line('</packagedElement>')

        # Process connections for associations, aggregations, compositions, and generalizations
        for conn in yolo_data.get("connections", []):
            from_id = classes.get(str(conn["from"]))
            to_id = classes.get(str(conn["to"]))
            conn_type = conn.get("type", "association")
            assoc_id = generate_guid()

            if from_id and to_id:
                if conn_type == "association":
                    writer.write_line(
                        '<packagedElement xmi:id="{}" visibility="public" xmi:type="uml:Association">'.format(assoc_id)
                    )
                    writer.indent()
                    writer.write_line('<memberEnd xmi:idref="{}" xmi:type="uml:Property"/>'.format(from_id))
                    writer.write_line('<memberEnd xmi:idref="{}" xmi:type="uml:Property"/>'.format(to_id))
                    writer.outdent()
                    writer.write_line('</packagedElement>')

                elif conn_type == "aggregation":
                    writer.write_line(
                        '<packagedElement xmi:id="{}" visibility="public" xmi:type="uml:Association">'.format(assoc_id)
                    )
                    writer.indent()
                    writer.write_line(
                        '<memberEnd xmi:idref="{}" xmi:type="uml:Property" aggregation="shared"/>'.format(from_id)
                    )
                    writer.write_line('<memberEnd xmi:idref="{}" xmi:type="uml:Property"/>'.format(to_id))
                    writer.outdent()
                    writer.write_line('</packagedElement>')

                elif conn_type == "composition":
                    writer.write_line(
                        '<packagedElement xmi:id="{}" visibility="public" xmi:type="uml:Association">'.format(assoc_id)
                    )
                    writer.indent()
                    writer.write_line(
                        '<memberEnd xmi:idref="{}" xmi:type="uml:Property" aggregation="composite"/>'.format(from_id)
                    )
                    writer.write_line('<memberEnd xmi:idref="{}" xmi:type="uml:Property"/>'.format(to_id))
                    writer.outdent()
                    writer.write_line('</packagedElement>')

                elif conn_type == "generalization":
                    # Place generalization directly inside the child class
                    writer.write_line('<packagedElement xmi:id="{}" xmi:type="uml:Class">'.format(from_id))
                    writer.indent()
                    writer.write_line('<generalization isSubstitutable="true" xmi:id="{}" xmi:type="uml:Generalization">'.format(assoc_id))
                    writer.indent()
                    writer.write_line('<general xmi:idref="{}"/>'.format(to_id))  # Reference to the parent class
                    writer.outdent()
                    writer.write_line('</generalization>')
                    writer.outdent()
                    writer.write_line('</packagedElement>')

        writer.outdent()
        writer.write_line('</packagedElement>')  # Close uml:Model
        writer.outdent()
        writer.write_line('</xmi:XMI>')

        with open(xmi_path, 'w') as f:
            f.write(writer.get_data())

    except FileNotFoundError:
        print(f"Error: The file '{yolo_json_path}' was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def generate_guid():
    return str(uuid.uuid4()).replace('-', '')



if __name__ == "__main__":
    # Specify image path and parameters
    image_path = "test1.jpg"
    yolo_json = save_bounding_boxes(image_path, confidence=40, overlap=30)

    if yolo_json:
        classes, relationships, connections = process_yolo_predictions(yolo_json)
        print("Connections Established:", json.dumps(connections, indent=4))

        # Save updated JSON results
        updated_json_path = 'updated_predictions.json'
        save_updated_json(updated_json_path, classes, relationships, connections)
        convert_yolo_to_class_xmi('updated_predictions.json', 'class_diagram.xmi')
