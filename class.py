import os
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from ClassDiagram import save_bounding_boxes, process_yolo_predictions, save_updated_json, convert_yolo_to_class_xmi
from flask_cors import CORS  # Import CORS from flask_cors

app = Flask(__name__)

# Enable CORS for the specific domain
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

# Set the upload folder for images and ensure it exists
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Endpoint to upload an image, detect bounding boxes, and return JSON and XMI
@app.route('/detect_image', methods=['POST'])
def detect_image():
    # Check if an image file is included in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    # Retrieve the image file from the request
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Secure the file name and save it to the upload folder
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)

    # Optional: Set confidence and overlap values from the form-data or use defaults
    confidence = int(request.form.get("confidence", 40))
    overlap = int(request.form.get("overlap", 30))

    # Detect bounding boxes and process them
    bounding_boxes = save_bounding_boxes(image_path, confidence, overlap)
    if bounding_boxes is None:
        return jsonify({"error": "Bounding box detection failed"}), 500

    # Process the detected bounding boxes into class, relationship, and connection data
    classes, relationships, connections = process_yolo_predictions(bounding_boxes)

    # Prepare JSON output
    json_response = {
        "bounding_boxes": bounding_boxes,
        "classes": classes,
        "relationships": relationships,
        "connections": connections
    }

    # Save JSON data to a file
    json_filename = f"{filename}_output.json"
    json_path = os.path.join(OUTPUT_FOLDER, json_filename)
    save_updated_json(json_path, classes, relationships, connections)

    # Convert to XMI format and save
    xmi_filename = f"{filename}_class_diagram.xmi"
    xmi_path = os.path.join(OUTPUT_FOLDER, xmi_filename)
    convert_yolo_to_class_xmi(json_path, xmi_path)

    # Send JSON response with links to download the XMI and JSON files
    return jsonify({
        "json_file_url": f"/download_json/{json_filename}",
        "xmi_file_url": f"/download_xmi/{xmi_filename}"
    })

# Endpoint to download the generated XMI file
@app.route('/download_xmi/<filename>', methods=['GET'])
def download_xmi(filename):
    xmi_path = os.path.join(OUTPUT_FOLDER, filename)
    try:
        return send_file(xmi_path, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "XMI file not found"}), 404

# Endpoint to download the generated JSON file
@app.route('/download_json/<filename>', methods=['GET'])
def download_json(filename):
    json_path = os.path.join(OUTPUT_FOLDER, filename)
    try:
        return send_file(json_path, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "JSON file not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=9000)
