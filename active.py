from flask import Flask, request, jsonify, send_file
import os
import cv2
import json
from ActivityDiagram import process_image, convert_yolo_to_xmi  # Import functions from activity.py
import uuid
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

# Folder paths for uploads and results
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/generate', methods=['POST'])
def generate_files():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    # Save the uploaded image
    image_file = request.files['image']
    image_filename = f"{uuid.uuid4()}.jpg"  # Unique filename
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    image_file.save(image_path)

    # Process the image and generate JSON and XMI files
    json_filename = f"{uuid.uuid4()}_activity.json"
    xmi_filename = f"{uuid.uuid4()}_activity.xmi"
    json_path = os.path.join(RESULT_FOLDER, json_filename)
    xmi_path = os.path.join(RESULT_FOLDER, xmi_filename)

    # Run object detection, OCR, and generate JSON and XMI files
    try:
        process_image(image_path, json_path)  # Generates JSON with bounding boxes and corrected text
        convert_yolo_to_xmi(json_path, xmi_path)  # Converts JSON data to XMI format

        # Return the JSON and XMI file paths
        return jsonify({
            "json_file": json_path,
            "xmi_file": xmi_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
