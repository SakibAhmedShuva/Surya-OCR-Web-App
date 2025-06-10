import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io

# --- Model Loading ---
# This is a time-consuming step, so we do it once when the application starts.
print("Initializing Flask app and loading Surya models...")

try:
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize models
    detection_predictor = DetectionPredictor()
    recognition_predictor = RecognitionPredictor()
    print("Surya models loaded successfully.")

except ImportError:
    print("\n---")
    print("Warning: `surya-ocr` or its dependencies are not installed.")
    print("The application will not function correctly.")
    print("Please install requirements with: pip install -r requirements.txt")
    print("---\n")
    detection_predictor = None
    recognition_predictor = None

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for the API

# --- API Endpoint ---
@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    if not recognition_predictor or not detection_predictor:
        return jsonify({"error": "OCR models are not loaded. Server setup is incomplete."}), 500
        
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400

    try:
        # Open the image from the in-memory file stream
        image = Image.open(file.stream).convert("RGB")
        langs = ["en"]  # Languages to detect

        # Perform OCR
        predictions = recognition_predictor([image], [langs], detection_predictor)
        ocr_result = predictions[0]

        # Format the output into a clean JSON structure
        formatted_lines = []
        for line in ocr_result.text_lines:
            formatted_lines.append({
                "text": line.text,
                "confidence": line.confidence,
                "bbox": line.bbox
            })

        return jsonify({"text_lines": formatted_lines})

    except Exception as e:
        print(f"An error occurred during OCR processing: {e}")
        return jsonify({"error": "Failed to process the image.", "details": str(e)}), 500

# --- Frontend Route ---
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')


if __name__ == '__main__':
    # Running on 0.0.0.0 makes the app accessible from other devices on the same network.
    # The default port is 5000.
    app.run(host='0.0.0.0', port=5000, debug=False)