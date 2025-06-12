# --- START OF FILE app.py ---

import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading ---
print("Initializing Flask app and loading Surya models...")

try:
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize models with error handling
    detection_predictor = DetectionPredictor()
    recognition_predictor = RecognitionPredictor()
    print("Surya models loaded successfully.")

except ImportError as e:
    print(f"\n--- ImportError: {e} ---")
    print("Warning: `surya-ocr` or its dependencies are not installed.")
    print("The application will not function correctly.")
    print("Please install requirements with: pip install -r requirements.txt")
    print("---\n")
    detection_predictor = None
    recognition_predictor = None

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

def validate_image(image):
    """Validate image format and dimensions"""
    try:
        # Check if image is valid
        if image.mode not in ['RGB', 'RGBA', 'L']:
            image = image.convert('RGB')
        
        # Check image dimensions (avoid very small images)
        width, height = image.size
        if width < 50 or height < 50:
            return False, "Image too small (minimum 50x50 pixels)"
        
        # Check if image is too large (memory issues)
        if width * height > 50000000:  # ~50MP limit
            return False, "Image too large (maximum ~50 megapixels)"
            
        return True, None
    except Exception as e:
        return False, f"Invalid image format: {str(e)}"

def safe_ocr_process(image):
    """Safely process OCR with comprehensive error handling"""
    try:
        # Validate image first
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            return None, error_msg

        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        logger.info(f"Processing image of size: {image.size}")
        
        # Step 1: Text Detection
        try:
            # Pass image as list to detection predictor
            detection_result = detection_predictor([image])
            
            if not detection_result or len(detection_result) == 0:
                logger.warning("No detection results returned")
                return [], None
                
            detected_boxes = detection_result[0]
            
            # Check if any text regions were detected
            if not hasattr(detected_boxes, 'bboxes') or len(detected_boxes.bboxes) == 0:
                logger.info("No text regions detected in image")
                return [], None
                
            logger.info(f"Detected {len(detected_boxes.bboxes)} text regions")
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return None, f"Text detection failed: {str(e)}"

        # Step 2: Text Recognition
        try:
            # Use the detection result for recognition
            recognition_result = recognition_predictor([image], det_predictor=detection_predictor)
            
            if not recognition_result or len(recognition_result) == 0:
                logger.warning("No recognition results returned")
                return [], None
                
            ocr_result = recognition_result[0]
            
            # Check if text_lines attribute exists and has content
            if not hasattr(ocr_result, 'text_lines') or len(ocr_result.text_lines) == 0:
                logger.info("No text lines found in recognition result")
                return [], None
                
            logger.info(f"Recognized {len(ocr_result.text_lines)} text lines")
            return ocr_result.text_lines, None
            
        except Exception as e:
            logger.error(f"Recognition failed: {str(e)}")
            # Try alternative approach if main recognition fails
            try:
                logger.info("Attempting alternative recognition approach...")
                # Sometimes re-running detection helps with shape mismatches
                detection_result = detection_predictor([image])
                recognition_result = recognition_predictor([image], det_predictor=detection_predictor)
                
                if recognition_result and len(recognition_result) > 0:
                    ocr_result = recognition_result[0]
                    if hasattr(ocr_result, 'text_lines') and len(ocr_result.text_lines) > 0:
                        return ocr_result.text_lines, None
                        
            except Exception as retry_e:
                logger.error(f"Retry also failed: {str(retry_e)}")
                
            return None, f"Text recognition failed: {str(e)}"
            
    except Exception as e:
        logger.error(f"Unexpected error in OCR process: {str(e)}")
        logger.error(traceback.format_exc())
        return None, f"OCR processing failed: {str(e)}"

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
        # Open and validate the image
        image = Image.open(file.stream)
        logger.info(f"Processing file: {file.filename}, Mode: {image.mode}, Size: {image.size}")
        
        # Perform safe OCR processing
        text_lines, error_msg = safe_ocr_process(image)
        
        if error_msg:
            logger.error(f"OCR processing error: {error_msg}")
            return jsonify({"error": error_msg}), 400
            
        if text_lines is None:
            return jsonify({"error": "Failed to process image - unknown error"}), 500
            
        # Handle case where no text was found (empty list)
        if len(text_lines) == 0:
            logger.info("No text detected in image")
            return jsonify({
                "text_lines": [],
                "message": "No text was detected in this image"
            })

        # Format the output into a clean JSON structure
        formatted_lines = []
        for line in text_lines:
            try:
                formatted_line = {
                    "text": getattr(line, 'text', ''),
                    "confidence": getattr(line, 'confidence', 0.0),
                    "bbox": getattr(line, 'bbox', [])
                }
                # Only add lines with actual text content
                if formatted_line["text"].strip():
                    formatted_lines.append(formatted_line)
            except Exception as e:
                logger.warning(f"Error formatting line: {e}")
                continue

        logger.info(f"Successfully processed {len(formatted_lines)} text lines")
        return jsonify({"text_lines": formatted_lines})

    except Exception as e:
        logger.error(f"Unexpected error in endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to process the image", 
            "details": str(e)
        }), 500

# --- Frontend Route ---
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- Health Check Route ---
@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy" if (detection_predictor and recognition_predictor) else "unhealthy",
        "models_loaded": bool(detection_predictor and recognition_predictor),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    return jsonify(status)

if __name__ == '__main__':
    # Running on 0.0.0.0 makes the app accessible from other devices on the same network.
    app.run(host='0.0.0.0', port=5004, debug=False)