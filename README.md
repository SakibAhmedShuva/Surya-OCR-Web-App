# Surya-OCR Demonstration

This repository contains a Jupyter Notebook (`surya-ocr.ipynb`) that demonstrates the usage of the [Surya OCR](https://github.com/VikParuchuri/surya) library for text detection and recognition. It serves as a practical example of how to set up and use Surya OCR for extracting text from images.

## About Surya OCR

Surya is a powerful, multilingual document OCR toolkit developed by Vik Paruchuri and contributors. It excels in accurately detecting text regions (bounding boxes) and recognizing the text content within those regions across various languages. It leverages deep learning models for high performance.

## Files in this Repository

* `surya-ocr.ipynb`: The main Jupyter Notebook demonstrating Surya OCR functionalities.
* `README.md`: This file, providing an overview and instructions.

## Notebook Overview (`surya-ocr.ipynb`)

The `surya-ocr.ipynb` notebook provides a step-by-step guide to using Surya OCR:

1. **Setup**:
   * Clones the official Surya OCR repository from GitHub. This makes the library's modules directly importable, especially in environments like Google Colab.
2. **Image Loading**:
   * Loads a sample image for OCR processing using the Pillow (PIL) library.
   * **Important Note**: The notebook uses a default image path: `/content/California -USA-_Front_f4dce8c3dfba3c893adafeec60cdd00d.jpg`. You will need to upload your own image to this path in your Colab environment or modify the path in the notebook to point to your image.
3. **OCR Processing**:
   * Initializes Surya's `DetectionPredictor` (for finding text bounding boxes) and `RecognitionPredictor` (for reading text within boxes).
   * Runs these predictors on the loaded image. The notebook specifies English (`"en"`) as the language, but Surya supports multilingual OCR (passing `None` for languages is often recommended for automatic language detection).
4. **Output Display**:
   * The notebook first shows the raw, structured output from the OCR process.
   * It then includes a section with Python code (using mocked data derived from an actual run) to parse and display these predictions in a more human-readable format, showing each detected text line along with its confidence score.

## Setup and Running the Notebook

### 1. Clone this Repository (Optional, if you want the notebook locally)
```bash
git clone https://github.com/SakibAhmedShuva/Surya-OCR.git
cd Surya-OCR
```

### 2. Environment Setup

#### Option A: Using Google Colab (Recommended for easy GPU access)
1. Open `surya-ocr.ipynb` in Google Colab. (You can upload it from this repository).
2. Ensure your Colab runtime is set to use a GPU for faster processing:
   * Go to Runtime -> Change runtime type.
   * Select GPU from the Hardware accelerator dropdown and click Save.
3. Upload your image:
   * The notebook currently expects an image at `/content/California -USA-_Front_f4dce8c3dfba3c893adafeec60cdd00d.jpg`.
   * Upload your desired image to your Colab session's `/content/` directory. You can do this by clicking the "Files" icon on the left sidebar, then "Upload to session storage".
   * Either rename your uploaded image to `California -USA-_Front_f4dce8c3dfba3c893adafeec60cdd00d.jpg` or update the path in the notebook cell where the image is loaded:
   ```python
   # In the notebook, find this line and update YOUR_IMAGE_NAME.jpg:
   image = Image.open("/content/YOUR_IMAGE_NAME.jpg")
   ```
4. Run the cells in the notebook sequentially. The first cell (`!git clone ...`) will clone the Surya library into your Colab environment. Model weights will be downloaded by Surya on the first run.

#### Option B: Local Setup
1. Create a Python Environment:
   * It's highly recommended to use a virtual environment (e.g., venv, conda).
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install PyTorch:
   * Follow the instructions at pytorch.org to install PyTorch. A version compatible with your system's CUDA (if you have an NVIDIA GPU) is recommended.

3. Install Surya OCR:
   * The notebook clones Surya directly. For a robust local setup where surya is properly installed:
   ```bash
   git clone https://github.com/VikParuchuri/surya.git
   cd surya
   pip install -r requirements.txt  # Install Surya's dependencies
   pip install -e .                 # Install surya in editable mode
   cd ..                            # Return to your project directory (e.g., Surya-OCR)
   ```
   * You will also need Pillow:
   ```bash
   pip install Pillow
   ```
   * Jupyter:
   ```bash
   pip install jupyterlab notebook
   ```

4. Prepare your image:
   * Place your image file (e.g., `my_document.jpg`) in a known location. Update the image path in the `surya-ocr.ipynb` notebook:
   ```python
   # In the notebook, find this line:
   # image = Image.open("/content/California -USA-_Front_f4dce8c3dfba3c893adafeec60cdd00d.jpg")
   # And change it to the local path of your image:
   image = Image.open("path/to/your/my_document.jpg")
   ```

5. Run Jupyter:
   ```bash
   jupyter lab  # or jupyter notebook
   ```
   * Open `surya-ocr.ipynb` and run the cells.
   * Note: If you installed Surya locally as per step 3, you might want to comment out the `!git clone https://github.com/VikParuchuri/surya` cell in the notebook to avoid re-cloning.

### 3. General Notes for Running
* The first time you run the OCR processing cell, Surya will download the necessary pre-trained models. This might take a few minutes depending on your internet connection.
* Using a GPU significantly speeds up both detection and recognition.

## Example Output Snippet

The notebook demonstrates how to format the OCR results for better readability. An example of this output (which will vary based on your input image) looks like this:

```
--- OCR Extracted Text ---

Document 1:
  Detected Languages: ['en']
  Recognized Text Lines (Text [Confidence]):
    Line 1: "CAN IN INC. REAL A DRIVER LICENSE" [43.95%]
    Line 2: "DL 1234568" [81.79%]
    Line 3: "CLASS C" [71.83%]
    Line 4: "EXP 08/31/2018" [84.62%]
    Line 5: "END NONE" [80.08%]
    Line 6: "LN CARDHOLDER" [89.84%]
    Line 7: "FN IMA" [66.02%]
    Line 8: "2570 24TH STREET Usa Template ESD" [96.39%]
    ... (and so on for other detected lines) ...
------------------------------
```

## Acknowledgements

This demonstration heavily relies on the Surya OCR library. All credit for the OCR technology goes to Vik Paruchuri and the contributors to the Surya project.

Please refer to the original Surya repository for detailed documentation, issues, advanced usage, and information about the models.
