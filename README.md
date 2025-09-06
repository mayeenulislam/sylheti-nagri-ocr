# Sylheti Nagri OCR Application

This application performs word detection and cropping on images containing Sylheti Nagri script, followed by generating an HTML reconstruction of the detected text.

## Features

- **Word Detection**: Detects words in an uploaded image.
- **Cropping**: Crops detected words and saves them as individual images.
- **OCR**: Performs Optical Character Recognition (OCR) on the cropped word images.
- **HTML Generation**: Generates an HTML reconstruction of the detected text.

## Demo

[![Watch the demo](https://img.shields.io/badge/Video-Demo-blue)](https://private-user-images.githubusercontent.com/67544093/486449635-1ffc7fe8-c187-4685-9d9e-35110daba9f1.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTcxODI3MDYsIm5iZiI6MTc1NzE4MjQwNiwicGF0aCI6Ii82NzU0NDA5My80ODY0NDk2MzUtMWZmYzdmZTgtYzE4Ny00Njg1LTlkOWUtMzUxMTBkYWJhOWYxLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTA2VDE4MTMyNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTRkZTNjMzRkYjJkYjc0MzIzZGZmODhhNmFmNTUzM2JmOTQwYTQ5NzA5NDA2MGYzZjU1MThlZWQwNGJlM2Q0YjcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.n8UVE3iTOsvsXgQHAT3jmS5R_80SoPt3D206FtvPDbg)

## Technologies Used

-   **Streamlit**: For creating the interactive web application interface.
-   **apsisocr**: Library for out-of-the-box text detection models (using PaddleDBNet).
-   **TensorFlow / Keras / mltu**: Frameworks used for the custom Sylheti Nagri OCR model training and loading. `mltu` provides utility functions for data handling, model architecture, and training components.
-   **OpenCV (`cv2`)**: For core image processing operations like decoding, cropping, resizing, and drawing visualizations.
-   **numpy**: For numerical operations, particularly with image data arrays.
-   **zipfile / io / base64 / json / collections / os**: Standard Python libraries used for file handling, data manipulation, and system interaction.

## Requirements

To run this application, you need to have the following Python packages installed. These are listed in the provided `requirements.txt` file:

- apsisocr==0.0.7
- fastdeploy-python==1.0.7
- numpy==1.23.5
- opencv-python==4.11.0.86
- streamlit
- onnxruntime
- mltu
- tensorflow

## Installation

1.  **Save the Application Files:**
    Ensure you have the `streamlit_app.py` and `requirements.txt` files saved in a local directory on your computer.

2.  **Create a Virtual Environment (Recommended):**
    It's highly recommended to use a virtual environment to isolate project dependencies. You can use `conda` (if installed) or Python's built-in `venv`.

    ```bash
    # Using Conda
    conda create -n sylheti_ocr_env python=3.9.20 # Or your preferred Python version (3.9 used in recent examples)
    conda activate sylheti_ocr_env

    # Using venv
    python -m venv sylheti_ocr_env
    # On macOS/Linux:
    source sylheti_ocr_env/bin/activate
    # On Windows:
    .\sylheti_ocr_env\Scripts\activate
    ```

3.  **Install Required Packages:**
    Navigate to the directory where you saved the files in your terminal with the virtual environment activated, and run:

    ```bash
    pip install -r requirements.txt
    ```


## How to Run the Application

1.  Ensure you have completed the installation steps and placed the required model and font files as described above.
2.  Open your terminal or command prompt.
3.  Activate the virtual environment you created:
    ```bash
    conda activate sylheti_ocr_env # Or the name of your venv
    ```
4.  Navigate to the directory where your `streamlit_app.py` file is located.
5.  Run the Streamlit application using the command:
    ```bash
    streamlit run streamlit_app.py
    ```
6.  Your default web browser should open automatically, displaying the application interface.

## How to Use the Application

1.  In the running Streamlit application in your browser, use the file uploader ("Upload an image...") to select and upload an image file (`.png`, `.jpg`, or `.jpeg`) containing Sylheti Nagri text.
2.  The application will automatically perform word detection. Wait for the "Detecting words..." and "Sorting detected words..." spinners to finish.
3.  A visualization of the detected word bounding boxes will be displayed on the uploaded image.
4.  Below the visualization, you will see the detected word metadata (in JSON format) and buttons to "Download Cropped Words (ZIP)" and "Download Metadata (JSON)".
5.  Click the "Run OCR and Create HTML" button.
6.  The application will process the cropped word images using the OCR model and generate the HTML reconstruction.
7.  A preview of the generated HTML document will be displayed directly within the Streamlit app, followed by a "Download HTML Reconstruction" button.
8.  You can download any of the output files as needed.

**Note:** This application is specifically designed for images containing **Sylheti Nagri script**. Performance and accuracy may vary for images with significantly different layouts.


## Troubleshooting

If you see the below error after running the command `streamlit run streamlit_app.py`:

```bash
Error loading Detector model: [WinError 3] The system cannot find the path specified: 'C:\Users\{YOUR_USERNAME}/.apsis_ocr/line'
```

Solution for this error: Run the command -
```bash
mkdir C:\Users\{YOUR_USERNAME}\.apsis_ocr\line
```

Then run the Streamlit app again -
```bash
streamlit run streamlit_app.py
```



