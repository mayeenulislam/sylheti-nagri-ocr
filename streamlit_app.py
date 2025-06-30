import streamlit as st
import cv2
import json
import numpy as np
import io
import zipfile
import base64
import collections
import os

# Assuming these dependencies are installed via requirements.txt
from apsisocr import PaddleDBNet
import tensorflow as tf

# Import necessary components from mltu
from mltu.configs import BaseModelConfigs
from mltu.transformers import ImageResizer
from mltu.annotations.images import CVImage

# Import the custom loss and metric used during training
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CWERMetric

# Import Keras layers and Model for defining the architecture
from keras import layers
from keras.models import Model
from mltu.tensorflow.model_utils import residual_block

# Enable unsafe deserialization for custom TF objects (CTCLoss, CWERMetric)
tf.keras.config.enable_unsafe_deserialization()

# --- Paths to your trained OCR model weights, config, and font ---
OCR_CONFIG_PATH = "configs.yaml"
WEIGHTS_LOCAL_PATH = "model_weights.weights.h5"
# FONT_PATH = "D:/Research/SN_OCR_APP/NotoSansSylotiNagri-Regular.ttf"
# FONT_PATH = "D:/Research/SN_OCR_APP/Surma-4.000/Surma-Regular.ttf"

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the font file
relative_font_path = "Surma-4.000/Surma-Regular.ttf"

# Join the script directory and the relative font path
FONT_PATH = os.path.join(script_dir, relative_font_path)


# --- Define ModelConfigs class (from your configs.py) ---
class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = ""
        self.vocab = ""
        self.height = 32
        self.width = 128
        self.max_text_length = 0
        self.batch_size = 1
        self.learning_rate = 1e-4
        self.train_epochs = 50
        self.train_workers = 2

# --- Define Your Model Architecture ---
def build_ocr_model(input_dim, output_dim, activation="leaky_relu", negative_slope=0.1, dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")
    input_tensor = layers.Lambda(lambda x: x / 255.0)(inputs)

    x1 = residual_block(input_tensor, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x2 = residual_block(x1, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x4 = residual_block(x3, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x6 = residual_block(x5, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x7 = residual_block(x6, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    squeezed = layers.Reshape((x7.shape[1] * x7.shape[2], x7.shape[3]))(x7)
    blstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(squeezed)
    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    model = Model(inputs=inputs, outputs=output)
    return model

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Sylheti Nagri OCR",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Sylheti Nagri OCR and HTML Reconstruction")

st.write("""
Upload an image containing Sylheti Nagri script to detect text, crop individual words, and generate an HTML reconstruction of the detected text.
""")

# --- Cached Models Loading ---
@st.cache_resource
def load_detector():
    try:
        detector_instance = PaddleDBNet(load_line_model=True)
        return detector_instance
    except Exception as e:
        st.error(f"Error loading Detector model: {e}")
        st.stop()

detector = load_detector()

@st.cache_resource
def load_ocr_model_and_configs(config_path, weights_path):
    try:
        if not os.path.exists(config_path):
            st.error(f"OCR Config file not found: {config_path}")
            st.stop()
        if not os.path.exists(weights_path):
            st.error(f"OCR Weights file not found: {weights_path}")
            st.stop()

        configs = ModelConfigs.load(config_path)
        input_shape = (configs.height, configs.width, 3)
        characters_num = len(configs.vocab)

        model = build_ocr_model(input_dim=input_shape, output_dim=characters_num)
        model.load_weights(weights_path)

        return configs, model
    except Exception as e:
        st.error(f"Error loading OCR model or configs: {e}")
        st.stop()

ocr_configs, ocr_model = load_ocr_model_and_configs(OCR_CONFIG_PATH, WEIGHTS_LOCAL_PATH)
mltu_image_resizer = ImageResizer(ocr_configs.width, ocr_configs.height)
ocr_vocab = ocr_configs.vocab

# --- Helper Functions ---
def sort_boxes_into_lines(boxes, y_threshold=15):
    word_entries = []
    for box in boxes:
        if isinstance(box[0], (list, tuple)):
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
        elif len(box) == 8:
            xs = box[::2]
            ys = box[1::2]
        elif len(box) == 4:
            xs = [box[0], box[2]]
            ys = [box[1], box[3]]
        else:
            continue

        if not xs or not ys:
            continue

        try:
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
        except ZeroDivisionError:
            continue

        x_min, y_min = int(min(xs)), int(min(ys))
        x_max, y_max = int(max(xs)), int(max(ys))

        if x_max <= x_min or y_max <= y_min:
            continue

        word_entries.append({"box": [x_min, y_min, x_max, y_max], "center_x": center_x, "center_y": center_y})

    if not word_entries:
        return []

    lines = []
    for word in sorted(word_entries, key=lambda w: w["center_y"]):
        placed = False
        for line in lines:
            line_y_min = min(w["box"][1] for w in line["words"])
            line_y_max = max(w["box"][3] for w in line["words"])
            word_y_min = word["box"][1]
            word_y_max = word["box"][3]

            if (word["center_y"] >= line["avg_y"] - y_threshold) and (word["center_y"] <= line["avg_y"] + y_threshold):
                line["words"].append(word)
                line["avg_y"] = np.mean([w["center_y"] for w in line["words"]])
                placed = True
                break

        if not placed:
            lines.append({"avg_y": word["center_y"], "words": [word]})

    sorted_lines = sorted(lines, key=lambda l: l["avg_y"])

    final_sorted = []
    for line in sorted_lines:
        sorted_words = sorted(line["words"], key=lambda w: w["center_x"])
        final_sorted.extend(sorted_words)

    for idx, word in enumerate(final_sorted, start=1):
        word["sort_key"] = idx

    return final_sorted

def visualize_word_order_on_image(img, word_data):
    vis_img = img.copy()
    if vis_img is None:
        return None

    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    height, width = vis_img.shape[:2]

    for word in word_data:
        if "bounding_box" not in word or len(word["bounding_box"]) != 4:
            continue

        x_min, y_min, x_max, y_max = word["bounding_box"]
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(width, x_max), min(height, y_max)

        if x_min >= x_max or y_min >= y_max:
            continue

        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return vis_img

def decode_predictions(preds: np.ndarray, vocab: str) -> list[str]:
    preds_tensor = tf.constant(preds, dtype=tf.float32)
    input_len = tf.ones(tf.shape(preds_tensor)[0], dtype=tf.int32) * tf.shape(preds_tensor)[1]
    decoded, _ = tf.keras.backend.ctc_decode(preds_tensor, input_length=input_len, greedy=True)

    if isinstance(decoded[0], tf.SparseTensor):
        decoded_dense = tf.sparse.to_dense(decoded[0]).numpy()
    else:
        decoded_dense = decoded[0].numpy()

    return ["".join([vocab[idx] for idx in pred if idx != -1 and idx < len(vocab)]) for pred in decoded_dense]

def embed_font_base64(font_path: str) -> str:
    try:
        with open(font_path, "rb") as font_file:
            return base64.b64encode(font_file.read()).decode("utf-8")
    except FileNotFoundError:
        st.error(f"Font file not found: {font_path}. HTML font embedding will fail.")
        return ""
    except Exception as e:
        st.error(f"Error embedding font: {e}")
        return ""

def perform_ocr_and_generate_html(sorted_word_data, cropped_images_in_memory, ocr_model, ocr_configs, font_path):
    """
    Performs OCR on cropped word images and generates HTML reconstruction.

    Args:
        sorted_word_data (list): List of dictionaries from sort_boxes_into_lines.
        cropped_images_in_memory (list): List of (filename, bytes) tuples.
        ocr_model (tf.keras.Model): The loaded mltu OCR model (with weights).
        ocr_configs (ModelConfigs): The loaded mltu model configs.
        font_path (str): Path to the font file for embedding.

    Returns:
        str: The generated HTML content, or None if processing fails.
    """
    if not sorted_word_data or not cropped_images_in_memory:
        st.warning("No word data or cropped images available for OCR.")
        return None

    # Create a dictionary for faster lookup of image bytes by filename
    cropped_images_dict = {filename: img_bytes for filename, img_bytes in cropped_images_in_memory}

    # Group words by line using their approximate vertical position for HTML structure
    # Use the original box y1 for grouping, NOT the padded box y1
    lines = collections.defaultdict(list)
    threshold = 15 # Vertical threshold for grouping words into lines

    # Sort by original box y1 then x1 before grouping
    words_for_grouping = sorted(sorted_word_data, key=lambda w: (w["bounding_box"][1], w["bounding_box"][0]))

    for word in words_for_grouping:
        y1 = word["bounding_box"][1] # Use original box y1 for initial grouping
        found_line = False
        # Find an existing line key that is close to this word's y1
        for key in list(lines.keys()): # Iterate over a copy of keys to safely modify dict
            if abs(key - y1) < threshold:
                lines[key].append(word)
                found_line = True
                break
        # If no close line key found, add this word with its y1 as a new line key
        if not found_line:
            lines[y1].append(word)

    # Sort lines by the line key (which is the approximate top y coordinate of the line)
    sorted_lines = sorted(lines.items(), key=lambda item: item[0])

    with open(font_path, "rb") as font_file:
        font_data = font_file.read()
        font_base64 = base64.b64encode(font_data).decode("utf-8")

    html_lines = [
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<style>",
        f"""
        @font-face {{
            font-family: 'SylhetiNagri';
            src: url(data:font/truetype;charset=utf-8;base64,{font_base64}) format('truetype');
        }}
        body {{
            font-family: 'SylhetiNagri', sans-serif;
            font-size: 14px;
            margin: 0;
            padding: 20px;
            background-color: white;
            color: black;
        }}
        .page-container {{
            position: relative;
            width: 800px; /* Reduced width for the preview screen */
            height: auto;
            margin: 0 auto;
            background-color: white;
        }}
        .text-line {{
            position: absolute;
            margin: 0;
            padding: 0;
            white-space: nowrap;
            color: black;
        }}
        """,
        "</style>",
        "</head>",
        "<body>",
        '<div class="page-container">'
    ]

    processed_word_count = 0
    for line_index, (line_key, line_words) in enumerate(sorted_lines):
        # Sort words horizontally within the line for correct text order
        line_words = sorted(line_words, key=lambda w: w["bounding_box"][0])

        if not line_words:
            continue # Skip empty lines

        # Get the left-most and top-most coordinates for the line div's position
        # Use original box coordinates for positioning
        line_x1 = min(w["bounding_box"][0] for w in line_words)
        line_y1 = min(w["bounding_box"][1] for w in line_words)

        # Apply vertical spacing between lines
        y1_adjusted = line_y1 + line_index * 12  # 12px extra gap per line

        # Calculate average font size for the line based on original box heights
        font_sizes = [(w["bounding_box"][3] - w["bounding_box"][1]) for w in line_words] # Use original box height
        # Add a minimum size or default in case of tiny boxes or empty list
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 14
        avg_font_size = max(avg_font_size, 8) # Ensure minimum font size

        # Use original y1 for the absolute top position of the line div
        div_style = f'style="left:{line_x1}px; top:{y1_adjusted}px; font-size:{avg_font_size}px;"'
        html_lines.append(f'<div class="text-line" {div_style}>')

        # Process each word in the line
        for word in line_words:
            try:
                filename = word["filename"] # Filename stored in metadata from cropping step

                if filename not in cropped_images_dict:
                    st.warning(f"Cropped image bytes not found for {filename}. Skipping OCR.")
                    html_lines.append(" [Image Not Found] ") # Add placeholder in HTML
                    continue

                img_bytes = cropped_images_dict[filename]

                # Decode image bytes to numpy array (BGR - 3 channels)
                img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                if img_np is None:
                    st.warning(f"Could not decode cropped image for {filename}. Skipping OCR.")
                    html_lines.append(" [Decode Error] ") # Add placeholder in HTML
                    continue

                # --- Image Preprocessing for Model Input ---
                # The model expects a 3-channel image of size (height, width, 3)
                # The mltu_image_resizer is instantiated with (configs.width, configs.height)
                # It resizes (width, height) and should preserve channels.

                # Wrap the 3-channel numpy array in CVImage
                cv_image = CVImage(img_np, "") # Second arg is annotation, not needed for resizing

                # Resize using mltu's ImageResizer (resizes to width, height)
                # It returns a list of processed CVImages, take the first element.
                # The output image array will be (height, width, 3)
                resized_cv_image = mltu_image_resizer(cv_image, "")[0]
                resized_img_np = resized_cv_image.image # Get the numpy array (height, width, 3)

                # The model's Lambda layer handles division by 255.
                # Just add the batch dimension (shape: 1, height, width, 3)
                input_image = np.expand_dims(resized_img_np, axis=0)

                # Perform OCR prediction
                # verbose=0 suppresses progress bar from prediction
                # The predict method expects a numpy array as input
                pred = ocr_model.predict(input_image, verbose=0)

                # Decode prediction
                # Use the global ocr_vocab loaded from configs
                # decode_predictions expects numpy array predictions
                decoded_word = decode_predictions(pred, ocr_vocab)[0]

                # Append decoded word to HTML with a space separator
                html_lines.append(decoded_word + " ")
                processed_word_count += 1

            except Exception as e:
                # Log error and add a placeholder in HTML if OCR fails for a word
                st.warning(f"Error processing {word.get('filename', 'unknown word')}: {str(e)}")
                html_lines.append(" [OCR Error] ")

        html_lines.append("</div>") # Close the text-line div

    html_lines.extend([
        "</div>", # Close page-container
        "</body>",
        "</html>"
    ])

    st.success(f"Successfully processed {processed_word_count} words and generated HTML.")
    return "\n".join(html_lines)



# --- Streamlit App Flow ---
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if 'original_img' not in st.session_state:
    st.session_state.original_img = None
    st.session_state.sorted_word_data = None
    st.session_state.cropped_images_in_memory = None
    st.session_state.metadata_for_json = None
    st.session_state.zip_bytes = None
    st.session_state.html_output = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original_img is None:
        st.error("Could not decode the image. Please try another file.")
        st.session_state.original_img = None
        st.session_state.sorted_word_data = None
        st.session_state.cropped_images_in_memory = None
        st.session_state.metadata_for_json = None
        st.session_state.zip_bytes = None
        st.session_state.html_output = None
    else:
        st.session_state.original_img = original_img.copy()
        st.subheader("Step 1: Detect Words & Crop")

        with st.spinner("Detecting words..."):
            word_boxes_raw = detector.get_word_boxes(st.session_state.original_img)

        if not word_boxes_raw:
            st.warning("No words detected in the image.")
            st.image(st.session_state.original_img, channels="BGR", caption="Original Image (No words detected)")
            st.session_state.sorted_word_data = None
            st.session_state.cropped_images_in_memory = None
            st.session_state.metadata_for_json = None
            st.session_state.zip_bytes = None
            st.session_state.html_output = None
        else:
            with st.spinner("Sorting detected words..."):
                sorted_word_data = sort_boxes_into_lines(word_boxes_raw)

            st.session_state.sorted_word_data = sorted_word_data

            if not st.session_state.sorted_word_data:
                st.warning("Detected words could not be sorted into lines.")
                st.image(st.session_state.original_img, channels="BGR", caption="Original Image (Detected words, but sorting failed)")
                st.session_state.cropped_images_in_memory = None
                st.session_state.metadata_for_json = None
                st.session_state.zip_bytes = None
                st.session_state.html_output = None
            else:
                metadata_for_json = []
                cropped_images_in_memory = []
                pad = 3

                with st.spinner("Cropping words and generating metadata..."):
                    if 'ocr_configs' not in locals() or ocr_configs is None:
                        st.error("OCR configurations not loaded. Cannot proceed with cropping based on model height.")
                        st.session_state.cropped_images_in_memory = None
                        st.session_state.metadata_for_json = None
                        st.session_state.zip_bytes = None
                        st.session_state.html_output = None
                        crops_generation_failed = True
                    else:
                        target_height = ocr_configs.height
                        crops_generation_failed = False

                    if not crops_generation_failed:
                        for word_entry in st.session_state.sorted_word_data:
                            x_min_orig, y_min_orig, x_max_orig, y_max_orig = word_entry['box']
                            sort_key = word_entry['sort_key']
                            center_x = word_entry['center_x']
                            center_y = word_entry['center_y']

                            h_img, w_img = st.session_state.original_img.shape[:2]
                            x1 = max(x_min_orig - pad, 0)
                            y1 = max(y_min_orig - pad, 0)
                            x2 = min(x_max_orig + pad, w_img)
                            y2 = min(y_max_orig + pad, h_img)

                            if x1 >= x2 or y1 >= y2:
                                continue

                            crop = st.session_state.original_img[y1:y2, x1:x2]

                            h_crop, w_crop = crop.shape[:2]
                            if h_crop > 0:
                                scale = target_height / h_crop
                                resized_w = max(1, int(w_crop * scale))
                                interp_method = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
                                resized_crop = cv2.resize(crop, (resized_w, target_height), interpolation=interp_method)
                            else:
                                continue

                            filename = f"word_{sort_key:04d}.png"
                            is_success, buffer = cv2.imencode(".png", resized_crop)
                            if not is_success:
                                st.error(f"Failed to encode image for word {sort_key}")
                                continue

                            image_bytes = buffer.tobytes()
                            cropped_images_in_memory.append((filename, image_bytes))

                            metadata_for_json.append({
                                "sort_key": sort_key,
                                "bounding_box": [x1, y1, x2, y2],
                                "center_x": int(center_x),
                                "center_y": int(center_y),
                                "filename": filename
                            })

                    st.session_state.cropped_images_in_memory = cropped_images_in_memory
                    st.session_state.metadata_for_json = metadata_for_json

                    if not st.session_state.cropped_images_in_memory:
                        st.warning("No valid word crops were generated.")
                        st.session_state.metadata_json_string = None
                        st.session_state.zip_bytes = None
                        st.session_state.html_output = None
                    else:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for filename, img_bytes in st.session_state.cropped_images_in_memory:
                                zipf.writestr(filename, img_bytes)
                        st.session_state.zip_bytes = zip_buffer.getvalue()

                        st.session_state.metadata_json_string = json.dumps({"word_data": st.session_state.metadata_for_json}, indent=2, ensure_ascii=False)

if st.session_state.sorted_word_data is not None and st.session_state.original_img is not None:
    if st.session_state.cropped_images_in_memory is not None:
        st.subheader("Results: ")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Word Location Visualization:")
            vis_img = visualize_word_order_on_image(st.session_state.original_img.copy(), st.session_state.metadata_for_json)
            if vis_img is not None:
                st.image(vis_img, channels="BGR", caption="Detected Word Bounding Boxes", use_container_width=True)
            else:
                st.warning("Could not generate visualization image.")

        with col2:
            st.write("Metadata and Downloads:")
            if st.session_state.metadata_for_json:
                st.json({"word_data": st.session_state.metadata_for_json})
            else:
                st.info("No metadata available.")

            if st.session_state.zip_bytes is not None:
                st.download_button(
                    label="Download Cropped Words (ZIP)",
                    data=st.session_state.zip_bytes,
                    file_name="cropped_words.zip",
                    mime="application/zip"
                )
            else:
                st.info("No cropped words to download.")

            if st.session_state.metadata_json_string is not None:
                st.download_button(
                    label="Download Metadata (JSON)",
                    data=st.session_state.metadata_json_string,
                    file_name="metadata.json",
                    mime="application/json"
                )
            else:
                st.info("No metadata to download.")

if st.session_state.cropped_images_in_memory and 'ocr_model' in locals() and ocr_model is not None:
    st.subheader("Step 2: Perform OCR & Generate HTML")
    if st.button("Run OCR and Create HTML"):
        if not os.path.exists(FONT_PATH):
            st.error(f"Font file not found at {FONT_PATH}. Cannot generate HTML.")
            st.session_state.html_output = None
        else:
            with st.spinner("Running OCR and generating HTML..."):
                html_content = perform_ocr_and_generate_html(
                    st.session_state.metadata_for_json,
                    st.session_state.cropped_images_in_memory,
                    ocr_model,
                    ocr_configs,
                    FONT_PATH
                )
                st.session_state.html_output = html_content

if st.session_state.html_output is not None:
    st.subheader("Step 2 Results: HTML Reconstruction")
    st.write("Preview (rendering may vary):")
    st.components.v1.html(st.session_state.html_output, height=600, width=800, scrolling=True)

    st.download_button(
        label="Download HTML Reconstruction",
        data=st.session_state.html_output,
        file_name="reconstructed_document.html",
        mime="text/html"
    )

