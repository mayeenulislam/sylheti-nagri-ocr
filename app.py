import os
import shutil
import base64
import collections
import io
import json
import tempfile
import zipfile
from functools import lru_cache

import cv2
import numpy as np
import gradio as gr
import spaces

from apsisocr import PaddleDBNet
import tensorflow as tf

from mltu.configs import BaseModelConfigs
from mltu.transformers import ImageResizer
from mltu.annotations.images import CVImage

from keras import layers
from keras.models import Model
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CWERMetric
from mltu.tensorflow.model_utils import residual_block

tf.keras.config.enable_unsafe_deserialization()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(SCRIPT_DIR, "Surma-4.000", "Surma-Regular.ttf")
BUNDLED_DETECTOR_DIR = os.path.join(SCRIPT_DIR, "models", "detector", "word")
DETECTOR_FILES = ["inference.pdmodel", "inference.pdiparams", "inference.pdiparams.info"]


def seed_detector_models(home_dir=None):
    base = home_dir or os.path.expanduser("~")
    target = os.path.join(base, ".apsis_ocr", "word")
    os.makedirs(target, exist_ok=True)
    for f in DETECTOR_FILES:
        dest = os.path.join(target, f)
        if not os.path.exists(dest):
            shutil.copy2(os.path.join(BUNDLED_DETECTOR_DIR, f), dest)


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


class VITModelConfigs(ModelConfigs):
    def __init__(self):
        super().__init__()

        # ViT specific parameters
        self.patch_size = 8
        self.embedding_dim = 192
        self.num_transformer_layers = 6
        self.num_heads = 6
        self.mlp_dim = 768
        self.transformer_dropout = 0.1
        self.embedding_dropout = 0.1

    @classmethod
    def load(cls, file_path):
        import yaml
        try:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            configs = cls()
            for key, value in config_dict.items():
                setattr(configs, key, value)
            return configs
        except FileNotFoundError:
            raise RuntimeError(f"Error: Config file not found at {file_path}")
            return cls()
        except Exception as e:
            raise RuntimeError(f"Error loading config file {file_path}: {e}")
            return cls()


# --- Define Your Model Architecture ---
class Patches(layers.Layer):
    def __init__(self, patch_size, image_width, image_height, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.image_width = image_width
        self.image_height = image_height
        self.num_patches_w = image_width // patch_size
        self.num_patches_h = image_height // patch_size
        self.num_patches = self.num_patches_w * self.num_patches_h

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "image_width": self.image_width,
            "image_height": self.image_height,
        })
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, embedding_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.dropout = layers.Dropout(embedding_dropout)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        projected_patches = self.projection(patches)
        encoded = projected_patches + self.position_embedding(positions)
        encoded = self.dropout(encoded)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
            "embedding_dropout": self.dropout.rate,
        })
        return config


def vit_transformer_block(x, embedding_dim, num_heads, mlp_dim, dropout_rate):
    x_norm1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=dropout_rate
    )(x_norm1, x_norm1)
    x_attention = layers.Add()([x, attention_output])

    x_norm2 = layers.LayerNormalization(epsilon=1e-6)(x_attention)
    mlp_output = layers.Dense(mlp_dim, activation=tf.nn.gelu)(x_norm2)
    mlp_output = layers.Dropout(dropout_rate)(mlp_output)
    mlp_output = layers.Dense(embedding_dim)(mlp_output)
    mlp_output = layers.Dropout(dropout_rate)(mlp_output)
    x_mlp = layers.Add()([x_attention, mlp_output])
    return x_mlp


def build_vit_model(input_dim, output_dim, patch_size, embedding_dim, num_transformer_layers, num_heads, mlp_dim, transformer_dropout, embedding_dropout):
    inputs = layers.Input(shape=input_dim, name="input_image")
    normalized_images = layers.Lambda(lambda x: x / 255.0, name="normalization")(inputs)
    patches = Patches(patch_size, input_dim[1], input_dim[0], name="patches")(normalized_images)
    num_patches = (input_dim[1] // patch_size) * (input_dim[0] // patch_size)
    encoded_patches = PatchEncoder(num_patches, embedding_dim, embedding_dropout, name="patch_encoder")(patches)
    x = encoded_patches
    for i in range(num_transformer_layers):
        x = vit_transformer_block(x, embedding_dim, num_heads, mlp_dim, transformer_dropout)
    ctc_output = layers.Dense(output_dim + 1, activation="softmax", name="output_ctc")(x)
    model = Model(inputs=inputs, outputs=ctc_output)
    return model


def generate_html(sorted_word_data, crops, model, resizer, vocab, font_path):
    """
    Performs OCR on cropped word images and generates HTML reconstruction.

    Args:
        sorted_word_data (list): List of dictionaries from sort_boxes_into_lines.
        crops (list): List of (filename, bytes) tuples.
        model (tf.keras.Model): The loaded mltu OCR model (with weights).
        resizer: The mltu ImageResizer instance.
        vocab (str): The model vocabulary string.
        font_path (str): Path to the font file for embedding.

    Returns:
        str: The generated HTML content, or None if processing fails.
    """
    if not sorted_word_data or not crops:
        print("No word data or crops available for OCR.")
        return None

    # Create a dictionary for faster lookup of image bytes by filename
    cropped_images_dict = {filename: img_bytes for filename, img_bytes in crops}

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
    max_bottom = 0
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

        max_bottom = max(max_bottom, y1_adjusted + avg_font_size)

        # Process each word in the line
        for word in line_words:
            try:
                filename = word["filename"] # Filename stored in metadata from cropping step

                if filename not in cropped_images_dict:
                    print("WARN:", f"Cropped image bytes not found for {filename}. Skipping OCR.")
                    html_lines.append(" [Image Not Found] ") # Add placeholder in HTML
                    continue

                img_bytes = cropped_images_dict[filename]

                # Decode image bytes to numpy array (BGR - 3 channels)
                img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                if img_np is None:
                    print("WARN:", f"Could not decode cropped image for {filename}. Skipping OCR.")
                    html_lines.append(" [Decode Error] ") # Add placeholder in HTML
                    continue

                # --- Image Preprocessing for Model Input ---
                # The model expects a 3-channel image of size (height, width, 3)
                # The resizer is instantiated with (configs.width, configs.height)
                # It resizes (width, height) and should preserve channels.

                # Wrap the 3-channel numpy array in CVImage
                cv_image = CVImage(img_np, "") # Second arg is annotation, not needed for resizing

                # Resize using mltu's ImageResizer (resizes to width, height)
                # It returns a list of processed CVImages, take the first element.
                # The output image array will be (height, width, 3)
                resized_cv_image = resizer(cv_image, "")[0]
                resized_img_np = resized_cv_image.image # Get the numpy array (height, width, 3)

                # The model's Lambda layer handles division by 255.
                # Just add the batch dimension (shape: 1, height, width, 3)
                input_image = np.expand_dims(resized_img_np, axis=0)

                # Perform OCR prediction
                # verbose=0 suppresses progress bar from prediction
                # The predict method expects a numpy array as input
                pred = model.predict(input_image, verbose=0)

                # Decode prediction
                # Use the vocab loaded from configs
                # decode_predictions expects numpy array predictions
                decoded_word = decode_predictions(pred, vocab)[0]

                # Append decoded word to HTML with a space separator
                html_lines.append(decoded_word + " ")
                processed_word_count += 1

            except Exception as e:
                # Log error and add a placeholder in HTML if OCR fails for a word
                print("WARN:", f"Error processing {word.get('filename', 'unknown word')}: {str(e)}")
                html_lines.append(" [OCR Error] ")

        html_lines.append("</div>") # Close the text-line div

    html_lines.extend([
        "</div>", # Close page-container
        "</body>",
        "</html>"
    ])

    print(f"Successfully processed {processed_word_count} words and generated HTML.")
    html = "\n".join(html_lines)
    return html.replace("height: auto;", f"height: {int(max_bottom) + 20}px;", 1)


from functools import lru_cache

RECOGNIZERS = {
    "CRNN": {"config": "configs.yaml", "weights": "model_weights.weights.h5"},
    "ViT": {"config": "vit_configs.yaml", "weights": "vit_model_weights.weights.h5"},
}


@lru_cache(maxsize=None)
def _load_detector():
    seed_detector_models()
    return PaddleDBNet(use_gpu=False, load_line_model=False)


@lru_cache(maxsize=None)
def _load_recognizer(name):
    cfg_path = os.path.join(SCRIPT_DIR, RECOGNIZERS[name]["config"])
    weights_path = os.path.join(SCRIPT_DIR, RECOGNIZERS[name]["weights"])
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    if name == "CRNN":
        configs = ModelConfigs.load(cfg_path)
        input_shape = (configs.height, configs.width, 3)
        model = build_ocr_model(input_dim=input_shape, output_dim=len(configs.vocab))
        resizer = ImageResizer(configs.width, configs.height)
    else:  # ViT
        configs = VITModelConfigs.load(cfg_path)
        if not (configs.vocab and configs.height and configs.width and configs.patch_size
                and configs.embedding_dim and configs.num_transformer_layers
                and configs.num_heads and configs.mlp_dim):
            raise ValueError("Required ViT configurations missing from vit_configs.yaml")
        input_shape = (configs.height, configs.width, 3)
        model = build_vit_model(
            input_dim=input_shape,
            output_dim=len(configs.vocab),
            patch_size=configs.patch_size,
            embedding_dim=configs.embedding_dim,
            num_transformer_layers=configs.num_transformer_layers,
            num_heads=configs.num_heads,
            mlp_dim=configs.mlp_dim,
            transformer_dropout=configs.transformer_dropout,
            embedding_dropout=configs.embedding_dropout,
        )
        resizer = ImageResizer(configs.width, configs.height, keep_aspect_ratio=True)

    model.load_weights(weights_path)
    return configs, model, resizer, configs.vocab


def get_recognizer(name):
    if name not in RECOGNIZERS:
        raise ValueError(f"Unknown model: {name}")
    return _load_recognizer(name)


SAMPLE_IMAGES = [
    {"file": "Gospel-Mathew.PNG", "label": "Gospel of Mathew"},
    {"file": "Book-Exodus.JPG", "label": "Book of Exodus"},
    {"file": "Sylheti-Folklore.PNG", "label": "Folklore: Shaat Koinar Bakhan"},
]
PAD = 3


def load_sample(sample_name):
    if not sample_name or sample_name == "Upload my own":
        return None
    path = os.path.join(SCRIPT_DIR, "sample_page", sample_name)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Sample image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_and_crop(image_rgb):
    if image_rgb is None:
        return (None, None, None, None, None, "Upload an image or pick a sample first.")

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    detector = _load_detector()

    try:
        word_boxes_raw = detector.get_word_boxes(img_bgr)
    except Exception as e:
        return (None, None, None, None, None, f"Word detection failed: {e}")

    if not word_boxes_raw:
        return (None, None, None, None, None, "No words detected in the image.")

    sorted_word_data = sort_boxes_into_lines(word_boxes_raw)
    if not sorted_word_data:
        return (None, None, None, None, None, "Detected words could not be sorted into lines.")

    target_height = get_recognizer("CRNN")[0].height
    h_img, w_img = img_bgr.shape[:2]
    crops, metadata = [], []

    for word_entry in sorted_word_data:
        x_min_orig, y_min_orig, x_max_orig, y_max_orig = word_entry["box"]
        sort_key = word_entry["sort_key"]
        x1, y1 = max(x_min_orig - PAD, 0), max(y_min_orig - PAD, 0)
        x2, y2 = min(x_max_orig + PAD, w_img), min(y_max_orig + PAD, h_img)
        if x1 >= x2 or y1 >= y2:
            continue

        crop = img_bgr[y1:y2, x1:x2]
        h_crop, w_crop = crop.shape[:2]
        if h_crop <= 0:
            continue

        scale = target_height / h_crop
        resized_w = max(1, int(w_crop * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized_crop = cv2.resize(crop, (resized_w, target_height), interpolation=interp)

        filename = f"word_{sort_key:04d}.png"
        is_success, buffer = cv2.imencode(".png", resized_crop)
        if not is_success:
            continue

        crops.append((filename, buffer.tobytes()))
        metadata.append({
            "sort_key": sort_key,
            "bounding_box": [x1, y1, x2, y2],
            "center_x": int(word_entry["center_x"]),
            "center_y": int(word_entry["center_y"]),
            "filename": filename,
        })

    if not crops:
        return (None, None, None, None, None, "No valid word crops were generated.")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename, img_bytes in crops:
            zipf.writestr(filename, img_bytes)
    json_str = json.dumps({"word_data": metadata}, indent=2, ensure_ascii=False)

    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip")
    json_fd, json_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(zip_fd, "wb") as f:
        f.write(zip_buffer.getvalue())
    with os.fdopen(json_fd, "w") as f:
        f.write(json_str)

    state = {"original_img": img_bgr, "crops": crops, "metadata": metadata}
    viz = visualize_word_order_on_image(img_bgr, metadata)
    return (state, viz, {"word_data": metadata}, zip_path, json_path, "")


def run_ocr(model_name, state):
    if not state or not state.get("crops"):
        return (None, None, "Run Step 1 (Detect & Crop) first.")
    try:
        configs, model, resizer, vocab = get_recognizer(model_name)
        html = generate_html(state["metadata"], state["crops"], model, resizer, vocab, FONT_PATH)
        if html is None:
            return (None, None, "OCR produced no output.")
        fd, html_path = tempfile.mkstemp(suffix=".html")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html)
        return (html, html_path, "")
    except Exception as e:
        return (None, None, f"OCR failed: {e}")


def _as_rgb(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


@spaces.GPU(duration=60)
def _ui_detect_and_crop(image_rgb):
    state, viz, meta, zip_path, json_path, status = detect_and_crop(image_rgb)
    return (
        state,
        _as_rgb(viz) if viz is not None else None,
        meta,
        zip_path,
        json_path,
        status if status else "Step 1 complete: words detected and cropped.",
    )


@spaces.GPU(duration=60)
def _ui_run_ocr(model_name, image_rgb, state):
    viz = meta = zip_path = json_path = None
    if not state or not state.get("crops"):
        state, viz, meta, zip_path, json_path, status = detect_and_crop(image_rgb)
        if status:
            return (None, None, None, None, None, None, None, None, f"Run Step 1 first. {status}")
    html, html_path, status = run_ocr(model_name, state)
    return (
        html,
        html_path,
        state,
        _as_rgb(viz) if viz is not None else None,
        meta,
        zip_path,
        json_path,
        status if status else "Step 2 complete: HTML generated.",
        html if html else "",
    )


def _ui_load_sample(sample_name):
    try:
        img = load_sample(sample_name)
        return img
    except FileNotFoundError as e:
        raise gr.Error(str(e))


with gr.Blocks(title="Sylheti Nagri OCR") as demo:
    logo_b64 = base64.b64encode(open(os.path.join(SCRIPT_DIR, "assets", "nagri-ocr-logo.png"), "rb").read()).decode()
    gr.Markdown(
        f'<div style="display: flex; align-items: center; gap: 12px;">'
        f'<img src="data:image/png;base64,{logo_b64}" style="width: 56px; height: 56px; border-radius: 12px; flex-shrink: 0;">'
        f'<div>'
        f'<h1 style="margin: 0; font-size: 1.6rem;">Sylheti Nagri OCR and HTML Reconstruction</h1>'
        f'<p style="margin: 0; color: #666;">Upload an image containing Sylheti Nagri script, or pick a sample, '
        f'to detect words, crop them, and generate an HTML reconstruction.</p>'
        f'</div>'
        f'</div>'
    )

    # ROW 1: settings (left) | image input + step 1 (right)
    with gr.Row():
        with gr.Column():
            model_choice = gr.Radio(["CRNN", "ViT"], value="CRNN", label="OCR Model")
            gr.Markdown("**Upload an Image Containing Nagri or Use a Sample:**")
            sample_buttons = []
            with gr.Row():
                for sample in SAMPLE_IMAGES:
                    with gr.Column(min_width=110):
                        gr.Image(
                            value=load_sample(sample["file"]),
                            interactive=False,
                            show_label=False,
                            height=90,
                            width=150,
                        )
                        btn = gr.Button(sample["label"], size="sm")
                        sample_buttons.append((btn, sample["file"]))
            upload_btn = gr.Button("Upload your Own")
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Image (upload or from sample)")
            detect_btn = gr.Button("Step 1: Detect & Crop")

    for btn, fname in sample_buttons:
        btn.click(lambda f=fname: _ui_load_sample(f), None, image_input)
    upload_btn.click(lambda: None, None, image_input)

    # ROW 2: detected boxes (left) | metadata + downloads (right)
    with gr.Row():
        with gr.Column():
            viz_out = gr.Image(label="Detected Word Bounding Boxes")
        with gr.Column():
            metadata_out = gr.JSON(label="Metadata (JSON)")
            with gr.Row():
                zip_out = gr.File(label="Download Cropped Words (ZIP)")
                json_out = gr.File(label="Download Metadata (JSON)")

    # ROW 3: step 2 (primary) + HTML output + download + copy
    ocr_btn = gr.Button("Step 2: Run OCR & Generate HTML", variant="primary")
    html_out = gr.HTML(label="HTML Output")
    with gr.Row():
        html_file_out = gr.File(label="Download HTML Reconstruction", scale=0)
        copy_btn = gr.Button("Copy HTML", scale=0)
    html_src = gr.Textbox(visible=False)
    copy_status = gr.Markdown()

    status = gr.Markdown()
    gr.Markdown(
        '<p style="text-align: center; color: #888; margin-top: 20px;">'
        'Made by Md Ariful Haque, Md. Mostafizur Rahman, Mayeenul Islam Mayeen, Fabiha Farzana</p>'
    )

    state = gr.State()

    detect_btn.click(
        _ui_detect_and_crop,
        [image_input],
        [state, viz_out, metadata_out, zip_out, json_out, status],
    )
    ocr_btn.click(
        _ui_run_ocr,
        [model_choice, image_input, state],
        [html_out, html_file_out, state, viz_out, metadata_out, zip_out, json_out, status, html_src],
    )
    copy_btn.click(
        None,
        [html_src],
        [copy_status],
        js="(src) => { navigator.clipboard.writeText(src || ''); return 'HTML copied to clipboard.'; }",
    )


if __name__ == "__main__":
    demo.launch(favicon_path=os.path.join(SCRIPT_DIR, "assets", "nagri-ocr-logo.png"))
