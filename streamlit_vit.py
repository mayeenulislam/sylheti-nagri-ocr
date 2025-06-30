import streamlit as st
import cv2
import json
import numpy as np
import io
import zipfile
import base64
import collections
import os
import tensorflow as tf
from apsisocr import PaddleDBNet
from keras import layers
from keras.models import Model
from mltu.configs import BaseModelConfigs
from mltu.transformers import ImageResizer
from mltu.annotations.images import CVImage
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CWERMetric

# Enable unsafe deserialization for custom TF objects (CTCLoss, CWERMetric, and custom layers)
tf.keras.config.enable_unsafe_deserialization()

# --- Paths to your trained ViT model weights, config, and font ---
OCR_CONFIG_PATH = "vit_configs.yaml"
WEIGHTS_LOCAL_PATH = "vit_model_weights.weights.h5"
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
            st.error(f"Error: Config file not found at {file_path}")
            return cls()
        except Exception as e:
            st.error(f"Error loading config file {file_path}: {e}")
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
        if not os.path.exists(FONT_PATH):
            st.warning(f"Font file not found at {FONT_PATH}. HTML font embedding will not work.")

        configs = ModelConfigs.load(config_path)
        if not hasattr(configs, 'vocab') or not configs.vocab:
            st.error("Vocabulary not found or is empty in the config file. Cannot build model.")
            st.stop()
        if not hasattr(configs, 'height') or not hasattr(configs, 'width'):
            st.error("Image dimensions (height, width) not found in the config file. Cannot build model.")
            st.stop()
        if not hasattr(configs, 'patch_size') or not hasattr(configs, 'embedding_dim') or \
           not hasattr(configs, 'num_transformer_layers') or not hasattr(configs, 'num_heads') or \
           not hasattr(configs, 'mlp_dim') or not hasattr(configs, 'transformer_dropout') or \
           not hasattr(configs, 'embedding_dropout'):
            st.error("Required ViT specific configurations not found in the config file. Cannot build model.")
            st.stop()

        input_shape = (configs.height, configs.width, 3)
        characters_num = len(configs.vocab)
        model = build_vit_model(
            input_dim=input_shape,
            output_dim=characters_num,
            patch_size=configs.patch_size,
            embedding_dim=configs.embedding_dim,
            num_transformer_layers=configs.num_transformer_layers,
            num_heads=configs.num_heads,
            mlp_dim=configs.mlp_dim,
            transformer_dropout=configs.transformer_dropout,
            embedding_dropout=configs.embedding_dropout
        )
        custom_objects_for_loading = {
            "CTCloss": CTCloss,
            "CWERMetric": CWERMetric(padding_token=len(configs.vocab)),
            "Patches": Patches,
            "PatchEncoder": PatchEncoder,
        }
        model.load_weights(weights_path)
        return configs, model
    except Exception as e:
        st.error(f"Error loading OCR model or configs: {e}")
        st.stop()

ocr_configs, ocr_model = load_ocr_model_and_configs(OCR_CONFIG_PATH, WEIGHTS_LOCAL_PATH)
mltu_image_resizer = ImageResizer(ocr_configs.width, ocr_configs.height, keep_aspect_ratio=True)
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
            x_min, y_min = int(min(xs)), int(min(ys))
            x_max, y_max = int(max(xs)), int(max(ys))
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
        except (ValueError, TypeError, ZeroDivisionError) as e:
            continue

        if x_max <= x_min or y_max <= y_min:
            continue

        word_entries.append({"original_box": [x_min, y_min, x_max, y_max], "center_x": center_x, "center_y": center_y})

    if not word_entries:
        return []

    lines = []
    for word in sorted(word_entries, key=lambda w: w["center_y"]):
        placed = False
        for line in lines:
            if (word["center_y"] >= line["avg_y"] - y_threshold) and (word["center_y"] <= line["avg_y"] + y_threshold):
                line["words"].append(word)
                line["avg_y"] = np.mean([w["center_y"] for w in line["words"]])
                placed = True
                break

        if not placed:
            lines.append({"avg_y": word["center_y"], "words": [word]})

    sorted_lines = sorted(lines, key=lambda l: l["avg_y"])
    final_sorted_words_with_filenames = []
    word_counter = 0
    for line in sorted_lines:
        sorted_words_in_line = sorted(line["words"], key=lambda w: w["center_x"])
        for word in sorted_words_in_line:
            word_counter += 1
            word_data_with_filename = {
                "sort_key": word_counter,
                "original_box": word["original_box"],
                "center_x": int(word["center_x"]),
                "center_y": int(word["center_y"]),
                "filename": f"word_{word_counter:04d}.png"
            }
            final_sorted_words_with_filenames.append(word_data_with_filename)

    return final_sorted_words_with_filenames

def visualize_word_order_on_image(img, word_data):
    vis_img = img.copy()
    if vis_img is None:
        return None

    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    height, width = vis_img.shape[:2]

    for word in word_data:
        box_to_draw = word.get("padded_box", word.get("original_box"))

        if box_to_draw is None or len(box_to_draw) != 4:
            continue

        x_min, y_min, x_max, y_max = box_to_draw
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(width, x_max), min(height, y_max)

        if x_min >= x_max or y_min >= y_max:
            continue

        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return vis_img

def decode_predictions(preds: np.ndarray, vocab: str) -> list[str]:
    preds_tensor = tf.constant(preds, dtype=tf.float32)
    if len(preds_tensor.shape) == 2:
        preds_tensor = tf.expand_dims(preds_tensor, axis=0)

    input_len = tf.ones(tf.shape(preds_tensor)[0], dtype=tf.int32) * tf.shape(preds_tensor)[1]
    decoded, _ = tf.keras.backend.ctc_decode(preds_tensor, input_length=input_len, greedy=True)
    decoded_tensor = decoded[0]

    if isinstance(decoded_tensor, tf.SparseTensor):
        decoded_dense = tf.sparse.to_dense(decoded_tensor).numpy()
    else:
        decoded_dense = decoded_tensor.numpy()

    decoded_texts = []
    for pred in decoded_dense:
        text = "".join([vocab[idx] for idx in pred if idx != -1 and idx < len(vocab)])
        decoded_texts.append(text)

    return decoded_texts

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

def perform_ocr_and_generate_html(sorted_word_data, cropped_images_in_memory, ocr_model, ocr_configs, font_path, mltu_image_resizer):
    if not sorted_word_data or not cropped_images_in_memory:
        st.warning("No word data or cropped images available for OCR.")
        return None

    cropped_images_dict = {filename: img_bytes for filename, img_bytes in cropped_images_in_memory}
    lines = collections.defaultdict(list)
    threshold = 15

    words_for_grouping = sorted(sorted_word_data, key=lambda w: (w["original_box"][1], w["original_box"][0]))

    for word in words_for_grouping:
        y1 = word["original_box"][1]
        found_line = False
        for key in list(lines.keys()):
            if abs(key - y1) < threshold:
                lines[key].append(word)
                found_line = True
                break
        if not found_line:
            lines[y1].append(word)

    sorted_lines = sorted(lines.items(), key=lambda item: item[0])

    font_base64 = embed_font_base64(font_path)
    font_style = ""
    if font_base64:
        font_style = f"""
        @font-face {{
            font-family: 'SylhetiNagri';
            src: url(data:font/truetype;charset=utf-8;base64,{font_base64}) format('truetype');
        }}
        body {{
            font-family: 'SylhetiNagri', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: white;
            color: black;
        }}
        """
    else:
        st.warning("Font embedding failed. Using default sans-serif font.")

    html_lines = [
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<style>",
        """
        .page-container {
            position: relative;
            width: 800px;
            height: auto;
            margin: 0 auto;
            background-color: white;
        }
        .text-line {
            position: absolute;
            margin: 0;
            padding: 0;
            white-space: nowrap;
            color: black;
        }
        """,
        font_style,
        "</style>",
        "</head>",
        "<body>",
        '<div class="page-container">'
    ]

    processed_word_count = 0
    for line_index, (line_key, line_words) in enumerate(sorted_lines):
        line_words = sorted(line_words, key=lambda w: w["original_box"][0])

        if not line_words:
            continue

        line_x1 = min(w["original_box"][0] for w in line_words)
        line_y1 = min(w["original_box"][1] for w in line_words)

        font_sizes = [(w["original_box"][3] - w["original_box"][1]) for w in line_words]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 14
        avg_font_size = max(avg_font_size, 8)

        div_style = f'style="left:{line_x1}px; top:{line_y1}px; font-size:{avg_font_size}px;"'
        html_lines.append(f'<div class="text-line" {div_style}>')

        for word in line_words:
            try:
                filename = word["filename"]

                if filename not in cropped_images_dict:
                    st.warning(f"Cropped image bytes not found for {filename}. Skipping OCR.")
                    html_lines.append(" [Image Not Found] ")
                    continue

                img_bytes = cropped_images_dict[filename]
                img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                if img_np is None:
                    st.warning(f"Could not decode cropped image for {filename}. Skipping OCR.")
                    html_lines.append(" [Decode Error] ")
                    continue

                cv_image = CVImage(img_np, "")
                resized_cv_image = mltu_image_resizer(cv_image, "")[0]
                resized_img_np = resized_cv_image.image

                input_image = np.expand_dims(resized_img_np, axis=0)
                pred = ocr_model.predict(input_image, verbose=0)
                decoded_word = decode_predictions(pred, ocr_vocab)[0]

                html_lines.append(decoded_word + " ")
                processed_word_count += 1

            except Exception as e:
                st.warning(f"Error processing {word.get('filename', 'unknown word')}: {str(e)}")
                html_lines.append(" [OCR Error] ")

        html_lines.append("</div>")

    html_lines.extend([
        "</div>",
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
    st.session_state.metadata_json_string = None

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
        st.session_state.metadata_json_string = None
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
            st.session_state.metadata_json_string = None
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
                st.session_state.metadata_json_string = None
            else:
                metadata_for_json = []
                cropped_images_in_memory = []
                pad = 3

                with st.spinner("Cropping words and generating metadata..."):
                    if 'ocr_configs' not in locals() or ocr_configs is None or not hasattr(ocr_configs, 'height') or not hasattr(ocr_configs, 'width'):
                        st.error("OCR configurations not loaded. Cannot proceed with cropping based on model height.")
                        st.session_state.cropped_images_in_memory = None
                        st.session_state.metadata_for_json = None
                        st.session_state.zip_bytes = None
                        st.session_state.html_output = None
                        st.session_state.metadata_json_string = None
                        crops_generation_failed = True
                    else:
                        target_height = ocr_configs.height
                        target_width = ocr_configs.width
                        crops_generation_failed = False

                    if not crops_generation_failed:
                        h_img, w_img = st.session_state.original_img.shape[:2]

                        for word_entry in st.session_state.sorted_word_data:
                            x_min_orig, y_min_orig, x_max_orig, y_max_orig = word_entry['original_box']
                            sort_key = word_entry['sort_key']
                            center_x = word_entry['center_x']
                            center_y = word_entry['center_y']

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
                                "original_box": [x_min_orig, y_min_orig, x_max_orig, y_max_orig],
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
                    FONT_PATH,
                    mltu_image_resizer
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
