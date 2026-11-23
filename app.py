import io
import json
import sys
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import tifffile

MODEL_DIR = Path(__file__).parent / "model"
sys.path.insert(0, str(MODEL_DIR))
from satfire_decision import decide


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_DIR / "best_model.keras", compile=False)
    with np.load(MODEL_DIR / "normalization.npz") as stats:
        mean = stats["mean"].astype("float32")
        std = stats["std"].astype("float32")
    config = json.loads((MODEL_DIR / "run_config.json").read_text())
    return model, mean, std, config


model, mean, std, config = load_assets()
TARGET_SIZE = tuple(model.input_shape[1:3])
CLASS_NAMES = {0: "Forest", 1: "Fire", 2: "Burned", 3: "Other", 255: "Ignore"}

COLORS = {
    0: (35, 132, 67),      # Forest - Green
    1: (239, 59, 44),      # Fire - Orange/Red
    2: (84, 39, 143),      # Burned - Purple
    3: (216, 179, 101),    # Other - Yellow
    255: (221, 221, 221),  # Ignore - Gray
}

def predict(image):
    valid = np.isfinite(image).all(axis=-1)
    safe_std = np.where(std == 0, 1.0, std)
    x = np.nan_to_num((image - mean) / safe_std, nan=0.0, posinf=0.0, neginf=0.0)
    x = x.astype("float32")
    probs = model(x[None], training=False).numpy()[0]
    labels = decide(probs, **config["decision"])
    labels[~valid] = 255
    return labels


def normalize_tiff(array):
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        # H, W, 8: keep VIIRS input bands 1-6
        if array.shape[-1] == 8:
            return array[..., :6]
        # H, W, 6
        if array.shape[-1] == 6:
            return array
        # 6, H, W
        if array.shape[0] == 6:
            return np.moveaxis(array, 0, -1)
        # 8, H, W
        if array.shape[0] == 8:
            return np.moveaxis(array[:6], 0, -1)
    raise ValueError(
        f"Unsupported raster shape: {array.shape}"
    )


def resize_image(image):
    if image.shape[:2] != TARGET_SIZE:
        image = cv2.resize(image, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST)
    return image.astype("float32")

def load_ground_truth():
    img = Image.open(Path(__file__).parent / "app_samples" / "expected.png")
    return img.crop((120, 73, 459, 411))


def load_uploads(files):
    if len(files) == 1:
        uploaded = files[0]
        name = uploaded.name.lower()

        if name.endswith(".npy"):
            image = np.load(io.BytesIO(uploaded.getvalue()))
            image = np.asarray(image)
            if image.shape[-1] != 6:
                raise ValueError(f"NumPy array shape {image.shape} must end in 6 bands.")
        elif name.endswith(".npz"):
            with np.load(io.BytesIO(uploaded.getvalue())) as archive:
                if "image" in archive:
                    image = archive["image"]
                elif "data" in archive:
                    image = archive["data"]
                else:
                    named_arrays = [archive[key] for key in archive.files]
                    image = named_arrays[0] if len(named_arrays) == 1 else np.stack(named_arrays, axis=-1)
            image = np.asarray(image)
            if image.shape[-1] != 6:
                raise ValueError(f"NumPy array shape {image.shape} must end in 6 bands.")
        elif name.endswith((".tif", ".tiff", ".tif")):
            image = normalize_tiff(tifffile.imread(io.BytesIO(uploaded.getvalue())))
        else:
            raise ValueError("Unsupported file. Upload .npy, .npz, .tif, or .tiff files.")
    else:
        if len(files) != 6:
            raise ValueError(f"Expected 6 single-band raster files, but received {len(files)}.")

        bands = []
        for uploaded in sorted(files, key=lambda item: item.name):
            if uploaded.name.lower().endswith((".npy", ".npz")):
                raise ValueError("Multiple uploads must be separate raster TIFF files.")
            band = normalize_tiff(tifffile.imread(io.BytesIO(uploaded.getvalue())))
            if band.ndim != 2:
                raise ValueError(f"{uploaded.name} must contain exactly one band.")
            bands.append(band)
        image = np.stack(bands, axis=-1)

    if image.ndim != 3 or image.shape[-1] != 6:
        raise ValueError(f"Loaded image shape {image.shape}; expected (height, width, 6).")

    return resize_image(image)


def colorize(labels):
    rgb = np.zeros((*labels.shape, 3), dtype="uint8")
    for label, color in COLORS.items():
        rgb[labels == label] = color
    return rgb

def display_image(image):
    rgb = image[..., [5, 1, 0]].copy()
    low, high = np.nanpercentile(rgb, [2, 98], axis=(0, 1))
    return np.nan_to_num(
        np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1)
    )

def find_ground_truth(uploaded_filename):
    samples_dir = Path(__file__).parent / "app_samples"

    for folder in samples_dir.iterdir():
        if not folder.is_dir():
            continue

        matching_tif = next(
            (
                f for f in folder.iterdir()
                if f.suffix.lower() in (".tif", ".tiff")
                and f.name.lower() == Path(uploaded_filename).name.lower()
            ),
            None,
        )

        if matching_tif:
            expected = folder / "expected.png"
            if expected.exists():
                return expected

    return None


st.set_page_config(
    page_title="SatFire Classification",
    layout="wide"
)

st.title("SatFire Satellite Image Classification")

left_col, right_col = st.columns(2, gap="large")

with right_col:
    st.subheader("Upload satellite image")

    uploaded_files = st.file_uploader(
        "Satellite raster",
        type=["tif", "tiff", "npy", "npz"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        try:
            image = load_uploads(uploaded_files)

            preview = (display_image(image) * 255).astype("uint8")

            st.image(
                preview,
                caption="Uploaded satellite image",
                width=380
            )

            run_prediction = st.button(
                "Run prediction",
                type="primary",
                use_container_width=True
            )

        except Exception as exc:
            st.error(str(exc))
            run_prediction = False
    else:
        run_prediction = False

with left_col:
    st.subheader("Classification results")

    if uploaded_files and run_prediction:
        try:
            with st.spinner("Running SatFire model..."):
                labels = predict(image)

            st.markdown("**CNN prediction**")

            st.image(
                colorize(labels),
                width=380
            )

            total = labels.size

            legend = " | ".join(
                f"**{CLASS_NAMES[label]}:** "
                f"{np.count_nonzero(labels == label) / total:.1%}"
                for label in [0, 1, 2, 3, 255]
            )

            st.markdown(legend)

            st.divider()

            st.markdown("**Ground truth (Cible)**")

            ground_truth = find_ground_truth(uploaded_files[0].name)

            if ground_truth:
                from PIL import Image

                img = Image.open(ground_truth)
                left_half = img.crop((0, 0, img.width // 2, img.height))

                st.image(left_half, width=380)
            else:
                st.warning("No matching ground truth found for this image.")

            st.image(
                load_ground_truth(),
                width=380
            )

            label_buffer = io.BytesIO()
            np.save(label_buffer, labels.astype("uint8"))

            st.download_button(
                "Download labels (.npy)",
                data=label_buffer.getvalue(),
                file_name="satfire_labels.npy",
                mime="application/octet-stream",
            )

        except Exception as exc:
            st.error(str(exc))

    else:
        st.info("Upload an image and run the prediction.")
