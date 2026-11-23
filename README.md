# SatFire: Active Fire Detection on Satellite Imagery Using CNN

A deep learning project that explores active fire detection and burned-area mapping using multispectral satellite imagery.

SatFire uses a Convolutional Neural Network (CNN) to perform pixel-wise classification, distinguishing between forests, active fires, burned areas, and other land-cover regions.

The project includes a trained TensorFlow/Keras model and an interactive Streamlit application for running predictions and visualizing classification results.

## Project Overview

Wildfires pose significant environmental and economic threats. Detecting active fires and identifying burned areas using satellite imagery can contribute to more effective environmental monitoring.

This project explores how deep learning and remote sensing can be combined to automate wildfire-related image classification.

The model processes six spectral input bands and generates a pixel-level classification map with four categories.

## Features

- **CNN-based classification:** Pixel-wise prediction using a trained TensorFlow/Keras model.
- **Multispectral imagery:** Processes six input spectral bands.
- **Four-class segmentation:** Identifies forest, active fire, burned areas, and other regions.
- **Interactive interface:** Upload satellite imagery and run predictions through Streamlit.
- **Visualization:** Display the input imagery and corresponding classification map.
- **Ground-truth comparison:** View available reference masks alongside predictions.
- **Prediction export:** Download classification labels as NumPy arrays.

## Classification Classes

Each pixel is assigned one of the following class IDs:

| Class ID | Category | Color |
|:---:|---|---|
| 0 | Forest | Green |
| 1 | Active Fire | Red |
| 2 | Burned Area | Purple |
| 3 | Other | Yellow |
| 255 | Ignore / Invalid Pixels | Gray |

Pixels containing invalid input values are assigned the Ignore label.

## Model Architecture

The project uses a convolutional neural network designed to extract spatial and spectral features from satellite imagery.

**Architecture:**
- Six convolutional blocks using 3×3 filters.
- Dilated convolutions with dilation rates of 1, 2, 4, 8, 16, and 1.
- Batch Normalization.
- A final 1×1 convolutional layer.
- Softmax activation for four-class prediction.

**Training configuration:**

| Parameter | Value |
|---|---|
| Input dimensions | 256 × 256 × 6 |
| Output classes | 4 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Batch size | 8 |
| Maximum epochs | 30 |
| Early stopping patience | 8 |
| Loss function | Weighted Cross-Entropy + 0.9 × Dice Loss |

The training configuration also uses `ReduceLROnPlateau` to adjust the learning rate when validation performance stops improving.

A configurable decision strategy is applied to refine predictions between related classes.

## Dataset

The project uses the **TS-SatFire** dataset.

- [TS-SatFire Dataset on Kaggle](https://www.kaggle.com/datasets/z789456sx/ts-satfire)
- [TS-SatFire Research Paper](https://doi.org/10.1038/s41597-025-06271-3)

The model uses six spectral input bands from the dataset, corresponding to the configured input channels.

## Streamlit Application

SatFire includes an interactive web interface built with Streamlit.

The application allows users to upload satellite imagery, execute the trained model, and visualize the resulting classification map.

### Supported Input Formats

| Format | Description |
|---|---|
| `.tif` / `.tiff` | Six-band or eight-band multispectral TIFF imagery |
| `.npy` | NumPy arrays containing six spectral bands |
| `.npz` | NumPy archives containing six-band imagery |
| Multiple TIFF files | Six individual single-band TIFF files |

For eight-band TIFF imagery, the application uses the first six bands.

Uploaded images are resized to the model's expected input dimensions (256 × 256) using nearest-neighbor interpolation.

### Running the Application

**1. Clone the repository**

```bash
git clone https://github.com/KhalilAbboud/Detection-du-feu-actif-sur-images-satellite-en-CNN.git

cd Detection-du-feu-actif-sur-images-satellite-en-CNN
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Linux / macOS:
```bash
source venv/bin/activate
```

**3. Install the dependencies**

```bash
pip install streamlit tensorflow==2.21.0 opencv-python numpy Pillow tifffile
```

**4. Launch the application**

```bash
streamlit run app.py
```

The application will open in your default web browser.

### Prediction Workflow

1. Launch the Streamlit application.
2. Upload a compatible satellite image.
3. Preview the uploaded imagery.
4. Click **Run prediction**.
5. Visualize the CNN classification map.
6. Inspect the class distribution and available reference masks.
7. Download the predicted labels as a `.npy` file.

## Project Structure

```text
Detection-du-feu-actif-sur-images-satellite-en-CNN/
│
├── app.py                      # Streamlit application
├── checktiff.py                # TIFF inspection utility
│
├── app_samples/                # Sample imagery and reference masks
│
├── docs/                       # Project documentation
│
└── model/
    ├── best_model.keras        # Trained CNN model
    ├── normalization.npz       # Input normalization statistics
    ├── run_config.json         # Model and training configuration
    └── satfire_decision.py     # Prediction decision logic
```

## Limitations

- The model expects compatible six-band satellite imagery.
- Input images are resized to 256 × 256 pixels, which may affect spatial detail.
- Predictions are exported as NumPy label arrays rather than georeferenced raster files.
- Classification performance depends on the quality and compatibility of the input imagery.
- The application is intended for experimentation and research, not as a standalone operational wildfire warning system.

## Contributors

Developed collaboratively by:

| Contributor | GitHub Profile |
|---|---|
| **Khalil Abboud** | [@KhalilAbboud](https://github.com/KhalilAbboud) |
| **ZAKARIAGHAZI** | [@ZAKARIAGHAZI](https://github.com/ZAKARIAGHAZI) |

Both contributors participated in the development of this project.

---

*SatFire — Applying Deep Learning to Satellite-Based Wildfire Monitoring.*
