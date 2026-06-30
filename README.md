# Screen Guard AI - Spot the Fake Photo (Real vs. Screen Recapture)

This repository contains a lightweight, high-performance machine learning pipeline designed to detect whether a photo is a real image or a recapture of a digital display (e.g., a photo taken of a laptop, phone, or monitor screen).

## Submission Summary & Metrics

* **Validation Accuracy:**94.97% on our train/validation split. Since the final evaluation uses a separate hidden dataset, real-world performance may vary depending on unseen screen types, lighting conditions, and camera devices.

* **5-Fold Cross-Validation Accuracy:** 94.70% (std: 1.29%)
* **Latency:** ~330 ms per image (on a standard laptop CPU, single-threaded).
  * Note: Over 95% of this time is consumed by image loading and decompression in OpenCV for high-resolution input JPEGs. The actual feature extraction and ensemble classifier prediction takes less than 5 ms.
* **Cost:** $0.00 (the model runs locally and is fully on-device). For cloud hosting (e.g., AWS Lambda), the cost is approximately $2.75 per 1,000,000 images.

---

## Getting Started

### Prerequisites

Install the required dependencies using the system package manager:
```bash
pip install -r requirements.txt
```

### Running Predictions

To run prediction on an image, pass the path of the file to `predict.py`. The script will output a single value between 0 and 1, where 0 indicates a real photo and 1 indicates a screen recapture:

```bash
python predict.py path/to/image.jpg
```

Example commands:
```bash
# Predict on a real photo (outputs near 0)
python predict.py "moire_classification/real/IMG20260630122135.jpg"

# Predict on a screen recapture (outputs near 1)
python predict.py "moire_classification/screen/IMG20260630122210.jpg"
```

### Running the Web Demo Dashboard

A local web application is included to test predictions interactively via a browser:
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser. Drag and drop any image to view its recapture probability score, latency breakdown, and classification verdict.

### Training the Model

If you wish to re-train the classifier or append new images to the training folders:
1. Place real images under `moire_classification/real/` or `moire_classification/real_world/`.
2. Place screen recapture images under `moire_classification/screen/` or `moire_classification/spoof/`.
3. Run the training script:
```bash
python train.py
```
The script will perform synthetic moiré data augmentation, run cross-validation, print the final validation matrix, and export the updated serialized pipeline to `detector_model.pkl`.

---

## File Structure

* `predict.py`: Main entrypoint script for running inference.
* `train.py`: Training script including dataset loading, synthetic moiré augmentation, and model optimization.
* `utils.py`: Hand-crafted feature extraction utilities (Fast Fourier Transform projections, Local Binary Patterns, color chromaticity, glare detection, and bezel line detection).
* `augment.py`: Synthetic moiré generator used to enrich the training set.
* `detector_model.pkl`: Serialized ensemble pipeline (SVM + Gradient Boosting + Random Forest).
* `app.py`: Flask local web application server.
* `submission_note.md`: Detailed report of the approach, metrics, and architecture.
* `requirements.txt`: Python package dependencies.
