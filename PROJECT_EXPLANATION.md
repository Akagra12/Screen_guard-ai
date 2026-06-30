# Screen Guard AI - Complete Technical Project Explanation

This document acts as a comprehensive reference guide to prepare for technical interviews regarding the implementation, architecture, and design decisions of this screen recapture detection system.

---

## 1. Problem Context & Objective

The goal of this project is display liveness detection (also known as anti-spoofing). Specifically, we want to distinguish between:
*   **Real Photos:** A direct capture of a physical scene (e.g., identity cards, facial biometrics, real-world objects).
*   **Screen Recaptures (Spoofing):** A photo taken of a digital display showing a photo of the target scene.

Recapture spoofing is a common method used to bypass KYC identity verification, face authentication, and fraud detection. Detecting this requires analyzing physical artifacts introduced during re-photography.

---

## 2. Key Challenges & Design Decisions

### Why not use a Deep Learning (CNN/ViT) model?
Although Deep Learning models (like ResNet or MobileNet) are standard for image classification, classical machine learning with hand-crafted features was selected for the following reasons:
*   **Zero Infrastructure Overhead:** The model runs locally in CPU-bound execution environments without requiring heavy deep-learning frameworks (no PyTorch, TensorFlow, or CUDA dependencies).
*   **Data Efficiency:** Deep neural networks require tens of thousands of images to generalize without overfitting. Our classical ML pipeline generalizes exceptionally well on small datasets (~340 images) because the extracted features directly isolate the known physical cues of recaptured displays.
*   **Low Cost & High Performance:** Features are extracted and processed in under 5 ms during model inference. The system runs locally for free.
*   **Explainability:** Unlike "black-box" deep learning models, we can trace exactly why a classification occurred by reviewing feature importances (e.g., spikes in specific Fourier frequency bands or Local Binary Pattern texture statistics).

---

## 3. Feature Extraction Pipeline (Under the Hood)

The image processing pipeline in `utils.py` standardizes the input image to 512×512 pixels and extracts a combined feature vector representing four categories of physical phenomena:

### A. Frequency Analysis via 2D Fast Fourier Transform (FFT)
When a camera captures a digital display, the interaction between the display's physical pixel grid and the camera sensor's color filter array (CFA) produces spatial interference patterns known as **moiré patterns**.
*   **Mechanism:** We transform the grayscale image into the frequency domain using the 2D FFT.
*   **Extraction:** We compute the magnitude spectrum. Since moiré grids introduce highly periodic, high-frequency structures, they manifest as distinct spikes in the frequency spectrum. We slice the spectrum into concentric circular radial bands (from low to high frequency) and calculate statistical moments (mean, standard deviation, max, and percentiles) for each band.
*   **Projections:** We also project the magnitude spectrum along the horizontal and vertical axes to isolate linear scanlines and display grid alignments.

### B. Texture Analysis via Local Binary Patterns (LBP)
Digital displays have a granular, pixelated surface texture that is distinct from natural surfaces.
*   **Mechanism:** LBP labels every pixel in the grayscale image by thresholding the 3×3 neighborhood against the center pixel value. It converts the result into an 8-bit binary number.
*   **Extraction:** We compute a normalized 64-bin histogram of these LBP codes. This captures the distribution of micro-textures (flat regions, edges, corners, and spots) characteristic of physical screen panels.

### C. Color Gamut & Chromaticity Shifts
LCD and OLED screens are emissive light sources with distinct spectral profiles compared to reflective natural lighting.
*   **Mechanism:** We convert the BGR image to the HSV color space to decouple intensity (Value) from color (Hue and Saturation).
*   **Extraction:** We compute statistical moments (mean, standard deviation, 10th and 90th percentiles) for BGR and HSV channels.
*   **Chromaticity:** We calculate normalized red ($r = R / (R+G+B)$) and green ($g = G / (R+G+B)$) chromaticity maps. Screen recaptures display narrow chromatic distributions due to the limited color gamut profile of electronic panels.

### D. Specular Glare & Bezel Detection
*   **Glare Detection:** Display screens are covered in glass or plastic, which reflects ambient light. We calculate the ratio of pixels exceeding an intensity value of 240, capturing these specular highlight reflections.
*   **Bezel Edge Detection:** Recaptured photos often contain physical screen borders. We downscale the grayscale image to 256x256, extract edges using the Canny edge detector, and run a Hough Line Transform to count straight lines. A high count of long, straight lines suggests the presence of a screen frame.

---

## 4. Synthetic Moiré Augmentation (`augment.py`)

A primary risk in anti-spoofing is overfitting to specific screens or backgrounds in the training set. To improve generalization, we implement custom data augmentation:
*   For every real photo, we generate **three synthetic spoof images** by applying a random chain of screen-recapture artifacts.
*   **Moiré Grid Simulation:** We overlay 2D sinusoidal grid patterns onto the image using randomized spatial frequencies and orientations.
*   **Color Shifts:** We apply non-linear gamma modifications and color temperature shifts (boosting blue, warm, or green channels) to simulate screen color filters.
*   **Camera Blur:** We apply Gaussian blur and slight brightness reduction to model focus degradation and display panel emission levels.

This expands our initial training set from **343 → 991 images**, balancing the classes and teaching the classifiers to identify frequency-domain grid patterns across diverse scene contents.

---

## 5. Model Architecture & Ensemble Voting

Instead of relying on a single classifier, we implement a **Soft-Voting Ensemble Classifier** to build a robust decision boundary:
1.  **Support Vector Machine (SVM):** Configured with an RBF kernel and balanced class weights. SVMs excel in high-dimensional feature spaces, focusing on the margins between classes.
2.  **Gradient Boosting (GB):** Trains sequential decision trees to minimize classification errors. It is effective at identifying non-linear patterns within tabular features.
3.  **Random Forest (RF):** Fits multiple decision trees on bootstrap samples of the data. It acts as a regularizer, reducing overall variance and preventing overfitting to noise.

**Feature Selection:** We scale the features using `StandardScaler` and select the top 80 features using ANOVA F-value feature selection (`SelectKBest`), retaining only the most statistically significant cues.

---

## 6. Common Technical Interview Questions

### Q1: Why did your original model fail on my device recaptures, and how did you resolve it?
*   **Answer:** "The model was originally trained on standard datasets featuring older screen displays with prominent, low-frequency pixel grids. Modern high-DPI (Retina/OLED) displays have pixels so dense that they do not generate identical low-frequency moiré patterns. To resolve this, I took custom photos of my laptop screen and integrated them into the training pipeline. I also added specular glare features (ratio of high-intensity pixels) and Hough Line Transform counts (detecting display borders). This allowed the model to correctly identify high-density displays."

### Q2: Why does a downloaded Google image sometimes get classified as a screen recapture?
*   **Answer:** "Google images undergo aggressive JPEG compression and resizing. The block-level discrete cosine transform (DCT) artifacts introduced by JPEG compression show periodic patterns in the frequency domain that mimic moiré grid lines. This is an out-of-distribution case. The model is designed to run on raw, uncompressed camera frames taken directly by the device's sensor, which do not have these compression artifacts."

### Q3: Why does your model have a ~330ms latency, and how would you optimize it for a real-time production system?
*   **Answer:** "95% of the latency is due to file I/O and JPEG decoding of large 8-13 Megapixel images in OpenCV, while the actual feature extraction and model inference takes under 5 ms. In a production environment, this can be optimized by:
    1. Resizing the image to 512×512 directly in memory (e.g. in the frontend or device camera buffer) before passing it to the prediction script, which eliminates the raw JPEG decoding overhead and reduces latency to under 15 ms.
    2. Using a hardware-accelerated JPEG decoder like `libjpeg-turbo` on the server."

### Q4: How did you handle serialization compatibility between python environments?
*   **Answer:** "Pickling estimators that use random states (like Random Forest or Gradient Boosting) stores internal NumPy `RandomState` and `BitGenerator` objects (such as `MT19937`). These objects are format-incompatible between NumPy 1.x and NumPy 2.x, causing runtime crashes when loaded across different Python environments. I implemented a graph traversal function in `train.py` that walks every attribute of the fitted estimators and replaces all NumPy random state instances with plain integer seeds, ensuring the pickle file is portable."
