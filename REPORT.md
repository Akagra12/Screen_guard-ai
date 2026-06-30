# Submission Note: Spot the Fake Photo (Real vs. Screen Recapture)

## 1. Approach
We implemented a **lightweight feature-engineering pipeline** combined with a **Soft-Voting Ensemble Classifier** (SVM + GradientBoosting + RandomForest). The solution is small, fast, and runs entirely on-device without heavy deep-learning frameworks (no PyTorch/TensorFlow required at inference time).

### Feature Extraction (Standardized to 512×512):
1. **Fourier Frequency Analysis (Moiré Grid Detection):** 2D FFT with radial band statistics and horizontal/vertical energy projections to capture periodic screen pixel grids.
2. **Local Binary Patterns (LBP):** Texture histograms capturing micro-textures unique to digital displays.
3. **Chromaticity & Color Distributions:** HSV channel statistics and r/g chromaticity maps to detect screen white-balance shifts and gamut limitations.
4. **Laplacian & Sobel Edge Statistics:** Blur variance and gradient magnitudes typical of re-photographed images.
5. **Specular Glare Detection:** Ratio of near-white pixels (>240 intensity) to detect screen reflections.
6. **Bezel/Line Detection:** Hough line counting to detect straight screen frame edges.

### Synthetic Data Augmentation:
To address the small dataset size (343 original images), we generated **3 synthetic spoof images per real photo** by overlaying:
- Moiré interference patterns (sinusoidal grids at screen-pixel frequencies)
- Color temperature shifts (blue/warm/greenish gamut simulation)
- Refocus Gaussian blur and brightness attenuation
- Non-linear gamma curve simulation

This expanded the training set from **343 → 991 samples**, significantly improving generalization.

### Model Pipeline:
- **Preprocessing:** StandardScaler → ANOVA F-value Feature Selector (`SelectKBest(k=80)`)
- **Ensemble:** Soft-Voting Classifier combining:
  - SVM (RBF kernel, C=50, class-weight balanced) — weight 2
  - GradientBoosting (200 estimators, max_depth=4) — weight 2
  - RandomForest (300 estimators, class-weight balanced) — weight 1
- **Final Fit:** Trained on the entire augmented 991-sample dataset for submission.

---

## 2. Internal Validation Results
The following metrics were measured on an internal train/validation split of the available dataset. Since the final evaluation will be performed on a separate hidden dataset, actual performance may vary depending on unseen devices, lighting conditions, and screen types.
| Metric | Value |
|---|---|
| **Validation Accuracy** | **94.97%** |
| **5-Fold CV Accuracy** | **94.70% (std: 1.29%)** |
| **Real Precision** | 90% |
| **Screen Precision** | 96% |
| **Real Recall** | 86% |
| **Screen Recall** | 97% |

**Confusion Matrix (199 validation samples):**
|  | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 37 | 6 |
| **Actual Screen** | 4 | 152 |

---

## 3. Required Metrics (Inference & Cost)
* **Latency:** **~330 ms** per image (on Laptop Intel CPU, single-threaded).
  * **Note on Latency:**Approximately 95% of the total execution time is spent decoding the original high-resolution (8–13 MP) JPEG image using OpenCV. Feature extraction, preprocessing, and ensemble prediction together require less than 5 ms. If images are resized to 512×512 before storage, the total latency is expected to **under 15 ms**.
    
* **Inference:** Fully deterministic and runs entirely offline. No network connection, external APIs, or cloud services are required during prediction.
* **Model Size:** `detector_model.pkl` – **2.84 MB**
* **Cost per Image:**
  * **On-device (local):** **$0** (runs free on client phone/laptop CPU).
  * **Cloud Server (Scale):** **~$2.75 per 1,000,000 images** (or **$0.00275 per 1,000 images**).
    * *Assumption:* Running on AWS Lambda (512MB RAM, $0.00000833 per 512MB-second) with 330ms execution time.

---

## 4. Future Improvements (With More Time)
1. **Pre-resize Pipeline:** Resize incoming images to 512×512 before feature extraction to cut latency from 330ms to <15ms.
2. **Transfer Learning with ONNX:** Fine-tune a lightweight CNN (MobileNetV3-Small or EfficientNet-Lite) in PyTorch, export to ONNX, and run via `cv2.dnn.readNetFromONNX` has the potential to improve accuracy on larger datasets while maintaining low latency.
3. **Larger & More Diverse Dataset:** Collect images across different screen types (LCD, OLED, E-ink), lighting conditions, and camera zoom levels for better real-world generalization.

## 5. Design Trade-offs

The primary goal was to build a lightweight, fast, and inexpensive solution that can run entirely on a standard CPU without requiring GPU acceleration or large deep-learning frameworks.
Compared with CNN-based approaches, the selected handcrafted feature pipeline and soft-voting ensemble provide the following advantages:
• Small model size
• Zero inference cost on-device
• Fast deployment
• Easy reproducibility
• No TensorFlow or PyTorch dependency during inference

A lightweight CNN (e.g., MobileNetV3) would likely improve accuracy further but would increase training complexity, model size, and deployment requirements.
