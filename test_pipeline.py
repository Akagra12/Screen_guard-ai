import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import load_dataset


def compute_lbp(gray):
    h, w = gray.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1)
    ]
    center = gray[1:h-1, 1:w-1]
    for idx, (dy, dx) in enumerate(offsets):
        neighbor = gray[1+dy:h-1+dy, 1+dx:w-1+dx]
        lbp += ((neighbor >= center) * (1 << idx)).astype(np.uint8)
    hist, _ = np.histogram(lbp, bins=32, range=(0, 256), density=True)
    return hist


def get_advanced_features(image_path):
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    img = cv2.resize(img_raw, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. FFT features + Projections (capturing moire grid lines)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude_log = np.log(1 + magnitude)
    
    # Radial bands
    cy, cx = 256, 256
    y, x = np.ogrid[:512, :512]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    fft_feats = []
    bands = [0, 15, 30, 60, 120, 200, 256]
    for i in range(len(bands) - 1):
        mask = (dist >= bands[i]) & (dist < bands[i+1])
        band_vals = magnitude_log[mask]
        if len(band_vals) > 0:
            fft_feats.extend([np.mean(band_vals), np.std(band_vals), np.max(band_vals)])
        else:
            fft_feats.extend([0, 0, 0])
            
    # Projections along axes to spot grid patterns
    proj_x = np.mean(magnitude_log, axis=0)
    proj_y = np.mean(magnitude_log, axis=1)
    
    fft_feats.extend([
        np.std(proj_x), np.max(proj_x) - np.mean(proj_x),
        np.std(proj_y), np.max(proj_y) - np.mean(proj_y)
    ])
    
    # 2. HOG (Histogram of Oriented Gradients) for bezel/edge structures
    # OpenCV HOGDescriptor
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    gray_128 = cv2.resize(gray, (128, 128))
    hog_feats = hog.compute(gray_128).flatten()
    
    # 3. LBP features
    lbp_feats = compute_lbp(gray)
    
    # 4. Color chromaticity
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    total = b + g + r + 1e-6
    r_chrom = r / total
    g_chrom = g / total
    
    color_feats = [
        np.mean(r_chrom), np.std(r_chrom),
        np.mean(g_chrom), np.std(g_chrom)
    ]
    
    # HSV statistics
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = img_hsv[:, :, 0]
    s_channel = img_hsv[:, :, 1]
    v_channel = img_hsv[:, :, 2]
    
    color_feats.extend([
        np.mean(h_channel), np.std(h_channel),
        np.mean(s_channel), np.std(s_channel),
        np.mean(v_channel), np.std(v_channel)
    ])
    
    # 5. Laplacian Blur variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edge_feats = [lap_var]
    
    return np.concatenate([fft_feats, hog_feats, lbp_feats, color_feats, edge_feats])


def test_pipeline():
    dataset_dir = "moire_classification"
    real_dir = os.path.join(dataset_dir, "real_world")
    spoof_dir = os.path.join(dataset_dir, "spoof")
    
    X, y = [], []
    print("Extracting advanced features...")
    
    for filename in os.listdir(real_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(real_dir, filename)
            X.append(get_advanced_features(path))
            y.append(0)
            
    for filename in os.listdir(spoof_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(spoof_dir, filename)
            X.append(get_advanced_features(path))
            y.append(1)
            
    X = np.array(X)
    y = np.array(y)
    print(f"Features extracted. X shape: {X.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train SVM
    svm = SVC(C=10.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)
    print(f"SVM Accuracy: {accuracy_score(y_val, svm.predict(X_val_scaled))*100:.2f}%")
    
    # Train RF
    rf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    rf.fit(X_train_scaled, y_train)
    print(f"RF Accuracy: {accuracy_score(y_val, rf.predict(X_val_scaled))*100:.2f}%")
    
    # Train Stacking
    estimators = [
        ('svm', SVC(C=10.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)),
        ('lr', LogisticRegression(C=1.0, class_weight='balanced', random_state=42))
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )
    stacking.fit(X_train_scaled, y_train)
    print(f"Stacking Accuracy: {accuracy_score(y_val, stacking.predict(X_val_scaled))*100:.2f}%")


if __name__ == "__main__":
    test_pipeline()
