import os
import cv2
import numpy as np

# Precompute FFT distance grid once at module load for speed
_CY, _CX = 256, 256
_Y, _X = np.ogrid[:512, :512]
_FFT_DIST_GRID = np.sqrt((_X - _CX)**2 + (_Y - _CY)**2)


def load_image_fast(image_path):
    """Loads an image using standard OpenCV to preserve high-frequency details, and resizes to 512x512."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return cv2.resize(img, (512, 512))


def compute_lbp(gray):
    """Computes an 8-neighbor Local Binary Pattern (LBP) histogram."""
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
        
    hist, _ = np.histogram(lbp, bins=64, range=(0, 256), density=True)
    return hist


def get_fft_features(gray):
    """Computes radial distribution and peak ratio statistics of the Fourier spectrum."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude_log = np.log(1 + magnitude)
    
    dist = _FFT_DIST_GRID
    
    # Define radial frequency bands
    bands = [0, 15, 30, 60, 120, 200, 256]
    features = []
    
    for i in range(len(bands) - 1):
        mask = (dist >= bands[i]) & (dist < bands[i+1])
        band_vals = magnitude_log[mask]
        if len(band_vals) > 0:
            features.extend([
                np.mean(band_vals),
                np.std(band_vals),
                np.max(band_vals),
                np.percentile(band_vals, 90),
                np.percentile(band_vals, 95),
                np.percentile(band_vals, 99)
            ])
        else:
            features.extend([0] * 6)
            
    # Projections along axes to capture horizontal/vertical moire grid lines
    proj_x = np.mean(magnitude_log, axis=0)
    proj_y = np.mean(magnitude_log, axis=1)
    
    features.extend([
        np.mean(proj_x), np.std(proj_x), np.max(proj_x),
        np.mean(proj_y), np.std(proj_y), np.max(proj_y)
    ])
            
    # Extract robust peak-to-background ratio statistics in high frequency (dist > 30)
    high_freq_mask = dist > 30
    high_freq_vals = magnitude_log[high_freq_mask]
    
    if len(high_freq_vals) > 0:
        mean_val = np.mean(high_freq_vals)
        std_val = np.std(high_freq_vals)
        max_val = np.max(high_freq_vals)
        
        max_to_mean = max_val / (mean_val + 1e-6)
        std_to_mean = std_val / (mean_val + 1e-6)
        
        peaks_2std = np.sum(high_freq_vals > (mean_val + 2 * std_val))
        peaks_3std = np.sum(high_freq_vals > (mean_val + 3 * std_val))
        peaks_4std = np.sum(high_freq_vals > (mean_val + 4 * std_val))
        
        features.extend([max_to_mean, std_to_mean, float(peaks_2std), float(peaks_3std), float(peaks_4std)])
    else:
        features.extend([0.0] * 5)
        
    return np.array(features)


def get_color_features(img_bgr):
    """Computes color distributions and chromaticity statistics."""
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    features = []
    channels = [
        img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2],
        img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    ]
    
    for channel in channels:
        mean = np.mean(channel)
        std = np.std(channel)
        p10 = np.percentile(channel, 10)
        p90 = np.percentile(channel, 90)
        features.extend([mean, std, p10, p90])
        
    # Chromaticity maps (R and G ratios)
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    
    total = b + g + r + 1e-6
    r_chrom = r / total
    g_chrom = g / total
    
    for chrom_channel in [r_chrom, g_chrom]:
        features.extend([
            np.mean(chrom_channel),
            np.std(chrom_channel),
            np.percentile(chrom_channel, 10),
            np.percentile(chrom_channel, 90)
        ])
        
    return np.array(features)


def get_edge_and_bezel_features(gray):
    """Extracts Laplacian focus, Sobel edges, specular glare, and line counts (bezels)."""
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # 1. Specular glare detection (screens show sharp high-intensity reflections)
    glare_mask = gray > 240
    glare_ratio = np.mean(glare_mask)
    
    # 2. Line detection (screens show straight rectangular bezels/borders)
    gray_small = cv2.resize(gray, (256, 256))
    edges = cv2.Canny(gray_small, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=30, maxLineGap=10)
    line_count = len(lines) if lines is not None else 0
    
    features = [
        lap_var,
        np.mean(edge_mag),
        np.std(edge_mag),
        np.max(edge_mag),
        glare_ratio,
        float(line_count)
    ]
    return np.array(features)


def extract_features_from_image(image_path):
    """Loads an image, resizes it to a standardized scale, and extracts features."""
    img = load_image_fast(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    fft_feat = get_fft_features(gray)
    lbp_feat = compute_lbp(gray)
    color_feat = get_color_features(img)
    edge_feat = get_edge_and_bezel_features(gray)
    
    return np.concatenate([fft_feat, lbp_feat, color_feat, edge_feat])


def extract_features_from_bgr(img_bgr):
    """Extracts features from a raw BGR numpy array (for synthetic/augmented images)."""
    img = cv2.resize(img_bgr, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    fft_feat = get_fft_features(gray)
    lbp_feat = compute_lbp(gray)
    color_feat = get_color_features(img)
    edge_feat = get_edge_and_bezel_features(gray)
    
    return np.concatenate([fft_feat, lbp_feat, color_feat, edge_feat])


def load_dataset(dataset_dir):
    """Scans dataset folders and extracts features for all valid images."""
    X = []
    y = []
    
    real_dir = os.path.join(dataset_dir, "real_world")
    spoof_dir = os.path.join(dataset_dir, "spoof")
    
    if not os.path.isdir(real_dir) or not os.path.isdir(spoof_dir):
        raise FileNotFoundError(
            f"Dataset must contain 'real_world' and 'spoof' directories in: {dataset_dir}"
        )
        
    print("Loading real world images...")
    real_count = 0
    for filename in os.listdir(real_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(real_dir, filename)
            try:
                feat = extract_features_from_image(path)
                X.append(feat)
                y.append(0) # 0 = real
                real_count += 1
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                
    print(f"Loaded {real_count} real world images.")
    
    print("Loading spoof (screen) images...")
    spoof_count = 0
    for filename in os.listdir(spoof_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(spoof_dir, filename)
            try:
                feat = extract_features_from_image(path)
                X.append(feat)
                y.append(1) # 1 = spoof
                spoof_count += 1
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                
    print(f"Loaded {spoof_count} spoof images.")
    
    return np.array(X), np.array(y)
