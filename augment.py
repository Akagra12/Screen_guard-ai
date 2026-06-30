"""
augment.py — Synthetic Moiré Augmentation

Creates realistic synthetic screen-recaptured images from real photos by
overlaying moiré interference patterns, screen color-temperature shifts,
slight refocus blur, and brightness attenuation.

This expands the minority class (spoof) and provides varied training
examples that help the classifier generalize to unseen screen types.
"""

import cv2
import numpy as np


def apply_moire_pattern(img, freq_x=None, freq_y=None, amplitude=None):
    """Overlay a synthetic moiré pattern (simulating screen pixel grid interference)."""
    h, w = img.shape[:2]

    if freq_x is None:
        freq_x = np.random.uniform(0.02, 0.12)
    if freq_y is None:
        freq_y = np.random.uniform(0.02, 0.12)
    if amplitude is None:
        amplitude = np.random.uniform(8, 35)

    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    phase_x = np.random.uniform(0, 2 * np.pi)
    phase_y = np.random.uniform(0, 2 * np.pi)

    pattern = amplitude * np.sin(2 * np.pi * freq_x * xx + phase_x)
    pattern += (amplitude * 0.6) * np.sin(2 * np.pi * freq_y * yy + phase_y)

    # Optionally add a diagonal component
    if np.random.random() > 0.5:
        freq_d = np.random.uniform(0.01, 0.08)
        pattern += (amplitude * 0.4) * np.sin(
            2 * np.pi * freq_d * (xx + yy) + np.random.uniform(0, 2 * np.pi)
        )

    result = img.astype(np.float32) + pattern[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_color_shift(img):
    """Simulate screen color-temperature and gamut shifts."""
    result = img.astype(np.float32)
    shift_type = np.random.choice(["blue", "warm", "desaturate", "greenish"])

    if shift_type == "blue":
        result[:, :, 0] *= np.random.uniform(1.03, 1.10)  # B channel boost
        result[:, :, 2] *= np.random.uniform(0.90, 0.97)  # R channel reduce
    elif shift_type == "warm":
        result[:, :, 0] *= np.random.uniform(0.90, 0.97)
        result[:, :, 2] *= np.random.uniform(1.03, 1.10)
    elif shift_type == "desaturate":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
        alpha = np.random.uniform(0.10, 0.30)
        result = (1 - alpha) * result + alpha * gray3
    elif shift_type == "greenish":
        result[:, :, 1] *= np.random.uniform(1.02, 1.08)  # G boost

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_screen_blur(img):
    """Simulate refocus blur that occurs when photographing a screen."""
    ksize = np.random.choice([3, 5])
    sigma = np.random.uniform(0.4, 1.2)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_brightness_reduction(img):
    """Screens are typically dimmer than the real scene, reduce brightness slightly."""
    factor = np.random.uniform(0.80, 0.95)
    result = img.astype(np.float32) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_gamma_shift(img):
    """Simulate non-linear gamma curve of LCD/OLED screens."""
    gamma = np.random.uniform(0.85, 1.20)
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype(np.uint8)
    return cv2.LUT(img, table)


def create_synthetic_spoof(img):
    """
    Apply a random combination of screen-recapture artifacts to a real image.
    Returns a synthetic spoof image.
    """
    result = img.copy()

    # Always apply moiré (the most distinctive artifact)
    result = apply_moire_pattern(result)

    # Randomly apply additional artifacts
    if np.random.random() > 0.3:
        result = apply_color_shift(result)

    if np.random.random() > 0.4:
        result = apply_screen_blur(result)

    if np.random.random() > 0.5:
        result = apply_brightness_reduction(result)

    if np.random.random() > 0.5:
        result = apply_gamma_shift(result)

    return result
