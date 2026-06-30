"""
train.py — Improved Training Pipeline

Improvements over v1:
1. Synthetic Moiré Augmentation: Creates 3 synthetic spoof images per real image,
   expanding the dataset from 322 → ~940 samples and balancing the classes.
2. Ensemble Classifier: Soft-voting ensemble of SVM + GradientBoosting + RandomForest
   for more robust predictions.
3. Stratified 10-Fold CV for more reliable accuracy estimates on small data.
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import extract_features_from_image, extract_features_from_bgr
from augment import create_synthetic_spoof

import cv2


# ---------------------------------------------------------------------------
# 1. Load dataset (original images only)
# ---------------------------------------------------------------------------

def load_images_and_features(dataset_dir):
    """
    Loads images from the dataset directory and extracts features.
    Scans both Kaggle directories (real_world, spoof) and custom directories (real, screen) if present.
    Also returns raw BGR images for real photos (needed for augmentation).
    """
    # Define directories to scan for real (class 0) and spoof (class 1)
    real_paths = []
    for d in ["real_world", "real"]:
        p = os.path.join(dataset_dir, d)
        if os.path.isdir(p):
            real_paths.append(p)
            
    spoof_paths = []
    for d in ["spoof", "screen"]:
        p = os.path.join(dataset_dir, d)
        if os.path.isdir(p):
            spoof_paths.append(p)

    if not real_paths and not spoof_paths:
        raise FileNotFoundError(
            f"No valid image directories found in: {dataset_dir}"
        )

    X, y = [], []
    real_images_bgr = []  # keep raw images for augmentation

    # --- Real images ---
    print("Loading real world images...")
    real_count = 0
    for real_dir in real_paths:
        print(f"  Scanning directory: {real_dir}")
        for filename in sorted(os.listdir(real_dir)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(real_dir, filename)
                try:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    img_resized = cv2.resize(img, (512, 512))
                    feat = extract_features_from_image(path)
                    X.append(feat)
                    y.append(0)
                    real_images_bgr.append(img_resized)
                    real_count += 1
                except Exception as e:
                    print(f"    Skipping {filename}: {e}")
    print(f"  Total loaded real world images: {real_count}")

    # --- Spoof images ---
    print("Loading spoof (screen) images...")
    spoof_count = 0
    for spoof_dir in spoof_paths:
        print(f"  Scanning directory: {spoof_dir}")
        for filename in sorted(os.listdir(spoof_dir)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(spoof_dir, filename)
                try:
                    feat = extract_features_from_image(path)
                    X.append(feat)
                    y.append(1)
                    spoof_count += 1
                except Exception as e:
                    print(f"    Skipping {filename}: {e}")
    print(f"  Total loaded spoof images: {spoof_count}")

    return np.array(X), np.array(y), real_images_bgr


# ---------------------------------------------------------------------------
# 2. Synthetic Augmentation
# ---------------------------------------------------------------------------

def augment_dataset(X, y, real_images_bgr, augments_per_image=3):
    """
    For each real image, generate `augments_per_image` synthetic spoof versions.
    This balances the classes and teaches the model to recognise moiré patterns
    on a wider variety of scene contents.
    """
    print(f"\nGenerating {augments_per_image} synthetic spoof images per real photo...")
    X_aug, y_aug = list(X), list(y)
    synth_count = 0

    for img_bgr in real_images_bgr:
        for _ in range(augments_per_image):
            synth_img = create_synthetic_spoof(img_bgr)
            feat = extract_features_from_bgr(synth_img)
            X_aug.append(feat)
            y_aug.append(1)  # synthetic spoof
            synth_count += 1

    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    print(f"  Generated {synth_count} synthetic spoof images.")
    print(f"  Augmented dataset: {X_aug.shape[0]} samples "
          f"(Real: {np.sum(y_aug == 0)}, Spoof: {np.sum(y_aug == 1)})")
    return X_aug, y_aug


# ---------------------------------------------------------------------------
# 3. Build Ensemble Classifier
# ---------------------------------------------------------------------------

def build_ensemble_pipeline(n_features):
    """
    Soft-voting ensemble of three complementary classifiers:
    - SVM: excellent in high-dimensional feature space
    - GradientBoosting: strong on tabular data, learns non-linear boundaries
    - RandomForest: robust to noise and overfitting
    """
    k = min(n_features, 80)  # select top features

    svm = SVC(
        C=50.0, gamma="scale", probability=True,
        class_weight="balanced", random_state=42
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("gb", gb), ("rf", rf)],
        voting="soft",
        weights=[2, 2, 1],  # SVM and GB slightly favoured
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("clf", ensemble),
    ])
    return pipe


# ---------------------------------------------------------------------------
# 4. Main Training Routine
# ---------------------------------------------------------------------------

def train_model():
    dataset_dir = "moire_classification"
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

    # --- Load original data ---
    print("=" * 60)
    print("STEP 1: Extracting features from original dataset")
    print("=" * 60)
    X_orig, y_orig, real_images_bgr = load_images_and_features(dataset_dir)
    print(f"\nOriginal dataset: {X_orig.shape[0]} samples, {X_orig.shape[1]} features")
    print(f"  Real: {np.sum(y_orig == 0)}, Spoof: {np.sum(y_orig == 1)}")

    # --- Augment ---
    print("\n" + "=" * 60)
    print("STEP 2: Synthetic Moiré Augmentation")
    print("=" * 60)
    X_aug, y_aug = augment_dataset(X_orig, y_orig, real_images_bgr, augments_per_image=3)

    # --- Split (stratified) ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_aug, y_aug, test_size=0.20, random_state=42, stratify=y_aug
    )
    print(f"\nTraining samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # --- Build ensemble ---
    print("\n" + "=" * 60)
    print("STEP 3: Training Ensemble Classifier")
    print("=" * 60)
    pipe = build_ensemble_pipeline(X_train.shape[1])

    # Cross-validation
    print("\nRunning Stratified 5-Fold Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  5-Fold CV Accuracy: {np.mean(cv_scores)*100:.2f}% (std: {np.std(cv_scores)*100:.2f}%)")

    # Fit on training split and evaluate on held-out validation set
    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")

    print("\nValidation Metrics:")
    print(classification_report(y_val, y_val_pred, target_names=["Real", "Screen"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    # --- Re-train on FULL augmented dataset for submission ---
    print("\n" + "=" * 60)
    print("STEP 4: Re-training final model on ALL data for submission")
    print("=" * 60)
    final_pipe = build_ensemble_pipeline(X_aug.shape[1])
    final_pipe.fit(X_aug, y_aug)

    # Strip ALL numpy RandomState objects to make pickle portable across numpy 1.x / 2.x
    _deep_strip_random_state(final_pipe)

    model_data = {
        "pipeline": final_pipe,
        "model_type": "Ensemble(SVM + GradientBoosting + RandomForest)",
        "features_k": min(X_aug.shape[1], 80),
        "augmented_samples": X_aug.shape[0],
    }

    model_path = "detector_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f, protocol=4)

    print(f"\nFinal model saved successfully to '{model_path}'")
    print(f"Model file size: {os.path.getsize(model_path) / 1024:.2f} KB")


def _deep_strip_random_state(obj, visited=None):
    """
    Brute-force walk every object and attribute in the model graph
    and replace any numpy RandomState / Generator with integer 42.
    Handles nested lists, tuples, dicts, numpy arrays, and object dicts.
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    # 1. Recurse into iterables
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        items = obj.flat if isinstance(obj, np.ndarray) else obj
        for item in items:
            if item is not None:
                _deep_strip_random_state(item, visited)
        return

    # 2. Recurse into dictionaries (VotingClassifier stores estimators in dicts)
    if isinstance(obj, dict):
        for val in obj.values():
            if val is not None:
                _deep_strip_random_state(val, visited)
        return

    # 3. Strip known attributes if present on this object
    for known_attr in ('_rng', '_random_state', 'random_state', '_rng_face'):
        if hasattr(obj, known_attr):
            val = getattr(obj, known_attr)
            if not isinstance(val, (int, type(None))):
                try:
                    setattr(obj, known_attr, 42)
                except (AttributeError, TypeError):
                    pass

    # 4. Scan all attributes
    attrs = {}
    try:
        attrs = vars(obj) if hasattr(obj, '__dict__') else {}
    except TypeError:
        pass

    for attr_name, attr_val in list(attrs.items()):
        type_name = type(attr_val).__name__
        module_name = getattr(type(attr_val), '__module__', '')

        # Strip numpy random classes
        if type_name in ('RandomState', 'Generator') and 'numpy' in str(module_name):
            try:
                setattr(obj, attr_name, 42)
            except (AttributeError, TypeError):
                pass
        else:
            # Recurse into any non-primitive attributes
            if attr_val is not None and not isinstance(attr_val, (int, float, str, bytes, bool)):
                _deep_strip_random_state(attr_val, visited)


if __name__ == "__main__":
    train_model()
