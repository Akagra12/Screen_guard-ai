"""Fill this in. That's the whole interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
A hard 0 or 1 is fine if your method gives a yes/no answer.
"""

import os
import sys
import pickle
import numpy as np

# Load our feature extraction utility
try:
    from utils import extract_features_from_image
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import extract_features_from_image

# Path to the serialized model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_model.pkl")

# Cache model in memory
_MODEL_DATA = None


def get_model():
    global _MODEL_DATA
    if _MODEL_DATA is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Please run train.py first to train the model."
            )
        with open(MODEL_PATH, "rb") as f:
            _MODEL_DATA = pickle.load(f)
    return _MODEL_DATA


def predict(image_path: str) -> float:
    """Predicts if the image is a photo of a screen (1) or a real photo (0)."""
    try:
        # 1. Load trained pipeline
        model_data = get_model()
        pipeline = model_data["pipeline"]

        # 2. Extract features (FFT, LBP, Color, Edge)
        features = extract_features_from_image(image_path)

        # 3. Predict probability of class 1 (screen/spoof)
        prob = pipeline.predict_proba([features])[0][1]
        return float(prob)
    except Exception as e:
        sys.stderr.write(f"Error during prediction: {e}\n")
        return 0.5


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
    
    score = predict(sys.argv[1])
    print(f"{score:.4f}")
