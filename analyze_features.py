import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import load_dataset


def analyze():
    dataset_dir = "moire_classification"
    X, y = load_dataset(dataset_dir)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Let's try selecting different numbers of top features
    k_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 141]
    best_acc = 0.0
    best_k = 0
    best_params = {}

    for k in k_options:
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Select top k features using ANOVA F-value
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_val_selected = selector.transform(X_val_scaled)

        # Hyperparameter search for SVM on this feature subset
        param_grid = {
            'C': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf']
        }
        grid = GridSearchCV(
            SVC(probability=True, class_weight='balanced', random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X_train_selected, y_train)
        
        # Test on validation set
        val_acc = accuracy_score(y_val, grid.predict(X_val_selected))
        print(f"k = {k:3d} | Best CV: {grid.best_score_*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Params: {grid.best_params_}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_k = k
            best_params = grid.best_params_

    print(f"\nOverall Best Validation Accuracy: {best_acc*100:.2f}% with k = {best_k} features.")


if __name__ == "__main__":
    analyze()
