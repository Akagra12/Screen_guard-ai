import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils import load_dataset


def test_voting():
    dataset_dir = "moire_classification"
    X, y = load_dataset(dataset_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 2. Select top 60 features
    selector = SelectKBest(score_func=f_classif, k=60)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_val_sel = selector.transform(X_val_scaled)

    # 3. Instantiate base classifiers
    svm = SVC(C=50.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
    hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=4, class_weight='balanced', random_state=42)

    # Fit and test individual classifiers
    svm.fit(X_train_sel, y_train)
    svm_acc = accuracy_score(y_val, svm.predict(X_val_sel))
    print(f"SVM Accuracy: {svm_acc*100:.2f}%")

    rf.fit(X_train_sel, y_train)
    rf_acc = accuracy_score(y_val, rf.predict(X_val_sel))
    print(f"RF Accuracy: {rf_acc*100:.2f}%")

    hgb.fit(X_train_sel, y_train)
    hgb_acc = accuracy_score(y_val, hgb.predict(X_val_sel))
    print(f"HGB Accuracy: {hgb_acc*100:.2f}%")

    # 4. Voting Classifier (Soft)
    voting_soft = VotingClassifier(
        estimators=[('svm', svm), ('rf', rf), ('hgb', hgb)],
        voting='soft'
    )
    voting_soft.fit(X_train_sel, y_train)
    voting_soft_acc = accuracy_score(y_val, voting_soft.predict(X_val_sel))
    print(f"Voting (Soft) Accuracy: {voting_soft_acc*100:.2f}%")

    # 5. Voting Classifier (Hard)
    voting_hard = VotingClassifier(
        estimators=[('svm', svm), ('rf', rf), ('hgb', hgb)],
        voting='hard'
    )
    voting_hard.fit(X_train_sel, y_train)
    voting_hard_acc = accuracy_score(y_val, voting_hard.predict(X_val_sel))
    print(f"Voting (Hard) Accuracy: {voting_hard_acc*100:.2f}%")

    # 6. Stacking Classifier
    estimators = [('svm', svm), ('rf', rf), ('hgb', hgb)]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
        cv=5,
        n_jobs=-1
    )
    stacking.fit(X_train_sel, y_train)
    stacking_acc = accuracy_score(y_val, stacking.predict(X_val_sel))
    print(f"Stacking Accuracy: {stacking_acc*100:.2f}%")
    
    print("\nStacking Classification Report:")
    print(classification_report(y_val, stacking.predict(X_val_sel), target_names=["Real", "Screen"]))


if __name__ == "__main__":
    test_voting()
