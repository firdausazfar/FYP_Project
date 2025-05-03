from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
import numpy as np
import joblib

def combine_features(twitter_data, spotify_data):
    if len(twitter_data) != len(spotify_data):
        raise ValueError("length mismatch")
    return np.hstack((twitter_data, spotify_data))

def train_model(X, y, clf=None):
    from collections import Counter
    print(f"Label distribution: {Counter(y)}")

    if clf is None:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

    print("=== 5-Fold Cross-Validation Scores ===")
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: {score:.2f}")
    print(f"Average Accuracy: {scores.mean():.2f}")

    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("\n=== Classification Report on Full Data ===")
    print(classification_report(y, y_pred, zero_division=0))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y, y_pred))
    joblib.dump(clf, "models/depression_model.pkl")
    print("Final model trained on full data and saved to models/depression_model.pkl")

    return clf, scores.mean()
