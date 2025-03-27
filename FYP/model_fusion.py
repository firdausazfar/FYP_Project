from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def combine_features(twitter_data, spotify_data):
    if len(twitter_data) != len(spotify_data):
        raise ValueError("length mismatch")
    return np.hstack((twitter_data, spotify_data))

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_predict, zero_division=0))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_predict))

    cm = confusion_matrix(y_test, y_predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Low", "Moderate", "High"],
                yticklabels=["Low", "Moderate", "High"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    accuracy = accuracy_score(y_test, y_predict)
    print( f"Accuracy: {accuracy: .2f}")

    joblib.dump(clf, "models/depression_model.pkl")
    print("✅ Model saved to models/depression_model.pkl")

    return clf, accuracy
