import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


import os

# Load feature vectors and labels
X = np.load("../data/cnn_features.npy")
y = np.load("../data/cnn_labels.npy")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label encoder
joblib.dump(le, "../models/label_encoder.pkl")

# Train/Test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# SVM model
svm = SVC(
    kernel='rbf',
    C=10,
    gamma=0.01,
    probability=True
)


# Train
print("Training SVM...")
svm.fit(X_train, y_train)

# Validate
y_pred = svm.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("Accuracy:", acc)

# Save model
joblib.dump(svm, "../models/svm_model.pkl")
print("SVM saved.")


def predict_with_unknown(model,label_encoder, feature_vector, threshold=0.60):
    feature_vector = np.array(feature_vector).reshape(1, -1)
    probs = model.predict_proba(feature_vector)[0]

    max_prob = np.max(probs)
    best_idx = np.argmax(probs)
    predicted_class = label_encoder.inverse_transform([best_idx])[0]

    if max_prob < threshold:
        return "unknown", max_prob

    return predicted_class, max_prob

# Class ID mapping
CLASS_TO_ID = {
    "glass": 0,
    "paper": 1,
    "cardboard": 2,
    "plastic": 3,
    "metal": 4,
    "trash": 5,
    "unknown": 6
}

# prediction
sample = X_val[0]
label, conf = predict_with_unknown(svm,le, sample)
class_id = CLASS_TO_ID[label]

print("Predicted:", label)
print("Class ID:", class_id)
print("Confidence:", conf)
