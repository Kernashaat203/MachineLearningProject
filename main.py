import cv2
import time
import numpy as np
import torch
import pickle
import joblib

from FeatureExtraction import CNNFeatureExtractor

CONF_THRESHOLD_SVM = 0.60
CONF_THRESHOLD_KNN = 0.65

SVM_MODEL_PATH = "models/svm_model.pkl"
SVM_LABEL_ENCODER_PATH = "models/label_encoder.pkl"

KNN_MODEL_PATH = "models/knn_model.pkl"
KNN_SCALER_PATH = "models/knn_scaler.pkl"
KNN_LABEL_ENCODER_PATH = "models/knn_label_encoder.pkl"
KNN_THRESHOLD_PATH = "models/knn_threshold.pkl"

CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
UNKNOWN_LABEL = "unknown"

#selects GPU if avilable else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print("Loading models....")

svm_model = joblib.load(SVM_MODEL_PATH)
svm_label_encoder = joblib.load(SVM_LABEL_ENCODER_PATH)

knn_model = pickle.load(open(KNN_MODEL_PATH, "rb"))
knn_scaler = pickle.load(open(KNN_SCALER_PATH, "rb"))
knn_label_encoder = pickle.load(open(KNN_LABEL_ENCODER_PATH, "rb"))
knn_thresholds = pickle.load(open(KNN_THRESHOLD_PATH, "rb"))
CONF_THRESHOLD_KNN = knn_thresholds["confidence_threshold"]

MODEL_TYPE = 'SVM'
print(f"Using model: {MODEL_TYPE.upper()}")

extractor = CNNFeatureExtractor()


def predict(features):
    if MODEL_TYPE == "svm":
        probs = svm_model.predict_proba(features.reshape(1, -1))[0]
        indx = np.argmax(probs)
        confidance = probs[indx]

        if confidance < CONF_THRESHOLD_SVM:
            return UNKNOWN_LABEL, confidance
        label = svm_label_encoder.inverse_transform([indx])[0]
        return label, confidance

    else:
        X = knn_scaler.transform(features.reshape(1, -1))
        probs = knn_model.predict_proba(X)[0]
        indx = np.argmax(probs)
        confidance = probs[indx]
        if confidance < CONF_THRESHOLD_KNN:
            return UNKNOWN_LABEL, confidance
        label = knn_label_encoder.inverse_transform([indx])[0]
        return label, confidance


#Camera loop
cap = cv2.VideoCapture(0)
print("Camera started. Press Q to exit!")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    start_time = time.time()

    features = extractor.extract_from_opencv(frame)
    label, confidence = predict(features)

    fps = 1.0 / (time.time() - start_time)
    text = f"{label} | {confidence:.2f} | FPS: {fps:.1f}"
    color = (0, 255, 0) if label != UNKNOWN_LABEL else (0, 0, 255)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Live Material Identification", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
