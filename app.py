import cv2
import time
import numpy as np
import torch
import pickle
import joblib
import os
from Preprocessing.FeatureExtraction import CNNFeatureExtractor

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

MODEL_TYPE = 'svm'
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


def predict_folder(folder_path):
    if not os.path.isdir(folder_path):
        print("ERROR: Folder not found")
        return
    print(f"\nPredicting images in folder: {folder_path}\n")

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(folder_path, file_name)
        try:
            features = extractor.extract_from_path(img_path)
            if features is None:
                continue
            label, confidence = predict(features)
            print(f"{file_name:25s} -> {label.upper():10s} | Confidence: {confidence:.2f}")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

def predict_image(img):
    img = cv2.resize(img, (224, 224))
    features = extractor.extract_from_opencv(img)
    return predict(features)


def run_camera():
    Region = 200
    Margin = 70
    #Camera loop
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Camera not accessible")
        exit()

    print("Camera started. Place object inside the square. Press q to exit!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        y1 = Margin
        x2 = w - Margin
        x1 = x2 - Region
        y2 = y1 + Region

        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        region_crop = frame[y1:y2, x1:x2]
        start_time = time.time()

        features = extractor.extract_from_opencv(frame)
        label, confidence = predict(features)

        fps = 1.0 / (time.time() - start_time)
        text = f"{label.upper()} | Confidence: {confidence:.2f} | FPS: {fps:.1f}"
        color = (0, 255, 0) if label != UNKNOWN_LABEL else (0, 0, 255)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Live Material Identification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\nChoose mode:")
    print("1- Real-time camera prediction")
    print("2- Predict images from folder")
    MODE = input("Enter choice (1 or 2): ").strip()

    if MODE == "1":
        run_camera()

    elif MODE == "2":
        folder = input("Enter folder path: ").strip()
        predict_folder(folder)

    else:
        print("Invalid choice. Exiting.")
