import os
import numpy as np
import joblib
from Preprocessing.FeatureExtraction import CNNFeatureExtractor

def predict(dataFilePath, bestModelPath):
    # load model
    model = joblib.load(bestModelPath)
    model_dir = os.path.dirname(bestModelPath)
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    #Label encoder
    if not os.path.exists(label_encoder_path):
        label_encoder_path = os.path.join(model_dir, "knn_label_encoder.pkl")
    label_encoder = joblib.load(label_encoder_path)
    #load scaler for KNN only
    scaler = None
    scaler_path = os.path.join(model_dir, "knn_scaler.pkl")
    if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)

    image_files = [f for f in os.listdir(dataFilePath)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Feature Extraction
    extractor = CNNFeatureExtractor()

    predictions = []

    for img_name in image_files:
        img_path = os.path.join(dataFilePath, img_name)
        try:
            features = extractor.extract_from_path(img_path)
            if features is None:
                continue
            X = features.reshape(1, -1)
            if scaler is not None:
                X = scaler.transform(X)
            probs = model.predict_proba(X)[0]
            idx = np.argmax(probs)
            label = label_encoder.inverse_transform([idx])[0]
            predictions.append(label)
        except Exception as e:
            print(e)
    return predictions
