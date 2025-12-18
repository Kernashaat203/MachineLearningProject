import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

FEATURES_FILE = "data/cnn_features.npy"
LABELS_FILE = "data/cnn_labels.npy"
MODEL_SAVE_PATH = "models/KNN/knn_model.pkl"
SCALER_SAVE_PATH = "models/KNN/knn_scaler.pkl"
LABEL_ENCODER_PATH = "models/KNN/knn_label_encoder.pkl"
THRESHOLD_PATH = "models/KNN/knn_threshold.pkl"

CONFIDENCE_THRESHOLD = 0.55

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

X = np.load(FEATURES_FILE)
y = np.load(LABELS_FILE)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}

knn_base = KNeighborsClassifier()
grid_search = GridSearchCV(
    knn_base,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

best_knn = grid_search.best_estimator_


def calculate_prediction_confidence(knn_model, X_scaled):

    if knn_model.weights == 'distance':
        # For distance weighting, use predict_proba if available
        if hasattr(knn_model, 'predict_proba'):
            probas = knn_model.predict_proba(X_scaled)
            confidences = np.max(probas, axis=1)
        else:
            # Fallback: use inverse of average distance
            distances, _ = knn_model.kneighbors(X_scaled)
            avg_distances = np.mean(distances, axis=1)
            confidences = 1.0 / (1.0 + avg_distances)
    else:
        # For uniform weighting, use class vote counts
        distances, indices = knn_model.kneighbors(X_scaled)
        predictions = knn_model.predict(X_scaled)

        confidences = []
        for i, pred in enumerate(predictions):
            neighbors = knn_model._y[indices[i]]
            vote_count = np.sum(neighbors == pred)
            confidence = vote_count / knn_model.n_neighbors
            confidences.append(confidence)
        confidences = np.array(confidences)

    return confidences


val_confidences = calculate_prediction_confidence(best_knn, X_val_scaled)
val_predictions = best_knn.predict(X_val_scaled)

# Try different confidence thresholds
thresholds_to_test = np.arange(0.3, 0.8, 0.05)
best_threshold = CONFIDENCE_THRESHOLD
best_accuracy = 0



for thresh in thresholds_to_test:
    adjusted_predictions = val_predictions.copy()
    unknown_mask = val_confidences < thresh

    if np.sum(~unknown_mask) > 0:
        accuracy = accuracy_score(y_val[~unknown_mask], adjusted_predictions[~unknown_mask])
        rejection_rate = np.sum(unknown_mask) / len(unknown_mask)
        if accuracy > best_accuracy and rejection_rate < 0.15:
            best_accuracy = accuracy
            best_threshold = thresh


def predict_with_unknown(knn_model, X_scaled, confidence_threshold):

    base_predictions = knn_model.predict(X_scaled)
    confidences = calculate_prediction_confidence(knn_model, X_scaled)

    final_predictions = base_predictions.copy()
    rejection_reasons = np.array(['accepted'] * len(base_predictions))

    low_confidence_mask = confidences < confidence_threshold
    final_predictions[low_confidence_mask] = 6  # Unknown class ID
    rejection_reasons[low_confidence_mask] = 'low_confidence'

    return final_predictions, confidences, rejection_reasons


y_val_pred_with_unknown, val_confidences, rejection_reasons = predict_with_unknown(best_knn, X_val_scaled, best_threshold)

known_mask = y_val_pred_with_unknown != 6
unknown_count = np.sum(~known_mask)

print(f"Validation Results:")
print(f"  Total samples: {len(y_val)}")
print(f"  Classified as known: {np.sum(known_mask)} ({np.sum(known_mask)/len(y_val)*100:.1f}%)")
print(f"  Classified as unknown: {unknown_count} ({unknown_count/len(y_val)*100:.1f}%)")

# Calculate accuracy on known classes only
if np.sum(known_mask) > 0:
    known_accuracy = accuracy_score(y_val[known_mask], y_val_pred_with_unknown[known_mask])
    print(f"\n✓ Accuracy on known classes (excluding unknown): {known_accuracy:.4f} ({known_accuracy*100:.2f}%)")
else:
    known_accuracy = 0.0
    print("\n⚠ Warning: All samples classified as unknown!")

val_accuracy = accuracy_score(y_val, y_val_pred_with_unknown)

print(f"VALIDATION ACCURACY (6 primary classes): {known_accuracy:.4f} ({known_accuracy*100:.2f}%)")

known_accuracy = accuracy_score(y_val[known_mask], y_val_pred_with_unknown[known_mask])
print(f"\n✓ Accuracy on accepted predictions: {known_accuracy:.4f} ({known_accuracy*100:.2f}%)")

overall_accuracy = accuracy_score(y_val, val_predictions)
print(f"✓ Overall accuracy (6 primary classes): {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

class_names = label_encoder.classes_
if np.sum(known_mask) > 0:
    print(classification_report(y_val[known_mask], y_val_pred_with_unknown[known_mask],
                                target_names=class_names, digits=4))


class_names_with_unknown = list(class_names) + ['unknown']

print("Rejection reasons:")
unique_reasons, reason_counts = np.unique(rejection_reasons, return_counts=True)
for reason, count in zip(unique_reasons, reason_counts):
    print(f"  {reason:20s}: {count:4d} ({count/len(rejection_reasons)*100:.1f}%)")

for i, class_name in enumerate(class_names):
    class_mask = (y_val == i) & known_mask
    if np.sum(class_mask) > 0:
        class_accuracy = accuracy_score(y_val[class_mask], y_val_pred_with_unknown[class_mask])
        rejected_in_class = np.sum((y_val == i) & (~known_mask))
        total_in_class = np.sum(y_val == i)
        print(f"{class_name:12s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) "
              f"[{rejected_in_class}/{total_in_class} rejected]")

with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(best_knn, f)


with open(SCALER_SAVE_PATH, 'wb') as f:
    pickle.dump(scaler, f)


with open(LABEL_ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)


thresholds = {
    'confidence_threshold': best_threshold
}
with open(THRESHOLD_PATH, 'wb') as f:
    pickle.dump(thresholds, f)

for weight in ['uniform', 'distance']:
    knn_test = KNeighborsClassifier(
        n_neighbors=best_knn.n_neighbors,
        weights=weight,
        metric=best_knn.metric,
        p=best_knn.p
    )
    knn_test.fit(X_train_scaled, y_train)
    acc = knn_test.score(X_val_scaled, y_val)
print(f"  - Confidence threshold: {best_threshold:.2f}")
