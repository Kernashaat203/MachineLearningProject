from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import joblib


FEATURE_FILE = "../data/cnn_features.npy"
LABELS_FILE = "../data/cnn_labels.npy"

# Load CNN features after data preprocessing
x = np.load(FEATURE_FILE)
y = np.load(LABELS_FILE)

# encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)



# data splitting
x_train, x_val, y_train, y_val = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# sclaing
scaler = StandardScaler()
# learn and sacale
x_scaled = scaler.fit_transform(x_train)
x_value_after_scale = scaler.transform(x_val)

# apply pca
pca = PCA(n_components=128)
# learn and scale
x_pca = pca.fit_transform(x_scaled)
x_val_after_pca = pca.transform(x_value_after_scale)

# apply svm
svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True)

# training data
print("training svm, this may take some time")
svm.fit(x_pca, y_train)

# prediction
y_pred = svm.predict(x_val_after_pca)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy: ", accuracy)

# save encoded labels
joblib.dump(le, "../models/SVM/label_encoder.pkl")
# save svm model
joblib.dump(svm, "../models/SVM/svm_model.pkl")
# save svm scalar
joblib.dump(scaler, "../models/SVM/svm_scaler.pkl")
# save svm pca
joblib.dump(pca, "../models/SVM/svm_pca.pkl")

print("svm model saved")
