# Smart Waste Material Classification System

An end‑to‑end Computer Vision project for **automatic recyclable material classification** into:

> **cardboard · glass · metal · paper · plastic · trash + unknown handling**

This project combines **CNN feature extraction (ResNet)** with **Machine Learning classifiers (SVM & KNN)** to classify waste in real‑time or from image datasets.

---

## 🧠 System Pipeline

1️⃣ **Data Augmentation** – balances dataset using rotation, flip, brightness, zoom & noise
2️⃣ **CNN Feature Extraction** – ResNet extracts meaningful feature vectors
3️⃣ **Feature Dataset Saving** – stores `cnn_features.npy` & `cnn_labels.npy`
4️⃣ **Model Training**

* **KNN** → GridSearch tuning + adaptive confidence threshold
* **SVM** → Scaling + PCA + RBF kernel
  5️⃣ **Deployment**
* Real‑time Webcam Classification
* Batch Folder Classification

---

## 📁 Project Structure

```
project/
│
├── Preprocessing/
│ ├── Run_Preprocessing.py # Augmentation + Feature extraction pipeline
│ ├── DataAugmentation.py # Dataset balancing using augmentations
│ └── FeatureExtraction.py # ResNet CNN feature extraction
│
├── TrainModels/
│ ├── KNN.py # Trains and saves KNN classifier
│ └── SVM.py # Trains and saves SVM classifier
│
├── test.py                    # Folder batch prediction tool
├── camera_app.py              # Real‑time + folder classification app
│
├── data/             # Train + augmented images
│   ├── cnn_features.npy
│   └── cnn_labels.npy
│
└── models/
    ├── KNN/
    │   ├── knn_model.pkl
    │   ├── knn_scaler.pkl
    │   ├── knn_label_encoder.pkl
    │   └── knn_threshold.pkl
    └── SVM/
        ├── svm_model.pkl
        ├── svm_scaler.pkl
        ├── svm_pca.pkl
        └── svm_label_encoder.pkl
```

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch + TorchVision
* OpenCV
* NumPy
* scikit‑learn
* joblib
* tqdm

Install dependencies:

```bash
pip install torch torchvision opencv-python numpy scikit-learn joblib tqdm pillow
```

---

## 🚀 Usage

### 1️⃣ Build Dataset (Augmentation + Feature Extraction)

```bash
python Run_Preprocessing.py
```

This will:

* Balance dataset
* Extract CNN features
* Save feature dataset

---

### 2️⃣ Train Models

#### Train KNN Model

```bash
python KNN.py
```

#### Train SVM Model

```bash
python SVM.py
```

Models will be saved inside `/models/`.

---

## 🎥 Real‑Time Classification (Webcam)

```bash
python camera_app.py
```

Select:

```
1 → Real‑time camera mode
2 → Predict from folder
```

Press **q** to exit camera mode.

---

## 📂 Predict From Folder

Run:

```bash
python camera_app.py
```

Choose option **2** and enter folder path.

or directly using test tool:

```bash
python test.py
```

---

## 🔍 Model Logic

### KNN

* GridSearch Hyperparameter Tuning
* Confidence threshold tuning
* Unknown rejection handling

### SVM

* Data Standardization
* PCA Dimensionality Reduction
* RBF Kernel
* Probability output enabled

---

## 🧪 Classes

```
cardboard
glass
metal
paper
plastic
trash
```

Objects below confidence threshold are labeled as **unknown**.

---

## ✅ Key Features

* Robust preprocessing & augmentation
* CNN powered feature extraction
* Real‑time intelligent classification
