import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import cv2

DATASET_PATH = '../data/dataset'
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CNN_FEATURES_FILE = "../data/cnn_features.npy"
CNN_LABELS_FILE = "../data/cnn_labels.npy"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class CNNFeatureExtractor:
    def __init__(self, model_name='resnet18'):
        print(f"\nLoading pre-trained {model_name}...")

        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=True)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")

        #remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        #move to GPU if available
        self.model = self.model.to(DEVICE)
        self.model.eval()  # Set to evaluation mode

        #define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]     # ImageNet std
            )
        ])

        print(f"Feature extractor ready!")
        print(f"Output dimension: {self.feature_dim}")
        print(f" Device: {DEVICE}")

    def extract_from_path(self, image_path): #extract features from image file path
        try:
            img = Image.open(image_path).convert('RGB')
            return self.extract_from_pil(img)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def extract_from_pil(self, pil_image): #Extract features from PIL Image
        #preprocess
        img_tensor = self.transform(pil_image)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(DEVICE)

        # extract features
        with torch.no_grad():
            features = self.model(img_tensor)

        #flatten and convert to numpy
        features = features.squeeze().cpu().numpy()
        return features

    def extract_from_opencv(self, cv2_image):  #Extract features from OpenCV image (BGR format)
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return self.extract_from_pil(pil_image)


def build_cnn_dataset(model_name='resnet18'):  #extract CNN features from all images in the dataset
    print("BUILDING CNN FEATURE DATASET")

    #initialize feature extractor
    extractor = CNNFeatureExtractor(model_name=model_name)

    feature_list = []
    labels = []
    total_images = 0

    #count total images
    for class_name in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        images = [img for img in os.listdir(class_path)
                  if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)

    print(f"\nTotal images to process: {total_images}")

    #process each class
    with tqdm(total=total_images, desc="Overall Progress") as pbar:
        for class_name in CLASSES:
            class_path = os.path.join(DATASET_PATH, class_name)
            images = [img for img in os.listdir(class_path)
                      if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

            print(f"\nProcessing: {class_name} ({len(images)} images)")

            #process images
            for img_name in images:
                img_path = os.path.join(class_path, img_name)

                #extract features
                features = extractor.extract_from_path(img_path)

                if features is not None:
                    feature_list.append(features)
                    labels.append(class_name)

                pbar.update(1)

    #convert to numpy arrays
    X = np.array(feature_list)
    y = np.array(labels)

    #save features
    np.save(CNN_FEATURES_FILE, X)
    np.save(CNN_LABELS_FILE, y)

    print(f"Features saved to: {CNN_FEATURES_FILE}")
    print(f"Labels saved to: {CNN_LABELS_FILE}")
    print("\n" + "="*70)
    print("DATASET BUILD SUMMARY")
    print(f"Total samples: {X.shape[0]}")
    print(f"Feature vector length: {X.shape[1]}")
    print(f"File size: {X.nbytes / (1024**2):.2f} MB")

    return X, y


def load_or_build_cnn_dataset(model_name='resnet18'):  #load existing CNN features or build new ones
    if os.path.exists(CNN_FEATURES_FILE) and os.path.exists(CNN_LABELS_FILE):
        print("Found existing CNN features!")
        print("Loading features...")
        X = np.load(CNN_FEATURES_FILE)
        y = np.load(CNN_LABELS_FILE)
        print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    else:
        print("CNN features not found - building new dataset...")
        return build_cnn_dataset(model_name=model_name)
