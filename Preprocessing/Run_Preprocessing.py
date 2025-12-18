from Preprocessing.DataAugmentation import CLASSES, DATASET_PATH, TRGT_IMGES_PER_CLASS, Augment_Img
import os
import random
import cv2
from Preprocessing.FeatureExtraction import load_or_build_cnn_dataset


print("\nStart Aug:\n")
for Class in CLASSES:
    Class_path = os.path.join(DATASET_PATH,Class)
    imges=os.listdir(Class_path)
    imges=[img for img in imges if img.lower().endswith((".jpg",".jpeg",".png"))]
    current_imge_size=len(imges)
    needed_nof_imgs=TRGT_IMGES_PER_CLASS-current_imge_size
    print(f"{Class}: {current_imge_size} imges so we need to add {needed_nof_imgs} more")
    if needed_nof_imgs<=0:
        print(f"{Class} have enough images already\n")
        continue

    for i in range(needed_nof_imgs):
        Img_Name=random.choice(imges)
        Img_Path=os.path.join(Class_path,Img_Name)
        img=cv2.imread(Img_Path)
        if img is None:
            continue
        Aug=Augment_Img(img)
        save_Img=f"aug_{i}_{Img_Name}"
        save_Path=os.path.join(Class_path,save_Img)
        if not cv2.imwrite(save_Path,Aug):
            print(f"{save_Path} failed")
    print(f"{needed_nof_imgs} are done for {Class}\n")
print("Data Augmentation is completed and all the classes are balanced\n")

print("\nStarting feature extraction...")
X, y = load_or_build_cnn_dataset(model_name='resnet18')