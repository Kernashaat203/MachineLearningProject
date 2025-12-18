import os
import numpy as np
import random
import cv2

CLASSES=["cardboard","glass","metal","paper","plastic","trash"]
TRGT_IMGES_PER_CLASS=500
DATASET_PATH = "data/dataset"

def random_rotation(img):
    angle=random.randint(-35,35)
    height,weight=img.shape[:2]
    matrix=cv2.getRotationMatrix2D((weight/2,height/2),angle,1)
    return cv2.warpAffine(img,matrix,(weight,height))

def random_flip(img):
    return cv2.flip(img,1)

def random_brightness(img):
    contrastAlpha=random.uniform(0.7,1.5)
    brightnessBeta=random.randint(-3,35)
    return cv2.convertScaleAbs(img,alpha=contrastAlpha,beta=brightnessBeta)


def random_zoom(img):
    height, weight=img.shape[:2]
    zoom=random.uniform(0.9,1.3)
    newheight=int(height*zoom)
    newweight=int(weight*zoom)
    newsize=cv2.resize(img,(newheight,newweight))

    if zoom>1: #zoom in
        y=(newheight-height)//2
        x=(newweight-weight)//2
        return newsize[y:y+newheight,x:x+newweight]
    else: # zoom<1 zoom out
        y=(height-newheight)//2
        x=(weight-newweight)//2
        zoomedimg=cv2.copyMakeBorder(newsize,y,height-newheight-y,x,weight-newweight-x,cv2.BORDER_CONSTANT,value=[0,0,0])
        return zoomedimg


def random_noise(img):
    noise=np.random.randint(0,30,img.shape,dtype=np.uint8)
    return cv2.add(img,noise)


def Augment_Img(img):
    functions = [random_rotation, random_flip, random_brightness, random_zoom, random_noise]
    image = img.copy()
    for func in functions:
        image=func(img)
    return image


DATASET_PATH = "data/dataset"

#main:
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

