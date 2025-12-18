import numpy as np
import random
import cv2

CLASSES=["cardboard","glass","metal","paper","plastic","trash"]
TRGT_IMGES_PER_CLASS=500
DATASET_PATH = "../data/dataset"

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


DATASET_PATH = "../data/dataset"


