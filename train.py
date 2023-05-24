import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imutils import paths

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\Users\umesh\OneDrive\Desktop\hacko pitch 2.0\dataset"
CATEGORIES = ["with_mask", "without_mask"]


data = []
labels = []


# for category in CATEGORIES:
#     path = os.path.join(DIRECTORY, category)
#     for img in os.listdir(path):
#     	img_path = os.path.join(path, img)
#         image = load_img(img_path, target_size=(224, 224))
#     	image = img_to_array(image)
#     	image = preprocess_input(image)
#     	data.append(image)
#     	labels.append(category)
# print("done")
for category in CATEGORIES:
    path = os.path.join(DIRECTORY,category)
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = load_img(img_path,target_size = (244,244))
        image = img_to_array(image)
        image = preprocess_input(image)
        labels.append(category)
        data.append(image)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


data = np.array(data, dtype="float32")
labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.30, stratify=labels, random_state=42)

aug = ImageDataGenerator(otation_range=20,width_shift_range=0.2,shear_range=0.15,zoom_range=0.15,height_shift_range=0.2,horizontal_flip=True,fill_mode="nearest")

baseModel = MobileNetV2( include_top=False,weights="imagenet",input_tensor=Input(shape=(224, 224, 3)))