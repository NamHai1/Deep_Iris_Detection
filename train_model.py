import os
import glob
import numpy as np
import json
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import time
import uuid
import albumentations as alb
import random
import shutil
from tensorflow.keras.models import load_model

# nohup python -u train_model.py &>train_model.log &
# jobs -l | grep nohup

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

train_images = tf.data.Dataset.list_files('/datausers/ioit/vvhieu/hainam/Nam_2023/GG/Deep_Iris_Detection/aug_data/train/images/*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (250,250)))
train_images = train_images.map(lambda x: x/255)

test_images = tf.data.Dataset.list_files('/datausers/ioit/vvhieu/hainam/Nam_2023/GG/Deep_Iris_Detection/aug_data/test/images/*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (250,250)))
test_images = test_images.map(lambda x: x/255)

val_images = tf.data.Dataset.list_files('/datausers/ioit/vvhieu/hainam/Nam_2023/GG/Deep_Iris_Detection/aug_data/val/images/*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (250,250)))
val_images = val_images.map(lambda x: x/255)

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
    return [label['keypoints']]

# tf.data.Dataset.list_files: lệt kê danh sách tệp nhãn có đuôi .json
# tf.py_function để sử dụng hàm load_labels và truyền vào đường dẫn của từng tệp nhãn

train_labels = tf.data.Dataset.list_files('/datausers/ioit/vvhieu/hainam/Nam_2023/GG/Deep_Iris_Detection/aug_data/train/labels/*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

test_labels = tf.data.Dataset.list_files('/datausers/ioit/vvhieu/hainam/Nam_2023/GG/Deep_Iris_Detection/aug_data/test/labels/*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

val_labels = tf.data.Dataset.list_files('/datausers/ioit/vvhieu/hainam/Nam_2023/GG/Deep_Iris_Detection/aug_data/val/labels/*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(4000) # Xáo trộn dữ liệu
train = train.batch(16) # Chia dữ liệu thành các batch, cho quá trình huấn luyện mini-batch gradient descent.
train = train.prefetch(4) # Tải dữ liệu cho batch tiếp theo

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1000)
test = test.batch(16)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1200)
val = val.batch(16)
val = val.prefetch(4)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Reshape, Dropout
from tensorflow.keras.applications import ResNet50

model = Sequential([
    Input(shape =(250, 250, 3)),
    ResNet50(include_top= False, weights= 'imagenet', input_shape=(250,250,3)),
    Conv2D(512, 3, padding='same', activation='relu'),
    Conv2D(512, 3, padding='same', activation='relu'),
    Conv2D(256, 3, 2, padding='same', activation='relu'),
    Conv2D(256, 2, 2, activation='relu'),
    Dropout(0.05),
    Conv2D(4, 2, 2),
    Reshape((4,))
])
model.summary()

batches_per_epoch = len(train)
lr_decay = (1./0.75 -1)/batches_per_epoch
print(lr_decay)
optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001, decay= lr_decay)
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer, loss)

hist = model.fit(train, epochs= 100, validation_data= val)
model.save('eyetrackerresnet.h5')