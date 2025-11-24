# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 19:07:54 2025

@author: sozaghba
"""


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import pandas as pd
from PIL import Image
import cv2

# Preprocessing function for images
def preprocess_image(image_path, to_rgb=True):
    image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3 if to_rgb else 1)
    image = tf.image.resize(image, (512, 512)) / 255.0
    return image

# Function to resize masks
def resize_mask(mask, target_shape=(512, 512)):
    mask_resized = cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)
    return mask_resized

# Function to load image and mask pairs
def load_image_and_mask(image_path, mask_path):
    image = preprocess_image(image_path, to_rgb=True)
    mask = preprocess_image(mask_path, to_rgb=False)
    mask = tf.image.resize(mask, (512, 512))  # Ensure correct mask shape
    mask = tf.cast(mask > 0.5, tf.uint8)  # Convert to binary mask
    return image, mask

# Function to load dataset
def load_dataset(image_dir, mask_dir, batch_size=16, shuffle=True):
    image_paths = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.jpg', '.png'))])
    mask_paths = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith(('.jpg', '.png'))])
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(lambda img, msk: load_image_and_mask(img, msk), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def FCN_ResNet50(input_shape=(None, None, 3), num_classes=1):

    model_input = layers.Input(shape=input_shape)

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=model_input
    )

    # Encoder feature maps (same as your original)
    conv1 = base_model.get_layer("conv1_relu").output           
    conv2 = base_model.get_layer("conv2_block3_out").output     
    conv3 = base_model.get_layer("conv3_block4_out").output     
    conv4 = base_model.get_layer("conv4_block6_out").output     
    conv5 = base_model.get_layer("conv5_block3_out").output    

    # Decoder
    x = layers.Conv2DTranspose(512, 3, strides=2, padding="same")(conv5)
    x = layers.Concatenate()([x, conv4])
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv3])
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv2])
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv1])
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    # FINAL LAYER: MUST CALL IT ON x
    x = layers.Conv2DTranspose(
        num_classes,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="sigmoid" if num_classes == 1 else "softmax"
    )(x)   

    return Model(inputs=model_input, outputs=x)



# Set learning rate and compile the model
learning_rate = 1e-4
dynamic_model = FCN_ResNet50(input_shape=(None, None, 3), num_classes=1)

# Optional: compile if you want metrics
dynamic_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

train_images_dir = ''
train_masks_dir  = ''
val_images_dir   = ''
val_masks_dir    = ''
test_images_dir  = ''
test_masks_dir   = ''


# Load datasets
batch_size = 16
train_dataset = load_dataset(train_images_dir, train_masks_dir, batch_size=batch_size, shuffle=True)
val_dataset = load_dataset(val_images_dir, val_masks_dir, batch_size=batch_size, shuffle=False)

# Training
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("fcn_resnet50_best.h5", save_best_only=True, monitor="val_loss", mode="min", verbose=1)
]

history = dynamic_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=callbacks
)
