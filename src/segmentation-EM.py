# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 10:56:30 2025

@author: sozaghba
"""



import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import os
import cv2
from PIL import Image
import numpy as np

def preprocess_image(image_path, to_rgb=True):
    # Decode image_path from Tensor -> string
    image_path = image_path.numpy().decode('utf-8')
    img = Image.open(image_path)
    img = img.convert("RGB") if to_rgb else img.convert("L")
    img = img.resize((256, 256))
    img = np.array(img).astype(np.float32) / 255.0

    if not to_rgb:
        img = np.expand_dims(img, axis=-1)

    return img
def preprocess_mask(mask_path):
    from PIL import Image
    import numpy as np

    mask_path = mask_path.numpy().decode('utf-8')

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((256, 256))
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


# Function to resize masks
def resize_mask(mask, target_shape=(256, 256)):
    mask_resized = cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)
    return mask_resized



def load_image_and_mask(image_path, mask_path):
    image = tf.py_function(func=preprocess_image, inp=[image_path], Tout=tf.float32)
    mask = tf.py_function(func=preprocess_mask, inp=[mask_path], Tout=tf.float32)

    # Set static shapes (required for model)
    image.set_shape([256, 256, 3])
    mask.set_shape([256, 256, 1])
    return image, mask


# Function to load dataset
def load_dataset(image_dir, mask_dir, batch_size=16, shuffle=True):
    image_paths = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.jpg', '.png','tiff'))])
    mask_paths = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith(('.jpg', '.png','tiff'))])
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(lambda img, msk: load_image_and_mask(img, msk), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset



def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([input_tensor, se])

def FCN_CoReNet(input_shape=(256, 256, 3), num_classes=1):
    model_input = layers.Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=model_input)

    # Encoder
    conv1 = base_model.get_layer("conv1_relu").output
    conv2 = base_model.get_layer("conv2_block3_out").output
    conv3 = base_model.get_layer("conv3_block4_out").output
    conv4 = base_model.get_layer("conv4_block6_out").output
    conv5 = base_model.get_layer("conv5_block3_out").output

    # Decoder with SE and Dropout
    x = layers.Conv2DTranspose(512, kernel_size=3, strides=2, padding="same")(conv5)
    x = layers.Concatenate()([x, conv4])
    x = layers.Conv2D(512, kernel_size=3, padding="same", activation="relu")(x)
    x = se_block(x)
    x = layers.Dropout(0.3)(x)  # 🔥 Dropout after SE block

    x = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv3])
    x = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = se_block(x)
    x = layers.Dropout(0.3)(x)  # 🔥 Dropout

    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv2])
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = se_block(x)
    x = layers.Dropout(0.3)(x)  # 🔥 Dropout

    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv1])
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = se_block(x)
    x = layers.Dropout(0.3)(x)  # 🔥 Dropout

    # Upsample to original size
    shared_features_up = layers.UpSampling2D(size=(2, 2))(x)

    # Output branches
    seg_output = layers.Conv2D(num_classes, kernel_size=1, activation='sigmoid', name='segmentation')(shared_features_up)

    conf = layers.Conv2D(32, 3, padding='same', activation='relu')(shared_features_up)
    conf = layers.Conv2D(1, 1, activation='sigmoid', name='confidence')(conf)

    refined_output = layers.Multiply(name='refined_output')([seg_output, conf])

    return Model(inputs=model_input, outputs=[refined_output, seg_output, conf])



train_images_dir = ''
train_masks_dir = ''
val_images_dir = ''
val_masks_dir = ''
test_images_dir = ''
test_masks_dir = ''



batch_size = 16



train_dataset = load_dataset(train_images_dir, train_masks_dir, batch_size=batch_size, shuffle=True)
val_dataset = load_dataset(val_images_dir, val_masks_dir, batch_size=batch_size, shuffle=False)


# Training
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("fcnn22LR4.h5", save_best_only=True, monitor="val_loss", mode="min", verbose=1)
]

model = FCN_CoReNet(input_shape=(256, 256, 3), num_classes=1)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={
        'refined_output': 'binary_crossentropy',
        'segmentation': 'binary_crossentropy',
        'confidence': 'mse'
    },
    loss_weights={
        'refined_output': 1.0,
        'segmentation': 0.5,
        'confidence': 0.1
    },
    metrics={
        'refined_output': ['accuracy'],
        'segmentation': ['accuracy'],
        'confidence': []
    }
)


# Start training
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,  # or any number of epochs you prefer
    callbacks=callbacks
)


