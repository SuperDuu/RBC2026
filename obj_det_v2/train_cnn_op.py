import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import albumentations as A
import cv2

DATASET_PATH = 'datasets/img_cnn'
MODEL_SAVE_PATH = 'models/classifier_v2.3.h5'
LABEL_SAVE_PATH = 'models/labels_v2.3.json'
LOG_SAVE_PATH = 'models/training_log_2.3.csv'
PLOT_LOSS_PATH = 'models/figure_v2.3_loss.png'
PLOT_ACC_PATH = 'models/figure_v2.3_accuracy.png'

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
MAX_EPOCHS = 150 

augmentor = A.Compose([
    A.SafeRotate(limit=45, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=128), 
    A.Perspective(scale=(0.05, 0.12), p=0.7), 
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 3), p=0.3), 
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
])

def augment_image(image):
    data = {"image": image.astype(np.uint8)}
    aug_data = augmentor(**data)
    return aug_data["image"].astype(np.float32)

class RoboconDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, subset="training", validation_split=0.2, batch_size=32):
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=validation_split
        )
        self.generator = self.datagen.flow_from_directory(
            directory, target_size=IMG_SIZE, color_mode='grayscale',
            batch_size=batch_size, class_mode='categorical', subset=subset, shuffle=True
        )
        self.classes = self.generator.classes
        self.class_indices = self.generator.class_indices

    def __len__(self): return len(self.generator)

    def __getitem__(self, index):
        x, y = self.generator[index]
        x_aug = np.array([augment_image(img * 255.0) / 255.0 for img in x])
        return x_aug, y

train_gen = RoboconDataGenerator(DATASET_PATH, subset="training")
val_gen = RoboconDataGenerator(DATASET_PATH, subset="validation")

with open(LABEL_SAVE_PATH, 'w') as f:
    json.dump(train_gen.class_indices, f)


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(64, 64, 1)),
        
        # Block 1: Trích xuất đặc trưng cơ bản
        layers.SeparableConv2D(32, (3, 3), padding='same', activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: Nhận diện hình khối
        layers.SeparableConv2D(64, (3, 3), padding='same', activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3: Đặc trưng bậc cao
        layers.SeparableConv2D(128, (3, 3), padding='same', activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='elu', kernel_regularizer=regularizers.l2(0.002)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model(len(train_gen.class_indices))

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0005), 
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
]

print(f"[+] Đang train v2.3 với {len(train_gen.class_indices)} classes...")
history = model.fit(
    train_gen, 
    validation_data=val_gen,
    epochs=MAX_EPOCHS, 
    callbacks=callbacks
)   

model.save(MODEL_SAVE_PATH)
print(f"[+] Đã lưu mô hình tại: {MODEL_SAVE_PATH}")