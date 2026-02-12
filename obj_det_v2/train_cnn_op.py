import tensorflow as tf
import json
import os
import numpy as np
from sklearn.utils import class_weight

DATASET_PATH = 'datasets/img_cnn'
MODEL_SAVE_PATH = 'models/classifier_v2.h5'
LABEL_SAVE_PATH = 'models/labels_v2.json'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
MAX_EPOCHS = 150 

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.25,    
    rotation_range=180,    
    zoom_range=[0.7, 1.3],   
    width_shift_range=0.2,    
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3], 
    fill_mode='constant',
    cval=0
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    color_mode='grayscale', class_mode='categorical', subset='training', shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    color_mode='grayscale', class_mode='categorical', subset='validation', shuffle=False
)

cls_indices = train_gen.classes
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(cls_indices),
    y=cls_indices
)
class_weights = dict(enumerate(weights))

with open(LABEL_SAVE_PATH, 'w') as f:
    json.dump(train_gen.class_indices, f)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 1)),
    
    # Block 1
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2), # Chống overfitting 
    
    # Block 2
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    
    # Block 3
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(), # Giảm tham số    

    # Dense Layers
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
)

model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=MAX_EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, lr_scheduler]
)

model.save(MODEL_SAVE_PATH)
print("--- OK ---")