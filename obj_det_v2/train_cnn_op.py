import tensorflow as tf
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

DATASET_PATH = 'datasets/img_cnn'
MODEL_SAVE_PATH = 'models/classifier_v2.1.h5'
LABEL_SAVE_PATH = 'models/labels_v2.1.json'
LOG_SAVE_PATH = 'models/training_log.csv'
PLOT_LOSS_PATH = 'models/figure8_loss.png'
PLOT_ACC_PATH = 'models/figure9_accuracy.png'

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
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

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
csv_logger = tf.keras.callbacks.CSVLogger(LOG_SAVE_PATH, append=False)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1
)

print(f"Bắt đầu huấn luyện mô hình cho {len(train_gen.class_indices)} lớp...")
history = model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=MAX_EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, lr_scheduler, csv_logger, checkpoint]
)

def plot_and_save_history(history):
    plt.rcParams["font.family"] = "serif"
    
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='train', color='steelblue', linewidth=1.5)
    plt.plot(history.history['val_accuracy'], label='test', color='chocolate', linewidth=1.5)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_ACC_PATH, dpi=300)
    print(f"Đã lưu biểu đồ Accuracy tại: {PLOT_ACC_PATH}")

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='train', color='steelblue', linewidth=1.5)
    plt.plot(history.history['val_loss'], label='test', color='chocolate', linewidth=1.5)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_LOSS_PATH, dpi=300)
    print(f"Đã lưu biểu đồ Loss tại: {PLOT_LOSS_PATH}")

plot_and_save_history(history)

model.save(MODEL_SAVE_PATH)
print("--- TẤT CẢ ĐÃ HOÀN TẤT: MODEL, LOGS VÀ FIGURES ĐÃ SẴN SÀNG ---")