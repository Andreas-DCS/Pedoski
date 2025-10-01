import os
os.environ['TMPDIR'] = '/tmp'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from cv2 import imread, resize
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, MaxPooling2D, SpatialDropout2D
from tensorflow.keras.applications import MobileNetV3Small
from keras import regularizers, layers, models
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import pandas as pd
import sys
import matplotlib
import Models.regnet as regnet
import time
from tensorflow.keras import backend as K



base_dir = "//home//user//My Code//Data//Perdoski_new_split_min10New"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print("Initialized classes:", class_names)

train_data = tf.keras.utils.image_dataset_from_directory(directory=train_dir,labels='inferred',label_mode='categorical',batch_size=32,image_size=(224, 224),shuffle=True,seed=2)
validation_data = tf.keras.utils.image_dataset_from_directory(directory=val_dir,labels='inferred',label_mode='categorical',batch_size=32,image_size=(224, 224),shuffle=True,seed=2)
test_data = tf.keras.utils.image_dataset_from_directory(directory=test_dir,labels='inferred',label_mode='categorical',batch_size=32,image_size=(224, 224),shuffle=False,seed=2)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(10/360),  
    layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    tf.keras.layers.Lambda(lambda x: tf.image.adjust_brightness(x, delta=tf.random.uniform([], -0.1, 0.1)))
])

train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=3)

train_data = train_data.prefetch(buffer_size=2)
validation_data = validation_data.prefetch(buffer_size=2)

# pretrained = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
pretrained = regnet.RegNetY_32GF(input_shape=(224, 224, 3), weights="//home//user//My Code//Pediatric//Models//RegNetY_32GF.h5", include_top=False)

for layer in pretrained.layers:
    layer.trainable = True

x = pretrained.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
output_layer = Dense(len(class_names))(x)

model = Model(inputs=pretrained.input, outputs=output_layer)

optimizer=SGD(learning_rate=0.001, momentum=0.95, nesterov=True)
model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),metrics=['accuracy'])

model_checkpoint = ModelCheckpoint(filepath='skin_disease_classification_model.keras', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-4, verbose=1)

history = model.fit(train_data, validation_data=validation_data, epochs=50, batch_size=32, callbacks=[model_checkpoint, reduce_lr], verbose=2)

time.sleep(10)

plt.style.use("ggplot")
fig = plt.figure(figsize=(12, 6))
epochs_range = range(len(history.history['accuracy']))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history["accuracy"], "go-")
plt.plot(epochs_range, history.history["val_accuracy"], "ro-")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train", "val"], loc="upper left")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history["loss"], "go-")
plt.plot(epochs_range, history.history["val_loss"], "ro-")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "val"], loc="upper left")
plt.savefig("model_result.png", dpi=600)

time.sleep(10)
model = load_model('skin_disease_classification_model.keras')
model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),metrics=['accuracy'])
scores = model.evaluate(test_data, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))

time.sleep(10)
y_pred = model.predict(test_data, verbose=2)
y_pred_labels = np.argmax(y_pred, axis=1) 
y_test_labels = []
for _, labels in test_data:
    y_test_labels.append(np.argmax(labels.numpy(), axis=1))

y_test_labels = np.concatenate(y_test_labels)
lesion_names = class_names
report = classification_report(y_test_labels, y_pred_labels, target_names=lesion_names, digits=4)
print(report)
with open('classification_report.txt', 'w') as f: f.write(report)

conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(12, 10))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(lesion_names))
plt.xticks(tick_marks, lesion_names, rotation=90)
plt.yticks(tick_marks, lesion_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png', dpi=600)
