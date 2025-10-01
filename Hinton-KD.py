import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cv2 import imread, resize
import tensorflow.compat.v1 as tf 
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from keras.models import Sequential, load_model
from tensorflow.keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, BatchNormalization, Flatten, concatenate, AveragePooling2D, MaxPool2D,Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet, MobileNetV3Small, EfficientNetV2B0, MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import DenseNet121,ResNet50
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import CategoricalAccuracy
import time
import csv
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from tensorflow.keras import backend as K



class SaveBestStudentModel(tf.keras.callbacks.Callback):
    def __init__(self, student_model, save_path="best_student_model.keras", monitor="val_loss", mode="min"):
        super(SaveBestStudentModel, self).__init__()
        self.student_model = student_model  
        self.best_loss = float("inf") if mode == "min" else -float("inf")
        self.monitor = monitor
        self.save_path = save_path
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            return

        if (self.mode == "min" and current_loss < self.best_loss) or (self.mode == "max" and current_loss > self.best_loss):
            print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best_loss:.4f} to {current_loss:.4f}. Saving student model...")
            self.best_loss = current_loss
            self.student_model.save(self.save_path)  
        else:
            print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best_loss:.4f}.")


class teacher_representative_extractor(tf.keras.Model):
    def __init__(self, teacher_base):
        super().__init__()
        self.model = teacher_base
        self.feature_map_layer = self.model.layers[-4]
        self.embedding_layer = self.model.layers[-3]
        self.logits_layer = self.model.layers[-1]

        self.multi_output_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[self.feature_map_layer.output, self.embedding_layer.output, self.logits_layer.output]
        )

    def call(self, x, training=True):
        feature_map, embedding, logits = self.multi_output_model(x, training=training)

        return {
            "feature_map": tf.stop_gradient(feature_map),
            "embedding": tf.stop_gradient(embedding),
            "logits": tf.stop_gradient(logits),
        }

class student_representative_extractor(tf.keras.Model):
    def __init__(self, student_base):
        super().__init__()
        self.model = student_base
        self.feature_map_layer = self.model.layers[-4]
        self.embedding_layer = self.model.layers[-3]
        self.logits_layer = self.model.layers[-1]

        self.multi_output_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[self.feature_map_layer.output, self.embedding_layer.output, self.logits_layer.output]
        )

    def call(self, x, training=True):
        feature_map, embedding, logits = self.multi_output_model(x, training=training)

        return {
            "feature_map": feature_map,  
            "embedding": embedding, 
            "logits": logits
        }

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.teacher_representative_extractor = teacher_representative_extractor(teacher)
        self.student_representative_extractor = student_representative_extractor(student)
        self.current_epoch = 0  

    def compile(self, student_optimizer, metrics, loss_fn, alpha=0.5, temperature=2):
        super(Distiller, self).compile(optimizer=student_optimizer, metrics=metrics, loss=loss_fn)
        self.loss_fn = loss_fn
        self.feature_loss_fn = tf.keras.losses.MeanSquaredError()
        self.student_optimizer = student_optimizer
        self.alpha = alpha
        self.temperature = temperature
        self._student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self._distillation_loss_tracker = tf.keras.metrics.Mean(name="distillation_loss")
        
    def call(self, x):
        return self.student(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            student_extractor = self.student_representative_extractor(x, training= True)
            teacher_extractor = self.teacher_representative_extractor(x, training=False)
            student_logits = student_extractor["logits"]   
            teacher_logits = teacher_extractor["logits"]   

            teacher_probs = tf.nn.softmax(teacher_logits / self.temperature)
            student_probs = tf.nn.softmax(student_logits / self.temperature)
            student_loss = self.loss_fn(y, student_logits)
            distillation_loss = tf.keras.losses.KLDivergence()(teacher_probs, student_probs) * (self.temperature ** 2)

            final_student_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss 

            self._student_loss_tracker.update_state(student_loss)
            self._distillation_loss_tracker.update_state(distillation_loss)
            self._loss_tracker.update_state(final_student_loss)

        trainable_vars = self.student.trainable_variables 
        student_gradients = tape.gradient(final_student_loss, trainable_vars)
        self.student_optimizer.apply_gradients(zip(student_gradients, trainable_vars))

        return self.compute_metrics(x, y, student_logits)
    
    def test_step(self, data):
        x, y = data
        x, y = data
        student_extractor = self.student_representative_extractor(x, training= False)
        teacher_extractor = self.teacher_representative_extractor(x, training=False)
        student_logits = student_extractor["logits"]   
        teacher_logits = teacher_extractor["logits"]   

        teacher_probs = tf.nn.softmax(teacher_logits / self.temperature)
        student_probs = tf.nn.softmax(student_logits / self.temperature)
        student_loss = self.loss_fn(y, student_logits)
        distillation_loss = tf.keras.losses.KLDivergence()(teacher_probs, student_probs) * (self.temperature ** 2)

        final_student_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss 

        self._student_loss_tracker.update_state(student_loss)
        self._distillation_loss_tracker.update_state(distillation_loss)
        self._loss_tracker.update_state(final_student_loss)

        return self.compute_metrics(x, y, student_logits)


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

train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=8)

train_data = train_data.prefetch(buffer_size=4)
validation_data = validation_data.prefetch(buffer_size=4)
test_data = test_data.prefetch(buffer_size=4)


pretrained = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in pretrained.layers:
    layer.trainable = True

feature = pretrained.output
x = GlobalAveragePooling2D()(feature)
x = Dense(512, activation='relu')(x)
output_layer = Dense(len(class_names))(x)

student_model = Model(inputs=pretrained.input, outputs=output_layer)
teacher_model = load_model("//home//user//My Code//Pediatric//Non-LS-RegNetY32GF-1.keras", compile=False)
teacher_model.name = "teacher"

for layer in teacher_model.layers:
    layer.trainable = False

student_optimizer = SGD(learning_rate=0.001, momentum=0.95, nesterov=True)

distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(student_optimizer=student_optimizer, metrics=['accuracy'], loss_fn=CategoricalCrossentropy(from_logits=True), alpha=0.4,temperature=2)
callbacks = [SaveBestStudentModel(student_model=student_model, save_path="best_student_model.keras", monitor="val_loss", mode="min"),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=15, min_lr=1e-4, verbose=2)]

history = distiller.fit(train_data,epochs=150,validation_data=validation_data,callbacks=callbacks,batch_size=32, verbose=2)

student_model.compile(optimizer=student_optimizer, loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])  

time.sleep(60)

student_model_accuracy = student_model.evaluate(test_data, verbose=0)
print("Student test accuracy: %.2f%%" % (student_model_accuracy[1] * 100))

student_model = load_model('best_student_model.keras')
student_model.compile(optimizer=student_optimizer,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
student_model.summary()

student_model_accuracy = student_model.evaluate(test_data, verbose=0)
print("student model test accuracy: %.2f%%" % (student_model_accuracy[1] * 100))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model_accuracy" + '.png')
plt.show()

plt.plot(history.history['student_loss'])
plt.plot(history.history['val_student_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model_loss" + '.png')
plt.show()


y_pred = student_model.predict(test_data)
y_pred_labels = np.argmax(y_pred, axis=1) 

y_test_labels = []
for _, labels in test_data:
    y_test_labels.append(np.argmax(labels.numpy(), axis=1))

y_test_labels = np.concatenate(y_test_labels)

lesion_names = class_names

report = classification_report(y_test_labels, y_pred_labels, target_names=lesion_names, digits=4)
print(report)

with open('classification_report.txt', 'w') as f:
    f.write(report)

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
plt.savefig('confusion_matrix.png')

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
plt.plot(epochs_range, history.history["student_loss"], "go-")
plt.plot(epochs_range, history.history["val_student_loss"], "ro-")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "val"], loc="upper left")
plt.savefig("model_result.png")
plt.show()