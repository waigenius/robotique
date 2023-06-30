import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer


#tf.keras.backend.clear_session()
train = pd.read_csv("/home/wai/Documents/Sign_MNIST_dataset/sign_mnist_train/sign_mnist_train.csv")
test = pd.read_csv("/home/wai/Documents/Sign_MNIST_dataset/sign_mnist_test/sign_mnist_test.csv")
y_train=train['label']
y_test=test['label']

#Binariser
label_binrizer = LabelBinarizer()
y_train = label_binrizer.fit_transform(y_train)
y_test = label_binrizer.fit_transform(y_test)


#Segmentation

X_train=train.drop(['label'],axis=1).values
X_test=test.drop(['label'],axis=1).values
print(f"Données entrainement caractéristique_train: {X_train.shape}, Label_train: {y_train.shape}")
print(f"Données entrainement caractéristique_test: {X_test.shape}, Label_test: {y_test.shape}")


X_train = X_train / 255.0
X_test = X_test / 255.0
print(f"Données normalisées entrainement: {X_train.shape}, Test: {X_test.shape}")

#Data augmentation
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)


#Redimensionner le dataset
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

datagen.fit(X_train)

#Architecture CNN
model1 = keras.Sequential()
print("Keras sequentiel")

#couche de convolution

model1.add(layers.Conv2D(64, kernel_size=(3,3), activation='swish', input_shape=(28,28,1)))

#couche de pooling
model1.add(layers.BatchNormalization())
model1.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model1.add(layers.Dropout(0.2))

#Repetition 
model1.add(layers.Conv2D(64, kernel_size=(3,3), activation='swish', input_shape=(28,28,1)))
model1.add(layers.BatchNormalization())
model1.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model1.add(layers.Dropout(0.25))

model1.add(layers.Conv2D(64, kernel_size=(3,3), activation='swish'))
model1.add(layers.BatchNormalization())
model1.add(layers.MaxPooling2D(pool_size=(2, 2)))
model1.add(layers.Dropout(0.3))

#Couche formaté par Flatten
model1.add(layers.Flatten())
#Couche de perte avec "softmax"
model1.add(layers.Dense(24, activation='softmax'))

#Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
)


#Compilation du modèle
model1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model1.summary()


#Entrainement du modèle
model1.fit(datagen.flow(X_train,y_train, batch_size = 128),validation_data=(X_test, y_test), epochs=5, batch_size=64,callbacks=[early_stop])