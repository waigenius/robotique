import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

#tf.keras.backend.clear_session()
data_train= pd.read_csv("/home/wai/Documents/Sign_MNIST_dataset/sign_mnist_train/sign_mnist_train.csv")
data_test= pd.read_csv("/home/wai/Documents/Sign_MNIST_dataset/sign_mnist_test/sign_mnist_test.csv")

total_cols = len(data_train.axes[1]) #Nombre de colonne

#Caractéristiques et la feature
X_train = data_train.iloc[:,1:total_cols]
Y_train = data_train['label']
print(f"Données entrainement caractéristique_train: {X_train.shape}, Label_train: {Y_train.shape}")

X_test = data_test.iloc[:, 1:total_cols]
Y_test = data_test['label']
print(f"Données entrainement caractéristique_test: {X_train.shape}, Label_test: {Y_train.shape}")

X_train = X_train.values
X_test = X_test.values

#Normaliser les valeurs X_train et Y_train puis diviser par 255
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"Données normalisées entrainement: {X_train.shape}, Test: {X_test.shape}")

#Redimension images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

#Architecture CNN
model1 = keras.Sequential()
print("Keras sequentiel")


#couche de convolution
model1.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28, 1)))

#couche de pooling
model1.add(layers.MaxPooling2D(pool_size=(2, 2)))

#Repetition des deux précédentes
model1.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model1.add(layers.MaxPooling2D(pool_size=(2, 2)))

#Couche formaté par Flatten
model1.add(layers.Flatten())
model1.add(layers.Dense(128, activation='relu'))


#Couche de perte avec "softmax"
model1.add(layers.Dense(10, activation='softmax'))

#Early stopping 
early_stop = EarlyStopping(monitor='val_loss', patience=2)


#Compilation du modèle
model1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model1.summary()


#Entrainement du modèle
model1.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test),epochs=25,callbacks=[early_stop])

