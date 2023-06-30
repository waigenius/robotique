import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 

from tensorflow.keras import regularizers
from keras import callbacks
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential



#tf.keras.backend.clear_session()
train = pd.read_csv("/home/wai/Documents/Sign_MNIST_dataset/sign_mnist_train/sign_mnist_train.csv")
test = pd.read_csv("/home/wai/Documents/Sign_MNIST_dataset/sign_mnist_test/sign_mnist_test.csv")


#Segmentation
y_train=train['label']
X_train=train.drop(['label'],axis=1).values

y_test=test['label']
X_test=test.drop(['label'],axis=1).values
print(f"Données entrainement caractéristique_train: {X_train.shape}, Label_train: {y_train.shape}")
print(f"Données entrainement caractéristique_test: {X_test.shape}, Label_test: {y_test.shape}")


X_train = X_train / 255
X_test = X_test / 255
print(f"Données normalisées entrainement: {X_train.shape}, Test: {X_test.shape}")

#Redimensionner le dataset
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)


#Binariser
label_binrizer = LabelBinarizer()
labels_train = label_binrizer.fit_transform(y_train)
labels_test = label_binrizer.fit_transform(y_test)
x_train, x_val, y_train, y_val = train_test_split(X_train, labels_train, test_size = 0.3)


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
model1.fit(x=X_train, y=y_train, validation_data=(X_test, Y_test),epochs=25,callbacks=[early_stop])

