import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#tf.keras.backend.clear_session()
data_train= pd.read_csv("/home/wai/Documents/Sign_MNIST_dataset/sign_mnist_train/sign_mnist_train.csv")

print(data_train.columns())

total_cols = len(data_train.axes[1]) #Nombre de colonne

X_train = data_train.iloc[:,1:total_cols]

Y_train = data_train['label']
#importer test
data_test= pd.read_csv("/home/wai/Documents/Sign_MNIST_dataset/sign_mnist_train/sign_mnist_test.csv")

#Formater les images(extraire X_train et Y_train)

#Normaliser les valeurs X_train et Y_train puis diviser par 255



#Architecture CNN
model1 = keras.Sequential()
print("Keras sequentiel")


#couche de convolution
model1.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28)))

#couche de pooling
model1.add(layers.MAxPooling2D(pool_size=(2, 2)))

#Repetition des deux précédentes
model1.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model1.add(layers.MAxPooling2D(pool_size=(2, 2)))

#Couche formaté par Flatten
model1.add(layers.Flatten())
model1.add(layers.Dense(128, activation='relu'))


#Couche de perte avec "softmax"
num_classes = [0,1,2] #Nom de classe dans Y_train
model1.add(layers.Dense(num_classes, activation='softmax'))

#Early stopping 
#early_stop = EarlyStopping(monitor='val_loss', patience=2)


#Compilation du modèle
model1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model1.summary()

#Entrainement du modèle
#model1.fit(x=X_train)

#Model.summary