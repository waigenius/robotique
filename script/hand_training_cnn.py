import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers




#dataset
df_train = pd.read_csv("/home/wai/Documents/Sign_MNIST_dataset/sign_mnist_train/sign_mnist_train.csv")
df_test = pd.read_csv("/home/wai/Documents/Sign_MNIST_dataset/sign_mnist_test/sign_mnist_test.csv")


#Information sur le dataset
df_train.info()

#Les six premiers ligne du dataset
#print(df_train.head())

#segmentation du dataset
y_train = df_train['label'].values
X_train = df_train.drop('label', axis = 1)

y_test = df_test['label'].values
X_test = df_test.drop('label', axis = 1)

print(f"X_train = {X_train}")
print(f"y_train = {y_train}")

#Redimensionner le dataset
images = X_train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

#Binariser
label_binrizer = LabelBinarizer()
y_train = label_binrizer.fit_transform(y_train)
y_test = label_binrizer.fit_transform(y_test)

print(f"y_train = {y_train}")


# Redimension des images
X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.values.reshape(X_train.shape[0],28,28,1)
X_test = X_test.values.reshape(X_test.shape[0],28,28,1)


# Architecture CNN

model = Sequential()
model.add(layers.Conv2D(filters = 64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1) ))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(layers.Conv2D(filters = 64, kernel_size=(3,3), activation='relu') )
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(layers.Conv2D(filters = 64, kernel_size=(3,3), activation='relu') )
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

# Couche formaté par Flatten
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

model.add(layers.Dense(24, activation='softmax'))


#Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
)


#Compilation du modèle
model.compile(loss="categorical_crossentropy",
              optimizer = Adam(),  
              metrics=['accuracy'])

# Entrainement du modèle
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10, batch_size = 128,callbacks=[early_stop])

model.summary()

loaded_model = keras.models.load_model('hand_trained_model.h5')

loaded_model.summary()