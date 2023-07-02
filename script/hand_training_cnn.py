import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator



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

# #LINitialisation des variables tensorflow
# batch_size = 128
# num_classes = 24
# epochs = 10

# Redimension des images
X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.values.reshape(X_train.shape[0],28,28,1)
X_test = X_test.values.reshape(X_test.shape[0],28,28,1)

# ### Augmentation des images avec un processing
# #Data augmentationw
# datagen = ImageDataGenerator(
#         rotation_range=10,  
#         zoom_range = 0.10,  
#         width_shift_range=0.1, 
#         height_shift_range=0.1)

# datagen.fit(X_train)


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

# #couche de pooling
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
# model.add(layers.Dropout(0.2))

# #Repetition 
# model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
# model.add(layers.Dropout(0.25))

# model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Dropout(0.3))



# #Early stopping
# early_stop = keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     min_delta=0.001, # minimium amount of change to count as an improvement
#     patience=5, # how many epochs to wait before stopping
#     restore_best_weights=True,
# )


#Compilation du modèle
model.compile(loss="categorical_crossentropy",
              optimizer = Adam(),  
              metrics=['accuracy'])

model.summary()

# Entrainement du modèle
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10, batch_size = 128)

# #Entrainement du modèle
# model.fit(datagen.flow(X_train,y_train, batch_size = 128),validation_data=(X_test, y_test), epochs=25, batch_size=64,callbacks=[early_stop])