from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from random import shuffle

import glob
import os
import cv2
import numpy as np

modelo_guardado = 'modelo_cancer.h5'

if os.path.exists(modelo_guardado):

    from tensorflow.keras.models import load_model
    model = load_model(modelo_guardado)
else:

    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    dataTrain = []

    for filename in glob.glob(os.path.join('data/train/malignant', '*.jpg')):
        dataTrain.append([1, cv2.imread(filename)])

    for filename in glob.glob(os.path.join('data/train/benign', '*.jpg')):
        dataTrain.append([0, cv2.imread(filename)])

    shuffle(dataTrain)

    for i, j in dataTrain:
        y_train.append(i)
        x_train.append(j)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for filename in glob.glob(os.path.join('data/test/malignant', '*.jpg')):
        x_test.append(cv2.imread(filename))
        y_test.append(1)

    for filename in glob.glob(os.path.join('data/test/benign', '*.jpg')):
        x_test.append(cv2.imread(filename))
        y_test.append(0)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))  # idealmente 100 a 1000 en lugar de 4


    model.save(modelo_guardado)


ruta = 'data/test/benign/1.jpg'
I = cv2.imread(ruta)

if round(model.predict(np.array([I]))[0][0]) == 1:
    print("CANCER")
    cv2.imshow("Cancer", I)
else:
    print("BENIGN")
    cv2.imshow("Benign", I)
