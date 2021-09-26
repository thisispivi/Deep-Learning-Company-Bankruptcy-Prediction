# Create the network
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_model():
    new_model = keras.models.Sequential()
    new_model.add(keras.layers.Dense(128, activation='relu', input_shape=(95,)))
    new_model.add(keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
    new_model.add(keras.layers.Dropout(0.5))
    new_model.add(keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
    new_model.add(keras.layers.Dropout(0.5))
    new_model.add(keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
    new_model.add(keras.layers.Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    new_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return new_model


def plot_loss(history):
    plt.subplots(figsize=(12, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.draw()
    plt.show()


def plot_accuracy(history):
    plt.subplots(figsize=(12, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.draw()
    plt.show()


def plot_conf_matr(model, x_test, y_test):
    predictions = model.predict(x_test)
    classes = predictions > 0.5
    cm = confusion_matrix(y_test, classes)

    # Plot
    plt.figure(figsize=(10, 7))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax,
                cmap="PuBu")  # annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['No Bankrupt', 'Bankrupt'])
    ax.yaxis.set_ticklabels(['No Bankrupt', 'Bankrupt'])
    print(classification_report(y_test, classes))
    plt.draw()
    plt.show()
