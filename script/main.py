# -*- coding: utf-8 -*-
"""Company Bankruptcy Prediction"""

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from analyze_dataset import *
from variables import *
from deep_learning import *

if __name__ == "__main__":

    # Read Files: In this section we import the csv files.
    df = pd.read_csv('data.csv')

    # Split the dataset into label and data
    labels = df['Bankrupt?']
    data = df.drop(['Bankrupt?'], axis=1)

    # Analyze Dataset: In this section we will analyze the dataset shape, balance and if it has null values in its rows

    # Shape: Check the shape of the dataset
    print('Data shape:', data.shape)
    print('Labels shape:', labels.shape)

    # Null values: Check if there are null values
    num_null = df.isnull().sum(axis=0).sum()
    if num_null == 0:
        print("There aren't null values in the dataset")
    else:
        print("There are null values in the dataset")

    # Balance: Check if the dataset is balanced
    balanced = True
    zero_percentage = balance(df['Bankrupt?'])
    if zero_percentage != 50.0:
        balanced = False

    # Normalize values
    df = normalize_dataset(df)

    # Update labels and data
    labels = df['Bankrupt?']
    data = df.drop(['Bankrupt?'], axis=1)

    # Balance Dataset using SMOTE
    data_new = data
    labels_new = labels

    if not balanced:
        sm = SMOTE()
        data_new, labels_new = sm.fit_resample(data, labels)

        print('Data shape:', data_new.shape)
        print('Labels shape:', labels_new.shape)

        balance(pd.Series(labels_new))

    # Split data into training, validation and test set
    x_train, x_test, y_train, y_test = train_test_split(data_new, labels_new, train_size=0.9)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8)

    # Print all the sizes
    print('Train data shape:', x_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Validation data shape:', x_valid.shape)
    print('Validation labels shape:', y_valid.shape)
    print('Test data shape:', x_test.shape)
    print('Test labels shape:', y_test.shape)
    input("PRESS ENTER TO CONTINUE.")

    if not load_model:

        model = get_model()

        # Train the network
        history = None
        if train_model:
            history = model.fit(x_train, y_train, epochs=200, validation_data=(x_valid, y_valid))

        # Loss graph of the model
        if model_loss:
            plot_loss(history)

        # Accuracy graph of the model
        if model_accuracy:
            plot_accuracy(history)
    else:
        # Load model
        model = keras.models.load_model(file_name)

    # Evaluate the model: Check how well the dataset perform on the test set
    if evaluate_model:
        model.evaluate(x_test, y_test)

    # Confusion Matrix: Compute the label prediction using the test set and plot the confusion matrix.
    if conf_matr:
        plot_conf_matr(model, x_test, y_test)
        # Test performance original value
        x_original_train, x_original_test, y_original_train, y_original_test = train_test_split(data, labels, train_size=0.9)
        plot_conf_matr(model, x_original_test, y_original_test)

    # Save the model
    if save_model:
        model.save(file_name)

    # Plot model
    if plot_model:
        dot_img_file = "network.png"
        keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
