import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def balance(df, smote):
    """
    Print the dataset info
    :param df: The pandas dataframe
    :param smote: A boolean that changes the title of the graph depending on if the dataset has been balanced or not
    :return: The percentage of elements of class 0 in the dataframe
    """
    result = df.value_counts()
    zero_percentage = round((result[0] * 100) / (result[0] + result[1]), 2)
    print("No. of 0: " + str(result[0]) + "\nNo. of 1: " + str(result[1]) +
          "\nPercentage of 0: " + str(zero_percentage) + " %\nPercentage of 1: " +
          str(round((100 - zero_percentage), 2)) + " %")

    plt.bar(x=["No Bankrupt", "Bankrupt"], height=[result[0], result[1]], color=["royalblue", "indianred"])
    plt.ylabel("Count")
    if smote == True:
        plt.title("Number of No Bankrupt rows vs number of Bankrupt rows / Balanced Dataset")
    else:
        plt.title("Number of No Bankrupt rows vs number of Bankrupt rows / Original Dataset")
    plt.draw()
    plt.show()

    plt.pie([result[0], result[1]], labels=["No Bankrupt", "Bankrupt"], explode=(0.1, 0), autopct='%1.2f%%',
            colors=["thistle", "paleturquoise"], radius=1.2)
    if smote == True:
        plt.title("Number of No Bankrupt rows vs number of Bankrupt rows / Balanced Dataset")
    else:
        plt.title("Number of No Bankrupt rows vs number of Bankrupt rows / Original Dataset")

    plt.draw()
    plt.show()
    return zero_percentage


def normalize_dataset(df):
    """
    Normalize the dataset
    :param df: The pandas dataframe
    :return: The normalized dataset
    """
    cols_for_scale = df.max()[df.max() > 1]
    # Take the columns with values less than 0
    var = df.min()[df.min() < 0]  # It is none there aren't negative values
    # Normalize values
    scale = StandardScaler()
    scaled = scale.fit_transform(df[cols_for_scale.keys()])
    # Substitute the old values with the normalized ones
    i = 0
    for column in cols_for_scale.keys():
        df[column] = scaled[:, i]
        i += 1

    return df
