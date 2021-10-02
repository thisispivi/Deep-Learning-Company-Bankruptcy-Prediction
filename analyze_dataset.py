import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np


def balance(df, smote, save_path_bar=None, save_path_pie=None):
    """
    Print the dataset info
    :param df: The pandas dataframe
    :param smote: A boolean that changes the title of the graph depending on if the dataset has been balanced or not
    :param save_path_bar: pathlib path. If the parameter is passed the plot image will be saved in the figures folder. Else the image will be showed
    :param save_path_pie: pathlib path. If the parameter is passed the plot image will be saved in the figures folder. Else the image will be showed
    :return: The percentage of elements of class 0 in the dataframe
    """
    result = df.value_counts()
    zero_percentage = round((result[0] * 100) / (result[0] + result[1]), 2)
    print("No. of 0: " + str(result[0]) + "\nNo. of 1: " + str(result[1]) +
          "\nPercentage of 0: " + str(zero_percentage) + " %\nPercentage of 1: " +
          str(round((100 - zero_percentage), 2)) + " %")

    # Bar
    if smote:
        plt.figure(figsize=(12, 8))
    else:
        plt.figure(figsize=(6, 4))
    plt.locator_params(axis="y", nbins=7)
    plt.bar(x=["No Bankrupt", "Bankrupt"], height=[
            result[0], result[1]], color=["royalblue", "indianred"])
    plt.ylabel("Count")
    if smote:
        plt.title("No. of No Bankrupt rows vs number of Bankrupt rows / Balanced")
    else:
        plt.title("No. of No Bankrupt rows vs number of Bankrupt rows / Original")
    plt.draw()
    if save_path_bar == None:
        plt.show()
    else:
        plt.savefig(save_path_bar)
        plt.close()

    # Pie
    if smote:
        plt.figure(figsize=(12, 8))
    else:
        plt.figure(figsize=(6, 4))
    plt.pie([result[0], result[1]], labels=["No Bankrupt", "Bankrupt"], explode=(0.1, 0), autopct='%1.2f%%',
            colors=["thistle", "paleturquoise"], radius=1)
    if smote:
        plt.title("No. of No Bankrupt rows vs number of Bankrupt rows / Balanced")
    else:
        plt.title("No. of No Bankrupt rows vs number of Bankrupt rows / Original")
    plt.draw()
    if save_path_pie == None:
        plt.show()
    else:
        plt.savefig(save_path_pie)
        plt.close()

    return zero_percentage


def plot_outliers(df, without_outlier, save_path=None):
    """
    Print the outliers
    :param df: The pandas dataframe
    :param without_outliers: boolean. True plot the graph after removing outliers, False plot the graph with the outliers
    :param save_path: pathlib path. If the parameter is passed the plot image will be saved in the figures folder. Else the image will be showed
    """
    fig, ax = plt.subplots(8, 12, figsize=(6, 4))
    fig.set_size_inches(55, 30)
    if without_outlier:
        fig.suptitle('Without outliers after capping and flooring')
    else:
        fig.suptitle('Checking Outliers')
    names = df.columns
    count = 1
    for i in range(8):
        for j in range(12):
            if count != 96:
                sns.boxplot(ax=ax[i, j], x=df[names[count]], data=df)
                sns.set(font_scale=1.5)
                count += 1
    fig.tight_layout(pad=4.0)
    plt.draw()
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def capping_flooring(df):
    """
    Perform capping and flooring
    :param df: The pandas dataframe
    :return: The datafraframe without outliers
    """
    for col in df:
        if col != "Bankrupt?":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            whisker_width = 1.5
            lower_whisker = q1 - (whisker_width * iqr)
            upper_whisker = q3 + (whisker_width * iqr)
            df[col] = np.where(df[col] > upper_whisker, upper_whisker,
                               np.where(df[col] < lower_whisker, lower_whisker, df[col]))
    return df


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
