import pandas as pd
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def import_world_values_data():
    """
    Reads the world values data into data frames.

    Returns:
        values_train: world_values responses on the training set
        hdi_train: HDI (human development index) on the training set
        values_test: world_values responses on the testing set
    """
    values_train = pd.read_csv('world-values-train2.csv')
    values_train = values_train.drop(['Country'], axis=1)
    values_test = pd.read_csv('world-values-test.csv')
    values_test = values_test.drop(['Country'], axis=1)
    hdi_train = pd.read_csv('world-values-hdi-train2.csv')
    hdi_train = hdi_train.drop(['Country'], axis=1)
    return values_train, hdi_train, values_test


def plot_hdi_vs_feature(training_features, training_labels, feature, color, title):
    """
    Input:
    training_features: world_values responses on the training set
    training_labels: HDI (human development index) on the training set
    feature: name of one selected feature from training_features
    color: color to plot selected feature
    title: title of plot to display

    Output:
    Displays plot of HDI vs one selected feature.
    """
    plt.scatter(training_features[feature],
    training_labels['2015'],
    c=color)
    plt.title(title)
    plt.show()


def calculate_correlations(training_features,
                           training_labels):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set

    Output:
        Prints correlations between HDI and each feature, separately.
        Displays plot of HDI vs one selected feature.
    """
    # Calculate correlations between HDI and each feature
    correlations = []
    for column in training_features.columns:
        print(column, training_features[column].corr(training_labels['2015']))
        correlations.append(round(training_features[column].corr(training_labels['2015']), 4))
    print(correlations)
    print()

    # Identify three features
    plot_hdi_vs_feature(training_features, training_labels, 'Protecting forests rivers and oceans',
                        'green', 'HDI versus ProtectingForestsRiversAndOceans')
    plot_hdi_vs_feature(training_features, training_labels, 'Better transport and roads',
                        'blue', 'HDI versus BetterTransportAndRoads')
    plot_hdi_vs_feature(training_features, training_labels, 'Access to clean water and sanitation',
                        'red', 'HDI versus AccessToCleanWaterAndSanitation')


def plot_pca(training_features,
             training_labels,
             training_classes):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set
        training_classes: HDI class, determined by hdi_classification(), on the training set

    Output:
        Displays plot of first two PCA dimensions vs HDI
        Displays plot of first two PCA dimensions vs HDI, colored by class
    """
    # Run PCA on training_features
    pca = PCA()
    transformed_features = pca.fit_transform(training_features)

    # Plot countries by first two PCA dimensions
    plt.scatter(transformed_features[:, 0],     # Select first column
                transformed_features[:, 1],     # Select second column
                c=training_labels)
    plt.colorbar(label='Human Development Index')
    plt.title('Countries by World Values Responses after PCA')
    plt.show()

    # Plot countries by first two PCA dimensions, color by class
    training_colors = training_classes.apply(lambda x: 'green' if x else 'red')
    plt.scatter(transformed_features[:, 0],     # Select first column
                transformed_features[:, 1],     # Select second column
                c=training_colors)
    plt.title('Countries by World Values Responses after PCA')
    plt.show()


def hdi_classification(hdi):
    """
    Input:
        hdi: HDI (human development index) value

    Output:
        high HDI vs low HDI class identification
    """
    if 1.0 > hdi >= 0.7:
        return 1.0
    elif 0.7 > hdi >= 0.30:
        return 0.0
    else:
        raise ValueError('Invalid HDI')
