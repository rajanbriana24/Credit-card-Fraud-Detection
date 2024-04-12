# Fraud Detection Pipeline

This project implements a fraud detection pipeline using various anomaly detection algorithms and a stacked ensemble model. The pipeline consists of several steps, including data loading, data cleaning, model training, evaluation, and ensemble stacking.

## Table of Contents
1. [Overview](#overview)
2. [Data](#data)
3. [Base Models](#base-models)
    - [Isolation Forest](#isolation-forest)
    - [One-Class SVM](#one-class-svm)
    - [Local Outlier Factor](#local-outlier-factor)
    - [Autoencoder](#autoencoder)
    - [Gaussian Mixture Models](#gaussian-mixture-models)
    - [DBSCAN](#dbscan)
    - [K-Nearest Neighbors](#k-nearest-neighbors)
    - [Minimum Covariance Determinant](#minimum-covariance-determinant)
    - [Angle-based Outlier Detection](#angle-based-outlier-detection)
4. [Ensemble Model](#ensemble-model)
5. [Methods](#methods)
    - [Data Loader](#data-loader)
    - [Data Cleaner](#data-cleaner)
    - [Model Trainer](#model-trainer)
    - [Ensemble Trainer](#ensemble-trainer)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Main Script](#main-script)
6. [Usage](#usage)
7. [References](#references)

## Overview

The fraud detection pipeline is designed to identify fraudulent transactions in a dataset containing both legitimate and fraudulent transactions. It utilizes various anomaly detection algorithms to identify unusual patterns in the data that may indicate fraudulent activity. These algorithms are trained on the dataset and evaluated based on their performance metrics. Additionally, a stacked ensemble model is trained using the predictions of the base models to improve detection accuracy.

## Data

The dataset used for fraud detection is sourced from Kaggle, specifically from the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset by Machine Learning Group at ULB. It contains numerical input variables resulting from a PCA transformation, including features V1 to V28, 'Time', and 'Amount', as well as the target variable 'Class' indicating whether a transaction is fraudulent (1) or legitimate (0).

## Base Models

The pipeline employs the following base anomaly detection models:

### Isolation Forest

Isolation Forest is an unsupervised learning algorithm that identifies anomalies by isolating instances in the dataset. It works by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. Anomalies are expected to have shorter average path lengths in the resulting trees.

### One-Class SVM

One-Class SVM is a support vector machine algorithm that learns the distribution of normal data points and identifies outliers based on deviations from this distribution. It finds the hyperplane that separates the data points from the origin with the maximum margin while still capturing most of the data.

### Local Outlier Factor

Local Outlier Factor (LOF) measures the local deviation of density of a data point with respect to its neighbors. It identifies outliers by comparing the local density of data points with the density of their neighbors. Points with significantly lower density than their neighbors are considered outliers.

### Autoencoder

Autoencoder is a type of artificial neural network used for unsupervised learning of efficient codings. In the context of anomaly detection, an autoencoder is trained to reconstruct normal data instances. Anomalies are detected when the reconstruction error is high.

### Gaussian Mixture Models

Gaussian Mixture Models (GMM) assume that the data is generated from a mixture of several Gaussian distributions with unknown parameters. It estimates these parameters using the Expectation-Maximization (EM) algorithm and identifies anomalies based on low probability densities.

### DBSCAN

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is a clustering algorithm that groups together closely packed points based on a density criterion. It identifies outliers as points that are not part of any cluster or are in low-density regions.

### K-Nearest Neighbors

K-Nearest Neighbors (KNN) is a non-parametric method used for classification and regression. In the context of anomaly detection, it classifies data points based on the majority class of their k-nearest neighbors. Outliers are identified as data points belonging to the minority class.

### Minimum Covariance Determinant

Minimum Covariance Determinant (MCD) is a robust estimator of multivariate location and scatter. It identifies outliers by fitting an ellipsoid to the data using the minimum covariance determinant method and flagging data points lying outside the ellipsoid.

### Angle-based Outlier Detection

Angle-based Outlier Detection (ABOD) measures the deviation of each data point from the average angle between pairs of data points. It identifies outliers based on the variance of these angles.

## Ensemble Model

A stacked ensemble model is trained using the predictions of the base anomaly detection models. This ensemble model combines the strengths of individual base models to improve detection accuracy. The predictions of base models are averaged, and the ensemble model's performance is evaluated using precision-recall curve and average precision score.

## Methods

The pipeline consists of the following methods:

### Data Loader

The data loader module loads the dataset from a specified path using pandas read_csv function.

### Data Cleaner

The data cleaner module preprocesses the dataset by removing rows with missing values using pandas dropna function.

### Model Trainer

The model trainer module trains various anomaly detection models on the preprocessed dataset and evaluates their performance using precision-recall curve and average precision score.

### Ensemble Trainer

The ensemble trainer module trains a stacked ensemble model using the predictions of base anomaly detection models and evaluates its performance using precision-recall curve and average precision score.

### Evaluation Metrics

The evaluation metrics module calculates various performance metrics for the trained models, including accuracy, F1 score, and confusion matrix.

### Main Script

The main script orchestrates the entire fraud detection pipeline by calling the data loader, data cleaner, model trainer, and ensemble trainer methods.

## Usage

To use the fraud detection pipeline, follow these steps:

1. Install the required dependencies using pip install -r requirements.txt.
2. Prepare your dataset in CSV format with the necessary features and target variable.
3. Update the data path in the main script to point to your dataset.
4. Run the main script using python main.py.

## References

- Liu, F. T., Ting, K. M., & Zhou, Z. (2008). Isolation Forest. In Proceedings of the Eighth IEEE International Conference on Data Mining (pp. 413-422).
- Machine Learning Group at ULB. (2013). Credit Card Fraud Detection. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
