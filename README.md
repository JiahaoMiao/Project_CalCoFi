# Project_CalCoFi

# Machine Learning Exam Project: CalCoFi Dataset Analysis and Modeling

This repository contains my machine learning project for the Machine Learning exam. The goal of this project is to explore and model the **CalCoFi** dataset using various machine learning techniques, including linear regression, regularization (Ridge and LASSO), and kernel methods.

## Project Overview

In this project, the **CalCoFi** dataset is explored, followed by the implementation of multiple machine learning models. The primary objectives are:
- Perform **Exploratory Data Analysis (EDA)** to understand the structure and characteristics of the dataset.
- Use **linear regression** to predict temperature and salinity based on several features.
- Explore how the **training set size** impacts the model fit.
- Tune the model using **grid search** to find optimal hyperparameters for regularized linear regression (Ridge and LASSO).
- Apply **kernel methods** to improve the model fit and handle non-linearity.
## Dataset

The **CalCoFi** dataset used in this project contains oceanographic data, including measurements of temperature (`T_degC`), salinity (`Salnty`), and various other environmental factors. The dataset is publicly available and can be downloaded from the following links:

- [Download CalCoFi Dataset from Kaggle](https://www.kaggle.com/datasets/sohier/calcofi)
- [Updated CalCoFi Dataset from official website](https://calcofi.org/data/oceanographic-data/bottle-database/)

Place the downloaded dataset in the `data` directory, which should be at the same level as the `notebooks` folder for proper execution.

EDA.ipynb uses the latest datasets from the official website named "bottle2023.cvs" and "cast2023.csv", while the main analysis was performed on the dataset from Kaggle "bottle.csv" and "cast.csv".

## `notebooks/EDA.ipynb`
This notebook contains the **Exploratory Data Analysis (EDA)** for the **CalCoFi** dataset. In this step, I performed data cleaning, examined the data distribution, and visualized key relationships between features and target variables. The purpose of the EDA is to better understand the dataset's structure and identify the features which are relevant for predicting temperature and salinity.

## `notebooks/calcofi.ipynb`
In this notebook, I implemented a **linear regression** model to predict the following targets:
- `T_degC` (Temperature in degrees Celsius)
- `Salnty` (Salinity)

I used the following features for the regression:
- `Depthm` (Depth in meters)
- `O2ml_L` (Oxygen in milliliters per liter)
- `STheta` (Sigma Theta, a measure of seawater density)
- `Bottom_D` (Bottom Depth)
- `Wind_Spd` (Wind Speed)
- `Lat_Dec` (Latitude Decimal)
- `Lon_Dec` (Longitude Decimal)
- `Quarter` (Season of the year)

#### Key Steps:
1. **Linear Regression:** A simple regression model was initially used to establish a baseline.
2. **Effect of Training Set Size:** The dependence of the fit was examined under the variation of training set size.
3. **Regularization:** I applied **Ridge** and **LASSO** regularization techniques to improve the generalization of the model and prevent overfitting.
4. **Grid Search:** A **grid search** was performed to find the optimal hyperparameters for the regularization models.
5. **Kernel Methods:** Non-linear kernels (e.g.polynomial kernel, RBF) were used to improve the model's performance on the data by mapping it to a higher-dimensional space.
   
## `notebooks/mymodule.py`
This Python module contains utility functions used throughout the notebooks:
- **prepare_data**: Splits the data into training and test sets, and scales the features.
- **calculate_metrics**: Calculates R², RMSE, and MAE metrics for regression models.
- **plot_density_scatter** and **plot_residuals_hist**: Functions for visualizing the regression results, including scatter plots and residuals histograms.
- **create_prediction_figure**: Generates separate diagnostic plots for temperature and salinity.
- **analyze_feature_cases**: Analyzes the effect of different feature combinations on model performance.
- **rbf_kernel** and **polynomial_kernel**: Kernel functions for kernel methods (RBF and polynomial).
- **kernel_train_predict**: Trains and predicts using a Nyström approximation to kernel methods.

## Requirements

To run this project, you will need to install the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `gsw`
## Results and Discussion

The results and conclusions can be found in the notebooks. Specifically:
- The effect of training set size on model performance is discussed and visualized.
- The grid search process for regularized linear regression methods (Ridge and LASSO) is presented, including the optimal hyperparameters.
- The kernel methods and their impact on the model's ability to fit the data are discussed.

