elta_task
Titanic Survival Prediction

This repository contains my submission for a Data Science Home Assignment.
It implements a full end-to-end machine learning pipeline for predicting passenger survival on the Titanic, including:

Data preprocessing and feature engineering

Model training using PyTorch

Model evaluation and logging

An interactive Streamlit web app for inference

Project Overview

The goal of this project is to predict whether a passenger survived the Titanic disaster based on demographic and ticket-related features.
The solution focuses on careful preprocessing of tabular data and a lightweight neural network architecture suitable for small datasets.

Project Structure
.
├── train.py            # Downloads data, preprocesses it, trains the model, and saves artifacts
├── ds_app.py           # Streamlit app for inference and visualization
├── eda.ipynb           # Exploratory Data Analysis (EDA)
├── requirements.txt    # Python dependencies
├── data/               # Dataset directory
├── titanic_model.pth   # Saved model weights (generated after training)
└── training_log.csv    # Training history (generated after training)
File Descriptions

train.py
Main training script. Handles data loading, preprocessing, model training, and saves:

Model weights (titanic_model.pth)

Training history (training_log.csv)

ds_app.py
Streamlit inference app that allows uploading a CSV file and viewing survival predictions.

eda.ipynb
Exploratory analysis including:

Feature distributions

Correlations

Missing value analysis

requirements.txt
All required Python packages.

data/
Directory for storing the Titanic dataset.

Setup & Installation

Clone the repository (or download the files).

Install dependencies:

pip install -r requirements.txt
How to Run
1. Train the Model

Run the training script to generate the model and logs:

python train.py

Note:
The script attempts to download the dataset automatically using the Kaggle API.
If the Kaggle API is not configured, you will be prompted to manually download train.csv and place it in the data/ folder.

2. Run the Streamlit App

After training is complete, launch the inference app:

python -m streamlit run ds_app.py

This will open a browser tab where you can:

Upload train.csv or another compatible test file

View survival predictions

Inspect training loss and accuracy curves

Design & Architecture Choices
Data Preprocessing

Special care was taken to engineer features that improve learning on tabular data:

Title Extraction
Extracted titles (e.g., Mr, Mrs, Miss) from passenger names.
Rare titles (e.g., Dr, Rev) were grouped into a single “Rare” category.

Missing Age Imputation
Missing ages were filled using the median age per Title group, rather than a global mean.

Fare Transformation
Applied a log transformation to the Fare feature due to heavy right skew.

Family Features
Created an IsAlone feature, since solo travelers showed different survival patterns.

Model Architecture

A lightweight feed-forward neural network was chosen over large pre-trained models, as the dataset is small and tabular.

Input Layer: Dynamically sized based on engineered features

Hidden Layer:

32 neurons

Batch Normalization (training stability)

ReLU activation

Regularization:

10% Dropout to reduce overfitting

Output Layer:

Single neuron with Sigmoid activation for binary classification

Loss Function:

BCEWithLogitsLoss

Positive class weighting to handle class imbalance (more non-survivors than survivors)

Evaluation & Logging

Data split: 80% training / 20% validation

Training metrics (loss and accuracy) are saved to training_log.csv

The Streamlit app visualizes these metrics over training epochs
