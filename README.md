# elta_task  
## Titanic Survival Prediction

This repository contains my submission for a **Data Science Home Assignment**.  
It implements an **end-to-end machine learning pipeline** for predicting passenger survival on the Titanic using **PyTorch** and **Streamlit**.

---

## Project Overview

The goal of this project is to predict whether a passenger survived the Titanic disaster based on demographic and ticket-related features.  
The solution focuses on careful preprocessing of tabular data and a lightweight neural network architecture suitable for small datasets.

---

## Project Structure
├── train.py # Trains the model and saves artifacts

├── ds_app.py # Streamlit inference app

├── eda.ipynb # Exploratory Data Analysis

├── requirements.txt # Python dependencies

├── data/ # Dataset directory

├── titanic_model.pth # Saved model weights (generated after training)

└── training_log.csv # Training history (generated after training)



### File Descriptions

- **train.py**  
  Main training script. Downloads data, preprocesses it, trains the model, and saves:
  - `titanic_model.pth`
  - `training_log.csv`

- **ds_app.py**  
  Streamlit app for running inference on uploaded CSV files.

- **eda.ipynb**  
  Exploratory Data Analysis including distributions, correlations, and missing values.

- **requirements.txt**  
  List of required Python libraries.

- **data/**  
  Folder for the Titanic dataset.

---

## Setup & Installation

1. Clone the repository (or download the files).
2. Install dependencies:

bash:
pip install -r requirements.txt


## How to Run
### 1. Train the Model

Run the training script:

python train.py


Note:
The script attempts to download the dataset using the Kaggle API.
If the API is not configured, manually download train.csv and place it in the data/ folder.

## 2. Run the Streamlit App

After training completes, launch the app:

python -m streamlit run ds_app.py

This opens a browser tab where you can upload a CSV file and view predictions.


# Design & Architecture Choices
## Data Preprocessing

### Title Extraction
Extracted titles (Mr, Mrs, Miss, etc.) from passenger names.
Rare titles (e.g., Dr, Rev) were grouped into a single Rare category.

### Age Imputation
Missing ages were filled using the median age per Title group.

F### are Transformation
Applied log transformation to reduce skewness.

### Family Features
Created an IsAlone feature to capture solo travelers

