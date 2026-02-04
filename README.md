# elta_task

# Titanic Survival Prediction

This is my submission for the Data Science Home Assignment. It is an end-to-end pipeline that predicts if a passenger survived the Titanic using a PyTorch neural network and a Streamlit web app.

## Project Structure

* `train.py`: The main script. It downloads data, trains the model, and saves the weights (`titanic_model.pth`) and history (`training_log.csv`).
* `ds_app.py`: The inference app. It lets you upload a CSV and see predictions.
* `eda.ipynb`: Jupyter notebook with my exploratory data analysis (plots, correlations, missing values).
* `requirements.txt`: List of libraries needed to run the code.
* `data/`: Folder for the dataset.

## Setup & Installation

1.  **Clone the repository** (or download these files).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### 1. Train the Model
You need to train the model first to generate the `.pth` file.
```bash
python train.py
Note: The script tries to download the data from Kaggle automatically. If you don't have the Kaggle API set up, it will ask you to manually download train.csv and put it in the folder.

2. Run the App
Once training is done, launch the Streamlit app:

Bash
python -m streamlit run ds_app.py
This will open a tab in your browser. You can upload train.csv (or a test file) to see the model in action.

Design & Architecture Choices
Preprocessing
I focused on cleaning the data to help the neural network learn better:

Titles: I extracted titles (Mr, Mrs, etc.) from names and grouped rare ones (like "Dr", "Rev") into a "Rare" category.

Missing Age: Instead of using a simple mean, I filled missing ages based on the median age of the passenger's Title group.

Fare: I used a Log transformation because the Fare distribution was very skewed.

Family Size: Created a new feature IsAlone because traveling alone seemed to impact survival chances.

Model Architecture
I chose a lightweight Feed-Forward Neural Network (PyTorch) instead of a heavy pre-trained model because the dataset is small/tabular.

Input Layer: Dynamic size based on features.

Hidden Layer: 32 neurons with Batch Normalization (to stabilize training) and ReLU activation.

Dropout: Added 10% dropout to prevent overfitting.

Output: Single neuron (Sigmoid) for binary classification (0 or 1).

Loss Function: BCEWithLogitsLoss with positive weights to handle the class imbalance (more people died than survived).

Evaluation
I used an 80/20 Train/Validation split.

The training script saves a training_log.csv so the App can display the Loss and Accuracy curves over time.

