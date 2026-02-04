import os
import sys
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score # added these
import subprocess

# --- defulat r values ---
DATA_FILE = 'data/train.csv'
MODEL_SAVE_PATH = 'titanic_model.pth'
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
PATIENCE = 5
SEED = 42

# --- lets make a seed so it will be 42 (42 the answer for every solution!!! oh yeah) ---
def seed_everything(seed=42):
    """
    setting seed so results wont change every time i run it
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"seed set to {seed}")

# --get the data ---
def fetch_data():
    """
    checking if data exists, if not try to download from   kaggle
    """
    if os.path.exists(DATA_FILE):
        print(f"found {DATA_FILE}, skipping download")
        return

    print(f"dataset {DATA_FILE} not found. trying to download...")
    
    try:
        # trying to run kaggle command
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "titanic"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            shell=True
        )
        
        print("download done. unzipping now...")
        subprocess.run(["unzip", "-o", "titanic.zip"], shell=True)
        print("data is ready")
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # if kaggle command fails or not found
        print("\n" + "!"*60)
        print("could not download data automatically")
        print("please download 'train.csv' manually from kaggle")
        print(f"put it in this folder: {os.getcwd()}")
        print("!"*60 + "\n")
        sys.exit(1)

# ---cleaning  the data ---
def preprocess_data(df):
    """
    function to clean the data. i use this for training and the app later
    """
    df = df.copy()
    
    # extracting title from name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # grouping rare titles together
    rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # filling missing age with median of title group
    df["Age"] = df["Age"].fillna(df.groupby("Title")["Age"].transform("median"))

    # feature engineering family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # filling embarked and log transform fare
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'] = df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

    # mapping string to int
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping).fillna(0)

    # drop columns i dont need
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Age_Bin']
    existing_cols = [c for c in drop_elements if c in df.columns]
    df = df.drop(existing_cols, axis=1)

    return df

# --my modell: the neural network ---
class TitanicNet(nn.Module):
    def __init__(self, input_size):
        super(TitanicNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.bc1 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- step 5: helper functions ---
def get_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    # convertingg to tensors
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion):
    """
    updated this to calculate precision and recall too
    """
    model.eval()
    running_loss = 0.0
    
    # gathering all preds to calc metrics at once
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # storing them on cpu
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    # calculating standard metrics
    val_acc = accuracy_score(all_targets, all_preds)
    val_prec = precision_score(all_targets, all_preds, zero_division=0)
    val_rec = recall_score(all_targets, all_preds, zero_division=0)
    
    return running_loss / len(loader), val_acc, val_prec, val_rec

# main function
def main():
    print("--- starting titanic training ---")
    
    # setup
    seed_everything(SEED)
    fetch_data()
    
    # load and clean data
    print("loading and cleaning data...")
    raw_df = pd.read_csv(DATA_FILE)
    clean_df = preprocess_data(raw_df)
    
    X = clean_df.drop("Survived", axis=1)
    y = clean_df["Survived"]
    
    # split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # get loaders
    train_loader, val_loader = get_dataloaders(X_train, y_train, X_val, y_val, BATCH_SIZE)
    
    # calculate weights for imbalance
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    
    # init model
    input_size = X_train.shape[1]
    model = TitanicNet(input_size)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2)
    
    # training loop
    best_val_loss = float('inf')
    counter = 0
    history = [] 

    print(f"\nstarting training for {NUM_EPOCHS} epochs...")
    print("-" * 95)
    print(f"{'Epoch':^5} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Precision':^10} | {'Recall':^10}")
    print("-" * 95)
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        # updated to unpack 4 values
        val_loss, val_acc, val_prec, val_rec = validate(model, val_loader, criterion)
        
        # --- saving extended stats ---
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec
        })

        scheduler.step(val_loss)
        
        # printing progress with new metrics
        if (epoch+1) % 5 == 0:
            print(f"{epoch+1:^5} | {train_loss:^10.4f} | {val_loss:^10.4f} | {val_acc:^10.4f} | {val_prec:^10.4f} | {val_rec:^10.4f}")
            
        # early stopping and saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'input_size': input_size}, MODEL_SAVE_PATH)
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"\nearly stopping at epoch {epoch+1}")
                break
    print(f"{epoch+1:^5} | {train_loss:^10.4f} | {val_loss:^10.4f} | {val_acc:^10.4f} | {val_prec:^10.4f} | {val_rec:^10.4f}")
    
    # --- Save history to CSV ---
    pd.DataFrame(history).to_csv("logs.csv", index=False)
    print(f"\ntraining done. best model saved to {MODEL_SAVE_PATH}")
    print("training history saved to logs.csv")

if __name__ == "__main__":

    main()

