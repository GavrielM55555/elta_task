import streamlit as st
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

# trying to import the stuff i wrote in train.py
try:
    from train import TitanicNet, preprocess_data
except ImportError:
    st.error("cant import from train.py. make sure the file is in the same folder.")
    st.stop()

# setting up the page layout
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

def load_saved_model(source):
    """
    function to load the actual model weights.
    it works with a path string or if i upload a file directly.
    """
    try:
        checkpoint = torch.load(source)
        # i need the input size to build the network structure correctly
        input_size = checkpoint['input_size']
        model = TitanicNet(input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # important to set this to eval mode so it doesnt learn
        return model
    except Exception as e:
        st.error(f"error loading model: {e}")
        return None

def show_sidebar():
    """
    handling the sidebar menu here. allowing model selection and showing graphs.
    """
    st.sidebar.header("1. Model Configuration")
    
    # option to use the default model i trained or upload a new one
    model_source = st.sidebar.radio("Model Source", ["Use Default Model", "Upload Model File"])
    
    model = None
    if model_source == "Use Default Model":
        # checking if the file actually exists on the disk
        if Path("titanic_model.pth").is_file():
            model = load_saved_model("titanic_model.pth")
            st.sidebar.success("Default model loaded")
        else:
            st.sidebar.warning("Default model not found. I need to train it first.")
    else:
        # allowing upload of a custom .pth file
        uploaded_model = st.sidebar.file_uploader("Upload .pth file", type="pth")
        if uploaded_model:
            model = load_saved_model(uploaded_model)
            st.sidebar.success("Custom model loaded")

    # showing the training stats if the log file exists
    if st.sidebar.checkbox("Show Training Performance", value=True):
        st.subheader("Training History")
        try:
            history_df = pd.read_csv("training_log.csv")
            
            # --- plotting graphs ---
            
            # 1. Loss Graph
            st.write("**Loss Curve**")
            st.line_chart(history_df.set_index('epoch')[['train_loss', 'val_loss']])
            
            # 2. Accuracy Graph
            st.write("**Accuracy Curve**")
            st.line_chart(history_df.set_index('epoch')['val_acc'])
            
            # 3. NEW: Precision & Recall Graph (if they exist in the csv)
            if 'val_precision' in history_df.columns and 'val_recall' in history_df.columns:
                st.write("**Precision & Recall**")
                # plotting both lines on the same chart
                st.line_chart(history_df.set_index('epoch')[['val_precision', 'val_recall']])
            
            # --- Final Stats Table ---
            st.write("**Final Model Stats (Last Epoch)**")
            
            last_epoch = history_df.iloc[-1]
            
            stats = {
                "Final Accuracy": [f"{last_epoch['val_acc']:.2%}"],
                "Final Loss": [f"{last_epoch['val_loss']:.4f}"]
            }
            
            if 'val_precision' in history_df.columns:
                 stats["Final Precision"] = [f"{last_epoch['val_precision']:.2%}"]
            if 'val_recall' in history_df.columns:
                 stats["Final Recall"] = [f"{last_epoch['val_recall']:.2%}"]

            st.dataframe(pd.DataFrame(stats), hide_index=True)

        except FileNotFoundError:
            st.warning("no training log found yet.")
            
    st.divider()
    return model

def display_results(y_true, preds):
    """
    calculating the metrics and plotting the confusion matrix.
    """
    st.write("#### Model Performance Metrics")

    # getting the full report as a dictionary so i can pick what i need
    report = classification_report(y_true, preds, output_dict=True)
    
    # extracting just the overall numbers
    metrics_data = {
        "Accuracy": [report['accuracy']],
        "Precision": [report['weighted avg']['precision']],
        "Recall": [report['weighted avg']['recall']]
    }
    
    # creating a simple dataframe for the table
    metrics_df = pd.DataFrame(metrics_data, index=["Overall Performance"])
    
    # formatting it nicely with percentages
    st.dataframe(metrics_df.style.format("{:.2%}"), use_container_width=True)
    
    # plotting the graphs side by side
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_true, preds.astype(int))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)
        
    with col2:
        st.write("**Prediction Balance**")
        fig2, ax2 = plt.subplots()
        plot_data = preds.astype(int)
        sns.countplot(
            x=plot_data, 
            hue=plot_data, 
            palette={0: "red", 1: "green", "0": "red", "1": "green"},
            legend=False
        )
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Deceased', 'Survived'])
        st.pyplot(fig2)

def main():
    st.title("Titanic Survival Prediction System")
    st.markdown("**Welcome.** This is my project using a PyTorch neural network to predict passenger survival.")

    # 1. getting the model from the sidebar
    model = show_sidebar()

    # 2. main section for data upload
    st.header("2. Upload Passenger Data")
    uploaded_file = st.file_uploader("Choose a CSV file (e.g., train.csv)", type="csv")

    if uploaded_file and model:
        # reading the raw csv file
        df_raw = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview")
        st.dataframe(df_raw.head())

        # 3. preprocessing the data
        try:
            with st.spinner("Preprocessing data..."):
                df_clean = preprocess_data(df_raw)
            st.success("Data processed successfully")
        except Exception as e:
            st.error(f"preprocessing failed: {e}")
            st.stop()

        # 4. running the predictions
        if st.button("Run Predictions", type="primary"):
            
            # preparing the data tensors
            if 'Survived' in df_clean.columns:
                X = df_clean.drop("Survived", axis=1)
                y_true = df_clean["Survived"]
                has_labels = True
            else:
                X = df_clean
                has_labels = False
            
            X_tensor = torch.tensor(X.values, dtype=torch.float32)

            # making predictions
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float().numpy().flatten()

            # adding results back to dataframe
            results_df = df_raw.copy()
            results_df['Predicted_Survival'] = preds.astype(int)
            # changed this from 'Confidence' to 'Model_Survival_Probability' to be less confusing
            results_df['Model_Survival_Probability'] = probs.numpy().flatten()
            
            # columns to show in the final table
            cols_to_show = ['Name', 'Sex', 'Age', 'Predicted_Survival', 'Model_Survival_Probability']

            st.divider()
            st.write("### Prediction Results")
            
            # 5. showing metrics if labels are present
            if has_labels:
                display_results(y_true, preds)
                
                # logic to add correct/incorrect column
                results_df['Actual_Survival'] = y_true
                results_df['Is_Correct'] = results_df['Actual_Survival'] == results_df['Predicted_Survival']
                
                # update the list of columns to show
                cols_to_show = ['Name', 'Sex', 'Age', 'Actual_Survival', 'Predicted_Survival', 'Is_Correct', 'Model_Survival_Probability']
            else:
                st.info("No 'Survived' column found. Showing predictions only.")
                st.write(f"Predicted Survivors: {int(preds.sum())} out of {len(preds)}")

            # 6. showing the table and download button
            st.write("### Individual Predictions")
            
            # formatting the float column to look like percentage for better readability
            st.dataframe(
                results_df[cols_to_show].head(20).style.format({
                    "Model_Survival_Probability": "{:.2%}" 
                })
            )
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions",
                csv,
                "titanic_predictions.csv",
                "text/csv"
            )
            
    elif uploaded_file is None:
        st.info("Please upload a CSV file to start.")
    elif model is None:
        st.warning("No model loaded. Please check the sidebar.")

if __name__ == "__main__":
    main()