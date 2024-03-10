# prompt: display the above cell

%%writefile app.py

import streamlit as st
import joblib
import numpy as np

# Define the ModelLoad_predict function
def ModelLoad_predict(final_df):
    """
    This Function Loads the pretrained model and performs ECG classification
    return the classification Type.
    """
    # Load the pretrained model from the pickle file
    loaded_model = joblib.load('/content/Heart_Disease_Prediction_using_ECG (4).pkl')
    
    # Perform prediction using the loaded model
    result = loaded_model.predict(final_df)
    
    # Translate prediction to human-readable format
    if result[0] == 1:
        return "Your ECG corresponds to Myocardial Infarction"
    elif result[0] == 0:
        return "Your ECG corresponds to Abnormal Heartbeat"
    elif result[0] == 2:
        return "Your ECG is Normal"
    else:
        return "Your ECG corresponds to History of Myocardial Infarction"

# Streamlit app
def main():
    st.title('ECG Classification App')

    # Input interface for ECG data
    ecg_data = st.text_area('Enter ECG data (comma-separated)', '')

    # Convert input data to numpy array
    if ecg_data:
        final_df = np.array([float(x) for x in ecg_data.split(',')]).reshape(1, -1)
    else:
        final_df = None

    # Call the ModelLoad_predict function to get the prediction
    if final_df is not None:
        prediction = ModelLoad_predict(final_df)
        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
