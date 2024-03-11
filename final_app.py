import streamlit as st
from Ecg import  ECG
#intialize ecg object
ecg = ECG()
#get the uploaded image
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  """#### **UPLOADED IMAGE**"""
  # call the getimage method
  ecg_user_image_read = ecg.getImage(uploaded_file)
  #show the image
  st.image(ecg_user_image_read)

  """#### **PASS TO PRETRAINED ML MODEL FOR PREDICTION**"""
  #call the Pretrainsed ML model for prediction
  ecg_model=ecg.ModelLoad_predict(ecg_final)
  my_expander5 = st.expander(label='PREDICTION')
  with my_expander5:
    st.write(ecg_model)
