import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage import color, filters, measure, transform

class ECG:
    def __init__(self):
        self.image = None
        self.gray_image = None
        self.processed_leads = []
        self.contour_leads = []
        self.pca_model = joblib.load('PCA_ECG.pkl')
        self.ml_model = joblib.load('Heart_Disease_Prediction_using_ECG.pkl')

    def load_image(self, uploaded_file):
        self.image = Image.open(uploaded_file)

    def to_gray_scale(self):
        self.gray_image = color.rgb2gray(self.image)

    def divide_leads(self):
        self.processed_leads = [
            self.gray_image[300:600, 150:643],  # Lead 1
            self.gray_image[300:600, 646:1135],  # Lead aVR
            self.gray_image[300:600, 1140:1625],  # Lead V1
            self.gray_image[300:600, 1630:2125],  # Lead V4
            self.gray_image[600:900, 150:643],  # Lead 2
            self.gray_image[600:900, 646:1135],  # Lead aVL
            self.gray_image[600:900, 1140:1625],  # Lead V2
            self.gray_image[600:900, 1630:2125],  # Lead V5
            self.gray_image[900:1200, 150:643],  # Lead 3
            self.gray_image[900:1200, 646:1135],  # Lead aVF
            self.gray_image[900:1200, 1140:1625],  # Lead V3
            self.gray_image[900:1200, 1630:2125],  # Lead V6
            self.gray_image[1250:1480, 150:2125]  # Long Lead
        ]

    def preprocess_leads(self):
        self.contour_leads = []
        for lead in self.processed_leads:
            blurred_image = filters.gaussian(lead, sigma=1)
            thresh = filters.threshold_otsu(blurred_image)
            binary_image = blurred_image < thresh
            binary_image = transform.resize(binary_image, (300, 450))
            self.contour_leads.append(binary_image)

    def extract_contours(self):
        for lead in self.contour_leads:
            contours = measure.find_contours(lead, 0.8)
            for contour in contours:
                if contour.shape in sorted([x.shape for x in contours])[-1:]:
                    self.contour_leads.append(transform.resize(contour, (255, 2)))

    def scale_data(self):
        scaled_data = []
        for lead in self.contour_leads:
            scaled_data.append(self.min_max_scaler(lead))
        return scaled_data

    def min_max_scaler(self, data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.reshape(-1, 1))

    def predict(self, uploaded_file):
        self.load_image(uploaded_file)
        self.to_gray_scale()
        self.divide_leads()
        self.preprocess_leads()
        self.extract_contours()
        scaled_data = self.scale_data()
        pca_result = self.pca_model.transform(scaled_data)
        prediction = self.ml_model.predict(pca_result)
        return prediction

if __name__ == "__main__":
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        ecg = ECG()
        prediction = ecg.predict(uploaded_file)

        st.write("### Prediction")
        st.write(prediction)

        st.write("### Uploaded Image")
        ecg_user_image_read = Image.open(uploaded_file)
        st.image(ecg_user_image_read)

        st.write("### Gray Scale Image")
        ecg_user_gray_image_read = ecg.to_gray_scale()
        st.image(ecg_user_gray_image_read)

        st.write("### Dividing Leads")
        dividing_leads = ecg.divide_leads()
        st.image('Leads_1-12_figure.png')
        st.image('Long_Lead_13_figure.png')

        st.write("### Preprocessed Leads")
        ecg_preprocessed_leads = ecg.preprocess_leads()
        st.image('Preprossed_Leads_1-12_figure.png')
        st.image('Preprossed_Leads_13_figure.png')

        st.write("### Extracting Signals (1-12)")
        ec_signal_extraction = ecg.extract_contours()
        st.image('Contour_Leads_1-12_figure.png')

        st.write("### 1D Signals")
        ecg_1dsignal = ecg.scale_data()
        st.write(ecg_1dsignal)

        st.write("### Dimensional Reduction")
        ecg_final = ecg.pca_model.transform(ecg_1dsignal)
        st.write(ecg_final)
