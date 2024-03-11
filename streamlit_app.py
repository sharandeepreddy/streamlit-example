from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu,gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted
from sklearn import linear_model, tree, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

class ECG:
	def  getImage(self,image):
		"""
		this functions gets user image
		return: user image
		"""
		image=imread(image)
		return image

	def ModelLoad_predict(self,final_df):
		loaded_model = joblib.load('Heart_Disease_Prediction_using_ECG (4).pkl')
		result = loaded_model.predict(final_df)
		if result[0] == 1:
			return "You ECG corresponds to Myocardial Infarction"
		elif result[0] == 0:
			return "You ECG corresponds to Abnormal Heartbeat"
		elif result[0] == 2:
			return "Your ECG is Normal"
		else:
			return "You ECG corresponds to History of Myocardial Infarction"
