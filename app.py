#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 15:42:35 2021

@author: francescaronci
"""

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.write("""
# Breast Cancer Prediction App
This app predicts if the breast cancer is **malignant** or **benign**
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    mean_radius = st.sidebar.slider('Mean Radius', 6.981, 28.11, 15.2)
    mean_texture = st.sidebar.slider('Mean Texture', 9.71, 39.28, 23.4)
    mean_perimeter = st.sidebar.slider('Mean Perimeter', 43.79, 188.5, 102.6)
    mean_area = st.sidebar.slider('Mean Area', 143.5, 2501.0, 1245.7)
    data = {'mean_radius': mean_radius,
            'mean_texture': mean_texture,
            'mean_perimeter': mean_perimeter,
            'mean_area': mean_area}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input values')
st.write(df)


cancer=datasets.load_breast_cancer()
X = cancer.data[:,0:4]
y = cancer.target

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)

sc.fit(X)
X_std= sc.transform(X)
df_std=sc.transform(df)

clf = RandomForestClassifier()
clf.fit(X_std, y)

prediction = clf.predict(df_std)
prediction_proba = clf.predict_proba(df_std)

st.subheader('Class labels and their corresponding index number')
st.write(cancer.target_names)

st.subheader('Prediction')
st.write(cancer.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

######## USIAMO LIME PER SPIEGARE LA CLASSIFICAZIONE #######

import sklearn
import numpy as np
import lime
import lime.lime_tabular
import plotly

np.random.seed(1)

explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names= cancer.feature_names, class_names=cancer.target_names, discretize_continuous=True)

i = np.random.randint(0, X.shape[0])
exp = explainer.explain_instance(X[i], clf.predict_proba, num_features=4, top_labels=1)

exp.show_in_notebook(show_table=True, show_all=False)

#fig=exp.save_to_file('figura.html')
#fig = exp.as_pyplot_figure(label=0)

st.write("""
# Explaining predictions for malignant cancer using **LIME**
""")
#st.pyplot(fig)


st.write(pd.DataFrame({
     'Class malignant': [exp.as_list(label=0)[0][0],exp.as_list(label=0)[1][0],exp.as_list(label=0)[2][0],exp.as_list(label=0)[3][0]] }))
# def user_input_features():
#     data = {'mean_radius': exp.as_list(label=0)[0],
#             'mean_texture': exp.as_list(label=0)[1],
#             'mean_perimeter': exp.as_list(label=0)[2],
#             'mean_area':exp.as_list(label=0)[3]}
    

# df = user_input_features()

