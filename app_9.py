#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:27:03 2021

@author: francescaronci
"""

# =============================================================================
# conda activate EXP_ML
# cd /Users/francescaronci/Desktop/PROJECT\ EXPLINABLE\ AI
# streamlit run app_5.py
# =============================================================================
## levare split da funzione 
## aggiungere bottone
## accuracy
# path imag /Users/francescaronci/Desktop/PROJECT\ EXPLINABLE\ AI/cosbi_logo.png
import matplotlib.image as mpimg
import streamlit as st
from sklearn import datasets
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split

# form = st.form("my_form")
# form.slider("Inside the form")
# st.slider("Outside the form")

#  # Now add a submit button to the form:
# form.form_submit_button("Submit")

form=st.sidebar.form("my_form")

img = mpimg.imread('/Users/francescaronci/Desktop/PROJECT EXPLINABLE AI/cosbi_logo.png')


form.image(img)

with form:
   

    
    form.header('User Input Parameters')
    
    st.write('Choose your dataset and your classifier then press submit to set the parameters')
    
    lista_dataset=['Iris','Breast Cancer','Wine']
    lista_classificatori=['Random Forest','SVM','Decision Tree','KNN']
    
    def user_input_features():
        
        dataset_option=form.selectbox('Dataset',lista_dataset)
        
        if dataset_option =='Breast Cancer':
         df=datasets.load_breast_cancer()

        elif dataset_option == 'Iris':
         df=datasets.load_iris()

        elif dataset_option=='Wine':
         df=datasets.load_wine()

        
        classification_option= form.selectbox('Classifier',lista_classificatori)
    
        if classification_option=='Random Forest':
 
             from sklearn.ensemble import RandomForestClassifier
        
             model = RandomForestClassifier()
        
        elif classification_option=='SVM':

             from sklearn import svm
        
             model = svm.SVC(probability=True)
    
        elif classification_option=='Decision Tree':

             from sklearn.tree import DecisionTreeClassifier
        
             model = DecisionTreeClassifier()
        elif classification_option=='KNN':
            
             from sklearn.neighbors import KNeighborsClassifier
        
             model = KNeighborsClassifier(n_neighbors=3)
             
            
        X = df.data[:,0:4]
        y = df.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        max_1=max(X_test[:,0])
        min_1=min(X_test[:,0])
        
        max_2=max(X_test[:,1])
        min_2=min(X_test[:,1])
    
        max_3= max(X_test[:,2])
        min_3=min(X_test[:,2])
    
    
        max_4=max(X_test[:,3])
        min_4=max((X_test[:,3]))
        
        #float(10)
        
        feature_1 = st.sidebar.slider(df.feature_names[0], min_1,max_1, float(((max_1+1)-min_1)/2))
        feature_2 = st.sidebar.slider(df.feature_names[1], min_2,max_2, float(((max_2+1)-min_2)/2))
        feature_3 = st.sidebar.slider(df.feature_names[2], min_3,max_3, float(((max_3+1)-min_3)/2))
        feature_4 = st.sidebar.slider(df.feature_names[3], min_4, max_4,float(((max_4+1)-min_4)/2))
        data = {df.feature_names[0]: feature_1,
                df.feature_names[1]: feature_2,
                df.feature_names[2]: feature_3,
                df.feature_names[3]: feature_4}
        features = pd.DataFrame(data, index=[0])
        
    
    
     
        return classification_option,features,X_train,y_train,model,df,dataset_option,X_test,y_test
    
    classification_option_scelta,dati_utente,X_train_scelto,y_train_scelto,model,dataframe,dataset_scelto,X_test_scelto,y_test_scelto = user_input_features()

    submitted = form.form_submit_button("Submit")
    if submitted:
        st.subheader('User Input values')
        #st.write(dati_utente)
        st.write('**Classifier:** ' + classification_option_scelta  , '**Dataset:** ' + dataset_scelto)

# st.write(
#     ' The classifier used by this App is the **' + classification_option_scelta +'**'
#   ) 

# st.write(
#     ' The dataset used by this App is the **' + dataset_scelto +'**'
#   ) 
# text= '''Questa applicazione consente la classifficazione di diversi dataset, scelti dall'utente. I classificatori tra cui l'utente può scegliere sono: Decsion Tree, Knn, SVM, Random Forest.'''
# st.download_button(
#      label="Download documentation",
#      data=text,
#      file_name='exp_app.txt',
#      mime='text',
#  )
txt = st.text_area('Application Overview', '''
The purpose of this application is to allow the user to choose a 
dataset from three possibilities (Iris, Breast Cancer, Wine) to 
perform a classification associated with two explicability methods. 
The methods used are UMAP and LIME. Uniform Manifold Approximation 
and Projection (UMAP) is a dimension reduction technique that can be
 used for visualization similarly to t-SNE, but also for general 
non-linear dimension reduction, (...)
     ''')

st.subheader('**Classification results**')    

#st.subheader('User Input values')
#st.write(dati_utente)
   
    

model.fit(X_train_scelto, y_train_scelto)
prediction = model.predict(dati_utente)

col1,col2= st.columns(2)

with col1:
 st.subheader('Class labels')
 st.write(dataframe.target_names)

with col2:
 st.subheader('Prediction')
 st.write(dataframe.target_names[prediction])
#st.write(prediction)

col3,col4= st.columns(2)

with col3:
 st.subheader('Prediction Probability')
 prediction_proba = model.predict_proba(dati_utente)
 st.write(prediction_proba)

from sklearn.metrics import accuracy_score
with col4:
    st.subheader('Classification Accuracy')
    y_pred = model.predict(X_test_scelto)    
    #st.write(accuracy_score(y_test_scelto, y_pred))
    acc=str(int(accuracy_score(y_test_scelto, y_pred)*100))+'%'
    st.metric(label='percentage of accuracy', value=acc)
    
    
    
# if classifier== 'Perceptron':
#    st.write(model.score(dati_utente_std,prediction))
# else :
#    prediction_proba = model.predict_proba(dati_utente_std)
#    st.write(prediction_proba)

######## USIAMO UMAP PER VISUALIZZARE I DATI #######

import umap
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd
import matplotlib.pyplot as plt
import colorcet
import matplotlib.colors
import matplotlib.cm
import bokeh.plotting as bpl
import bokeh.transform as btr
import holoviews as hv
import holoviews.operation.datashader as hd
import umap.plot

st.write("""
 # Explaining data using **UMAP**
 """)


    
def user_input_values():
    value_1 = st.slider("Pick a number of neighbors", 2,100)

    return value_1

scelta_utente = user_input_values()

with st.spinner('Wait for it...'):
    
 mapper = umap.UMAP(random_state=42,metric='euclidean',min_dist=0.1, n_components=2, n_epochs=None, n_neighbors=scelta_utente).fit(dataframe.data)

 fig= umap.plot.points(mapper, labels=dataframe.target,theme='fire').figure

st.success('Done!')

#plt.legend([dataframe.target_names[0],dataframe.target_names[1]])
#umap.plot.connectivity(mapper, edge_bundling='hammer')

st.pyplot(fig)
 

######## USIAMO LIME PER SPIEGARE LA CLASSIFICAZIONE #######

import numpy as np
import lime
import lime.lime_tabular


numero_classi= len(dataframe.target_names)


explainer = lime.lime_tabular.LimeTabularExplainer(X_train_scelto, mode="classification",feature_names= dataframe.feature_names, class_names=dataframe.target_names, discretize_continuous=True)

#i = np.random.randint(0, X_scelto.shape[0])


exp = explainer.explain_instance(np.array(dati_utente)[0], model.predict_proba, num_features=np.shape(X_train_scelto)[1], top_labels=numero_classi)

#exp.save_to_file('figura.html')



st.write("""
 # Explaining predictions using **LIME**
 """)

for classe in range(numero_classi):
 st.pyplot(exp.as_pyplot_figure(label=classe))

