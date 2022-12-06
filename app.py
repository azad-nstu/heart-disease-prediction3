import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image

# Load  model a 
model = joblib.load(open("decision_tree_model.joblib","rb"))



st.write("""
# Heart Disease Prediction using ML Model 
by **Abul Kalam Azad.**
This app predicts the possibility of heart attck by analyzing different habitual and physical condition using Machine Learning model.
""")

#read in wine image and render with streamlit
image = Image.open('heart_attack.jpg')
st.image(image, caption='A heart disease scenario',use_column_width=True)

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Heart Disease Prediction'],
                          icons=['heart'],
                          default_index=0)
    
    

# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        Smoking = st.text_input('Smoking')
        
    with col2:
        Stroke = st.text_input('Stroke')
        
    with col3:
        DiffWalking = st.text_input('DiffWalking')
        
    with col4:
        Sex = st.text_input('Sex')
        
    with col1:
        Diabetic = st.text_input('Diabetic')
        
    with col2:
        PhysicalActivity = st.text_input('PhysicalActivity')
        
    with col3:
        KidneyDisease = st.text_input('KidneyDisease')
        
    with col4:
        SkinCancer = st.text_input('SkinCancer')
        
    with col1:
        AgeCategory = st.text_input('AgeCategory')
        
    with col2:
        GenHealth = st.text_input('GenHealth')
        
    with col3:
        PhysicalHealth = st.text_input('PhysicalHealth')
        
    with col4:
        BMI = st.text_input('BMI')
        
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[Smoking, Stroke, DiffWalking, Sex, Diabetic, PhysicalActivity, KidneyDisease, SkinCancer, AgeCategory, GenHealth, PhysicalHealth, BMI]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
