import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image

# Load  model a 
model = joblib.load(open("decision_tree_model.sav","rb"))

def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    df.Smoking = df.Smoking.map({'Yes':1, 'No':0})
    df.Stroke = df.Stroke.map({'Yes':1, 'No':0})
    df.DiffWalking = df.DiffWalking.map({'Yes':1, 'No':0})
    df.Sex = df.Sex.map({'Yes':1, 'No':0})
    df.Diabetic = df.Diabetic.map({'Yes':1, 'No':0})
    df.PhysicalActivity = df.PhysicalActivity.map({'Yes':1, 'No':0})
    df.KidneyDisease = df.KidneyDisease.map({'Yes':1, 'No':0})
    df.SkinCancer = df.SkinCancer.map({'Yes':1, 'No':0})
    df.AgeCategory = df.SkinCancer.map({'18-24':0,'25-29':1,'30-34':2,'35-39':3,'40-44':4,'45-49':5,'50-54':6,'55-59':7,'60-64':8,'65-69':9,'70-74':10,'75-79':11,'80 or older':12})
    df.GenHealth = df.GenHealth.map({'Excellent':0,'Very good':1,'Good':2,'Fair':3,'Poor':4})
   
    return df

st.write("""
# Heart Disease Prediction using ML Model 
This app predicts the ** Quality of Wine **  using **wine features** input via the **side panel** 
""")

#read in wine image and render with streamlit
image = Image.open('wine_image.png')
st.image(image, caption='wine company',use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar

def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe

    """
    Smoking = st.sidebar.selectbox("Do you smoke?",("Yes", "No"))
    Stroke = st.sidebar.selectbox("Did you stroke ever?",("Yes", "No"))
    DiffWalking = st.sidebar.selectbox("Do you have difficulty in walking?",("Yes", "No"))
    Sex = st.sidebar.selectbox("Gender?",("Yes", "No"))
    Diabetic = st.sidebar.selectbox("Are you a diabetic patient?",("Yes", "No"))
    PhysicalActivity = st.sidebar.selectbox("Do you exercise?",("Yes", "No"))
    KidneyDisease = st.sidebar.selectbox("Do you have kidney disease?",("Yes", "No"))
    SkinCancer = st.sidebar.selectbox("Do you have skin cancer?",("Yes", "No"))
    AgeCategory = st.sidebar.selectbox("Whats your age range?",("18-24", "25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80 or older"))
    GenHealth = st.sidebar.selectbox("What's your general health condition?",("Excellent", "Very good","Good","Fair","Poor"))
    
    PhysicalHealth = st.sidebar.slider('PhysicalHealth', 0.0, 30, 1)
    BMI = st.sidebar.slider('BMI', 12.0, 94.85, 0.05)
     
    features = {'Smoking': Smoking,
            'Stroke': Stroke,
            'DiffWalking': DiffWalking,
            'Sex': Sex,
            'Diabetic': Diabetic,
            'PhysicalActivity': PhysicalActivity,
            'KidneyDisease': KidneyDisease,
            'SkinCancer': SkinCancer,
            'AgeCategory': AgeCategory,
            'GenHealth': GenHealth,
            'PhysicalHealth': PhysicalHealth,
            'BMI': BMI
            }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.subheader('User Input parameters')
st.write(user_input_df)

prediction = model.predict(processed_user_input)

# creating a button for Prediction 
heart_diagnosis = ''
if st.button('Heart Disease Diagnosis'):       
    if (prediction[0] == 1):
        heart_diagnosis = 'The person is having heart disease'
    else:
        heart_diagnosis = 'The person does not having heart disease'
        
st.success(heart_diagnosis)


