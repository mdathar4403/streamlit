# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:11:07 2024

@author: MD ATHAR
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('D:/Downloads/Deploying Machine Learning/trained_model.sav', 'rb'))

#creating a function for Prediction

def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'


def main():
    # giving a title 
    st.title('Diabetes Prediction Web App')
    
    # getting the input data from the user
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
    Pregnancies = st.text_input('Enter the no. of Pregnancies')
    Glucose = st.text_input('Enter the Glucose')
    BloodPressure = st.text_input('Enter the BloodPressure')
    SkinThickness = st.text_input('Enter the SkinThickness')
    Insulin = st.text_input('Enter the Insulin')
    BMI = st.text_input('Enter the BMI')
    DiabetesPedigreeFunction = st.text_input('Enter the DiabetesPedigreeFunction')
    Age = st.text_input('Enter the Age')
    
    # Code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    

if __name__=='__main__':
    main()
    
    
    