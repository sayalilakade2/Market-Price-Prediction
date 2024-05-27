# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:36:04 2024

@author: sayal
"""

import pickle
import numpy as np
import streamlit as st


# Load the model
try:
    with open('model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except Exception as e:
    st.error(f"Error loading the model: {e}")

def DecisionTreeRegressor(input_data):
    input_data_asarray = np.asarray(input_data)
    input_data_reshaped = input_data_asarray.reshape(1, -1) 
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def predict_price(entries):
    try:
        # Get user input
        input_data = [int(entries[0]), float(entries[1]), int(entries[2]), int(entries[3]), 
                      float(entries[4]), int(entries[5]), int(entries[6])]
        
        # Perform prediction
        predicted_price = DecisionTreeRegressor(input_data)[0]
        return f"The predicted price is ${predicted_price:,.2f}"
    except ValueError:
        return "Please enter valid inputs."
    except Exception as e:
        return f"Error during prediction: {e}"
    

def main():
    st.title("Market Price Prediction")
    entries = []
    for feature in ['District Name:', 'Market Name:', 'Commodity:', 'Variety:', 'Grade:', 
                    'temp:', 'precip:']:
        entries.append(st.text_input(feature))
    
    if st.button('Predict Price'):
        result = predict_price(entries)
        st.success(result)

if __name__ == '__main__':
    main()

