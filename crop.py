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
        if not hasattr(loaded_model, 'predict'):
            raise ValueError("The loaded object is not a model with a predict method.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

def DecisionTreeRegressor(input_data):
    input_data_asarray = np.asarray(input_data)
    input_data_reshaped = input_data_asarray.reshape(1, -1) 
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def predict_price(entries):
    try:
        # Convert user input to appropriate data types
        input_data = [
            int(entries[0]),  # District Name
            int(entries[1]),  # Market Name
            int(entries[2]),  # Commodity
            int(entries[3]),  # Variety
            float(entries[4]),  # Temperature
            float(entries[5])   # Precipitation
        ]
        
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
    for feature in ['District Name', 'Market Name', 'Commodity', 'Variety', 'Temperature', 'Precipitation']:
        entries.append(st.text_input(feature))
    
    if st.button('Predict Price'):
        result = predict_price(entries)
        st.success(result)

if __name__ == '__main__':
    main()
