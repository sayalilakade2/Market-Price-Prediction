# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:36:04 2024

@author: sayal
"""

import os
import pickle
import numpy as np
import streamlit as st

# Load the model
model_path = 'model.pkl'
loaded_model = None

st.title("Market Price Prediction")

# Load the model with detailed debugging
if os.path.exists(model_path):
    try:
        st.write(f"Attempting to load model from {model_path}...")
        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
            st.write(f"Loaded model type: {type(loaded_model)}")  # Debug: Print the type of the loaded model
            if not hasattr(loaded_model, 'predict'):
                raise ValueError("The loaded object is not a model with a predict method.")
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
else:
    st.error(f"Model file not found at path: {model_path}")

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
        
        # Prepare input data
        input_data_asarray = np.asarray(input_data)
        input_data_reshaped = input_data_asarray.reshape(1, -1)
        
        # Perform prediction
        predicted_price = loaded_model.predict(input_data_reshaped)[0]
        return f"The predicted price is ${predicted_price:,.2f}"
    except ValueError:
        return "Please enter valid inputs."
    except Exception as e:
        return f"Error during prediction: {e}"

def main():
    entries = []
    for feature in ['District Name', 'Market Name', 'Commodity', 'Variety', 'Temperature', 'Precipitation']:
        entries.append(st.text_input(feature))
    
    if st.button('Predict Price'):
        if loaded_model is not None:
            result = predict_price(entries)
            st.success(result)
        else:
            st.error("Model is not loaded properly.")

if __name__ == '__main__':
    main()

