# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:36:04 2024

@author: sayal
"""

import pickle
import numpy as np
import streamlit as st
import requests
import os

# Load the model from the GitHub raw content URL
model_url = "https://github.com/sayalilakade2/Market-Price-Prediction/raw/main/model.pkl"

# Ensure the model is downloaded
model_path = 'model.pkl'
if not os.path.exists(model_path):
    r = requests.get(model_url)
    if r.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(r.content)
    else:
        st.error("Failed to download the model file")
        st.stop()

# Load the model
try:
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
        if not hasattr(loaded_model, 'predict'):
            raise ValueError("The loaded object is not a model with a predict method.")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

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
    st.title("Market Price Prediction")
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
