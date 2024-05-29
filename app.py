import pickle
import scikit-learn
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the model
model_path = "model.pkl"
encoder_path = "label_encoders.pkl"

with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open(encoder_path, 'rb') as encoder_file:
    label_encoders = pickle.load(encoder_file)

def preprocess_input(input_data):
    for column in label_encoders.keys():
        # Check if the input_data contains new unseen labels
        unseen_labels = set(input_data[column]) - set(label_encoders[column].classes_)
        if unseen_labels:
            # Append new unseen labels to the classes of the encoder
            label_encoders[column].classes_ = np.append(label_encoders[column].classes_, list(unseen_labels))
        input_data[column] = label_encoders[column].transform(input_data[column].astype(str))
    return input_data

def predict_price(entries):
    try:
        # Prepare input data
        input_data = pd.DataFrame([entries], 
                                  columns=['District Name', 'Market Name', 'Commodity', 'Variety', 'Grade', 'temp', 'precip'])
        
        # Encode the categorical features using the label encoders
        input_data = preprocess_input(input_data)
        
        # Perform prediction
        input_data = input_data.values.flatten().reshape(1, -1)
        predicted_price = loaded_model.predict(input_data)
        
        return f"The predicted Min price is {predicted_price[0][0]:,.2f} to Max price is {predicted_price[0][1]:,.2f}"
    except ValueError as e:
        return f"Please enter valid inputs. Error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    st.title("Market Price Prediction")
    
    district_name = st.text_input("District Name")
    market_name = st.text_input("Market Name")
    commodity = st.text_input("Commodity")
    variety = st.text_input("Variety")
    grade = st.text_input("Grade")
    temp = st.text_input("Temperature")
    precip = st.text_input("Precipitation")
    
    if st.button('Predict Price'):
        entries = [district_name, market_name, commodity, variety, grade, temp, precip]
        result = predict_price(entries)
        st.success(result)

if __name__ == '__main__':
    main()
