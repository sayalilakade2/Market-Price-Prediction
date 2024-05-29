# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:52:35 2024

@author: sayal
"""

import pickle
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your data
file_path = r'C:\Users\sayal\Downloads\Banana-Commodity database (merge).csv'
data = pd.read_csv(file_path)

# Drop rows where target variables are NaN
data = data.dropna(subset=['Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)'])

# Extract features and target variables
X = data[['District Name', 'Market Name', 'Commodity', 'Variety', 'Grade', 'temp', 'precip']]
y = data[['Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)']]

# Encode categorical variables
label_encoders = {}
for column in ['District Name', 'Market Name', 'Commodity', 'Variety', 'Grade']:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Train the model
model = DecisionTreeRegressor()
model.fit(X, y)

# Save the model
with open(r'C:\Users\sayal\internship\Farmoid\model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the label encoders
with open(r'C:\Users\sayal\internship\Farmoid\label_encoders.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoders, encoder_file)
