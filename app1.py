import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained regression model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Salary Prediction using ANN')

# User inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 65, 30)
balance = st.number_input('Balance', min_value=0.0, value=10000.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Encode Gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])

# Combine all into a DataFrame
input_data = pd.DataFrame([[
    credit_score,
    gender_encoded,
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active_member,
    estimated_salary
] + list(geo_encoded[0])], 
columns=[
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
] + list(geo_cols))

# Scale the input data
input_scaled = scaler.transform(input_data)

# Predict salary
prediction = model.predict(input_scaled)
predicted_salary = prediction[0][0]

# Display result
st.subheader(f"Predicted Estimated Salary: RM {predicted_salary:,.2f}")