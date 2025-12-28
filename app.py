import streamlit as st
import numpy as np
import joblib

st.title("🌸 Iris Flower Classifier")
st.write("Predict Setosa, Versicolor, or Virginica based on flower measurements.")

# Load model
model = joblib.load("iris_model.joblib")

# Input sliders
sepal_length = st.slider("Sepal length (cm)", 4.3, 7.9, 5.8)
sepal_width  = st.slider("Sepal width (cm)", 2.0, 4.4, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 6.9, 4.3)
petal_width  = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

if st.button("Predict"):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(X)[0]
    names = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Prediction: {names[pred]}")