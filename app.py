import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Classifying Iris Flowers')
st.markdown('Toy model to classify iris flowers into (setosa, versicolor, virginica) based on their sepal/petal and length/width.')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_length = st.slider('Sepal Length (cm)', 1.0, 8.0, 5.0)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.4, 3.0)

with col2:
    st.text("Petal characteristics")
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

st.text('')
if st.button("Predict type of Iris"):
    result = predict(
        np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
    st.success(f'Predicted Iris Species: {result[0]}')