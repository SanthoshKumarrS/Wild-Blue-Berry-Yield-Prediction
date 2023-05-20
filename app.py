import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prediction import get_prediction


# Load the model
model = joblib.load(r'C:/Santhosh/Wild-Blueberry-Yield-prediction-Project/Model/gb_yield_prediction_model.joblib')

st.set_page_config(page_title="Wild Blueberry Yield Prediction",
                   page_icon="🫐", layout="wide")


features = ['fruitset',
            'seeds',
            'fruitmass',
            'osmia',
            'rainingdays',
            'averagerainingdays',
            'averageoflowertrange',
            'minoflowertrange',
            'minofuppertrange',
            'andrena']


st.markdown("<h1 style='text-align: center;'>Wild-Blueberry-Yield-prediction-App 🫐</h1>", unsafe_allow_html=True)


def main():
    with st.form('prediction_form'):
        st.subheader("Enter the input for following features:")
        
        fruitset = st.number_input("Fruit Set:", value=0.0, min_value=0.0, max_value=0.999999, format="%.2f")
        seeds = st.number_input("seeds:", value=0.0, min_value=0.0, max_value=0.999999, format="%.2f")
        fruitmass = st.number_input("fruitmass:", value=0.0, min_value=0.0, max_value=0.999999, format="%.2f")
        osmia = st.selectbox("Osmia:", (0.25, 0.38, 0.5, 0.63, 0.75, 0.058, 0.101, 0, 0.033, 0.021, 0.585, 0.117))
        rainingdays = st.selectbox("rainingdays:", (16.0, 1.0, 24.0, 34.0, 3.77))
        averagerainingdays = st.selectbox("averagerainingdays:", (0.26, 0.1, 0.39, 0.56, 0.06))
        averageoflowertrange = st.selectbox("averageoflowertrange:", (50.8, 55.9, 45.8, 41.2, 45.3))
        minoflowertrange = st.selectbox("minoflowertrange:", (30.0, 33.0, 27.0, 24.3, 28.0))
        minofuppertrange = st.selectbox("minofuppertrange:", (52.0, 57.2, 46.8, 42.1, 39.0))
        andrena = st.number_input("andrena:", value=0.0, min_value=0.0, max_value=0.999999, format="%.2f")

        submit = st.form_submit_button("Predict")

    if submit:
        data = np.array([fruitset, seeds, fruitmass, osmia, rainingdays, averagerainingdays, averageoflowertrange,
                         minoflowertrange, minofuppertrange, andrena]).reshape(1, -1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted yield is: {pred[0]}")

if __name__ == '__main__':
    main()

