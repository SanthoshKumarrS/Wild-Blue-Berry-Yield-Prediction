import joblib
import numpy as np

def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)