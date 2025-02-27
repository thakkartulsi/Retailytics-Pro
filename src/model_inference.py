import pandas as pd
import joblib
import os
import sys

# Ensure 'src' is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import preprocess_data

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/retail_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "../models/retail_encoders.pkl")

def load_model():
    return joblib.load(MODEL_PATH)

def load_encoder():
    return joblib.load(ENCODER_PATH)

def predict(input_data):
    """Preprocess input data and make predictions using the trained model."""
    # Load encoder and preprocess data
    encoder = load_encoder()
    input_data = preprocess_data(input_data, train=False, encoder_path=ENCODER_PATH)
    
    # Load model and predict
    model = load_model()
    return model.predict(input_data)
