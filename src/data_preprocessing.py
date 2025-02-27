import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, train=True, encoder_path="models/retail_encoders.pkl"):

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    if train:
        # Fit and save encoders
        encoders = {}
        for col in categorical_cols:
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col])
            encoders[col] = enc
        joblib.dump(encoders, encoder_path)
    else:
        # Load pre-trained encoders
        encoders = joblib.load(encoder_path)
        for col in categorical_cols:
            if col in df.columns: # Ensure column exists in input data
                df[col] = encoders[col].transform(df[col])
    return df


if __name__ == '__main__':
    df = load_data('data/train.csv')
    df = preprocess_data(df, train=True)
    df.to_csv("data/processed_train.csv", index=False)

