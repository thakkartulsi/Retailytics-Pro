import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_model():
    df = pd.read_csv("data/processed_train.csv")

    X = df.drop(columns=["Item_Outlet_Sales"])
    y = df["Item_Outlet_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")

    joblib.dump(model, "models/retail_model.pkl")

if __name__ == "__main__":
    train_model()
