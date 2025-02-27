import streamlit as st
import pandas as pd
import joblib
from src.model_inference import predict

# Load encoder
encoder = joblib.load("models/retail_encoders.pkl")

st.markdown("""
    <style>
    /* Main background with black shiny gradient */
    .stApp {
        background: linear-gradient(to right, #0f0f0f, #292929, #3d3d3d);
    }

    /* Sidebar with black shiny gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(to right, #0f0f0f, #292929, #3d3d3d);
    }

    /* Heading and Labels (white text) */
    h1, h2, h3, h4, h5, h6, label {
        color: white !important;
    }

    /* Sidebar labels (white text) */
    [data-testid="stSidebar"] label {
        color: white !important;
    }

    /* Input fields - Keep text BLACK */
    input, textarea {
        color: black !important;
        background-color: white !important;
        border: 1px solid white !important;
    }

    /* Dropdown and Selectbox - Keep default */
    div[data-baseweb="select"] > div {
        color: black !important;
    }

    
            
   /* Button Styling */
    div.stButton > button {
        background: linear-gradient(to right, #0f0f0f, #292929, #3d3d3d);
        color: white !important;
        border-radius: 8px;
        border: 2px solid white !important;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        width: 100%;
        transition: all 0.3s ease-in-out;
    }

    /* Button Hover Effect */
    div.stButton > button:hover {
        background: linear-gradient(to right, #ffffff, #f0f0f0); /* Subtle white gradient */
        color: black !important;
        border: 2px solid black !important;
    }

  div[data-testid="stSlider"] * {
        color: white !important;
    }
    
    h3 {
        text-align: center;
        font-size: 30px;
    }

    <div style="
        background-color: #4CAF50; 
        color: white; 
        padding: 10px; 
        border-radius: 5px; 
        font-size: 18px; 
        text-align: center;">
        💰 <b>Predicted Sales: ₹{prediction[0]:,.2f}</b>
    </div>
                 
    </style>
""", unsafe_allow_html=True)




st.markdown("<h3>🛒 Retailytics Pro - Big Mart Sales Prediction 📊</h3>", unsafe_allow_html=True)

st.sidebar.header("Input Features")

# Taking inputs
Item_Identifier = st.text_input("Item Identifier")
Item_Weight = st.sidebar.slider("Item Weight", 0.1, 25.0, 5.0)
Item_Fat_Content = st.sidebar.selectbox("Item Fat Content", ["Low Fat", "Regular"])
Item_Visibility = st.sidebar.slider("Item Visibility", 0.0, 0.9, 0.05)
Item_Type = st.sidebar.selectbox("Item Type", [
    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
    "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast",
    "Health and Hygiene", "Hard Drinks", "Canned", "Breads", "Starchy Foods",
    "Others", "Seafood"
])
Item_MRP = st.number_input("Item MRP", min_value=0.0, max_value=500.0, value=100.0)
Outlet_Identifier = st.text_input("Outlet Identifier")
# Outlet_Establishment_Year = st.date_input("Outlet Establishment Year").year
Outlet_Establishment_Year = st.number_input(
    "Outlet Establishment Year", min_value=1985, max_value=2025, value=2000, step=1
)
Outlet_Size = st.sidebar.selectbox("Outlet Size", ["Small", "Medium", "High"])
Outlet_Location_Type = st.sidebar.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
Outlet_Type = st.selectbox("Outlet Type", ["Supermarket Type1", "Supermarket Type2", "Grocery Store", "Supermarket Type3"])

# Convert input into DataFrame
input_data = pd.DataFrame([[
    Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility,
    Item_Type, Item_MRP, Outlet_Identifier, Outlet_Establishment_Year,
    Outlet_Size, Outlet_Location_Type, Outlet_Type
]], columns=[
    "Item_Identifier", "Item_Weight", "Item_Fat_Content", "Item_Visibility",
    "Item_Type", "Item_MRP", "Outlet_Identifier", "Outlet_Establishment_Year",
    "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"
])

# Predict
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Predict Sales", use_container_width=True):
        prediction = predict(input_data)
        st.warning(f"💰 Predicted Sales: ₹ {prediction[0]:,.2f}")