import pandas as pd
import streamlit as st
import pickle
import numpy as np

# =================================
# Load trained ML pipeline & dataset
# =================================
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")
st.title("💻 Laptop Price Prediction using Quality Analysis")

st.markdown(
    """
    This application analyzes **laptop quality parameters**
    (RAM, Storage, CPU, GPU, Display, etc.)
    and predicts the **expected market price**
    using historical dataset patterns.
    """
)

# =============================
# User Inputs (Laptop Quality)
# =============================
company = st.selectbox("Brand", df['Company'].unique())
type_name = st.selectbox("Laptop Type", df['TypeName'].unique())
ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64, 128, 256])
weight = st.number_input(
    "Weight of Laptop (kg)",
    min_value=0.8,
    max_value=4.0,
    step=0.1
)

touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS Display", ["No", "Yes"])

screen_size = st.slider(
    "Screen Size (inches)",
    10.0, 18.0, 15.6
)

resolution = st.selectbox(
    "Screen Resolution",
    [
        "1920x1080", "1366x768", "1600x900",
        "3840x2160", "3200x1800",
        "2880x1800", "2560x1600",
        "2560x1440", "2304x1440"
    ]
)

cpu = st.selectbox("CPU Brand", df['Cpu brand'].unique())
hdd = st.selectbox("HDD Storage (GB)", [0, 128, 256, 512, '1TB', '2TB', '4TB'])
ssd = st.selectbox("SSD Storage (GB)", [0, 128, 256, '1TB', '2TB'])
gpu = st.selectbox("GPU Brand", df['Gpu brand'].unique())
os = st.selectbox("Operating System", df['os'].unique())

# =============================
# Prediction & Analysis
# =============================
if st.button("🔍 Analyze Laptop & Predict Price"):

    # Convert categorical values
    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

    # (You can keep query creation if needed later)
    query = pd.DataFrame(
        [[
            company, type_name, ram, weight,
            touchscreen_val, ips_val, ppi,
            cpu, hdd, ssd,
            gpu, os
        ]],
        columns=[
            'Company','TypeName','Ram','Weight',
            'Touchscreen','Ips','ppi',
            'Cpu brand','HDD','SSD',
            'Gpu brand','os'
        ]
    )

    # =============================
    # Dataset-based Final Prediction
    # =============================
    similar_laptops = df[
        (df['Ram'] == ram) &
        (df['SSD'] == ssd) &
        (df['Gpu brand'] == gpu)
    ]

    if len(similar_laptops) > 0:
        avg_price = int(similar_laptops['Price'].mean())
        min_price = int(similar_laptops['Price'].min())
        max_price = int(similar_laptops['Price'].max())

        st.success(f"💰 Estimated Laptop Price (from Dataset): ₹ {avg_price}")

        st.info(
            f"Price range for this configuration is "
            f"₹{min_price} – ₹{max_price}"
        )
    else:
        st.warning(
            "No exact match found in dataset. Price estimation may be inaccurate."
        )

