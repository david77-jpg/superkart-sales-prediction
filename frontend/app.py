import streamlit as st
import pandas as pd
import requests

# Pointing directly to the Flask backend container's host and port on the Docker network
BACKEND_URL = "http://backend:7860"

st.title("SuperKart System")
st.write("Enter store details below to predict the total sales.")

Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])

# Create JSON payload
store_data = {
    "Store_Type": Store_Type,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Size": Store_Size
}

# Single Prediction
if st.button("Predict", type='primary'):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=store_data)
        if response.status_code == 200:
            result = response.json()
            predicted_sales = result["Sales"]
            st.success(f"Predicted Product Store Sales Total: ₹{predicted_sales:.2f}")
        else:
            st.error(f"API error: Status code {response.status_code}")
    except Exception as e:
        st.error(f"Unable to connect to the prediction API: {e}")

# Batch Prediction
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict for Batch", type='primary'):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": uploaded_file}
            )
            if response.status_code == 200:
                results = response.json()
                st.success("Predictions completed successfully!")
                df = pd.DataFrame(list(results.items()), columns=['Index', 'Predicted_Sales'])
                st.dataframe(df, use_container_width=True)
            else:
                st.error("Unable to get batch predictions from the backend.")
        except Exception as e:
            st.error(f"Unable to connect to backend: {e}")
