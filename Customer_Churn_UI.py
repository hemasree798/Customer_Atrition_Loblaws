import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the saved scaler and model
scaler_file = "min_max_scaler.pkl"
model_file = "random_forest_model.pkl"

scaler = joblib.load(scaler_file)
model = joblib.load(model_file)

# Define the columns to scale
columns_to_scale = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
                    'SatisfactionScore', 'NumberOfAddress']

# Function to predict churn
def predict_churn(data):
    data_scaled = scaler.transform(data[columns_to_scale])
    data[columns_to_scale] = data_scaled
    prediction = model.predict(data)
    return prediction
def main():
    # Streamlit UI
    st.title("Customer Churn Prediction")

    # Input form for user to enter customer data
    st.header("Enter Customer Details")

    customer_id = st.number_input("Customer ID", value=50001)
    tenure = st.number_input("Tenure", value=1)
    city_tier = st.selectbox("City Tier", options=[1, 2, 3])
    warehouse_to_home = st.number_input("Warehouse to Home Distance", value=4.0)
    hour_spend_on_app = st.number_input("Hours Spent on App", value=3.0)
    preferred_payment_mode = st.selectbox("Preferred Payment Mode", options=["Debit Card","Credit Card","E Wallet","UPI","Cash On Delivery"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    number_of_device_registered = st.number_input("Number of Devices Registered", value=3.0)
    prefered_order_cat = st.selectbox("Preferred Order Category", options=["Laptop & Accessory", "Mobile", "Fashion","Grocery","Others"])
    satisfaction_score = st.number_input("Satisfaction Score", value=2)
    marital_status = st.selectbox("Marital Status", options=["Single", "Married"])
    number_of_address = st.number_input("Number of Address", value=9)
    complaint_status = st.selectbox("Complaint Status", options=[0, 1])
    preferred_login_device = st.selectbox("Preferred Login Device", options=["Mobile Phone", "Phone"])
    order_amount_hike_from_last_year = st.number_input("Order Amount Hike From Last Year", value=10.0)
    coupon_used = st.number_input("Coupon Used", value=0)
    order_count = st.number_input("Order Count", value=5)
    day_since_last_order = st.number_input("Day Since Last Order", value=30)
    cashback_amount = st.number_input("Cashback Amount", value=50.0)


    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'CustomerID': [customer_id],
        'Tenure': [tenure],
        'CityTier': [city_tier],
        'WarehouseToHome': [warehouse_to_home],
        'HourSpendOnApp': [hour_spend_on_app],
        'Gender': [gender],
        'NumberOfDeviceRegistered': [number_of_device_registered],
        'SatisfactionScore': [satisfaction_score],
        'PreferredOrderCat': [prefered_order_cat],
        'MaritalStatus': [marital_status],
        'NumberOfAddress': [number_of_address],
        'ComplaintStatus': [complaint_status],
        'PreferredLoginDevice': [preferred_login_device],
        'PreferredPaymentMode': [preferred_payment_mode],
        'OrderAmountHikeFromlastYear': [order_amount_hike_from_last_year],
        'CouponUsed': [coupon_used],
        'OrderCount': [order_count],
        'DaySinceLastOrder': [day_since_last_order],
        'CashbackAmount': [cashback_amount]
    })

    # Predict churn
    if st.button("Predict Churn"):
        prediction = predict_churn(input_data)
        if prediction == 0:
            st.success("Customer is not likely to churn.")
        else:
            st.warning("Customer is likely to churn.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
