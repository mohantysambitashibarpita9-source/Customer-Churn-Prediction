
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("customer_churn_model.pkl")

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction System")

st.write(
    "Enter customer information to predict "
    "whether the customer is likely to leave."
)

st.divider()

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (Months)", 1, 100, 12
    )

    monthly_charges = st.number_input(
        "Monthly Charges (₹)", 0.0, 10000.0, 999.0
    )

    support_calls = st.number_input(
        "Support Calls", 0, 20, 2
    )

    late_payments = st.number_input(
        "Late Payments", 0, 20, 1
    )

with col2:
    contract = st.selectbox(
        "Contract Type",
        ["Monthly", "Annual", "Two-year"]
    )

    usage_hours = st.number_input(
        "Usage Hours Per Month", 0.0, 200.0, 30.0
    )

    add_on_services = st.number_input(
        "Add-on Services", 0, 10, 2
    )

contract_mapping = {
    "Monthly": 0,
    "Annual": 1,
    "Two-year": 2
}

if st.button("Predict Customer Churn"):

    features = pd.DataFrame([{
        "Tenure_Months": tenure,
        "Monthly_Charges": monthly_charges,
        "Support_Calls": support_calls,
        "Late_Payments": late_payments,
        "Contract_Type": contract_mapping[contract],
        "Usage_Hours_Per_Month": usage_hours,
        "Add_On_Services": add_on_services
    }])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    st.divider()

    if prediction == 1:

        st.error("⚠️ Customer is likely to leave!")

        st.write(
            f"Estimated churn probability: {probability:.2%}"
        )

        st.info(
            "Suggested action: Contact the customer "
            "and review their service experience."
        )

    else:

        st.success("✅ Customer is likely to stay!")

        st.write(
            f"Estimated retention probability: "
            f"{(1-probability):.2%}"
        )
