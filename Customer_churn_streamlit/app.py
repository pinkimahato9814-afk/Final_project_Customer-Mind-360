import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import home
from models import predict_churn


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


def knn_prediction_page():
    features, target, model, Y_pred, cr, cm = predict_churn()

    st.title("KNN Customer Churn Prediction")
    st.write("Enter customer details from the sidebar and click the prediction button.")

    st.sidebar.header("Enter Customer Details")

    Customer_Service_Calls = st.sidebar.slider(
        "Customer Service Calls",
        min_value=0,
        max_value=20,
        value=4,
        step=1
    )

    Cart_Abandonment_Rate = st.sidebar.slider(
        "Cart Abandonment Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    )

    Pages_Per_Session = st.sidebar.slider(
        "Pages Per Session",
        min_value=0.0,
        max_value=50.0,
        value=3.0,
        step=0.1
    )

    Session_Duration_Avg = st.sidebar.slider(
        "Average Session Duration",
        min_value=0.0,
        max_value=60.0,
        value=5.5,
        step=0.1
    )

    Email_Open_Rate = st.sidebar.slider(
        "Email Open Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
    )

    Mobile_App_Usage = st.sidebar.slider(
        "Mobile App Usage",
        min_value=0,
        max_value=1,
        value=1,
        step=1
    )

    Login_Frequency = st.sidebar.slider(
        "Login Frequency",
        min_value=0,
        max_value=100,
        value=2,
        step=1
    )

    Total_Purchases = st.sidebar.slider(
        "Total Purchases",
        min_value=0,
        max_value=200,
        value=5,
        step=1
    )

    Days_Since_Last_Purchase = st.sidebar.slider(
        "Days Since Last Purchase",
        min_value=0,
        max_value=365,
        value=60,
        step=1
    )

    Signup_Quarter = st.sidebar.selectbox(
        "Signup Quarter",
        ["Q1", "Q2", "Q3", "Q4"]
    )

    X_new = pd.DataFrame({
        "Customer_Service_Calls": [Customer_Service_Calls],
        "Cart_Abandonment_Rate": [Cart_Abandonment_Rate],
        "Pages_Per_Session": [Pages_Per_Session],
        "Session_Duration_Avg": [Session_Duration_Avg],
        "Email_Open_Rate": [Email_Open_Rate],
        "Mobile_App_Usage": [Mobile_App_Usage],
        "Login_Frequency": [Login_Frequency],
        "Total_Purchases": [Total_Purchases],
        "Days_Since_Last_Purchase": [Days_Since_Last_Purchase],
        "Signup_Quarter": [Signup_Quarter]
    })

    st.markdown(
        """
        <style>
        .table-container {
            width: 100%;
            overflow-x: auto;
            margin-bottom: 20px;
        }

        .custom-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 15px;
        }

        .custom-table th {
            background-color: #fff3cd;
            color: #000000;
            font-weight: bold;
            text-align: center;
            padding: 12px;
            border: 1px solid #dddddd;
            white-space: nowrap;
        }

        .custom-table td {
            text-align: center;
            padding: 10px;
            border: 1px solid #dddddd;
            white-space: nowrap;
        }

        .custom-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        .custom-table tr:hover {
            background-color: #f1f1f1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Customer Input")

    customer_input_table = X_new.to_html(
        index=False,
        classes="custom-table"
    )

    st.markdown(
        f"""
        <div class="table-container">
            {customer_input_table}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Prediction Result")

    if st.button("Predict Customer Churn"):
        prediction = model.predict(X_new)

        if prediction[0] == 1:
            st.error("Customer is likely to churn.")
        else:
            st.success("Customer is not likely to churn.")

    st.markdown("---")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        xticklabels=["Predicted Not Churned", "Predicted Churned"],
        yticklabels=["Actual Not Churned", "Actual Churned"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.subheader("Classification Report")

    cr_df = pd.DataFrame(cr).transpose()
    cr_df = cr_df.round(2)

    cr_df = cr_df.reset_index()
    cr_df = cr_df.rename(columns={"index": "Class / Metric"})

    classification_report_table = cr_df.to_html(
        index=False,
        classes="custom-table"
    )

    st.markdown(
        f"""
        <div class="table-container">
            {classification_report_table}
        </div>
        """,
        unsafe_allow_html=True
    )


st.sidebar.title("Customer Churn Dashboard")

page = st.sidebar.radio(
    "Go to",
    ["Home", "KNN Churn Prediction"]
)

if page == "Home":
    home.app()

elif page == "KNN Churn Prediction":
    knn_prediction_page()