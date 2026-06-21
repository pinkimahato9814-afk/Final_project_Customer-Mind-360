# import streamlit as st


# def app():
#     st.title("Customer Churn Prediction Dashboard")

#     st.markdown(
#         """
# <div style="background: linear-gradient(135deg, #eef7ff, #f8fbff); padding: 24px; border-radius: 16px; border-left: 7px solid #1f77b4; box-shadow: 0px 4px 12px rgba(0,0,0,0.08); margin-top: 15px; margin-bottom: 25px;">

# <h3 style="color: #1f4e79; margin-bottom: 12px; font-size: 26px;">
# Smart Ecommerce Customer Churn Prediction
# </h3>

# <p style="font-size: 17px; line-height: 1.7; color: #333333; margin-bottom: 0;">
# This Streamlit dashboard is designed to predict whether an 
# <b style="color:#d63384;">ecommerce customer will churn or stay</b>. 
# The project uses a 
# <b style="color:#1f77b4;">K-Nearest Neighbors (KNN) Classification Model</b> 
# to analyze customer behavior, purchase activity, engagement level, 
# and service interaction patterns.
# <br><br>
# The main goal of this dashboard is to help businesses identify 
# <b>high-risk customers</b> early and take better decisions for 
# customer retention.
# </p>

# </div>
#         """,
#         unsafe_allow_html=True
#     )

#     st.markdown("---")

#     st.header("Project Workflow")

#     st.write("""
#     The project was completed step by step:

#     1. **Data Cleaning, Preprocessing and Analysis**
#        - Loaded the ecommerce customer churn dataset.
#        - Checked missing values, duplicate values, and data types.
#        - Cleaned the dataset for further analysis.

#     2. **Customer Segmentation**
#        - Analyzed customer behavior using purchase, engagement, and activity features.
#        - Helped understand different customer groups.

#     3. **Hypothesis Testing**
#        - Used two-sample t-test for numerical features.
#        - Used chi-square test for categorical features.
#        - Selected important features related to customer churn.

#     4. **KNN Model Training**
#        - Used selected features to train a KNN classification model.
#        - Used elbow method to choose the K value.
#        - Evaluated the model using accuracy, precision, recall, F1-score, and confusion matrix.
#     """)

#     st.markdown("---")

#     st.header("Selected Features Used in KNN Model")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Numerical Features")
#         st.write("""
#         - Customer_Service_Calls
#         - Cart_Abandonment_Rate
#         - Pages_Per_Session
#         - Session_Duration_Avg
#         - Email_Open_Rate
#         - Mobile_App_Usage
#         - Login_Frequency
#         - Total_Purchases
#         - Days_Since_Last_Purchase
#         """)

#     with col2:
#         st.subheader("Categorical Feature")
#         st.write("""
#         - Signup_Quarter
#         """)

#         st.subheader("Target Variable")
#         st.write("""
#         The target variable is **Churned**.

#         - **0 = Not Churned**
#         - **1 = Churned**
#         """)

#     st.markdown("---")

#     st.info("Use the KNN Churn Prediction page from the sidebar to train the model and predict new customers.")





import streamlit as st


def app():
    st.title("Customer Churn Prediction Dashboard")

    st.markdown(
        """
<div style="background: linear-gradient(135deg, #eef7ff, #f8fbff); padding: 24px; border-radius: 16px; border-left: 7px solid #1f77b4; box-shadow: 0px 4px 12px rgba(0,0,0,0.08); margin-top: 15px; margin-bottom: 25px;">

<h3 style="color: #1f4e79; margin-bottom: 12px; font-size: 26px;">
Smart Ecommerce Customer Churn Prediction
</h3>

<p style="font-size: 17px; line-height: 1.7; color: #333333; margin-bottom: 0;">
This Streamlit dashboard is designed to predict whether an 
<b style="color:#d63384;">ecommerce customer will churn or stay</b>. 
The project uses a 
<b style="color:#1f77b4;">K-Nearest Neighbors (KNN) Classification Model</b> 
to analyze customer behavior, purchase activity, engagement level, 
and service interaction patterns.
<br><br>
The main goal of this dashboard is to help businesses identify 
<b>high-risk customers</b> early and take better decisions for 
customer retention.
</p>

</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
<div style="background-color:#ffffff; padding:22px; border-radius:14px; box-shadow:0px 4px 12px rgba(0,0,0,0.08); margin-top:20px; margin-bottom:25px; border-left:6px solid #28a745;">

<h3 style="color:#1f4e79; margin-bottom:15px;">
Final Code Analysis Summary
</h3>

<ul style="font-size:16px; line-height:1.8; color:#333333; margin-bottom:0;">
    <li><b>Clean Project Structure:</b> Separate files are used for app navigation, home page, model training, and requirements.</li>
    <li><b>User-Friendly Dashboard:</b> Sidebar navigation and sliders make the app simple and easy to use.</li>
    <li><b>Proper ML Pipeline:</b> Numerical features are scaled and categorical features are encoded correctly.</li>
    <li><b>KNN Model:</b> The dashboard uses K-Nearest Neighbors classification for customer churn prediction.</li>
    <li><b>Prediction Output:</b> The app clearly shows whether a customer is likely to churn or not.</li>
    <li><b>Model Evaluation:</b> Confusion matrix and classification report help explain model performance.</li>
    <li><b>Future Improvement:</b> Model caching can be added to make the dashboard faster during reload.</li>
</ul>

</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.header("Project Workflow")

    st.write("""
    The project was completed step by step:

    1. **Data Cleaning, Preprocessing and Analysis**
       - Loaded the ecommerce customer churn dataset.
       - Checked missing values, duplicate values, and data types.
       - Cleaned the dataset for further analysis.

    2. **Customer Segmentation**
       - Analyzed customer behavior using purchase, engagement, and activity features.
       - Helped understand different customer groups.

    3. **Hypothesis Testing**
       - Used two-sample t-test for numerical features.
       - Used chi-square test for categorical features.
       - Selected important features related to customer churn.

    4. **KNN Model Training**
       - Used selected features to train a KNN classification model.
       - Used elbow method to choose the K value.
       - Evaluated the model using accuracy, precision, recall, F1-score, and confusion matrix.
    """)

    st.markdown("---")

    st.header("Selected Features Used in KNN Model")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Numerical Features")
        st.write("""
        - Customer_Service_Calls
        - Cart_Abandonment_Rate
        - Pages_Per_Session
        - Session_Duration_Avg
        - Email_Open_Rate
        - Mobile_App_Usage
        - Login_Frequency
        - Total_Purchases
        - Days_Since_Last_Purchase
        """)

    with col2:
        st.subheader("Categorical Feature")
        st.write("""
        - Signup_Quarter
        """)

        st.subheader("Target Variable")
        st.write("""
        The target variable is **Churned**.

        - **0 = Not Churned**
        - **1 = Churned**
        """)

    st.markdown("---")

    st.info("Use the KNN Churn Prediction page from the sidebar to train the model and predict new customers.")



st.markdown("---")

st.markdown(
        """
<div style="background: linear-gradient(135deg, #fffaf0, #ffffff); padding:22px; border-radius:14px; border-left:6px solid #ff9800; box-shadow:0px 4px 12px rgba(0,0,0,0.08); margin-top:25px; margin-bottom:25px;">

<h3 style="color:#b45f06; margin-bottom:15px;">
Final Output Observation Report
</h3>

<ul style="font-size:16px; line-height:1.8; color:#333333; margin-bottom:0;">
    <li><b>Cleaned Dataset:</b> The final dataset is ready for analysis and model training.</li>
    <li><b>Churn Pattern:</b> A noticeable number of customers are leaving, so churn prediction is useful.</li>
    <li><b>Customer Engagement:</b> Low email open rate and low login frequency may increase churn risk.</li>
    <li><b>Service Calls:</b> Customers with more service calls are more likely to churn.</li>
    <li><b>Purchase Behavior:</b> Total purchases and last purchase gap are important churn indicators.</li>
    <li><b>Segmentation:</b> Customer groups help understand different behavior patterns.</li>
    <li><b>Model Result:</b> KNN classification is used to predict whether a customer will churn or not.</li>
</ul>

</div>
        """,
        unsafe_allow_html=True
    )

st.info("Use the KNN Churn Prediction page from the sidebar to train the model and predict new customers.")