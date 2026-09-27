import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Car Insurance Premium",
    page_icon="🚗",
    layout="wide"
)


# LOAD MODEL
model=joblib.load("car_insurance_premium_model.pkl")


# CSS
st.markdown("""
<style>
.main-title{
    text-align:center;
    color:#2c3e50;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#5d6d7e;
    font-size:18px;
    margin-bottom:30px;
}

.section-title{
    color:#2c3e50;
    font-size:28px;
    font-weight:bold;
}

.card{
    padding:25px;
    border-radius:12px;
    background-color:#f5f7f9;
    margin-bottom:20px;
}

.prediction{
    text-align:center;
    font-size:35px;
    font-weight:bold;
    color:#1f618d;
}

.small-text{
    text-align:center;
    color:#566573;
    font-size:16px;
}
</style>
""",unsafe_allow_html=True)


# SIDEBAR
st.sidebar.title("🚗 Car Insurance")
st.sidebar.write("Navigate through the application.")

page=st.sidebar.selectbox("Select a Section",["Home","Data Insights","Premium Predictor",])

# HOME PAGE
if page=="Home":
    st.markdown('<div class="main-title">🚗 Car Insurance Premium Predictor</div>',unsafe_allow_html=True)

    st.markdown('<div class="subtitle">Estimate the insurance premium using driver and vehicle information.</div>',unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7",use_container_width=True)

    st.markdown("---")

    col1,col2,col3=st.columns(3)
    with col1:
        st.markdown(
            '<div class="card">'
            '<h3>👤 Driver Information</h3>'
            '<p>Use of age, driving experience and accident history.</p>'
            '</div>',unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="card">'
            '<h3>🚘 Vehicle Information</h3>'
            '<p>Use of mileage, manufacturing year and car age.</p>'
            '</div>',unsafe_allow_html=True)

    with col3:
        st.markdown(
            '<div class="card">'
            '<h3>🤖 ML Prediction</h3>'
            '<p>Linear Regression estimates the insurance premium.</p>'
            '</div>',unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("How It Works")

    st.write("""
    1. Enter the driver's details.
    2. Enter the vehicle details.
    3. Click **Predict Premium**.
    4. The trained Linear Regression model estimates the insurance premium.""")


# DATA INSIGHTS PAGE
elif page=="Data Insights":
    st.markdown(
        '<div class="main-title">📊 Data Insights</div>',unsafe_allow_html=True)

    st.markdown(
        '<div class="subtitle">'
        'Overview of the dataset used to train the prediction model.'
        '</div>',unsafe_allow_html=True)


    df=pd.read_csv("car_insurance_premium.csv")
    col1,col2,col3=st.columns(3)

    with col1:
        st.metric("Total Records",len(df))

    with col2:
        st.metric("Total Features",len(df.columns)-1)

    with col3:
        st.metric("Target","Insurance Premium")

    st.markdown("---")
    st.subheader("Dataset Preview")

    st.dataframe(df.head(10),use_container_width=True)

    st.subheader("Insurance Premium Distribution")

    st.bar_chart(df["Insurance Premium ($)"].value_counts().sort_index())



# PREMIUM PREDICTOR PAGE
elif page=="Premium Predictor":

    st.markdown(
        '<div class="main-title">💰 Premium Predictor</div>',unsafe_allow_html=True)

    st.markdown(
        '<div class="subtitle">'
        'Enter the driver and vehicle details below.'
        '</div>',unsafe_allow_html=True)

    # DRIVER INFORMATION
    st.markdown(
        '<div class="section-title">👤 Driver Information</div>',unsafe_allow_html=True)

    st.markdown("---")

    col1,col2,col3=st.columns(3)
    with col1:
        driver_age=st.number_input("Driver Age", min_value=18, max_value=100, value=30)

    with col2:
        driver_experience=st.number_input("Driver Experience", min_value=0, max_value=80, value=5)

    with col3:
        previous_accidents=st.number_input("Previous Accidents", min_value=0, max_value=20, value=0)

    st.markdown("")

    # VEHICLE INFORMATION
    st.markdown(
        '<div class="section-title">🚘 Vehicle Information</div>',
        unsafe_allow_html=True)

    st.markdown("---")

    col1,col2,col3=st.columns(3)
    with col1:
        annual_mileage=st.number_input("Annual Mileage (x1000 km)", min_value=0.0, max_value=200.0, value=15.0)

    with col2:
        car_manufacturing_year=st.number_input("Manufacturing Year", min_value=1950, max_value=2026, value=2020)

    with col3:
        car_age=st.number_input("Car Age", min_value=0, max_value=100, value=5)

    st.markdown("")
    st.markdown("")

    # PREDICTION BUTTON
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        predict=st.button("🚗 PREDICT PREMIUM",use_container_width=True)

    if predict:
        input_data=pd.DataFrame([[
            driver_age,
            driver_experience,
            previous_accidents,
            annual_mileage,
            car_manufacturing_year,
            car_age
        ]],
        columns=[
            "Driver Age",
            "Driver Experience",
            "Previous Accidents",
            "Annual Mileage (x1000 km)",
            "Car Manufacturing Year",
            "Car Age"
        ])

        prediction=model.predict(input_data)

        premium=prediction[0]

        st.markdown("---")

        st.markdown(
            '<div class="section-title" style="text-align:center;">'
            'Predicted Premium'
            '</div>',unsafe_allow_html=True)

        st.markdown(f'<div class="prediction">${premium:.2f}</div>',unsafe_allow_html=True)

        st.markdown(
            '<div class="small-text">'
            'Estimated insurance premium based on the entered details.'
            '</div>',unsafe_allow_html=True)

