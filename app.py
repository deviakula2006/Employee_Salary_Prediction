import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
MODEL_PATH = "income_model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Employee Salary Prediction",
    layout="wide"
)

# --------------------------------------------------
# Title & Description
# --------------------------------------------------
st.title("Employee Salary Prediction Application")

st.markdown(
    """
    This application predicts whether an employee’s annual income is 
    **greater than 50K** or **less than or equal to 50K** based on 
    demographic and work-related information.

    The model was trained using the Adult Income Dataset and deployed 
    as a web application using Streamlit.
    """
)

st.markdown("---")

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Employee Details")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)

education = st.sidebar.selectbox(
    "Education Level",
    ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"]
)

occupation = st.sidebar.selectbox(
    "Occupation",
    [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces"
    ]
)

hours_per_week = st.sidebar.number_input(
    "Hours Worked Per Week",
    min_value=1,
    max_value=100,
    value=40,
    step=1
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

# --------------------------------------------------
# Encoding Maps (must match training logic)
# --------------------------------------------------
education_map = {
    "HS-grad": 9,
    "Some-college": 10,
    "Bachelors": 13,
    "Masters": 14,
    "Doctorate": 16
}

occupation_map = {
    "Tech-support": 1,
    "Craft-repair": 2,
    "Other-service": 3,
    "Sales": 4,
    "Exec-managerial": 5,
    "Prof-specialty": 6,
    "Handlers-cleaners": 7,
    "Machine-op-inspct": 8,
    "Adm-clerical": 9,
    "Farming-fishing": 10,
    "Transport-moving": 11,
    "Priv-house-serv": 12,
    "Protective-serv": 13,
    "Armed-Forces": 14
}

gender_map = {
    "Male": 1,
    "Female": 0
}

# --------------------------------------------------
# Build input in training feature format
# --------------------------------------------------
input_df = pd.DataFrame(columns=model.feature_names_in_)
input_df.loc[0] = 0

input_df.loc[0, "age"] = age
input_df.loc[0, "educational-num"] = education_map[education]
input_df.loc[0, "occupation"] = occupation_map[occupation]
input_df.loc[0, "hours-per-week"] = hours_per_week
input_df.loc[0, "gender"] = gender_map[gender]

# --------------------------------------------------
# Main Layout
# --------------------------------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Input Summary")

    summary_df = pd.DataFrame({
        "Feature": ["Age", "Education", "Occupation", "Hours per Week", "Gender"],
        "Value": [age, education, occupation, hours_per_week, gender]
    })

    st.table(summary_df)

with col2:
    st.subheader("Prediction Result")

    if st.button("Predict Salary Class", use_container_width=True):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df).max()

        if prediction == 1 or prediction == ">50K":
            st.success("Predicted Income: Greater than 50K")
        else:
            st.warning("Predicted Income: Less than or equal to 50K")

        st.write(f"Prediction Confidence: {probability * 100:.2f}%")

    if st.button("Reset Inputs", use_container_width=True):
        st.experimental_rerun()

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="text-align: center; font-size: 14px;">
        <p>
            <b>GitHub Repository:</b>
            <a href="https://github.com/deviakula2006/Employee_Salary_Prediction.git" target="_blank">
                Employee Salary Prediction
            </a>
        </p>
        <p>
            <b>LinkedIn Profile:</b>
            <a href="https://www.linkedin.com/in/devi-ganga-bhavani-akula-192065291/" target="_blank">
                Devi Ganga Bhavani Akula
            </a>
        </p>
        <p>Version 1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
