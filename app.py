import streamlit as st
import pandas as pd
import joblib

# Load model

model = joblib.load("income_model.pkl")

st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="wide"
)


st.markdown(
    """
    <h1 style='text-align: center;'>💼 Employee Salary Classification</h1>
    <p style='text-align: center; font-size:18px;'>
    Predict whether an employee earns <b>&gt;50K</b> or <b>≤50K</b>
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)


# Sidebar

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
    "Job Role",
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



# Encoding Maps

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

gender_map = {"Male": 1, "Female": 0}


# Build input in training format

input_df = pd.DataFrame(columns=model.feature_names_in_)
input_df.loc[0] = 0

input_df.loc[0, "age"] = age
input_df.loc[0, "educational-num"] = education_map[education]
input_df.loc[0, "occupation"] = occupation_map[occupation]
input_df.loc[0, "hours-per-week"] = hours_per_week
input_df.loc[0, "gender"] = gender_map[gender]


# Main layout

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📊 Input Summary")
    st.dataframe(
        pd.DataFrame({
            "Feature": ["Age", "Education", "Occupation", "Hours/Week", "Gender"],
            "Value": [age, education, occupation, hours_per_week, gender]
        }),
        use_container_width=True
    )

with col2:
    st.subheader("Prediction")

    if st.button("Predict Salary Class", use_container_width=True):
        prediction = model.predict(input_df)[0]

        if prediction == 1 or prediction == ">50K":
            st.success("💰 **Predicted Income: > 50K**")
        else:
            st.warning("📉 **Predicted Income: ≤ 50K**")


# Footer

st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray;'>
    Built by ganga using Machine Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
