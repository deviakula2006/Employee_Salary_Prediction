import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Salary Prediction System",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("income_model.pkl")

# -----------------------------
# Header Section
# -----------------------------
st.markdown(
    """
    <div style="text-align:center;">
        <h1>Employee Salary Prediction System</h1>
        <p style="font-size:17px; color:#555;">
        A machine learning–based application to classify employee income levels
        </p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar - Input Section
# -----------------------------
st.sidebar.title("Employee Information")

age = st.sidebar.slider("Age", 18, 70, 30)

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

hours_per_week = st.sidebar.slider("Working Hours per Week", 1, 80, 40)
gender = st.sidebar.radio("Gender", ["Male", "Female"])

# -----------------------------
# Encoding Maps
# -----------------------------
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

# -----------------------------
# Build Input DataFrame
# -----------------------------
input_df = pd.DataFrame(columns=model.feature_names_in_)
input_df.loc[0] = 0

input_df.loc[0, "age"] = age
input_df.loc[0, "educational-num"] = education_map[education]
input_df.loc[0, "occupation"] = occupation_map[occupation]
input_df.loc[0, "hours-per-week"] = hours_per_week
input_df.loc[0, "gender"] = gender_map[gender]

# -----------------------------
# Main Layout
# -----------------------------
left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.subheader("Input Summary")
    st.dataframe(
        pd.DataFrame({
            "Feature": ["Age", "Education", "Occupation", "Hours per Week", "Gender"],
            "Value": [age, education, occupation, hours_per_week, gender]
        }),
        use_container_width=True
    )

with right_col:
    st.subheader("Prediction Result")

    if st.button("Predict Salary Category", use_container_width=True):
        prediction = model.predict(input_df)[0]

        st.markdown(
            "<div style='padding:20px; border-radius:10px; background:#f5f7fa; border:1px solid #ddd;'>",
            unsafe_allow_html=True
        )

        if prediction == 1 or prediction == ">50K":
            st.markdown(
                "<h3 style='color:green;'>Predicted Income: Above 50K</h3>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h3 style='color:#c0392b;'>Predicted Income: 50K or Below</h3>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray;">
    Developed by <b>Devi Ganga Bhavani Akula</b> | Machine Learning Project
    </p>
    """,
    unsafe_allow_html=True
)
