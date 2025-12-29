import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("income_model.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f4f6fb;
}

.main-title {
    text-align: center;
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
}

.result-success {
    background: #e8f7ee;
    border-left: 6px solid #2ecc71;
    padding: 20px;
    border-radius: 10px;
}

.result-fail {
    background: #fdecea;
    border-left: 6px solid #e74c3c;
    padding: 20px;
    border-radius: 10px;
}

.footer {
    text-align: center;
    padding: 25px;
}

.footer img {
    width: 35px;
    margin: 0 12px;
    transition: transform 0.2s;
}

.footer img:hover {
    transform: scale(1.2);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("""
<div class="main-title">
    <h1>Employee Salary Prediction System</h1>
    <p>Machine Learning application to classify employee income levels</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Employee Details")

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

hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
gender = st.sidebar.radio("Gender", ["Male", "Female"])

# -----------------------------
# Encoding
# -----------------------------
education_map = {
    "HS-grad": 9,
    "Some-college": 10,
    "Bachelors": 13,
    "Masters": 14,
    "Doctorate": 16
}

occupation_map = {
    "Tech-support": 1, "Craft-repair": 2, "Other-service": 3, "Sales": 4,
    "Exec-managerial": 5, "Prof-specialty": 6, "Handlers-cleaners": 7,
    "Machine-op-inspct": 8, "Adm-clerical": 9, "Farming-fishing": 10,
    "Transport-moving": 11, "Priv-house-serv": 12, "Protective-serv": 13,
    "Armed-Forces": 14
}

gender_map = {"Male": 1, "Female": 0}

# -----------------------------
# Input DataFrame
# -----------------------------
input_df = pd.DataFrame(columns=model.feature_names_in_)
input_df.loc[0] = 0

input_df.loc[0, "age"] = age
input_df.loc[0, "educational-num"] = education_map[education]
input_df.loc[0, "occupation"] = occupation_map[occupation]
input_df.loc[0, "hours-per-week"] = hours_per_week
input_df.loc[0, "gender"] = gender_map[gender]

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Input Summary")

    st.table(pd.DataFrame({
        "Feature": ["Age", "Education", "Occupation", "Hours / Week", "Gender"],
        "Value": [age, education, occupation, hours_per_week, gender]
    }))
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction")

    if st.button("Predict Salary Category", use_container_width=True):
        prediction = model.predict(input_df)[0]

        if prediction == 1 or prediction == ">50K":
            st.markdown("""
            <div class="result-success">
                <h3>Income Prediction: Above 50K</h3>
                <p>This employee is likely to earn more than 50K annually.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-fail">
                <h3>📉 Income Prediction: 50K or Below</h3>
                <p>This employee is likely to earn 50K or below annually.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer with Social Links
# -----------------------------
st.markdown("""
<hr>
<div class="footer">
    <p><b>Developed by Devi Ganga Bhavani Akula</b></p>
    <a href="https://www.linkedin.com/in/devi-ganga-bhavani-akula-192065291/" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png">
    </a>
    <a href="https://github.com/deviakula2006" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png">
    </a>
</div>
""", unsafe_allow_html=True)
