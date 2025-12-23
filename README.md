# Employee Salary Prediction

## Overview

This project is an end-to-end Machine Learning application that predicts whether an employee’s income is **greater than 50K** or **less than or equal to 50K** based on demographic and work-related attributes.
The model is trained using the Adult Income Dataset and deployed as a web application using Streamlit.

---

## Live Application

The deployed application can be accessed at the link below:

[https://employee-salary-prediction-f3nvt5hsfbjdpuwv4ucjvx.streamlit.app/](https://employee-salary-prediction-f3nvt5hsfbjdpuwv4ucjvx.streamlit.app/)

---

## Problem Statement

Given a set of employee attributes such as age, education level, occupation, working hours, and gender, the objective is to classify whether the employee earns more than 50K per year.

This is a binary classification problem.

---

## Dataset

* Dataset Name: Adult Income Dataset
* Source: UCI Machine Learning Repository
* Target Variable: Salary (`>50K`, `<=50K`)

The dataset contains demographic and employment-related features commonly used for income prediction tasks.

---

## Machine Learning Approach

* Data Cleaning and Preprocessing

  * Handling missing values
  * Removing redundant features
  * Encoding categorical variables
  * Feature selection

* Model Training

  * Multiple models were evaluated
  * Random Forest Classifier was selected based on performance

* Model Persistence

  * The trained model was saved using `joblib`
  * The saved model is reused in the deployed application

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

---

## Application Functionality

* Accepts user input through a web interface
* Preprocesses input to match training feature format
* Uses the trained model to make predictions
* Displays the predicted salary class in real time

---

## Project Structure

```
Employee_Salary_Prediction/
│
├── app.py                      # Streamlit web application
├── salary prediction.ipynb     # Model training and analysis notebook
├── adult.csv                   # Dataset
├── income_model.pkl            # Trained machine learning model
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation


## Running the Project Locally

1. Clone the repository:

```bash
git clone https://github.com/deviakula2006/Employee_Salary_Prediction.git
cd Employee_Salary_Prediction


2. Install required dependencies:

```bash
pip install -r requirements.txt


3. Run the Streamlit application:

```bash
streamlit run app.py




## Deployment

The application is deployed using Streamlit Community Cloud.
This allows the project to be accessed directly through a web browser without requiring local installation.



## Key Learnings

* Building a complete machine learning pipeline
* Feature engineering and preprocessing
* Model selection and evaluation
* Saving and loading trained models
* Developing web interfaces for ML models
* Deploying machine learning applications to the cloud

---

## Author

Devi Ganga Bhavani Akula
