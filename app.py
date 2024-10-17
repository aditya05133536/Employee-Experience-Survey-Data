import streamlit as st
import pandas as pd
import pickle

# Load the trained model and feature set
with open('job_satisfaction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model_features.pkl', 'rb') as feature_file:
    model_features = pickle.load(feature_file)


def preprocess_employee_data(employee_data, model_features):
    # Create a DataFrame from the input data
    df_employee = pd.DataFrame(employee_data)

    # Map Likert scale responses to numeric values
    likert_mapping = {
        'Strongly Disagree': 1,
        'Disagree': 2,
        'Neutral': 3,
        'Agree': 4,
        'Strongly Agree': 5
    }

    # Apply mapping to Likert scale columns
    likert_columns = [
        'Work-Life Balance', 'Management Support', 'Team Collaboration',
        'Workload Fairness', 'Career Development Opportunities',
        'Workplace Inclusivity', 'Company Communication',
        'Compensation Satisfaction', 'Job Security'
    ]

    for column in likert_columns:
        if column in df_employee.columns:
            df_employee[column] = df_employee[column].map(likert_mapping)

    # One-Hot Encode categorical columns
    categorical_columns = ['Age Bracket', 'Gender', 'Ethnicity', 'Department']
    df_employee_encoded = pd.get_dummies(df_employee, columns=categorical_columns, drop_first=True)

    # Reindex to ensure the new data has the same columns as the training data
    df_employee_encoded = df_employee_encoded.reindex(columns=model_features, fill_value=0)

    return df_employee_encoded


# Streamlit app interface
st.title("Employee Job Satisfaction Predictor")

# Input fields for user survey responses
work_life_balance = st.selectbox("Work-Life Balance",
                                 options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
management_support = st.selectbox("Management Support",
                                  options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
team_collaboration = st.selectbox("Team Collaboration",
                                  options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
workload_fairness = st.selectbox("Workload Fairness",
                                 options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
career_dev_opportunities = st.selectbox("Career Development Opportunities",
                                        options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
workplace_inclusivity = st.selectbox("Workplace Inclusivity",
                                     options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
company_communication = st.selectbox("Company Communication",
                                     options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
compensation_satisfaction = st.selectbox("Compensation Satisfaction",
                                         options=["Strongly Disagree", "Disagree", "Neutral", "Agree",
                                                  "Strongly Agree"])
job_security = st.selectbox("Job Security",
                            options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
age_bracket = st.selectbox("Age Bracket", options=["Under 25", "25-34", "35-44", "45-54", "55-64", "65 and above"])
gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
ethnicity = st.selectbox("Ethnicity", options=["Asian", "Black", "Hispanic", "White", "Other"])
department = st.selectbox("Department", options=["HR", "IT", "Finance", "Marketing", "Other"])

# Create a button to make predictions
if st.button("Predict Job Satisfaction"):
    user_input = {
        'Work-Life Balance': work_life_balance,
        'Management Support': management_support,
        'Team Collaboration': team_collaboration,
        'Workload Fairness': workload_fairness,
        'Career Development Opportunities': career_dev_opportunities,
        'Workplace Inclusivity': workplace_inclusivity,
        'Company Communication': company_communication,
        'Compensation Satisfaction': compensation_satisfaction,
        'Job Security': job_security,
        'Age Bracket': age_bracket,
        'Gender': gender,
        'Ethnicity': ethnicity,
        'Department': department
    }

    new_employee_data = [user_input]

    # Preprocess the data to ensure feature consistency
    new_employee_features = preprocess_employee_data(new_employee_data, model_features)

    # Make the prediction
    predicted_job_satisfaction = model.predict(new_employee_features)

    # Display the result
    st.success(f"Predicted Job Satisfaction for the new employee: {predicted_job_satisfaction[0]:.2f}")
