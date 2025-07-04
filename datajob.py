import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and feature columns
try:
    with open('salary_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('model_features.pkl', 'rb') as file:
        model_features = pickle.load(file)
    st.success("Model and features loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model files not found. Make sure 'salary_prediction_model.pkl' and 'model_features.pkl' are in the same directory.")
    st.stop() # Stop the app if files are not found

# Title of the Streamlit app
st.title('Data Analyst Salary Predictor')
st.markdown('Enter the details below to predict the average salary for a Data Analyst position.')

# User Input for Prediction
st.header('Job Details')

rating = st.slider('Company Rating (on a scale of 1 to 5)', 1.0, 5.0, 3.5, 0.1)

# Mapping for Size (ensure consistency with training data)
size_options = ['Unknown', '1 to 50 employees', '51 to 200 employees', '201 to 500 employees',
                '501 to 1000 employees', '1001 to 5000 employees', '5001 to 10000 employees',
                '10000+ employees']
selected_size = st.selectbox('Company Size', options=size_options)

ownership_options = ['Unknown', 'Company - Public', 'Company - Private', 'Nonprofit Organization',
                     'Subsidiary or Business Segment', 'Government', 'Hospital', 'College / University']
selected_ownership = st.selectbox('Type of Ownership', options=ownership_options)

# Ensure these options match the `Industry_Cleaned` categories from your training data,
# including the 'Other_Industry' category.
# The list comprehension `list(model_features[model_features.str.startswith('Industry_Cleaned_')].str.replace('Industry_Cleaned_', '').unique())`
# dynamically extracts them based on the loaded model_features.
industry_options_dynamic = list(pd.Series(model_features)[pd.Series(model_features).str.startswith('Industry_Cleaned_')].str.replace('Industry_Cleaned_', '').unique())
if 'Other_Industry' not in industry_options_dynamic: # Ensure 'Other_Industry' is an explicit option
    industry_options_dynamic.append('Other_Industry')
selected_industry = st.selectbox('Industry', options=industry_options_dynamic)

# Ensure these options match the `Sector_Cleaned` categories from your training data
sector_options_dynamic = list(pd.Series(model_features)[pd.Series(model_features).str.startswith('Sector_Cleaned_')].str.replace('Sector_Cleaned_', '').unique())
if 'Other_Sector' not in sector_options_dynamic: # Ensure 'Other_Sector' is an explicit option
    sector_options_dynamic.append('Other_Sector')
selected_sector = st.selectbox('Sector', options=sector_options_dynamic)

revenue_options = ['Unknown / Non-Applicable', '$1 to $5 million (USD)', '$5 to $10 million (USD)',
                   '$10 to $25 million (USD)', '$25 to $50 million (USD)', '$50 to $100 million (USD)',
                   '$100 to $500 million (USD)', '$500 million to $1 billion (USD)',
                   '$1 to $2 billion (USD)', '$2 to $5 billion (USD)', '$5 to $10 billion (USD)',
                   '$10+ billion (USD)']
selected_revenue = st.selectbox('Revenue', options=revenue_options)

job_level_options = ['Associate/Mid-level', 'Junior', 'Senior', 'Lead', 'Manager', 'Director']
selected_job_level = st.selectbox('Job Level', options=job_level_options)

# Ensure these options match the `State_Cleaned` categories from your training data
state_options_dynamic = list(pd.Series(model_features)[pd.Series(model_features).str.startswith('State_Cleaned_')].str.replace('State_Cleaned_', '').unique())
if 'Other_State' not in state_options_dynamic: # Ensure 'Other_State' is an explicit option
    state_options_dynamic.append('Other_State')
selected_state = st.selectbox('State', options=state_options_dynamic)

company_founded_year = st.number_input('Company Founded Year', min_value=1800, max_value=2025, value=2000, step=1)
company_age = 2025 - company_founded_year
if company_age < 0:
    company_age = 0 # Handle cases where founded year is in the future or incorrect

st.subheader('Skills (Select all that apply)')
skill_python = st.checkbox('Python')
skill_r = st.checkbox('R')
skill_sql = st.checkbox('SQL')
skill_excel = st.checkbox('Excel')
skill_tableau = st.checkbox('Tableau')
skill_power_bi = st.checkbox('Power BI')
skill_aws = st.checkbox('AWS')
skill_azure = st.checkbox('Azure')
skill_gcp = st.checkbox('GCP')
skill_spark = st.checkbox('Spark')
skill_hadoop = st.checkbox('Hadoop')
skill_machine_learning = st.checkbox('Machine Learning')
skill_statistics = st.checkbox('Statistics')


# Create input DataFrame matching the model's training features
input_data = pd.DataFrame(0, index=[0], columns=model_features)

# Populate numerical features
# You MUST use the scaler object that was fitted during training for 'Rating_scaled'
# For simplicity here, we're re-initializing StandardScaler. In a real app, save/load the scaler.
# For now, let's assume a rough min/max for rating scale.
# A robust solution requires saving the actual StandardScaler object used for 'Rating' in training.
# Since the model expects `Rating_scaled`, we'll need to scale the input rating.
# Let's assume a simplified scaling for demonstration if the original scaler isn't saved.
# If your training script didn't save the scaler, this will be an approximation.
# A better way: Pass the original `scaler` object (used for 'Rating') to the Streamlit app.
# For now, manually scale based on min/max or mean/std if you know them from training.
# A simple min-max scaling for a 1-5 rating: (rating - min) / (max - min)
# Or, if StandardScaler was used, we need its mean and std.
# For demonstration, we'll re-fit a dummy scaler for 'Rating' based on the 1-5 range:
temp_scaler_rating = StandardScaler()
temp_scaler_rating.fit(np.array([[1.0],[5.0]])) # Fit on min and max of expected rating
input_data['Rating_scaled'] = temp_scaler_rating.transform(np.array([[rating]]))[0][0]

input_data['Company_Age'] = company_age

# The 'Founded_scaled' column is important. We need the original scaler used on the training data.
# For a real deployment, you'd save the scaler object too.
# For simplicity here, we'll just set it to 0 or derive from company_age if 'Founded' itself isn't an input.
# If 'Founded_scaled' was explicitly used in the model, and the scaler for 'Founded' wasn't saved,
# this becomes an issue. Let's assume its effect is primarily captured by 'Company_Age'.
if 'Founded_scaled' in input_data.columns:
    # If original scaler was not saved, this is an approximation.
    # To be accurate, save the original scaler fitted on 'Founded' during training.
    # For now, setting to 0 or relating it to company age.
    # A safer approach is to pass it through a scaler if founded_year is given.
    # Here, we'll try to re-scale founded year based on a rough range.
    temp_scaler_founded = StandardScaler()
    # Assume 'Founded' years range from e.g., 1800 to 2024 in training data.
    temp_scaler_founded.fit(np.array([[1800],[2024]]))
    input_data['Founded_scaled'] = temp_scaler_founded.transform(np.array([[company_founded_year]]))[0][0]


# Populate skill features
input_data['skill_python'] = 1 if skill_python else 0
input_data['skill_r'] = 1 if skill_r else 0
input_data['skill_sql'] = 1 if skill_sql else 0
input_data['skill_excel'] = 1 if skill_excel else 0
input_data['skill_tableau'] = 1 if skill_tableau else 0
input_data['skill_power_bi'] = 1 if skill_power_bi else 0
input_data['skill_aws'] = 1 if skill_aws else 0
input_data['skill_azure'] = 1 if skill_azure else 0
input_data['skill_gcp'] = 1 if skill_gcp else 0
input_data['skill_spark'] = 1 if skill_spark else 0
input_data['skill_hadoop'] = 1 if skill_hadoop else 0
input_data['skill_machine_learning'] = 1 if skill_machine_learning else 0
input_data['skill_statistics'] = 1 if skill_statistics else 0


# Populate one-hot encoded features
if f'Size_{selected_size}' in input_data.columns:
    input_data[f'Size_{selected_size}'] = 1

if f'Type of ownership_{selected_ownership}' in input_data.columns:
    input_data[f'Type of ownership_{selected_ownership}'] = 1

# Handle 'Other' categories for industry, sector, state - only set 1 if specific category matches
# If the selected category is 'Other_Industry', then none of the specific industry columns should be 1.
# This is implicitly handled because input_data is initialized with all zeros.
if selected_industry != 'Other_Industry' and f'Industry_Cleaned_{selected_industry}' in input_data.columns:
    input_data[f'Industry_Cleaned_{selected_industry}'] = 1

if selected_sector != 'Other_Sector' and f'Sector_Cleaned_{selected_sector}' in input_data.columns:
    input_data[f'Sector_Cleaned_{selected_sector}'] = 1

if f'Revenue_{selected_revenue}' in input_data.columns:
    input_data[f'Revenue_{selected_revenue}'] = 1

if f'Job_Level_{selected_job_level}' in input_data.columns:
    input_data[f'Job_Level_{selected_job_level}'] = 1

if selected_state != 'Other_State' and f'State_Cleaned_{selected_state}' in input_data.columns:
    input_data[f'State_Cleaned_{selected_state}'] = 1

# Ensure input_data columns are in the same order as model_features
input_data = input_data[model_features]

if st.button('Predict Salary'):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f'Predicted Average Annual Salary: **${prediction:,.2f}**')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Input data for debugging:")
        st.write(input_data)
