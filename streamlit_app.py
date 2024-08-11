import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.title('Startup Status Predictor')

st.write("""
This app estimates the probability of different statuses for a startup based on key features.
Please fill in the information below:
""")

founded_at = st.number_input('Founded Year', min_value=1900, max_value=datetime.now().year, value=2023)
funding_total_usd = st.number_input('Total Funding (USD)', min_value=0, value=500000000000000)
first_funding_at = st.number_input('First Funding Year', min_value=1900, max_value=datetime.now().year, value=2023)
last_funding_at = st.number_input('Last Funding Year', min_value=1900, max_value=datetime.now().year, value=2023)
funding_rounds = st.number_input('Number of Funding Rounds', min_value=0, value=100)
milestones = st.number_input('Number of Milestones', min_value=0, value=10000)

categories = ['analytics', 'biotech', 'cleantech', 'ecommerce', 'enterprise', 'games_video',
              'hardware', 'health', 'medical', 'mobile', 'social', 'software', 'web', 'other']
selected_category = st.selectbox('Select the primary category', options=categories, index=categories.index('software'))

countries = ['USA', 'GBR', 'CAN', 'DEU', 'FRA', 'CHN', 'IND', 'ESP', 'IRL', 'ISR', 'NLD', 'RUS', 'SGP', 'SWE', 'Other']
selected_country = st.selectbox('Select the country', options=countries, index=countries.index('USA'))

age_in_days = (datetime.now().year - founded_at) * 365


def estimate_probabilities(age, funding, rounds, milestones, category, country):
    base_probs = {'Acquired': 0.2, 'Closed': 0.1, 'IPO': 0.05, 'Operating': 0.65}

    if age > 3650:  # More than 10 years
        base_probs['Acquired'] += 0.1
        base_probs['IPO'] += 0.05
        base_probs['Operating'] -= 0.15

    if funding > 100000000:  # More than 100M
        base_probs['Acquired'] += 0.15
        base_probs['IPO'] += 0.1
        base_probs['Operating'] -= 0.25

    if rounds > 5:
        base_probs['Acquired'] += 0.1
        base_probs['IPO'] += 0.05
        base_probs['Operating'] -= 0.15

    if milestones > 10:
        base_probs['Operating'] += 0.1
        base_probs['Closed'] -= 0.1

    if category in ['software', 'enterprise', 'mobile']:
        base_probs['Acquired'] += 0.05

    if country in ['USA', 'GBR', 'CHN']:
        base_probs['Acquired'] += 0.05
        base_probs['IPO'] += 0.05
        base_probs['Operating'] -= 0.1

    total = sum(base_probs.values())
    return {k: v / total for k, v in base_probs.items()}


if st.button('Estimate Status Probabilities'):
    probabilities = estimate_probabilities(age_in_days, funding_total_usd, funding_rounds, milestones,
                                           selected_category, selected_country)

    st.write("Estimated Probabilities:")
    for status, prob in probabilities.items():
        st.write(f"{status}: {prob:.2%}")

    chart_data = pd.DataFrame({
        'Status': list(probabilities.keys()),
        'Probability': list(probabilities.values())
    })
    st.bar_chart(chart_data.set_index('Status'))

    max_prob = max(probabilities.values())
    max_status = max(probabilities, key=probabilities.get)

    st.write(f"The highest estimated probability ({max_prob:.2%}) is for the status: {max_status}")
    st.write("Please consider all probabilities when interpreting the results.")

st.info("""
This app estimates the probabilities of different statuses for a startup based on key features including funding information, 
company category, and location. The possible status outcomes are:
- Acquired
- Closed
- IPO (Initial Public Offering)
- Operating
""")








# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime
#
# # Load the trained model
# model = joblib.load('best_startup_status_model.joblib')
#
# # Set up the Streamlit app
# st.title('Startup Status Predictor')
#
# st.write("""
# This app predicts the status of a startup based on key features.
# Please fill in the information below:
# """)
#
# # Create input fields for the most important features
# founded_at = st.number_input('Founded Year', min_value=1900, max_value=datetime.now().year, value=2023)
# funding_total_usd = st.number_input('Total Funding (USD)', min_value=0, value=1000000000000)
# first_funding_at = st.number_input('First Funding Year', min_value=1900, max_value=datetime.now().year, value=2023)
# last_funding_at = st.number_input('Last Funding Year', min_value=1900, max_value=datetime.now().year, value=2023)
#
# # Dropdown for company category
# categories = ['analytics', 'biotech', 'cleantech', 'ecommerce', 'enterprise', 'games_video',
#               'hardware', 'health', 'medical', 'mobile', 'social', 'software', 'web', 'other']
# selected_category = st.selectbox('Select the primary category', options=categories)
#
# # Dropdown for country
# countries = ['USA', 'GBR', 'CAN', 'DEU', 'FRA', 'CHN', 'IND', 'ESP', 'IRL', 'ISR', 'NLD', 'RUS', 'SGP', 'SWE', 'Other']
# selected_country = st.selectbox('Select the country', options=countries)
#
# # Calculate Age_in_Days
# age_in_days = (datetime.now().year - founded_at) * 365
#
# # Prepare the input data
# input_data = {
#     'founded_at': founded_at,
#     'funding_total_usd': funding_total_usd,
#     'first_funding_at': first_funding_at,
#     'last_funding_at': last_funding_at,
#     'Age_in_Days': age_in_days
# }
#
# # Add categorical features
# for category in categories:
#     input_data[category] = 1 if category == selected_category else 0
#
# # Add country features
# for country in ['USA', 'GBR', 'CAN', 'DEU', 'FRA', 'CHN', 'IND', 'ESP', 'IRL', 'ISR', 'NLD', 'RUS', 'SGP', 'SWE']:
#     input_data[country] = 1 if country == selected_country else 0
#
# # Add other features that might be expected by the model but not collected here
# additional_features = ['investment_rounds', 'funding_rounds', 'first_milestone_at', 'last_milestone_at',
#                        'milestones', 'relationships', 'lat', 'lng', 'other.1']
#
# for feature in additional_features:
#     input_data[feature] = 0  # Set to 0 or another appropriate default value
#
# # Create a DataFrame from the input data
# input_df = pd.DataFrame([input_data])
#
# # Make prediction when the user clicks the button
# if st.button('Predict Status'):
#     try:
#         # Make prediction
#         prediction = model.predict(input_df)
#         prediction_proba = model.predict_proba(input_df)
#
#         # Get the predicted status
#         status_mapping = {0: 'acquired', 1: 'closed', 2: 'ipo', 3: 'operating'}
#         predicted_status = status_mapping[prediction[0]]
#
#         # Display the prediction
#         st.success(f'The predicted status of the company is: {predicted_status.upper()}')
#
#         # Display prediction probabilities
#         st.write("Prediction Probabilities:")
#         for i, status in status_mapping.items():
#             st.write(f"{status.capitalize()}: {prediction_proba[0][i]:.4f}")
#
#         # Display feature importances if available
#         if hasattr(model, 'feature_importances_'):
#             feature_importance = model.feature_importances_
#             feature_names = model.feature_names_in_
#             importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
#             importance_df = importance_df.sort_values('importance', ascending=False)
#             st.write("Top 10 Feature Importances:")
#             st.write(importance_df.head(10))
#
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         st.error("Please make sure the model is compatible with the input data.")
#
# # Add some information about the model and its usage
# st.info("""
# This model predicts the status of a startup based on key features including funding information,
# company category, and location. The possible status outcomes are:
# - Acquired
# - Closed
# - IPO (Initial Public Offering)
# - Operating
#
# Please note that this is a predictive model and its accuracy may vary. The prediction should be used
# as one of many factors in assessing a company's status.
# """)




# import streamlit as st
# import pandas as pd
# import joblib
#
# from pipeline import BinaryPipeline, MulticlassPipeline, MulticlassClassifier, ProbabilityExtractor, BinaryClassifier
#
# # Load the pre-trained pipeline
# pipeline = joblib.load('full_pipeline.joblib')
#
#
# # Define the Streamlit app
# def main():
#     st.title("Startup Acquisition Status Prediction")
#
#     # Create a form to input features
#     with st.form("prediction_form"):
#         st.header("Enter the details of the startup:")
#
#         # Input fields for key features
#         founded_at = st.number_input('Founded At (Year):', min_value=1900, max_value=2024)
#         first_funding_at = st.number_input('First Funding At (Year):', min_value=1900, max_value=2024)
#         last_funding_at = st.number_input('Last Funding At (Year):', min_value=1900, max_value=2024)
#         funding_total_usd = st.number_input('Funding Total USD:', min_value=0.0)
#
#         # Country field as dropdown
#         country = st.selectbox('Country:', [
#             'USA', 'CAN', 'CHN', 'DEU', 'ESP', 'FRA', 'GBR', 'IND', 'IRL',
#             'ISR', 'NLD', 'RUS', 'SGP', 'SWE', 'Other'
#         ])
#
#         # Product feature field as dropdown
#         product = st.selectbox('Product:', [
#             'analytics', 'biotech', 'cleantech', 'ecommerce', 'enterprise',
#             'games_video', 'hardware', 'health', 'medical', 'mobile', 'other',
#             'social', 'software', 'web'
#         ])
#
#         # Submit button
#         submit = st.form_submit_button("Predict")
#
#     if submit:
#         # Prepare the input data
#         input_data = {'founded_at': [founded_at], 'first_funding_at': [first_funding_at],
#                       'last_funding_at': [last_funding_at], 'funding_total_usd': [funding_total_usd],
#                       'last_milestone_at': [0], 'lat': [0.0], 'lng': [0.0], 'funding_rounds': [0], 'relationships': [0],
#                       'Age_in_Days': [0], 'milestones': [0], 'investment_rounds': [0], 'first_milestone_at': [0],
#                       'analytics': [False], 'biotech': [False], 'cleantech': [False], 'ecommerce': [False],
#                       'enterprise': [False], 'games_video': [False], 'hardware': [False], 'health': [False],
#                       'medical': [False], 'mobile': [False], 'other': [False], 'social': [False], 'software': [False],
#                       'web': [False], 'CAN': [False], 'CHN': [False], 'DEU': [False], 'ESP': [False], 'FRA': [False],
#                       'GBR': [False], 'IND': [False], 'IRL': [False], 'ISR': [False], 'NLD': [False], 'RUS': [False],
#                       'SGP': [False], 'SWE': [False], 'USA': [False], 'other.1': [False], country: [True],
#                       product: [True]}
#
#         # Set the correct country
#
#         # Set the correct product feature
#
#         # Convert to DataFrame
#         input_data_df = pd.DataFrame(input_data)
#
#         # Predict the acquisition status
#         prediction = pipeline.predict(input_data_df)
#
#         # Map the numerical prediction to actual labels
#         label_map = {0: 'Acquired', 1: 'Closed', 2: 'IPO', 3: 'Operating'}
#         prediction_label = label_map[prediction[0]]
#
#         # Display the prediction
#         st.subheader(f"The predicted acquisition status is: {prediction_label}")
#
#
# if __name__ == "__main__":
#     main()






# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from pipeline import BinaryPipeline, MulticlassPipeline, MulticlassClassifier, ProbabilityExtractor, BinaryClassifier
#
# # Load the pre-trained pipeline
# pipeline = joblib.load('full_pipeline.joblib')
#
#
# # Define the Streamlit app
# def main():
#     st.title("Startup Acquisition Status Prediction")
#
#     # Create a form to input features
#     with st.form("prediction_form"):
#         st.header("Enter the details of the startup:")
#
#         # Assume the features are the same as in the dataset
#         # Replace these with actual feature names and types
#         features = {}
#         features['feature1'] = st.text_input('Feature 1:')
#         features['feature2'] = st.number_input('Feature 2:', min_value=0, max_value=100)
#         features['feature3'] = st.number_input('Feature 3:', min_value=0, max_value=100)
#         features['feature4'] = st.selectbox('Feature 4:', ['Option 1', 'Option 2', 'Option 3'])
#         # Add more fields as necessary
#
#         # Submit button
#         submit = st.form_submit_button("Predict")
#
#     if submit:
#         # Convert the form data into a dataframe
#         input_data = pd.DataFrame([features])
#
#         # Preprocess categorical features if any
#         input_data['feature4'] = input_data['feature4'].map({'Option 1': 0, 'Option 2': 1, 'Option 3': 2})
#
#         # Predict the acquisition status
#         prediction = pipeline.predict(input_data)
#
#         # Map the numerical prediction to actual labels
#         label_map = {0: 'Acquired', 1: 'Closed', 2: 'IPO', 3: 'Operating'}
#         prediction_label = label_map[prediction[0]]
#
#         # Display the prediction
#         st.subheader(f"The predicted acquisition status is: {prediction_label}")
#
#
# if __name__ == "__main__":
#     main()
