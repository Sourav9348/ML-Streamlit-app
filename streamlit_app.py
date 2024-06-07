import streamlit as st
import pandas as pd
import joblib
from pipeline import BinaryPipeline, MulticlassPipeline, MulticlassClassifier, ProbabilityExtractor, BinaryClassifier

# Load the pre-trained pipeline
pipeline = joblib.load('full_pipeline.joblib')


# Define the Streamlit app
def main():
    st.title("Startup Acquisition Status Prediction")

    # Create a form to input features
    with st.form("prediction_form"):
        st.header("Enter the details of the startup:")

        # Input fields for key features
        founded_at = st.number_input('Founded At (Year):', min_value=1900, max_value=2024)
        first_funding_at = st.number_input('First Funding At (Year):', min_value=1900, max_value=2024)
        last_funding_at = st.number_input('Last Funding At (Year):', min_value=1900, max_value=2024)
        funding_total_usd = st.number_input('Funding Total USD:', min_value=0.0)

        # Country field as dropdown
        country = st.selectbox('Country:', [
            'USA', 'CAN', 'CHN', 'DEU', 'ESP', 'FRA', 'GBR', 'IND', 'IRL',
            'ISR', 'NLD', 'RUS', 'SGP', 'SWE', 'Other'
        ])

        # Product feature field as dropdown
        product = st.selectbox('Product:', [
            'analytics', 'biotech', 'cleantech', 'ecommerce', 'enterprise',
            'games_video', 'hardware', 'health', 'medical', 'mobile', 'other',
            'social', 'software', 'web'
        ])

        # Submit button
        submit = st.form_submit_button("Predict")

    if submit:
        # Prepare the input data
        input_data = {'founded_at': [founded_at], 'first_funding_at': [first_funding_at],
                      'last_funding_at': [last_funding_at], 'funding_total_usd': [funding_total_usd],
                      'last_milestone_at': [0], 'lat': [0.0], 'lng': [0.0], 'funding_rounds': [0], 'relationships': [0],
                      'Age_in_Days': [0], 'milestones': [0], 'investment_rounds': [0], 'first_milestone_at': [0],
                      'analytics': [False], 'biotech': [False], 'cleantech': [False], 'ecommerce': [False],
                      'enterprise': [False], 'games_video': [False], 'hardware': [False], 'health': [False],
                      'medical': [False], 'mobile': [False], 'other': [False], 'social': [False], 'software': [False],
                      'web': [False], 'CAN': [False], 'CHN': [False], 'DEU': [False], 'ESP': [False], 'FRA': [False],
                      'GBR': [False], 'IND': [False], 'IRL': [False], 'ISR': [False], 'NLD': [False], 'RUS': [False],
                      'SGP': [False], 'SWE': [False], 'USA': [False], 'other.1': [False], country: [True],
                      product: [True]}

        # Set the correct country

        # Set the correct product feature

        # Convert to DataFrame
        input_data_df = pd.DataFrame(input_data)

        # Predict the acquisition status
        prediction = pipeline.predict(input_data_df)

        # Map the numerical prediction to actual labels
        label_map = {0: 'Acquired', 1: 'Closed', 2: 'IPO', 3: 'Operating'}
        prediction_label = label_map[prediction[0]]

        # Display the prediction
        st.subheader(f"The predicted acquisition status is: {prediction_label}")


if __name__ == "__main__":
    main()

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
