import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

# Function for direct preprocessing
def preprocess_input(data):
    # Mapping feature names to fit with model
    column_mapping = {
        'cons_conf_idx': 'cons.conf.idx', 
        'cons_price_idx': 'cons.price.idx'
    }
    # Rename columns if necessary
    data = data.rename(columns=column_mapping)

    # Categorical columns
    cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'age_group']
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode each categorical column
    for column in cat_columns:
        data[column] = label_encoder.fit_transform(data[column])

    # Numerical columns
    num_columns = ['duration', 'campaign', 'pdays', 'previous', 'cons.price.idx', 'cons.conf.idx']
    # Initialize StandardScaler
    scaler = StandardScaler()
    data[num_columns] = scaler.fit_transform(data[num_columns])

    return data

# Function to make predictions
def predict(data):
    prediction = model.predict(data)
    return prediction[0]

# Function to recommend based on prediction
def recommend(prediction):
    if prediction == 1:
        return "Recommend contacting this customer for the marketing campaign."
    else:
        return "Do not recommend contacting this customer for the marketing campaign."

# Main Streamlit app
def main():
    st.title("Bank Marketing Campaign Recommendation System")

    # Center the input form on the page
    input_container = st.container()
    with input_container:
        st.header("User Input")

        # Input form
        job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
        marital = st.selectbox("Marital Status", ['divorced', 'married', 'single'])
        education = st.selectbox("Education", ['basic.education', 'high.school', 'illiterate', 'professional.course', 'university.degree'])
        default = st.selectbox("Has Credit in Default", ['no', 'yes'])
        housing = st.selectbox("Housing Loan", ['no', 'yes'])
        loan = st.selectbox("Personal Loan", ['no', 'yes'])
        duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=10000, value=500, step=1)
        campaign = st.number_input("Number of Contacts Performed During This Campaign", min_value=0, max_value=100, value=1, step=1)
        pdays = st.number_input("Number of Days Since Last Contact from Previous Campaign", min_value=0, max_value=1000, value=0, step=1)
        previous = st.number_input("Number of Contacts Performed Before This Campaign", min_value=0, max_value=100, value=0, step=1)
        cons_price_idx = st.number_input("Consumer Price Index", min_value=0.0, max_value=100.0, value=93.0, step=0.1)
        cons_conf_idx = st.number_input("Consumer Confidence Index", min_value=-100.0, max_value=100.0, value=-40.0, step=0.1)
        age_group = st.selectbox("Age Group", ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'])

        # Button to make prediction
        if st.button("Get Recommendation"):
            # Process input data
            data = {
                'job': job,
                'marital': marital,
                'education': education,
                'default': default,
                'housing': housing,
                'loan': loan,
                'duration': duration,
                'campaign': campaign,
                'pdays': pdays,
                'previous': previous,
                'cons_price_idx': cons_price_idx,
                'cons_conf_idx': cons_conf_idx,
                'age_group': age_group
            }

            df = pd.DataFrame([data])

            # Preprocess input data
            df = preprocess_input(df)

            # Make prediction
            prediction = predict(df)

            # Recommendation based on prediction
            recommendation = recommend(prediction)

            # Display prediction and recommendation
            st.write("")
            st.write("## Prediction")
            if prediction == 1:
                st.write("The model predicts that this customer will subscribe to the term deposit.")
            else:
                st.write("The model predicts that this customer will not subscribe to the term deposit.")
            
            st.write("## Recommendation")
            st.write(recommendation)

# Run the app
if __name__ == "__main__":
    main()
