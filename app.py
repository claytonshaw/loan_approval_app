import streamlit as st
import joblib
import pandas as pd

st.title("Loan Prediction App")
st.write("This app predicts the loan status based on your input using a pre-trained XGBoost model.")

# Load the model with caching and error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("xgboost_model.joblib")
        return model
    except Exception as e:
        st.error("Error loading model: " + str(e))
        return None

model = load_model()

# Load training data for default input values with caching and error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("train.csv")
        return data
    except Exception as e:
        st.error("Error loading data: " + str(e))
        return pd.DataFrame()

data = load_data()

# If data loading fails, halt the app
if data.empty:
    st.stop()

st.sidebar.header("Input Features")

def user_input_features():
    # Group inputs into a form for a better user experience
    with st.sidebar.form("input_form"):
        person_age = st.slider(
            "Person Age",
            int(data["person_age"].min()),
            max(int(data["person_age"].max()), 100),
            int(data["person_age"].median())
        )
        person_income = st.number_input(
            "Person Income",
            min_value=1,  # Avoid division by zero
            value=int(data["person_income"].median())
        )
        person_home_ownership = st.selectbox(
            "Home Ownership",
            data["person_home_ownership"].unique()
        )
        person_emp_length = st.slider(
            "Employment Length (Years)",
            int(data["person_emp_length"].min()),
            max(int(data["person_emp_length"].max()), 65),
            int(data["person_emp_length"].median())
        )
        loan_intent = st.selectbox(
            "Loan Intent",
            data["loan_intent"].unique()
        )
        loan_grade = st.selectbox(
            "Loan Grade",
            data["loan_grade"].unique()
        )
        loan_amnt = st.number_input(
            "Loan Amount",
            min_value=0,
            value=int(data["loan_amnt"].median())
        )
        loan_int_rate = st.number_input(
            "Loan Interest Rate",
            min_value=0.0,
            value=float(data["loan_int_rate"].median())
        )
        cb_person_default_on_file = st.selectbox(
            "Person Default on File",
            data["cb_person_default_on_file"].unique()
        )
        cb_person_cred_hist_length = st.number_input(
            "Credit History Length",
            min_value=0,
            value=int(data["cb_person_cred_hist_length"].median())
        )
        
        submit_button = st.form_submit_button("Predict")
    
    # Calculate derived feature safely (person_income is always ≥1)
    loan_percent_income = loan_amnt / person_income

    # Build the dictionary and convert to DataFrame
    input_dict = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }
    features = pd.DataFrame(input_dict, index=[0])
    
    # Ensure categorical consistency
    for col in features.select_dtypes(include=["object"]).columns:
        features[col] = features[col].astype("category")
    
    return features, submit_button

input_df, submitted = user_input_features()

# Display the input summary for user confirmation
if submitted:
    st.subheader("Input Summary")
    st.write(input_df)

# Make prediction once the user submits the form
if submitted:
    if model:
        try:
            with st.spinner("Predicting..."):
                prediction = model.predict(input_df)
            st.subheader("Prediction")
            if prediction[0] == 0:
                st.write("Predicted Loan Status: Not Approved")
            elif prediction[0] == 1:
                st.write("Predicted Loan Status: Approved")
            else:
                st.write("Predicted Loan Status:", prediction[0])
        except Exception as e:
            st.error("An error occurred during prediction: " + str(e))
    else:
        st.error("Model could not be loaded.")

# Optional: Expandable section with additional information about the app
with st.expander("About this App"):
    st.write("""
        This Loan Prediction App uses a pre-trained XGBoost model to determine whether a loan is likely to be approved or not.
        The model was trained on historical loan data and leverages features like age, income, home ownership, employment length, 
        and loan-specific details. Use the sidebar to adjust the inputs and click "Predict" to see the outcome.
    """)
