import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Model Predictor App", page_icon="🤖", layout="centered"
)


# Load the model with caching to optimize performance
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model


model = load_model()

# Streamlit UI
st.title("🤖 Machine Learning Model Deployment")
st.write(
    "Enter the required feature values below to get a prediction from the trained model."
)

st.divider()

# Input Fields (Adjust features according to your model)
col1, col2 = st.columns(2)

with col1:
    feature_1 = st.number_input(
        "Feature 1 (e.g., Age / Value)", value=0.0, step=0.1
    )
    feature_2 = st.number_input(
        "Feature 2 (e.g., Score / Input)", value=0.0, step=0.1
    )

with col2:
    feature_3 = st.number_input(
        "Feature 3 (e.g., Count / Rate)", value=0.0, step=0.1
    )
    feature_4 = st.number_input(
        "Feature 4 (e.g., Dimension)", value=0.0, step=0.1
    )

# Prediction Logic
if st.button("Generate Prediction", type="primary"):
    # Reshape input into a 2D array expected by the model
    input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])

    try:
        prediction = model.predict(input_data)
        st.success(f"**Prediction Result:** {prediction[0]}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
