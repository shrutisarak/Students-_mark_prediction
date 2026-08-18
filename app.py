import pickle
import numpy as np
import streamlit as st

# १. पेज सेटअप
st.set_page_config(
    page_title="Naive Bayes Prediction App", page_icon="🤖", layout="centered"
)


# २. `naive_model.pkl` लोड करणारा फंक्शन
@st.cache_resource
def load_model():
    with open("naive_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model


# मॉडेल लोड करा
model = load_model()

# ३. अ‍ॅप टायटल
st.title("🤖 Naive Bayes Model Deployment")
st.write("खाली दिलेल्या इनपुट फिल्ड्समध्ये व्हॅल्यू भरा:")

st.divider()

# ४. इनपुट्स
col1, col2 = st.columns(2)

with col1:
    feature_1 = st.number_input("Feature 1", value=0.0, step=0.1)
    feature_2 = st.number_input("Feature 2", value=0.0, step=0.1)

with col2:
    feature_3 = st.number_input("Feature 3", value=0.0, step=0.1)
    feature_4 = st.number_input("Feature 4", value=0.0, step=0.1)

# ५. प्रेडिक्शन बटण
if st.button("Predict", type="primary"):
    input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])

    try:
        prediction = model.predict(input_data)
        st.success(f"**Prediction Result:** {prediction[0]}")
    except Exception as e:
        st.error(f"एरर: {e}")
        
