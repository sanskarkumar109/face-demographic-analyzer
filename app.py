import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = load_model("face_multi_output_model.keras")

# Labels
age_labels = ["Child", "Teen", "Adult", "Middle", "Senior"]
ethnicity_labels = ["White", "Black", "Asian", "Indian", "Other"]

st.title("Face Demographic Analyzer")
st.write("Upload an image to predict Age Group, Gender, and Ethnicity")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image
    img = np.array(image)
    img = cv2.resize(img, (96, 96))
    img = img.astype("float32")
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Predict
    gender_pred, age_pred, eth_pred = model.predict(img)

    gender = "Male" if gender_pred[0][0] < 0.5 else "Female"
    age_group = age_labels[np.argmax(age_pred)]
    ethnicity = ethnicity_labels[np.argmax(eth_pred)]

    st.subheader("Prediction Results")
    st.write("Gender:", gender)
    st.write("Age Group:", age_group)
    st.write("Ethnicity:", ethnicity)