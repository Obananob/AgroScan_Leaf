import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from io import BytesIO

# Load your TFLite model
interpreter = tflite.Interpreter(model_path="agroscan_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Target_Spot",
    "Tomato___healthy"
]

def preprocess_image(image):
    image = image.resize((160, 160))  # Your trained size
    image = np.array(image) / 255.0
    image = np.expand_dims(image.astype(np.float32), axis=0)
    return image

def predict(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0]
    return prediction

# Streamlit UI
st.title("🌿 AgroScan - Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf', use_column_width=True)

    with st.spinner("Analyzing..."):
        img = preprocess_image(image)
        prediction = predict(img)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"🧠 Prediction: **{predicted_class}**")
        st.info(f"🔍 Confidence: **{confidence:.2f}**")

        if confidence < 0.7:
            st.warning("Confidence is low. Please upload a clearer image.")
