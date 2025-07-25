from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import PlainTextResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tflite_runtime.interpreter as tflite
import requests
import os
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()

app = FastAPI(title="AgroScan – Plant Disease Detector")

# Load TFLite model
interpreter = tflite.Interpreter(model_path="agroscan_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
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

# Twilio setup
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB").resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = output_data[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    if confidence < 0.7:
        return {
            "class": "Uncertain",
            "confidence": confidence
        }

    return {
        "class": predicted_class,
        "confidence": confidence
    }

# Whatsapp Webhook
@app.post("/hook", response_class=PlainTextResponse)
async def whatsapp_hook(request: Request):
    data = await request.form()
    user_msg = data.get("Body", "").strip().lower()

    if "hi" in user_msg or "hello" in user_msg:
        reply = "Welcome to AgroScan! Send me a plant leaf image and I’ll tell you if it’s sick and what to do."
    else:
        reply = "Please upload a clear leaf image so I can analyze it."

    response = MessagingResponse()
    response.message(reply)
    return PlainTextResponse(str(response))

@app.get("/")
def root():
    return {"message": "Welcome to AgroScan FastAPI backend!"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
