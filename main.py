from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import PlainTextResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
import os
from dotenv import load_dotenv
from twilio.rest import Client
from huggingface_hub import InferenceClient
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()

app = FastAPI(title= "AgroScan – Plant Disease Detector")

# Load model
MODEL = tf.keras.models.load_model("model/agroscan_model.keras")

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

# # Hugging Face setup
# Client = InferenceClient(
#     provider = "featherless-ai",
#     api_key = os.getenv("HF_TOKEN")
# )
# HF_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    # filter inputs with low confidence
    if confidence < 0.7:
        return {
            "class": "Uncertain",
            "confidence": float(confidence),
          #  "treatment": "Please Upload a clear Plant Leaf Image"
        }
   # treatment = get_treatment_recommendation(predicted_class)

    return {
        "class": predicted_class,
        "confidence": float(confidence),
        # "treatment": treatment
    }

# # LLM Integration
# def get_treatment_recommendation(disease: str):
#     prompt = (
#         f"Think carefully. A plant has benn diagnosed with the disease:{disease}."
#     "What is a simple and actionable treatment recommendation?"
#     "Respond in the same language as the disease name"
#     )
#
#     try:
#         completion = Client.chat.completions.create(
#             model=HF_MODEL,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             temperature = 0.6,
#             max_tokens = 200,
#             timeout= 60
#         )
#         return completion.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error from Hugging Face LLM: {str(e)}"
#

# Whatsapp Webhook
@app.post("/hook",
response_class=PlainTextResponse)
async def whatsapp_hook(request: Request):
        data = await request.form()
        user_msg = data.get("Body", "").strip().lower()

        if "hi" in user_msg or "hello" in user_msg:
            reply = "Welcome to AgroScan! Send me a plant leaf image and I’ll tell you if it’s sick and what to do."
        else:
            reply = "Please upload a clear leaf image so I can analyze it."

        response= MessagingResponse()
        response.message(reply)
        return PlainTextResponse(str(response))

@app.get("/")
def root():
    return {"message": "Welcome to AgroScan FastAPI backend!"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)