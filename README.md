AgroScan – CNN-Based Plant Disease Detection 
AgroScan is a deep learning-powered solution that detects crop leaf diseases from images. The model is designed to be deployed via a WhatsApp chatbot using Twilio, making it accessible to farmers and agricultural workers in remote or low-connectivity areas.

AgroScan is currently being developed as part of the Africa Deep Tech Challenge 2025.

Project Goals
Build an image classifier that detects plant diseases from leaf photos
Deploy the model through a WhatsApp bot for easy farmer access
Promote early detection and minimize crop loss in agriculture

Model Details
Model: Convolutional Neural Network (CNN)
Trained on: PlantVillage Dataset (Kaggle)
Output: Disease classification (e.g., Early Blight, Late Blight, Healthy)
Evaluation: Accuracy, Confusion Matrix, Precision/Recall

Tech Stack
Python
TensorFlow / Keras
Pandas, NumPy
Twilio API (WhatsApp integration)
FastAPI for hosting

📊 Dataset
The model is trained using the PlantVillage Dataset, a labeled image dataset of healthy and diseased crop leaves.

Dataset is not included in this repo due to size.
