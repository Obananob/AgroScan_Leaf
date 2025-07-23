import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image):
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)
