import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("terrain_model.h5")

class_names = ['gravel', 'rock', 'sand', 'smooth']

def predict_image(image_path):
    # Load image
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

# Example
predict_image("test5.jpg")