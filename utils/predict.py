import numpy as np
from tensorflow.keras.models import load_model

CLASS_NAMES = ['gravel', 'rock', 'sand', 'smooth']

def load_trained_model(model_path):
    return load_model(model_path)

def predict(model, image):
    preds = model.predict(image, verbose=0)[0]

    print("\nRaw Probabilities:")
    for i, p in enumerate(preds):
        print(f"{CLASS_NAMES[i]}: {p*100:.2f}%")

    class_index = np.argmax(preds)
    confidence = preds[class_index]

    return CLASS_NAMES[class_index], confidence