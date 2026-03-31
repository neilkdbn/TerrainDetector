import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

# Load saved model
model = tf.keras.models.load_model("models/final_model.h5")

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train_balanced",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

class_names = val_ds.class_names
print("Classes:", class_names)

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)

    y_pred.extend(preds)
    y_true.extend(labels.numpy())

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# MSE
mse = mean_squared_error(y_true, y_pred)
print("\nMean Squared Error (MSE):", mse)