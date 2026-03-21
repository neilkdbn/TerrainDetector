import tensorflow as tf
import os

# 1. Load dataset with validation split
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train_balanced",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train_balanced",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

# Print class names (IMPORTANT for weight mapping)
print("Class order:", train_ds.class_names)

# 2. Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# 3. Base model (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# 4. Full model
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4, activation='softmax')
])

# 5. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Correct class weights (based on your dataset)
# ['gravel', 'rock', 'sand', 'smooth']
class_weight = {
    0: 2.0,   # gravel (93)
    1: 1.0,   # rock (150)
    2: 1.0,   # sand (150)
    3: 4.0    # smooth (44)
}

# 7. Train with validation
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight
)

print("\n--- Starting Fine-Tuning ---\n")

# Unfreeze base model
base_model.trainable = True

# Freeze most layers, only train top ones
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with LOW learning rate (CRITICAL)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train again (short training)
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weight
)
os.makedirs("models", exist_ok=True)
model.save("models/best_model.h5")
print("Model saved to models/best_model.h5")