import os
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(R"C:\Users\KOLIPAKA HEMASUNDAR\Desktop\fullcode\8fullseq.h5", compile=False)

# Load the base model
loaded_base_model = tf.keras.models.load_model(R"C:\Users\KOLIPAKA HEMASUNDAR\Desktop\fullcode\mobilenet_basemodel")

# Ensure that the loaded base model is not trainable
loaded_base_model.trainable = False

# Define the classes for prediction
classes = ["Colon Adenocarcinoma", "Colon Benign Tissue", "Lung Adenocarcinoma", "Lung Benign Tissue", "Lung Squamous Cell Carcinoma"]

# Define a function for preprocessing the image
def preprocess_image(image):
    try:
        image = image.convert("RGB")
        image = image.resize((224, 224))  # Resize the image to match the model input size
        image = np.array(image) / 255.0  # Normalize the pixel values
        return image
    except Exception as e:
        print("Error processing image:", e)
        return None

# Define a function for making predictions with the model
def predict_with_model(image):
    # Preprocess the image
    image = preprocess_image(image)
    if image is None:
        return {'error': 'Error processing image'}

    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = classes[predicted_class_index]
    confidence_percentage = prediction[0][predicted_class_index] * 100

    # Return the prediction results
    return {
        'predicted_class': predicted_class_label,
        'confidence_percentage': confidence_percentage
    }
