from flask import Flask, request, jsonify, render_template
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("8fullseq.h5", compile=False)

# Define the classes for prediction
classes = ["Colon Adenocarcinoma", "Colon Benign Tissue", "Lung Adenocarcinoma", "Lung Benign Tissue", "Lung Squamous Cell Carcinoma"]

# Define a function for preprocessing the image
def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        # print(image)
        image = image.convert("RGB")
        image = image.resize((224, 224))  # Resize the image to match the model input size
        image = np.array(image) / 255.0  # Normalize the pixel values
        return image
    except Exception as e:
        print("Error processing image:", e)
        return None

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('front_test.html')

@app.route('/predict', methods=['POST'])
def predict_with_model():
    file = request.files['file']
    if file:
        # Save the uploaded file
        file_path = os.path.join("static", file.filename)
        file.save(file_path)

        # Print the file path to check
        print("Image Path:", file_path)

        image = Image.open(file_path)

        # Preprocess the image
        image_rgb = image.convert("RGB")  # Ensure image is in RGB format
        img = image_rgb.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = classes[predicted_class_index]
        confidence_percentage = predictions[0][predicted_class_index] * 100

        # Check if the predicted class is related to lung or colon cancer
        lung_classes = ["Lung Adenocarcinoma", "Lung Benign Tissue", "Lung Squamous Cell Carcinoma"]
        colon_classes = ["Colon Adenocarcinoma", "Colon Benign Tissue"]

        if predicted_class_label in lung_classes:
            cancer_type = "Lung Cancer"
        elif predicted_class_label in colon_classes:
            cancer_type = "Colon Cancer"
        else:
            cancer_type = "Other"

        # Pass prediction data to the result template
        return render_template('prediction.html',
                               image_path=file_path,
                               predicted_class=predicted_class_label,
                               confidence_percentage=confidence_percentage,
                               cancer_type=cancer_type)
    else:
        return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)
