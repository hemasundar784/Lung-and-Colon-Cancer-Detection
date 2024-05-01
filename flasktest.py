from flask import Flask, request, jsonify, render_template
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model (8fullseq)
model = tf.keras.models.load_model("8fullseq.h5", compile=False)

# Load the ResNet50 model
model2 = tf.keras.models.load_model(r"ResnetModel3.h5", compile=False)

# Define the classes for prediction
classes = ["Colon Adenocarcinoma", "Colon Benign Tissue", "Lung Adenocarcinoma", "Lung Benign Tissue", "Lung Squamous Cell Carcinoma"]

# Define a function for preprocessing the image
def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        # Convert the image to RGB
        image = image.convert("RGB")
        # Resize the image to match the model input size
        image = image.resize((224, 224))
        # Normalize the pixel values
        image = np.array(image) / 255.0
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


        image = Image.open(file_path)

        # Preprocess the image
        image_rgb = image.convert("RGB")  # Ensure image is in RGB format
        img = image_rgb.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        # Perform predictions separately for each model
        
        
        # Call prediction functions and obtain results
        
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = classes[predicted_class_index]
        confidence_percentage = predictions[0][predicted_class_index] * 100

        
        predictions2 = model2.predict(img_array)
        predicted_class_index2 = np.argmax(predictions2)
        predicted_class_label2 = classes[predicted_class_index2]
        confidence_percentage2 = predictions2[0][predicted_class_index2] * 100

        # Define function to determine cancer type
        def determine_cancer_type(predicted_class):
            if predicted_class in ["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma" , "Lung Benign Tissue" ]:
                return "Lung Cancer"
            elif predicted_class in ["Colon Adenocarcinoma", "Colon Benign Tissue"]:
                return "Colon Cancer"
            else:
                return "Other"
        
        # Determine cancer type for each model's prediction
        cancer_type_8fullseq = determine_cancer_type(predicted_class_label)
        cancer_type_resnet50 = determine_cancer_type(predicted_class_label2)
        
        # Return results to the template
        return render_template('prediction1.html',
                               image_path=file_path,
                               predicted_class_8fullseq=predicted_class_label,
                               confidence_percentage_8fullseq=confidence_percentage,
                               cancer_type_8fullseq=cancer_type_8fullseq,
                               predicted_class_resnet50=predicted_class_label2,
                               confidence_percentage_resnet50=confidence_percentage2,
                               cancer_type_resnet50=cancer_type_resnet50)
    else:
        return "No file uploaded"


if __name__ == '__main__':
    app.run(debug=True)
