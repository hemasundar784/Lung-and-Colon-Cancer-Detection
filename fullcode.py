
import pathlib
import itertools
from PIL import Image
import shutil

import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import all of TensorFlow
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.optimizers import Adam, Adamax, AdamW
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras import regularizers

# Set random seed for reproducibility
np.random.seed(123)
tf.random.set_seed(123)

# Data directory
data_dir =r"C:\Users\KOLIPAKA HEMASUNDAR\Desktop\lung_colon_image_set"

# Filepaths and labels
filepaths = []
labels = []
folds = os.listdir(data_dir)

# Limit for taking images from each category
image_limit = 2500

for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    flist = os.listdir(foldpath)
    for f in flist:
        f_path = os.path.join(foldpath, f)
        filelist = os.listdir(f_path)[:image_limit]  # Limiting images
        for file in filelist:
            fpath = os.path.join(f_path, file)
            filepaths.append(fpath)
            if f == "colon_aca":
                labels.append("Colon Adenocarcinoma")
            elif f == "colon_n":
                labels.append("Colon Benign Tissue")
            elif f == "lung_aca":
                labels.append("Lung Adenocarcinoma")
            elif f == "lung_n":
                labels.append("Lung Benign Tissue")
            elif f == "lung_scc":
                labels.append("Lung Squamous Cell Carcinoma")

# Create DataFrame
Fseries = pd.Series(filepaths, name="file_paths")
Lseries = pd.Series(labels, name="Labels")
df = pd.concat([Fseries, Lseries], axis=1)


import pandas as pd

# Load the split dataframes from pickle files
train_df = pd.read_pickle(r"C:\Users\KOLIPAKA HEMASUNDAR\Desktop\fullcode\train_df.pkl")
valid_df = pd.read_pickle(r"C:\Users\KOLIPAKA HEMASUNDAR\Desktop\fullcode\valid_df.pkl")
test_df = pd.read_pickle(r"C:\Users\KOLIPAKA HEMASUNDAR\Desktop\fullcode\test_df.pkl")



# Image preprocessing function
def preprocess_image(image_path, img_size):
    # Read image using OpenCV
    img = cv2.imread(image_path)
    # Resize image
    img = cv2.resize(img, img_size)
    # Convert image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Median filtering
    img = cv2.medianBlur(img, 3)
    # Normalize image
    img = img / 255.0
    return img

# Image generators
batch_size = 64
img_size = (224, 224)






    
def move_images(df, target_dir):
    for _, row in df.iterrows():
        src = row['file_paths']
        dst = os.path.join(target_dir, row['Labels'], os.path.basename(src))
        shutil.copy(src, dst) 
# Define the target directory where you want to copy the images for the training set
target_train_dir = r"C:\Users\KOLIPAKA HEMASUNDAR\Desktop\fullcode"

# Call the move_images function with the training DataFrame and the target directory
move_images(train_df, target_train_dir)

# Move images for train, validation, and test sets
move_images(train_df, train_df)
move_images(valid_df, valid_df)
move_images(test_df, test_df) 


# Now, you can use image_dataset_from_directory
train_gen = tf.keras.utils.image_dataset_from_directory(
    train_df,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

valid_gen = tf.keras.utils.image_dataset_from_directory(
    valid_df,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

test_gen = tf.keras.utils.image_dataset_from_directory(
    test_df,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)
# Get class names
classes = train_gen.class_names

# Create subdirectories for each class
for label in classes:
    os.makedirs(os.path.join(train_df, label), exist_ok=True)
    os.makedirs(os.path.join(valid_df, label), exist_ok=True)
    os.makedirs(os.path.join(test_df, label), exist_ok=True)

#base_model.save("/content/drive/MyDrive/fullcode/mobilenet_basemodel")

import tensorflow as tf

# Load the saved MobileNetV2 base model
loaded_base_model = tf.keras.models.load_model(r"C:\Users\KOLIPAKA HEMASUNDAR\Desktop\fullcode\mobilenet_basemodel")

# Ensure that the loaded base model is not trainable
loaded_base_model.trainable = False


g_dict =test_gen.class_indices
classes=list(g_dict.keys())
# CNN Model
model = Sequential([
    loaded_base_model,
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])





#Get Predictions

preds=model.predict_generator(test_gen)
y_pred = np.argmax(preds , axis=1 )

#Confusion Matrics and Classification Report



#print(classification_report(test_gen.classes ,y_pred , target_names=classes))



model= tf.keras.models.load_model(r"C:\Users\KOLIPAKA HEMASUNDAR\Desktop\fullcode\ffullseq.h5",

                                           compile=False)
model.compile(Adamax(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

def predict_with_model(image_path, model, classes):
    # Load the image
    try:
        image = Image.open(image_path)
        image_rgb = image.convert("RGB")
    except Exception as e:
        print("Error:", e)
        return

    # Check if the image is in RGB format
    if image_rgb.mode != "RGB":
        print("The uploaded image does not appear to be in the RGB format.")
        return

    # Resize the image
    img = image_rgb.resize((224, 224))

    # Convert the image to array and expand dimensions
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = classes[predicted_class_index]
    confidence_percentage = predictions[0][predicted_class_index] * 100

    # Print the prediction results
    print(f"Predicted Class: {predicted_class_label}")
    print(f"Confidence: {confidence_percentage:.2f}%")

    # Check if the predicted class is related to lung or colon cancer
    lung_classes = ["Lung Adenocarcinoma", "Lung Benign Tissue", "Lung Squamous Cell Carcinoma"]
    colon_classes = ["Colon Adenocarcinoma", "Colon Benign Tissue"]

    if predicted_class_label in lung_classes:
        print("The uploaded image is related to lung cancer.")
    elif predicted_class_label in colon_classes:
        print("The uploaded image is related to colon cancer.")
    else:
        print("The uploaded image does not appear to be a histopathological image of lung or colon cancer.")

image_path = r"/content/drive/MyDrive/Colab Notebooks/Lung and Colon cancer /lc25k/lung_colon_image_set/colon_image_sets/colon_aca/colonca4949.jpeg"
predict_with_model(image_path, model, classes)









"""#**Gradio**"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the pre-trained model and classes
loaded_model3 = tf.keras.models.load_model(r"C:\Users\KOLIPAKA HEMASUNDAR\Desktop\fullcode\ffullseq.h5", compile=False)
loaded_model3.compile(tf.keras.optimizers.Adamax(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Define the predict_with_model function
def predict_with_model(image):
    # Convert the image data to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Load the image from bytes
    try:
        image = Image.open(image_bytes)
    except Exception as e:
        return "Error: " + str(e)

    # Preprocess the image
    image_rgb = image.convert("RGB")  # Ensure image is in RGB format
    img = image_rgb.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Predict the class
    predictions =loaded_model3.predict(img_array)
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

    return f"Predicted Class: {predicted_class_label}, Confidence: {confidence_percentage:.2f}%, Cancer Type: {cancer_type}"

# Create the Gradio interface
image_input = gr.Image(label="Upload Image", type="pil")
output_text = gr.Textbox(label="Prediction")

gr.Interface(fn=predict_with_model, inputs=image_input, outputs=output_text, title="Histopathological Image Classifier").launch()