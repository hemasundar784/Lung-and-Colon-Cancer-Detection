# -*- coding: utf-8 -*-
"""Copy of 3models of Lung_colon_cancer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FuNvjJDxSKf0y45EWLQ1PgxPsG-t4gO6
"""

'''
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'lung-and-colon-cancer-histopathological-images:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F601280%2F1079953%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240408%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240408T035346Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3De200247809216ceca5230a27434ea499c0dbb961dec23385facddb4e7a6b872e1c5e61f8146829d1b9717785cf805f06ee85b03729270e3935ae98e9fedc2719d4e658e4f52121a22dce3c623cbfbd60787341043712c760238b525851b7fe255a8dfe75b5896e1ebd8dcc0b92228c3132bacfc06be219252197c0cbd755d2eac281dc31dc7b1075a95cd7849a2fba9f9de20fdf8a8081f310487d52b738cc6eeda39c77b4f2e17481bc1900f20de617ac7c945530c945c39ad3d1b71ee25596ef4266025de491b62590ccb6d877ab936ed5bd3f5f5ec6f2d61625b44e672de83db5a2db256579fd6caaa8bf0ca88443d7f6d74c09945a0ea8fafe293557a449'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')
'''

import os
import time
import shutil
import pathlib
import itertools
import PIL as Image

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot  as plt
from sklearn.model_selection  import   train_test_split
from sklearn.metrics import confusion_matrix , classification_report

import tensorflow  as tf
from tensorflow import  keras
from tensorflow.keras.models import  Sequential
\
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam ,Adamax ,AdamW
from tensorflow.keras.layers import Conv2D ,MaxPooling2D,Flatten,Dense ,Activation, Dropout,BatchNormalization
from tensorflow.keras import regularizers
print("perfect")

data_dir="/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set"
filepaths = []
labels = []
folds=os.listdir(data_dir)
for fold in folds:
    foldpath= os.path.join(data_dir, fold)
    flist=os.listdir(foldpath)
    for f in flist:
        f_path=os.path.join(foldpath, f)
        filelist=os.listdir(f_path)
        for file in filelist:
            fpath = os.path.join(f_path, file)
            filepaths.append(fpath)
            if f=="colon_aca" :
                labels.append("Colon Adenocarcinoma")
            elif f == "colon_n":
                labels.append("Colon Benign Tissue")
            elif f=="lung_aca":
                labels.append("Lung Adenocarcinoma")
            elif f == "lung_n":
                labels.append("Lung Benign Tissue")
            elif f=="lung_scc":
                labels.append("Lung Squamous Cell Carcinoma")

Fseries= pd.Series(filepaths ,name= "file_paths")
Lseries =pd.Series(labels ,name ="Labels")
df= pd.concat ([Fseries ,Lseries],axis =1)
#df

"""# split data into train , valid , test"""

strat =df["Labels"]

train_df ,dummy_df = train_test_split(df ,train_size= 0.8,shuffle=True , random_state =123 ,stratify =strat )

strat=dummy_df["Labels"]
valid_df,test_df = train_test_split(dummy_df ,train_size=0.5,shuffle =True , random_state =123,stratify =strat )



"""# Creat data image generator"""

batch_size =64
img_size=(224,224)
channels=3
img_shape=(224 ,224 ,3)

tre_gen =ImageDataGenerator()
ts_gen =ImageDataGenerator()

train_gen = tre_gen.flow_from_dataframe(train_df , x_col = 'file_paths' , y_col = 'Labels' , target_size = img_size ,
                                   class_mode = 'categorical' , color_mode = 'rgb' , shuffle = True , batch_size = 64)
valid_gen = tre_gen.flow_from_dataframe(valid_df , x_col = 'file_paths' , y_col = 'Labels' , target_size = img_size ,
                                   class_mode = 'categorical' , color_mode = 'rgb' , shuffle = True , batch_size = 64)
test_gen= tre_gen.flow_from_dataframe(test_df , x_col = 'file_paths' , y_col = 'Labels' , target_size = img_size ,
                                   class_mode = 'categorical' , color_mode = 'rgb' , shuffle = False , batch_size = 64)

"""# show sample from train data

"""

g_dict =train_gen.class_indices
classes= list(g_dict)
images, labels = next(train_gen)
'''
plt.figure(figsize=(20,20))
for i in range (16) :
        plt.subplot(4,4,i+1)
        image = images[i] /255
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name , color = "blue", fontsize =12 )
        plt.axis("off")
plt.show()  '''

"""# Model Structure
> **generic model creation**
"""

'''
img_size = (224,224)
channels = 3
img_shape = (224,224,3)
class_count = 5

model = tf.keras.Sequential([Conv2D(filters =64 , kernel_size=(3,3) , padding ="same" ,activation="relu" ,input_shape= img_shape ),
Conv2D(filters =64 , kernel_size=(3,3) , padding = "same" , activation ="relu"),
MaxPooling2D((2,2)),

Conv2D(filters =128 , kernel_size=(3,3) , padding ="same" ,activation="relu" ),
Conv2D(filters =128, kernel_size=(3,3) , padding = "same" , activation ="relu"),
Conv2D(filters =128, kernel_size=(3,3) , padding = "same" , activation ="relu"),
MaxPooling2D((2,2)),

Conv2D(filters =256, kernel_size=(3,3) , padding ="same" ,activation="relu" ),
Conv2D(filters =256, kernel_size=(3,3) , padding = "same" , activation ="relu"),
Conv2D(filters =256, kernel_size=(3,3) , padding = "same" , activation ="relu"),
MaxPooling2D((2,2)),

Conv2D(filters =512 , kernel_size=(3,3) , padding ="same" ,activation="relu" ),
Conv2D(filters =512, kernel_size=(3,3) , padding = "same" , activation ="relu"),
Conv2D(filters =512, kernel_size=(3,3) , padding = "same" , activation ="relu"),
MaxPooling2D((2,2)),

Conv2D(filters =512 , kernel_size= (3,3) , padding ="same" ,activation="relu" ),
Conv2D(filters =512, kernel_size=(3,3) , padding = "same" , activation ="relu"),
Conv2D(filters =512, kernel_size=(3,3) , padding = "same" , activation ="relu"),
MaxPooling2D((2,2)),

Flatten(),
Dense(256 ,activation= "relu"),
Dense(64 ,activation="relu"),
Dense(256, activation ="relu"),
Dense(64 , activation = "relu"),
Dense(64,activation ="relu"),
Dense(class_count , activation="softmax")])

model.compile(Adamax(learning_rate=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])  '''

"""# Train Model"""

'''
epochs =20
history = model.fit(x=train_gen ,epochs= epochs , verbose= 1, validation_data = valid_gen ,validation_steps =None , shuffle =False)  '''

"""# Desplay Model Performance"""

'''
tr_acc = history.history["accuracy"]
tr_loss = history.history["loss"]
val_acc = history.history["val_accuracy"]
val_loss = history.history["val_loss"]
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest =val_acc[index_acc]
Epochs = [i+1 for i in range (len(tr_acc))]
loss_label = f"best epoch= {str(index_loss+1)}"
acc_label = f"best epoch={str(index_acc+1)}"

plt.figure(figsize =(20,8))
plt.style.use("fivethirtyeight")

plt.subplot(1,2,1)
plt.plot(Epochs ,tr_loss ,'r',label = "Training Loss")
plt.plot(Epochs , val_loss ,'g',label="Validation Loss")
plt.scatter(index_loss+1 ,val_lowest , s=150 ,c= "blue",label = loss_label)
plt.title("Training and Validation Loss ")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.ylim(0,6)
plt.legend()

plt.subplot(1,2,2)
plt.plot(Epochs ,tr_acc ,'r',label = "Training Accuracy")
plt.plot(Epochs , val_acc ,'g',label="Validation Accuracy")
plt.scatter(index_acc+1 ,acc_highest , s=150 ,c= "blue",label = acc_label)
plt.title("Training and Validation Accuracy ")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.legend()

plt.tight_layout
plt.show()  '''

"""# Evaluate Model

"""

'''
ts_length= len(test_df)
test_batch_size =max(sorted([ts_length//n for n in range (1, ts_length +1)if ts_length%n == 0 and ts_length/n <=80]))
test_steps =ts_length // test_batch_size

train_score =model.evaluate(train_gen ,steps= test_steps , verbose =1 )
valid_score =model.evaluate(valid_gen ,steps= test_steps , verbose =1 )
test_score =model.evaluate(test_gen ,steps= test_steps , verbose =1 )

print("Train Loss:" , train_score[0])
print("Traing Accuracy :",train_score[1])
print("-"*20)

print("Valid Loss:" , valid_score[0])
print("Valid Accuracy :", valid_score[1])
print("-"*20)

print("Test Loss:" , test_score[0])
print("Test Accuracy :",test_score[1])

'''

"""# Get Predictions"""

preds=model.predict_generator(test_gen)
y_pred = np.argmax(preds , axis=1 )

"""# Confusion Matrics and Classification Report"""

'''
g_dict =test_gen.class_indices
classes=list(g_dict.keys())

cm = confusion_matrix(test_gen.classes, y_pred)

plt.figure(figsize=(10,10))
plt.imshow(cm,interpolation="nearest",cmap = plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks =np.arange(len(classes))
plt.xticks(tick_marks, classes ,rotation =45)
plt.yticks(tick_marks ,classes)

thresh =cm.max()/2
for i ,j in itertools.product (range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j,i,cm[i,j],horizontalalignment = "center",color = "white" if cm[i,j]>thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("predicted Label")

    plt.show
    '''

'''
print(classification_report(test_gen.classes ,y_pred , target_names=classes))
'''

"""# Save Model

"""

'''
model.save("/content/drive/MyDrive/Model.5h")
print("perfect") '''

"""#**2nd model** **vgg**"""

'''
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
import matplotlib.pyplot as plt

# Define constants
img_shape = (224, 224, 3)
class_count = 5
epochs = 5

# Define the input layer
inputs = Input(shape=img_shape)

# Define the convolutional layers
conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(conv1)
pool1 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(pool1)
conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(conv3)
pool2 = MaxPooling2D((2, 2))(conv4)

conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(pool2)
conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv5)
conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv6)
pool3 = MaxPooling2D((2, 2))(conv7)

conv8 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool3)
conv9 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv8)
conv10 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv9)
pool4 = MaxPooling2D((2, 2))(conv10)

conv11 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool4)
conv12 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv11)
conv13 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv12)
pool5 = MaxPooling2D((2, 2))(conv13)

# Flatten layer
flatten = Flatten()(pool5)

# Dense layers
dense1 = Dense(256, activation="relu")(flatten)
dense2 = Dense(64, activation="relu")(dense1)
dense3 = Dense(256, activation="relu")(dense2)
dense4 = Dense(64, activation="relu")(dense3)
dense5 = Dense(64, activation="relu")(dense4)

# Output layer
outputs = Dense(class_count, activation="softmax")(dense5)

# Create the model
model2 = Model(inputs=inputs, outputs=outputs)

# Compile the model
model2.compile(optimizer=Adamax(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
history = model2.fit(train_gen, epochs=epochs, verbose=1, validation_data=valid_gen)

# Visualize training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Performance')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''

'''
#2nd evalute model
ts_length= len(test_df)
test_batch_size =max(sorted([ts_length//n for n in range (1, ts_length +1)if ts_length%n == 0 and ts_length/n <=80]))
test_steps =ts_length // test_batch_size

train_score =model2.evaluate(train_gen ,steps= test_steps , verbose =1 )
valid_score =model2.evaluate(valid_gen ,steps= test_steps , verbose =1 )
test_score =model2.evaluate(test_gen ,steps= test_steps , verbose =1 )

print("Train Loss:" , train_score[0])
print("Traing Accuracy :",train_score[1])
print("-"*20)

print("Valid Loss:" , valid_score[0])
print("Valid Accuracy :", valid_score[1])
print("-"*20)

print("Test Loss:" , test_score[0])
print("Test Accuracy :",test_score[1])
'''

##metrices2

  from sklearn.metrics import classification_report
  g_dict =test_gen.class_indices
  classes=list(g_dict.keys())
'''
  cm2 = confusion_matrix(test_gen.classes, y_pred)

  plt.figure(figsize=(10,10))
  plt.imshow(cm2,interpolation="nearest",cmap = plt.cm.Blues)
  plt.title("Confusion Matrix")
  plt.colorbar()
  tick_marks =np.arange(len(classes))
  plt.xticks(tick_marks, classes ,rotation =45)
  plt.yticks(tick_marks ,classes)

  thresh =cm2.max()/2
  for i ,j in itertools.product (range(cm2.shape[0]),range(cm2.shape[1])):
      plt.text(j,i,cm2[i,j],horizontalalignment = "center",color = "white" if cm2[i,j]>thresh else "black")

      plt.tight_layout()
      plt.ylabel("True Label")
      plt.xlabel("predicted Label")

      plt.show

  # Evaluate the model on the test generator
  #prediction
  preds=model2.predict(test_gen)
  y_pred = np.argmax(preds , axis=1 )

  # Print the classification report
  report = classification_report(test_gen.classes, y_pred, target_names=classes)

  # Print the classification report in the specified format
  print(report)
'''

'''
model2.save("/content/drive/MyDrive/vggModel2.5h")
print("perfect")
'''



"""#**3rd model** **resnet50**"""

from tensorflow.keras.applications import ResNet50

# Constants
img_size = (224, 224)
channels = 3
class_count = 5
epochs = 6
batch_size = 36  # Adjust according to your batch size

# Calculate validation steps
validation_steps = len(valid_gen) // batch_size

# Load pre-trained ResNet50 model without top (fully connected layers)
model3 = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], channels))

# Freeze the layers of ResNet50 (optional)
for layer in model3.layers:
    layer.trainable = False

# Flatten the output of the ResNet50 model
flatten = Flatten()(model3.output)

# Add dense layers for classification
dense1 = Dense(256, activation='relu')(flatten)
dense2 = Dense(64, activation='relu')(dense1)
output = Dense(class_count, activation='softmax')(dense2)

# Create the model
model3 = tf.keras.Model(inputs=model3.input, outputs=output)

# Compile the model
model3.compile(optimizer=Adamax(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with explicit validation_steps
history = model3.fit(train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, validation_steps=validation_steps)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Performance')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

'''
from google.colab import drive
drive.mount('/content/drive')
'''

'''
#3rd evalute model
ts_length= len(test_df)
test_batch_size =max(sorted([ts_length//n for n in range (1, ts_length +1)if ts_length%n == 0 and ts_length/n <=80]))
test_steps =ts_length // test_batch_size

train_score =model3.evaluate(train_gen ,steps= test_steps , verbose =1 )
valid_score =model3.evaluate(valid_gen ,steps= test_steps , verbose =1 )
test_score =model3.evaluate(test_gen ,steps= test_steps , verbose =1 )

print("Train Loss:" , train_score[0])
print("Traing Accuracy :",train_score[1])
print("-"*20)

print("Valid Loss:" , valid_score[0])
print("Valid Accuracy :", valid_score[1])
print("-"*20)

print("Test Loss:" , test_score[0])
print("Test Accuracy :",test_score[1])

'''

'''
##metrices3
preds = model3.predict(test_gen)
y_pred = np.argmax(preds , axis=1 )

from sklearn.metrics import classification_report
g_dict =test_gen.class_indices
classes=list(g_dict.keys())

cm3 = confusion_matrix(test_gen.classes, y_pred)

plt.figure(figsize=(10,10))
plt.imshow(cm3,interpolation="nearest",cmap = plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks =np.arange(len(classes))
plt.xticks(tick_marks, classes ,rotation =45)
plt.yticks(tick_marks ,classes)

thresh =cm3.max()/2
for i ,j in itertools.product (range(cm3.shape[0]),range(cm3.shape[1])):
    plt.text(j,i,cm3[i,j],horizontalalignment = "center",color = "white" if cm3[i,j]>thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("predicted Label")

    plt.show

# Evaluate the model on the test generator
# Print the classification report
report = classification_report(test_gen.classes, y_pred, target_names=classes)

# Print the classification report in the specified format
print(report)
'''

'''
model3.save("/content/drive/MyDrive/ResnetModel3.5h")
print("perfect")
'''

"""# Prediction using loaded_model

from google.colab import drive
drive.mount('/content/drive')
"""

'''loaded_model = tf.keras.models.load_model("/content/drive/MyDrive/Lungcolon/seqmodel" ,compile=False )

loaded_model.compile (Adamax(learning_rate=0.0001), loss ="categorical_crossentropy", metrics= ["accuracy"])

print("perfect")'''

'''
from PIL import Image

image_path = r"/content/drive/MyDrive/Colab Notebooks/colonca72.jpeg"
image = Image.open(image_path)
img = image.resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = loaded_model.predict(img_array)                                               #1st model test result
class_labels = classes
score = tf.nn.softmax(predictions[0])

# Get the predicted class index and the corresponding label
predicted_class_index = tf.argmax(score)
predicted_class_label = class_labels[predicted_class_index]

# Get the confidence score (percentage)
confidence_percentage = score[predicted_class_index] * 100

print(f"Predicted Class: {predicted_class_label}")
print(f"Confidence: {confidence_percentage:.2f}%")

'''

'''
from PIL import Image

image_path = r"/content/drive/MyDrive/Colab Notebooks/lungn23.jpeg"
image = Image.open(image_path)
img = image.resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = loaded_model.predict(img_array)
class_labels = classes
score = tf.nn.softmax(predictions[0])
print("for model1",f"{class_labels[tf.argmax(score)]}")
'''

loaded_model3 = tf.keras.models.load_model(r"/content/drive/MyDrive/Lungcolon/keravgg16model.keras",
                                           compile=False)
loaded_model3.compile(Adamax(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

def predict_with_model(image_path, model, classes):
    image = Image.open(image_path)
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = classes[predicted_class_index]
    confidence_percentage = predictions[0][predicted_class_index] * 100

    print(f"Predicted Class: {predicted_class_label}")
    print(f"Confidence: {confidence_percentage:.2f}%")


# Perform Prediction
image_path = r"/content/drive/MyDrive/Lungcolon/lungscc23.jpeg"
predict_with_model(image_path, loaded_model3, classes)

'''
from PIL import Image
loaded_model3 = tf.keras.models.load_model(r"/content/drive/MyDrive/keraseqmodel.keras",
                                           compile=False)
loaded_model3.compile(Adamax(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

def predict_with_model(image_path, model, classes):
    image = Image.open(image_path)
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = classes[predicted_class_index]
    confidence_percentage = predictions[0][predicted_class_index] * 100

    print(f"Predicted Class: {predicted_class_label}")
    print(f"Confidence: {confidence_percentage:.2f}%")


# Perform Prediction
image_path = r"/content/drive/MyDrive/Colab Notebooks/Lung and Colon cancer /lc25k/lung_colon_image_set/colon_image_sets/colon_aca/colonca4950.jpeg"
predict_with_model(image_path, loaded_model3, classes)
'''

'''
####******
loaded_model3 = tf.keras.models.load_model(r"/content/drive/MyDrive/Lungcolon/keraResnetmodel.keras",
                                           compile=False)
loaded_model3.compile(Adamax(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

def predict_with_model(image_path, model, classes):
    image = Image.open(image_path)
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = classes[predicted_class_index]
    confidence_percentage = predictions[0][predicted_class_index] * 100

    print(f"Predicted Class: {predicted_class_label}")
    print(f"Confidence: {confidence_percentage:.2f}%")


# Perform Prediction
image_path = r"/content/drive/MyDrive/Lungcolon/lungn23.jpeg"
predict_with_model(image_path, loaded_model3, classes)  '''

'''
#### Perform Prediction
image_path = r"/content/drive/MyDrive/Lungcolon/lungscc23.jpeg"
predict_with_model(image_path, loaded_model3, classes)  '''

'''
def predict_with_model(image_path, model, classes):
    # Load the image
    image = Image.open(image_path)
    img = image.resize((224, 224))
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
image_path = r"/content/drive/MyDrive/Lungcolon/lungscc23.jpeg"
predict_with_model(image_path, loaded_model3, classes)

'''

'''
def predict_with_model(image_path, model, classes):
    # Load the image
    image = Image.open(image_path)
    # Convert the image to RGB format
    image = image.convert("RGB")
    img = image.resize((224, 224))
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
        print("The uploaded image does not appear to be a histopathological image of lung or colon cancer. Please upload the correct image.")
image_path = r"/content/drive/MyDrive/My photos/FB_IMG_1560788861566.jpg"
predict_with_model(image_path, loaded_model3, classes)
'''

'''
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
image_path = r"/content/drive/MyDrive/Colab Notebooks/test/xray_dataset_covid19/test/COVID/SARS-10.1148rg.242035193-g04mr34g0-Fig8c-day10.jpeg"
predict_with_model(image_path, loaded_model3, classes) '''





'''
from flask import Flask, render_template, request
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__, template_folder=(r"/content/drive/MyDrive/Lungcolon/templates"))

# Load the pre-trained model
model = tf.keras.models.load_model(r"/content/drive/MyDrive/Lungcolon/ResnetModel3.5h", compile=False)
model.compile(tf.keras.optimizers.Adamax(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Define image size
img_size = (224, 224)

# Define classes
classes = ['Colon Adenocarcinoma', 'Colon Benign Tissue', 'Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction_text="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', prediction_text="No selected file")

        if file:
            # Read the image
            image = Image.open(file)
            img = image.resize(img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, 0)

            # Predict
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            # Get the predicted class index and label
            predicted_class_index = tf.argmax(score)
            predicted_class_label = classes[predicted_class_index]

            # Get confidence percentage
            confidence_percentage = score[predicted_class_index] * 100

            return render_template('index.html', prediction_text=f"Predicted Class: {predicted_class_label}, Confidence: {confidence_percentage:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)
'''



"""#Gradio"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the pre-trained model and classes
loaded_model3 = tf.keras.models.load_model("/content/drive/MyDrive/keraseqmodel.keras", compile=False)
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

