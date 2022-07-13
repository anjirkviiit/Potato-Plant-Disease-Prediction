import numpy as np
from flask import Flask, request, render_template, flash, redirect
import h5py
import os
from tensorflow import keras
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from werkzeug.utils import secure_filename
import warnings
from io import BytesIO
warnings.filterwarnings('ignore')

app = Flask(__name__)

IMAZE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
MODEL_PATH = 'potatoDiseaseModel.h5'
#loading the trained model
model = models.load_model(MODEL_PATH)
    
dataset = image_dataset_from_directory(
    "LeafData",
    shuffle=True,
    image_size = (IMAZE_SIZE,IMAZE_SIZE),
    batch_size = BATCH_SIZE
)

class_names = dataset.class_names
def predictClass(model, img_path):
    img = image.load_img(img_path, target_size=(IMAZE_SIZE,IMAZE_SIZE))
    #preprocessing the image
    imgArray = image.img_to_array(img)   

    #Scaling
    imgArray = imgArray/255
    imgArray = np.expand_dims(imgArray,0)    
    predictions = model.predict(imgArray)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.argmax(predictions[0])), 2)
    return predicted_class, confidence

@app.route('/',methods=['GET'])
def home():
    #Rendering Home Page
    return render_template('input.html')

@app.route("/displayPrediction",methods=['GET','POST'])
def displayPrediction():
    if request.method == 'POST':
        file = request.files['file']
        #Save the file to ./uploads
        rootPath = os.path.dirname(__file__)
        file_path = os.path.join(rootPath, 'uploads', secure_filename(file.filename))
        file.save(file_path)

        #Make Prediction
        predicted_class, confidence = predictClass(model, file_path)
        if predicted_class==0:
            result="Potato plant is infected with Early Blight disease"
        elif predicted_class==1:
            result="Potato plant is infected with Late Blight disease"
        elif predicted_class==2:
            result="Potato plant is Healthy and Fresh"
        else:
            result="Potato plant is Healthy and Fresh"
        return render_template('input.html', predicted_text='Predicted:{}.\n Confidence: {}%'.format(result,confidence))

if __name__=="__main__":
    app.run(debug=True, threaded=True)