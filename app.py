# for various functions related to import
from flask import Flask,request,jsonify,render_template
from __future__ import division, print_function
import pickle
import sys
import os
import glob
import re
import numpy as np
# for image preprocessing
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Flask related functions
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# define app
app = Flask(__name__)
# define file name
MODEL_PATH ='malaria.h5'
# loading trained model
model = load_model(MODEL_PATH)
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(60,60))
    #  for converting image to array and scaling it
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
        preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Person is Infected With Pneumonia"
    else:
        preds="The Person is not Infected With Pneumonia"
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        # Get the file from post request
        f = request.files['file']

        # to save the file to upload
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # for prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None
if __name__ == '__main__':
    app.run(debug=True)