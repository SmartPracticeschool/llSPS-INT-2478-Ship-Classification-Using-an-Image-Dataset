from __future__ import division, print_function
from flask import Flask, redirect, url_for, request, render_template, jsonify
import json
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
import os


app = Flask(__name__, static_url_path='')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict_classes(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        model = load_model('ship_classification.h5')
        preds = model_predict(file_path, model)
        
        
        ls=["Cargo-Ship","Air-Carrier","Cruise-Ship"]
        print(preds)
        result = ls[preds[0]]
        print(result)
        return result
    return "AbC"


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
