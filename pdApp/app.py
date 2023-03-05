from flask import Flask, request, jsonify
import pickledb 
import requests
import os
import tensorflow as tf
import numpy as np
import cv2
import logging
import base64
from paddymodels.checker import Predictor
from PIL import Image
from io import BytesIO

global isPaddy
global ClassPredictor

isPaddy = None
ClassPredictor = None

def getModels(url,modelName = "model.h5"):
    request = requests.get(url)
    if(request.status_code==requests.codes.ok):
        model = request.content
        file=open("MLmodels/"+modelName,"wb")
        file.write(request.content)
        file.close()
        return model
    else:
        print(request.status_code)
        print('modelRetrivalFailed')

app = Flask(__name__)
db = pickledb.load('example.db', False)
db.set("Count",0)


@app.route('/')
def home():
    return 'Hello, paddydisease'


@app.route('/image/<id>')
def getimage(id):
    imageuri = db.get(""+str(id))
    print(imageuri)
    if(imageuri):
        return '<img src="data:image/jpeg;base64,' + imageuri + '">'
    else:
        return jsonify({'error':"No image found"})

@app.route('/check', methods=['POST'])
def checkkk():
    return {'mil':'djhsbd'}

@app.route('/predict', methods=['POST'])
def predict():
    print(os.curdir)
    input_json = request.get_json(force=True)
    uri = input_json['imguri']
    count = int(db.get("Count"))
    if(count == False):
        count = 0
    count += 1
    db.set("" + str(count),uri)
    db.set("Count",count)
    binary_data = base64.b64decode(uri) 
    # print(binary_data)
    img = Image.open(BytesIO(binary_data))
    # Save the image to a file
    img.save(f"images/image{count}.jpg")
    pred= Predictor(tf.keras.models.load_model('MLmodels/isPaddyorNot.h5'),tf.keras.models.load_model('MLmodels/ClassPredictor.h5'))
    result = pred.Predict(f'images/image{count}.jpg')
    return jsonify(result)
    # return jsonify({'image':"image/"+str(count)})


@app.route('/loadmodels/<modelName>')
def getmodels(modelName):
    global isPaddy
    global ClassPredictor
    if(modelName.lower()=='ispaddyornot' or modelName == "1" or modelName == "all"):
        
        url="https://gitlab.com/paddydisease/paddydiseasefinder/-/raw/dd9103394c7833a385a3c130da3a320c75c54c24/Models/isPaddyorNot.h5"
        isPaddy = getModels(url,modelName="isPaddyorNot.h5")
    if(modelName.lower=='classpredictor' or modelName == "2" or modelName == "all"):
        url="https://gitlab.com/paddydisease/paddydiseasefinder/-/raw/dd9103394c7833a385a3c130da3a320c75c54c24/Models/ClassPredictor.h5"
        ClassPredictor = getModels(url,modelName="ClassPredictor.h5")
    print(os.listdir())
    return jsonify(os.listdir())

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# if __name__ == '__main__':
# 	app.run(host="0.0.0.0", port=int("5000"), debug=True)