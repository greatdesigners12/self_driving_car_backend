from flask import Flask, request
import eventlet
import socketio
from io import BytesIO 
import base64
from PIL import Image
import numpy as np
from keras.models import load_model
import cv2
import csv
import numpy as np
import time

sio = socketio.Server()
app = Flask(__name__)
speed_limit = 10

def log_time(start_processing, end_processing, start_predict, end_predict):
    processing_time = end_processing - start_processing
    predict_time = end_predict - start_predict
    csv_path = "connection/logtime.csv"
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([start_processing, end_processing, processing_time, start_predict, end_predict, predict_time])

def img_preprocess(img):
    # image -> (height, width, number of channels)
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    # normalization
    img = img/255
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    speed_limit = 100  # assuming a speed limit value

    start_processing = time.time()
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    end_processing = time.time()

    start_predict = time.time()
    steering_angle = float(model.predict(image))
    end_predict = time.time()

    throttle = 1 - speed / speed_limit
    send_control(steering_angle, throttle)

    log_time(start_processing, end_processing, start_predict, end_predict)

@sio.on('connect')
def connect(sid, environ):
    print("connected")
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data ={
        'steering_angle' : steering_angle.__str__(),
        'throttle' : throttle.__str__()
    })

@app.route('/telemetry', methods=['POST'])
def handle_telemetry():
    data = request.json
    sid = data['sid']
    telemetry(sid, data)

if __name__ == "__main__":
    model = load_model("self_driving_car_newest_1.h5")
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)