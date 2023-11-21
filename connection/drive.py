from flask import Flask
import eventlet
import socketio
from io import BytesIO 
import base64
from PIL import Image
import numpy as np
from keras.models import load_model
import cv2


sio = socketio.Server()
app = Flask(__name__)
speed_limit = 10

def img_preprocess(img) :
    
    # image -> (height, width, number of channels)
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    # normalization
    img = img/255
    return img

@sio.on('telemetry')
def telemetry(sid, data) :
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1 - speed/speed_limit
    send_control(steering_angle, throttle)

# def img_preprocess(img) :

#     # image -> (height, width, number of channels)
#     img = img[60:135, :, :]
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#     img = cv2.GaussianBlur(img, (3, 3), 0)
#     img = cv2.resize(img, (200, 66))
#     # normalization
#     img = img/255
#     return img

@sio.on('connect') # message, disconect
def connect(sid, environ) :
    print("connected")
    send_control(0, 0)

def send_control(steering_angle, throttle) :
    sio.emit('steer', data ={
        'steering_angle' : steering_angle.__str__(),
        'throttle' : throttle.__str__()
    })

if __name__ == "__main__" :
    model = load_model("self_driving_car_model.h5")
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
