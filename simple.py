from flask import Flask,render_template,Response
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
import numpy as np



app=Flask(__name__)
camera=cv2.VideoCapture('/Users/danukathaja/Downloads/Flask-Web-Framework-main/Tutorial 7/WhatsApp Video 2022-04-01 at 13.47.05_Trim.mp4')
#address="https://192.168.43.1:8080/video"
#camera.open(address)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

Autoencoder = Autoencoder = tf.keras.models.load_model('autoencoder-model_justCSA5.h5')
#Autoencoder.summary()
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)



def generate_frames():
    list =[]
    while True:

        ret, frame = camera.read()
    
    # Resize image
        img = frame.copy()
    #img = img[680:2000, 400:640]
        h, w, c = img.shape
        print(h,w)
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 640,352)
        input_img = tf.cast(img, dtype=tf.int32)
        #frame = tf.cast(frame,dtype =tf.float32)
        #print(frame.dtype)
    
        # Detection section
        results = movenet(input_img)
        keypoints_with_scores_v = results['output_0'].numpy()[:,0,:51].reshape((1,17,3))
        keypoints_with_scores_A = results['output_0'].numpy()[:,0,:51].reshape(51)
        list.append(keypoints_with_scores_A)
        data = np.array(list)
        a,b = data.shape
        #frame = frame[680:2000, 400:640]

    
        # Render keypoints 
        loop_through_people(frame, keypoints_with_scores_v, EDGES, 0.1)
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        #print('waiting')
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
#@exception_handler
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
#@exception_handler

if __name__=="__main__":
    app.run(debug=True)