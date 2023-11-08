from ultralytics import YOLO
import cv2
import math
import time
from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    """Vision"""
    return render_template('index.html')

def gen():
    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(3,480)

    # Load our custom bottle detection deep learning model
    model = YOLO("runs/detect/train4/weights/best.pt")

    # Define possible classes
    class_names = ["can", "plastic bottle", "nothing", "nothing"]

    #frames = 0

    while True:
        success, img= cap.read()

        results = model(img, stream=True)
        
        success, jpg = cv2.imencode('.jpg', img)
        frame = jpg.tobytes()

        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port =5000, debug=True, threaded=True)
