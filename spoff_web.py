from flask import Flask
from flask import render_template
from flask import Response
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import model_from_json

app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#Cargar face detection model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
     "haarcascade_frontalface_default.xml")

#Cargar el Anti-Spoofing Model graph
json_file = open('antispoofing_models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#Cargar el antispofing model peso
model.load_weights('antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")
def generate():
     while True:
          ret, frame = cap.read()
          if ret:
               gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               faces = face_detector.detectMultiScale(gray, 1.3, 5)
               for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
               (flag, encodedImage) = cv2.imencode(".jpg", frame)
               if not flag:
                    continue
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
     return render_template("index.html")

@app.route("/video_feed")
def video_feed():
     return Response(generate(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
     app.run(debug=False)

cap.release()