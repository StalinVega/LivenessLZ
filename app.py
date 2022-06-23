from logging import captureWarnings
import cv2

##Especificamos que vamos a realizar un video stremming
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#especificamos el modelo que vamos a usar
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#detector de fotogramas
while True:
    ret, frame = cap.read()
    #tranformacion a escala de grises
    if ret:
        gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces= face_detector.detectMultiScale(gray, 1.3,5)
        #dibujamos un rectangulo en el rostro
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))
            cv2.imshow("Frame",frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()