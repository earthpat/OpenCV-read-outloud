import cv2
import numpy as numpy
import pyttsx3

eng = pyttsx3.init()


def startSpeech(name):
    eng.say("I see a " + name + " approaching.")
    eng.runAndWait()
    return


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
carCascade = cv2.CascadeClassifier("cars.xml")


cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    cars = carCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "Human", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        startSpeech("human")
       
    for(x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "Car", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        startSpeech("Car")



    cv2.imshow('camera', img)

    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
