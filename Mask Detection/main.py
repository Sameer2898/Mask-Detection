from cv2 import cv2
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('modle.h5')
img_width, img_height = 200, 200

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
img_count_full = 0

font = cv2.FONT_HERSHEY_SIMPLEX

org = (1, 1)
class_label = ' '
font_scale = 1 #0.5
color = (255, 0, 0)
thickness = 2 #1

while True:
    img_count_full += 1
    response, color_img = cap.read()
    if response == False:
        break
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 6)
    img_count = 0
    for (x, y, w, h) in faces:
        org = (x - 10, y - 10)
        img_count += 1
        color_face = color_img[y:y+h, x:x+w]
        cv2.imwrite('faces/input/%d%dface.jpg' %(img_count_full, img_count), color_face)
        img = load_img('faces/input/%d%dface.jpg'%(img_count_full, img_count), target_size=(img_width, img_height))
        img = img_to_array(img)/255
        img = np.expand_dims(img, axis=0)
        pred_prob = model.predict(img)
        pred = np.argmax(pred_prob)
        if pred == 0:
            print('Wearing mask - ', pred_prob[0][0])
            class_label = 'Mask'
            color = (255, 0, 0)
            cv2.imwrite('faces/with_mask/%d%dface.jpg'%(img_count_full, img_count), color_face)
        else:
            print('Now Wearing Mask - ', pred[0][1])
            class_label = 'No Mask'
            color = (0, 255, 0)
            cv2.imwrite('faces/without_mask/%d%dface.jpg'%(img_count_full, img_count), color_face)
        
        cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(color_img, class_label, org ,font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Mask Detector', color_img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()