import cv2
import os
import numpy as np

data_path = 'dataset'
people = os.listdir(data_path) #join path -> dataset (for training)

faces = [] # cropped grayscale face images
labels = []
label_map = {}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#here, haarcascade -> pre-trained model

for label_id, person in enumerate(people):
    label_map[label_id] = person
    person_folder = os.path.join(data_path, person)
    
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        faces_rect = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces_rect:
            face = img[y:y+h, x:x+w]
            faces.append(face)
            labels.append(label_id)

# Train the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("face_model.yml")
np.save("label_map.npy", label_map)
