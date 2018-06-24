# 识别人脸

import cv2, os
import numpy as np
from PIL import Image
import sys

font = cv2.FONT_HERSHEY_SIMPLEX

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('yml/trainner.yml')

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face = []
f = open("yml/label.txt", 'r')
label = f.readlines()
margin = 15
path = ""

for i in range(len(label)):
    label[i] = label[i].strip().replace('\n', '')


def prin(face):
    for i in range(len(face)):
        cv2.imshow('img', face[i])
        #         cv2.imwrite(path +  "out_pics\\" + str(i)+ ".jpg",face[i])#储存识别集

        #         print(face[i].shape)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def getImagesAndLabels(path):
    faceSamples = []

    cap = cv2.VideoCapture(0)
    while (1):

        ret, pilImage = cap.read()
        if ret == True:

            imageNp = cv2.cvtColor(pilImage, cv2.COLOR_BGR2GRAY)
            # print(imageNp)

            faces = detector.detectMultiScale(imageNp, 1.05, 5)
            # cv2.imshow('img', imageNp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print(len(faces))
            
            for (x, y, w, h) in faces:

                #             faceSamples.append(imageNp[y:y+h,x:x+w])

                #             Ids.append(Id)
                
                # print("OKOK")

                Id, confidence = recognizer.predict(imageNp[y:y + h, x:x + w])

                cv2.rectangle(pilImage, (x, y), (x + w, y + h), (0, 255, 0), 1)

                the_id = label[Id - 1].split(':')[1]
                #             print(Id)

                if confidence > 50:
                    cv2.putText(pilImage, the_id, (x, y + margin), font, 0.5, (255, 255, 255), 2)
                else:
                    cv2.putText(pilImage, 'ID', (x, y + margin), font, 0.5, (255, 255, 255), 2)

                faceSamples.append(pilImage)

                #             break#提取一张人脸
            break

    return faceSamples


faces = getImagesAndLabels('recognise')


# def remove_file(path):
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     for imagePath in imagePaths:
#         Id = os.path.split(imagePath)[1]
#         x = path + "\\" + Id
#         os.remove(x)


# remove_file(path + 'out_pics')


prin(faces)

f.close()
