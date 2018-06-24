# 提取人脸,更新标签





import cv2, os
import numpy as np
from PIL import Image
import sys

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
f = open("yml/label.txt", 'w+')
face = []


def prin(face):
    for i in range(len(face)):
        cv2.imshow('img', face[i])
        print(face[i].shape)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, x) for x in os.listdir(path)]
    faceSamples = []
    Ids = []
    count = 1
    Id = 0

    for imagePath in imagePaths:
        #         pilImage=Image.open(imagePath).convert('L')

        #         imageNp=np.array(pilImage,'uint8')

        pilImage = cv2.imread(imagePath, cv2.IMREAD_COLOR)

        imageNp = cv2.cvtColor(pilImage, cv2.COLOR_BGR2GRAY)

        name = os.path.split(imagePath)[-1].split(".")[0]

        #         Id = int(os.path.split(imagePath)[-1].split(".")[0])

        #         Id = os.path.split(imagePath)[-1].split(".")[0]

        Id += 1
        f.write(str(Id) + ':' + name + '\n')  # 更新标签

        faces = detector.detectMultiScale(imageNp, 1.05, 30)



        for (x, y, w, h) in faces:
            cv2.rectangle(pilImage, (x, y), (x + w, y + h), (0, 255, 0), 1)

            faceSamples.append(imageNp[y:y + h, x:x + w])
            face.append(pilImage)

            Ids.append(Id)
            break  # 一张图片提取一张人脸

    return faceSamples, Ids


faces, Ids = getImagesAndLabels('Test_pics')

recognizer.train(faces, np.array(Ids))
recognizer.save('yml/trainner.yml')
prin(face)

f.seek(0, 0)
a = f.readlines()
for i in range(len(a)):
    a[i] = a[i].strip().replace('\n', '')
    a[i] = a[i].split(':')[0]
print(a)

# f.write('gouzi2\ngouzi')

f.close()
