# 提取人脸,更新标签





import cv2, os
import numpy as np
from PIL import Image
import sys

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
f = open("yml/label.txt", 'w+')
face = []
font = cv2.FONT_HERSHEY_SIMPLEX
margin = 15
pilImage2 = []

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

        faces = detector.detectMultiScale(imageNp, 1.03, 10)



        for (x, y, w, h) in faces:
            cv2.rectangle(pilImage, (x, y), (x + w, y + h), (0, 255, 0), 1)

            faceSamples.append(imageNp[y:y + h, x:x + w])
            # face.append(pilImage)

            Ids.append(Id)
            break  # 一张图片提取一张人脸

    return faceSamples, Ids

def get_train():
    # faceSamples = []
    a = input("请输入名称")
    cap = cv2.VideoCapture(0)
    while (1):
        ret, pilImage = cap.read()
        pilImage2 = pilImage
        if ret == True:

            imageNp = cv2.cvtColor(pilImage, cv2.COLOR_BGR2GRAY)
            # print(imageNp)

            faces = detector.detectMultiScale(imageNp, 1.05, 10)
            print(len(faces))


            for (x, y, w, h) in faces:

                #             faceSamples.append(imageNp[y:y+h,x:x+w])

                #             Ids.append(Id)

                # Id, confidence = recognizer.predict(imageNp[y:y + h, x:x + w])

                # cv2.rectangle(pilImage, (x, y), (x + w, y + h), (0, 255, 0), 1)

                # the_id = label[Id - 1].split(':')[1]
                #             print(Id)

                cv2.putText(pilImage2, a, (0, 0 + margin), font, 0.5, (255, 255, 255), 2)
                cv2.rectangle(pilImage2, (x, y), (x + w, y + h), (0, 255, 0), 1)

                face.append(pilImage2)

                # cv2.imshow('img', pilImage2)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                cv2.imwrite("Test_pics\\"+a+".jpg", pilImage2)
                # faceSamples.append(pilImage)

                #             break#提取一张人脸
            break


get_train()

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
