import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime


path='/home/abhinav/Desktop/computer vision/Project_attendance/images'

images=[]       
class_names=[]

mylist=os.listdir(path)
# print(mylist)

for cl in mylist:
    currentimage=cv2.imread(f'{path}/{cl}') # reading the images one by one and store it to the current_image

    images.append(currentimage)             # append only omages to the images list

    class_names.append(os.path.splitext(cl)[0])
# print(class_names)
# print(images)


def findencodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist
encodelistknownfaces=findencodings(images)
print('encoding completed......')
# print(encodelistknownfaces)


def markattendence(name):
    with open('/home/abhinav/Desktop/computer vision/Project_attendance/Attendence.csv','r+') as f:
        myDatalist = f.readlines()
        # print(myDatalist)
        namelist=[]
        for line in myDatalist:
            entry=line.split(',')
            namelist.append(entry[0])

        if name not in namelist:
            now=datetime.now()
            dt=now.strftime('%d-%m-%y')
            time=now.strftime('%I,%M,%P')
            cv2.putText(img,'attendance marked',(x1+2,y1-15),cv2.FONT_HERSHEY_COMPLEX,.5,(0,0,255),1)
            f.writelines(f'\n{name},{time},{dt}')



cm=cv2.VideoCapture(0)
while True: 
    sucess,img=cm.read()
    image_small=cv2.resize(img,(0,0),None,0.25,0.25)
    image_small=cv2.cvtColor(image_small,cv2.COLOR_BGR2RGB)

    faceinframe=face_recognition.face_locations(image_small)
    faceencode=face_recognition.face_encodings(image_small,faceinframe)

    # print(faceencode)

    for facencode,face_loc in zip(faceencode,faceinframe):
        matches=face_recognition.compare_faces(encodelistknownfaces,facencode)

        face_distance=face_recognition.face_distance(encodelistknownfaces,facencode)
        print('Matches :',matches)
        #print('Face distsnce : ',face_distance)

        match_index=np.argmin(face_distance)
        #print('Match index :' ,match_index)
        name=class_names[np.argmin(face_distance)]

        if matches[match_index]:
            name=class_names[match_index]
            print(name)
            # print(face_distance)

            y1,x2,y2,x1=face_loc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4

            markattendence(name)
            now=datetime.now()
            dt=(now.strftime('%d-%m-%y'))
            time=(now.strftime('%I:%M %p'))
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2+3),(x2,y2+27),(255,255,255),-2)
            cv2.putText(img,name,(x1,y2+20),cv2.FONT_HERSHEY_COMPLEX,.6,(0,0,255),1)
            cv2.putText(img,dt,(x1,y2+38),cv2.FONT_HERSHEY_COMPLEX,.4,(0,255,255),1)
            cv2.putText(img,time,(x1,y2+52),cv2.FONT_HERSHEY_COMPLEX,.4,(0,255,255),1)

    cv2.imshow('face',img)
    if cv2.waitKey(1) & 0xFF == 27: 
        break
