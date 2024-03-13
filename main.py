import cv2
import numpy as np
import face_recognition

imageKev = face_recognition.load_image_file('StudentImages/Kevin - Copy.jpg')
imageKev = cv2.cvtColor(imageKev, cv2.COLOR_BGR2RGB)
imageTest = face_recognition.load_image_file('StudentImages/random - Copy.jpg')
imageTest = cv2.cvtColor(imageTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imageKev)[0]
encodeKev = face_recognition.face_encodings(imageKev)[0]
cv2.rectangle(imageKev, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imageTest)[0]
encodeTest = face_recognition.face_encodings(imageTest)[0]
cv2.rectangle(imageTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

faceDis = face_recognition.face_distance([encodeKev], encodeTest)
results = face_recognition.compare_faces([encodeKev], encodeTest)
print(results, faceDis)
cv2.putText(imageTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Kevin Godoy', imageKev)
cv2.imshow('Kevin Godoy Test', imageTest)
cv2.waitKey(0)
