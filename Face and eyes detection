import cv2
# reading image.
image = cv2.imread("PRI_223554170.webp")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# getting pretrained models for detecting face and eyes detction.
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# detecting faces
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(15, 15)
)
# going through each face and detecting eyes for each of the faces
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

print("[INFO] Found {0} Faces!".format(len(faces)))

# changing color of rectangle from blue to red
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)

#saving detected image to 'besiktas2.jpg' file
status = cv2.imwrite('besiktas2.jpg', image)
print("[INFO] Image besiktas2.jpg written to filesystem: ", status)
