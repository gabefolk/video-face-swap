import cv2 as cv # computer vision module
import numpy as np # numpy module for processing images as arrays

# Face class to store face data
class Face():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.realHeight = y + h
        self.realWidth = x + w

# Path to OpenCV's Haar Cascades
path = "/Users/gabrielfolk/opt/anaconda3/share/opencv4/haarcascades/"
face_cascade = cv.CascadeClassifier(path + 'haarcascade_frontalface_alt.xml')
catface_cascade = cv.CascadeClassifier(path + 'haarcascade_frontalcatface.xml')

# capture a video frame from user's primary camera
cap = cv.VideoCapture(0)

while True:
    #make wait time short so that video feed seems constant
    waitTime = 1

    # capture frame-by-frame
    _, frame = cap.read() # use underscore to indicate insignificant variable for boolean return value (if out of frames)

    # face_cascade will return an array of numbers indicating the rectangle coordinates
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3)    
    catface_rects = catface_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=4)

    # make array list to hold the faces
    fL = []  

    # get all the faces and store in an array list
    if len(face_rects) > 1:
        for (x, y, w, h) in face_rects:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # draw rectangle around face        
            fL.append(Face(x, y, w, h))

    if len(catface_rects) > 1:
        for (x, y, w, h) in catface_rects:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # draw rectangle around face        
            fL.append(Face(x, y, w, h))

    # swap the two faces, whether human or cat
    if len(face_rects) + len(catface_rects) == 2 and fL[0].w == fL[1].w and fL[0].h == fL[1].h:
        # have to create a new numpy array here, or else the subarray will be overwritten by the assignment operation
        tempFace = np.array(frame[fL[0].y:fL[0].realHeight, fL[0].x:fL[0].realWidth])
        frame[fL[0].y:fL[0].realHeight, fL[0].x:fL[0].realWidth] = frame[fL[1].y:fL[1].realHeight, fL[1].x:fL[1].realWidth]
        frame[fL[1].y:fL[1].realHeight, fL[1].x:fL[1].realWidth] = tempFace
        waitTime = 2000

    # display the resulting frame
    cv.imshow('frame', frame)
    # using & 0xFF ensures only last 8 bits of variable are read
    if cv.waitKey(waitTime) & 0xFF == ord('q'):     # waits for a key press input 
        break

# release the capture and destroy windows
cap.release()
cv.destroyAllWindows()