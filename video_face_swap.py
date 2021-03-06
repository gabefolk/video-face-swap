import cv2 as cv
import numpy as np

# Path to OpenCV's Haar Cascades
path = "/Users/gabrielfolk/opt/anaconda3/share/opencv4/haarcascades/"
face_cascade = cv.CascadeClassifier(path + 'haarcascade_frontalface_alt.xml')
#catface_cascade = cv.CascadeClassifier(path + 'haarcascade_frontalcatface.xml')

cap = cv.VideoCapture(0)

while True:
    waitTime = 1
    # Capture frame-by-frame
    _, frame = cap.read() # use underscore to indicate insignificant variable for boolean return value (if out of frames)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3)    
    #catface_rects = catface_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=4)
      
    if len(face_rects) == 2 and face_rects[0][2] == face_rects[1][2] and face_rects[0][3] == face_rects[1][3]:
        x = face_rects[0][0] 
        y = face_rects[0][1] 
        w = face_rects[0][2] 
        h = face_rects[0][3] 
        xN = face_rects[1][0] 
        yN = face_rects[1][1]
        # have to create a new numpy array here, or else subarray will be overwritten after assignment* operation
        face1 = np.array(frame[y:y+h, x:x+w])
        frame[y:y+h, x:x+w] = frame[yN:yN+h, xN:xN+w] # the assignment* operation
        frame[yN:yN+h, xN:xN+w] = face1
        waitTime = 2000
    # Display the resulting frame
    cv.imshow('frame', frame)
    # waits for a key press input 
    # using & 0xFF ensures only last 8 bits of variable are read
    if cv.waitKey(waitTime) & 0xFF == ord('q'): 
        break

# Release the capture
cap.release()
cv.destroyAllWindows()