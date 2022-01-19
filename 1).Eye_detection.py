# importing modules which will be required for simple eye detection
import cv2
import numpy as np
import dlib

# we used the detector to detect the frontal face
detector = dlib.get_frontal_face_detector()

# it will dectect the facial landwark points
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#We create a function that we will need later on to detect the medium point.
#On this function we simply put the coordinates of two points and will return the medium point
#(the points in the middle between the two points).
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


# to open webcab to capture the image
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()

    # change the color of the frame captured by webcam to grey
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # to detect faces from grey color frame
    faces = detector(gray)
    for face in faces:
        # to detect the landmarks of a face
        landmarks = predictor(gray, face)

        # to detect the left_side of a left eye
        left_point = (landmarks.part(36).x, landmarks.part(36).y)

        # to detect the right_side of the left eye
        right_point = (landmarks.part(39).x, landmarks.part(39).y)

        # to detect the mid point for the center of top in left eye
        center_top = midpoint(landmarks.part(37), landmarks.part(38))

        # to detect the mid point for the center of the bottom in left eye
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        # creating the horizontal line from the left_side to the right_side of the left eye
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)

        # creating the vertical line from  the center_top to center_bottom
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    # close the webcam when escape key is pressed
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()