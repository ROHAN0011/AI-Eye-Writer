import cv2
import numpy as np
import dlib
from math import hypot

# we used the detector to detect the frontal face
detector = dlib.get_frontal_face_detector()

# it will dectect the facial landwark points
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

#We create a function that we will need later on to detect the medium point.
#On this function we simply put the coordinates of two points and will return the medium point
#(the points in the middle between the two points).
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def get_blinking_ratio(eye_points, facial_landmarks):
    # to detect the left_side of a left eye
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)

    # to detect the right_side of the left eye
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    # to detect the mid point for the center of top in left eye
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))

    # to detect the mid point for the center of the bottom in left eye
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # to calculate horizontal line distance
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))

    # to calculate vertical line distance
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    # to calculate ratio
    ratio = hor_line_lenght / ver_line_lenght

    return ratio


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

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))

        # Gaze detection
        # getting the area from the frame of the left eye only
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        # cv2.polylines(frame, [left_eye_region], True, 255, 2)
        height, width, _ = frame.shape

        # create the mask to extract xactly the inside of the left eye and exclude all the sorroundings.
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)

        # We now extract the eye from the face and we put it on his own window.Onlyt we need to keep in mind that wecan only cut
        # out rectangular shapes from the image, so we take all the extremes points of the eyes to get the rectangle
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        gray_eye = left_eye[min_y: max_y, min_x: max_x]

        # threshold to seperate iris and pupil from the white part of the eye.
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        eye = cv2.resize(gray_eye, None, fx=5, fy=5)

        cv2.imshow("EYE", eye)
        cv2.imshow("THRESHOLD", threshold_eye)
        cv2.imshow("LEFT_EYE", left_eye)
        cv2.imshow("mask", mask)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    # close the webcam when escape key is pressed
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()