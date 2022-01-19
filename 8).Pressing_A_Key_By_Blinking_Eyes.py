import cv2
import numpy as np
import dlib
from math import hypot

# we used the detector to detect the frontal face
detector = dlib.get_frontal_face_detector()

# it will dectect the facial landwark points
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

# Keyboard setting
keyboard = np.zeros((600,1000,3),np.uint8)
# dictionary containing the letters, each one associated with an index.
keys_set_1 = {0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
              5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
              10: "Z", 11: "X", 12: "C", 13: "V", 14: "B"}


def letter(letter_index, text, letter_light):
    # Keys
    # Each key is simply a rectangle containing some text. So we define the sizes and we draw the rectangle.
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0

    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0

    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200

    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200

    elif letter_index == 10:
        x = 0
        y = 400
    elif letter_index == 11:
        x = 200
        y = 400
    elif letter_index == 12:
        x = 400
        y = 400

    elif letter_index == 13:
        x = 600
        y = 400
    elif letter_index == 14:
        x = 800
        y = 400
    width = 200
    height = 200
    th = 3  # thickness

    if letter_light == True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)

    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)

    # Inside the rectangle now we put the letter. So we define the sizes and style of the text and we center it.
    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_th)


# We create a function that we will need later on to detect the medium point.
# On this function we simply put the coordinates of two points and will return the medium point
# (the points in the middle between the two points).
def midpoint(p1,p2):
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


def get_gaze_ratio(eye_points, facial_landmarks):
    # Gaze detection
    # getting the area from the frame of the left eye only
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)

    # cv2.polylines(frame, [left_eye_region], True, 255, 2)
    height, width, _ = frame.shape

    # create the mask to extract xactly the inside of the left eye and exclude all the sorroundings.
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    # We now extract the eye from the face and we put it on his own window.Onlyt we need to keep in mind that wecan only cut
    # out rectangular shapes from the image, so we take all the extremes points of the eyes to get the rectangle
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]

    # threshold to seperate iris and pupil from the white part of the eye.
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    # dividing the eye into 2 parts .left_side and right_side.
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1

    elif right_side_white == 0:
        gaze_ratio = 5

    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


# to open webcab to capture the image
cap = cv2.VideoCapture(0)

# Going to create a white image which is going to be the board where we will put the letters we click from the virtual keyboard.
board = np.zeros((500, 500,), np.uint8)
board[:] = 255

# Counters
# To count the number of the frames
frames = 0

# The blinking_frames variable will keep track of the frames in a row in which the eyes are blinking.
blinking_frames = 0

letter_index = 0

# Text is going to contain all the letter that we will press when we blink our eyes.
text = ""

while True:
    _, frame = cap.read()
    frames = frames + 1
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    keyboard[:] = (0, 0, 0)

    # showing direction
    new_frame = np.zeros((500, 500, 3), np.uint8)

    # change the color of the frame captured by webcam to grey
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    active_letter = keys_set_1[letter_index]

    # to detect faces from grey color frame
    faces = detector(gray)
    for face in faces:

        # to detect the landmarks of a face
        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 4, (255, 0, 0), thickness=3)
            blinking_frames = blinking_frames + 1
            frames = frames - 1

            if blinking_frames == 5:
                text = text + active_letter
        else:
            blinking_frames = 0

    # letters
    # Then just after the blinking detection we add +1 to the blinking frames and when the blinking is detected for 5 frames in a
    # row then we add the selected letter to the text.
    if frames == 15:
        letter_index = letter_index + 1
        frames = 0
    if letter_index == 15:
        letter_index = 0

    for i in range(15):
        if i == letter_index:
            light = True
        else:
            light = False
        letter(i, keys_set_1[i], light)

    # Finally we display the text on the board.
    cv2.putText(board, text, (10, 100), font, 4, 0, 3)

    cv2.imshow("Frame", frame)
    # cv2.imshow("NEW_Frame",new_frame)
    cv2.imshow("Keyboard", keyboard)
    cv2.imshow("Board", board)

    key = cv2.waitKey(1)
    # close the webcam when escape key is pressed
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()