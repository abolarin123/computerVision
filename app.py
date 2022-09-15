import cv2 as cv
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
# print(cv.__version__)
solutions = mp.solutions
mpHands = solutions.hands
mpDraw = solutions.drawing_utils
mpHandsStyle = solutions.drawing_styles
hands = mpHands.Hands(
    static_image_mode = False,
    model_complexity = 1,
    min_detection_confidence = 0.75,
    min_tracking_confidence = 0.75,
    max_num_hands = 2
)
# start capturing webcam
capture = cv.VideoCapture(0)

while True:
    success, img = capture.read()
    # flip image frame
    img = cv.flip(img, 1)

    # change bgr image to rgb image
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        if len(results.multi_handedness) == 2:
            cv.putText(img, "Both hands", (250, 250), cv.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
        else:
            for i in results.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']
                print(label)
                if label == 'Left':
                    # print("left")
                     cv.putText(img, label+' Hand',
                                (20, 50),
                                cv.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
                if label == 'Right':
                     
                    # Display 'Left Hand'
                    # on left side of window
                    cv.putText(img, label+' Hand', (460, 50),
                                cv.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
    # show video
    cv.imshow('Webcam', img)
    if cv.waitKey(1) & 0xff == ord('q'):
        break
