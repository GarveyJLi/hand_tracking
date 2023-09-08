import cv2
import time
import mediapipe as mp
from cleaning import *
import sys

# Something goes weird when light turns green 

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils


capture = cv2.VideoCapture(sys.argv[1])
if capture.isOpened() == False:
    print("Check filepath")
    raise TypeError

# initialize time for FPS calc
previousTime = 0
currentTime = 0

while capture.isOpened():
    #capture frame by frame
    ret, frame = capture.read()

    #resize frame
    frame = cv2.resize(frame, (800, 600))

    # Converting from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using holistic model
    # To improve performance, optionally mark image as not writeable to pass by refernce
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable=True

    #Converting from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    """
    # Drawing facial landmarks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(
            color=(255, 0, 255),
            thickness=1,
            circle_radius=1
        ),
        mp_drawing.DrawingSpec(
            color=(0, 255, 255),
            thickness=1,
            circle_radius=1
        )
    )
    """

    # Drawing hand landmarks

    #Right
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    # LEft
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    
    """
    # Calc fps
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime

    

    # Display FPS
    cv2.putText(image, str(int(fps)) + " FPS", (10, 70), \
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    """

    """if results.right_hand_landmarks:
       cv2.putText(image, str(results.right_hand_landmarks.landmark[0].x - results.right_hand_landmarks.landmark[4].x), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    """
    print(joints_dists(results))

    # Display image
    cv2.imshow("Facial and Hand landmarks", image)

    

    # q to break loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    time.sleep(0.25)

    # When done, release capture and destroy windows

capture.release()
cv2.destroyAllWindows()









