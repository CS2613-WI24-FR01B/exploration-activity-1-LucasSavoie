#import libraries, cv2 is image processing, mediapipe is machine learning/hand tracking
import cv2
import mediapipe as mp

#capture real-time video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

#mp_drawing is to draw/edit current picture
mp_drawing = mp.solutions.drawing_utils
#mp_hands is hand tracking from mediapipe
mp_hands = mp.solutions.hands

#hand is an object, or objects if there are multiple hands
hand = mp_hands.Hands()

#infinite loop
while True:
    #two variables that have current image, frame will be modified
    success, frame = cap.read()
    if success:
        #processes the image and checks for hands
        result = hand.process(frame)
        #if there are multiple hands
        if result.multi_hand_landmarks:
            #keep track of number of fingers
            count = 0
            #loop over hands
            for hand_landmarks in result.multi_hand_landmarks:
                #determines if the thumb is folded or straight (1 or 0)
                if(abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[2].x) > 0.1):
                    count += 1
                #determines if other 4 fingers are folded or straight
                for i in range(6, 20, 4):
                    if hand_landmarks.landmark[i].y > hand_landmarks.landmark[i + 2].y:
                        count += 1
                #draws the different connections, link to picture in markdown
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            toString = "Num: " + str(count)
            #put text onto screen
            cv2.putText(frame, toString, (0, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        #window with image capturing
        cv2.imshow("Hand Detection", frame)
        #if 'q' is pressed, program is quit
        if cv2.waitKey(1) == ord('q'):
            break





cv2.destroyAllWindows()