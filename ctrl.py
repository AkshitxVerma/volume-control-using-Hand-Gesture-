import cv2
import mediapipe as mp
import time
import numpy as np  # Import NumPy for mathematical operations
import math
from pynput.keyboard import Key, Controller

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList

def main():
    # Camera settings
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    pTime = 0
    detector = handDetector(detectionCon=0.7)
    keyboard = Controller()

    # Hand range settings
    minHand = 30
    maxHand = 200

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        if len(lmList) != 0:
            # Get the positions of the thumb tip (id 4) and index finger tip (id 8)
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            
            # Draw a line between the thumb and index finger
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # Calculate the distance between the two points
            length = math.hypot(x2 - x1, y2 - y1)

            # Map the hand distance to volume level (0 to 100)
            volume_level = np.interp(length, [minHand, maxHand], [0, 100])

            # Control the system volume based on the calculated volume level
            if volume_level < 50:  # When the hand is close
                for _ in range(int(volume_level)):
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
            else:  # When the hand is open wide
                for _ in range(100 - int(volume_level)):
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)

            print(f"Volume Level: {volume_level}")

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
