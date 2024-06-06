import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import autopy
import time
import speech_recognition as sr

class GestureRecognizer:
    def __init__(self, activeMode=False, maxHands=1, detectionConfidence=False, trackingConfidence=0.5):
        self.activeMode = activeMode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mediaPipeHands = mp.solutions.hands
        self.handProcessor = self.mediaPipeHands.Hands(self.activeMode, self.maxHands, self.detectionConfidence, self.trackingConfidence)
        self.mediaPipeDrawing = mp.solutions.drawing_utils
        self.fingerTipIndices = [4, 8, 12, 16, 20]

    def detectHands(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.processResults = self.handProcessor.process(frameRGB)

        if self.processResults.multi_hand_landmarks:
            for handLandmarks in self.processResults.multi_hand_landmarks:
                if draw:
                    self.mediaPipeDrawing.draw_landmarks(frame, handLandmarks, self.mediaPipeHands.HAND_CONNECTIONS)
        return frame

    def getPositions(self, frame, handIndex=0, draw=True):   
        coordXList, coordYList, boundingBox, self.landmarkList  = [], [], [], []

        if self.processResults.multi_hand_landmarks:
            selectedHand = self.processResults.multi_hand_landmarks[handIndex]
            
            for idx, landmark in enumerate(selectedHand.landmark):
                height, width, _ = frame.shape
                coordX, coordY = int(landmark.x * width), int(landmark.y * height)
                coordXList.append(coordX)
                coordYList.append(coordY)
                self.landmarkList.append([idx, coordX, coordY])
                if draw:
                    cv2.circle(frame, (coordX, coordY), 5, (255, 0, 255), cv2.FILLED)

            xMin, xMax = min(coordXList), max(coordXList)
            yMin, yMax = min(coordYList), max(coordYList)
            boundingBox = xMin, yMin, xMax, yMax

            if draw:
                cv2.rectangle(frame, (xMin - 20, yMin - 20), (xMax + 20, yMax + 20),
                              (0, 255, 0), 2)

        return self.landmarkList, boundingBox

    def fingersRaised(self):
        fingers = []

        if self.landmarkList[self.fingerTipIndices[0]][1] > self.landmarkList[self.fingerTipIndices[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for i in range(1, 5):
            if self.landmarkList[self.fingerTipIndices[i]][2] < self.landmarkList[self.fingerTipIndices[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    w = 640
    h = 480
    frameR = 100
    smooth = 8
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0
    prev_time = 0
    stab_buf = []
    stab_thresh = 10
    stab_rad = 10
    scroll_down_speed = -60
    scroll_up_speed = 60
    interval = 0.01 # interval (in seconds) between keystrokes when typing.
    timeout = 5 # duration (in seconds) that the speech recognizer waits for the speaker to start talking before it times out.
    phrase_time_limit = 10 # duration (in seconds) for capturing speech after it has started, regardless of any pauses or breaks in speaking.
    
    r = sr.Recognizer()
    m = sr.Microphone()

    def speech_to_text():
        with m as source:
            r.adjust_for_ambient_noise(source)
            try:
                audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                return r.recognize_google(audio)
            except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
                return ''

    cap = cv2.VideoCapture(0)
    cap.set(3, w)
    cap.set(4, h)
    cap.set(cv2.CAP_PROP_FPS, 60)

    detector = GestureRecognizer()               
    scr_w, scr_h = autopy.screen.size()

    hold = False

    while True:
        _, img = cap.read()
        img = detector.detectHands(img)
        lmList, _ = detector.getPositions(img)           

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            fingers = detector.fingersRaised()
            cv2.rectangle(img, (frameR, frameR), (w - frameR, h - frameR), (255, 0, 255), 2)
            
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, w - frameR), (0, scr_w))
                y3 = np.interp(y1, (frameR, h - frameR), (0, scr_h))

                curr_x = prev_x + (x3 - prev_x) / smooth
                curr_y = prev_y + (y3 - prev_y) / smooth

                autopy.mouse.move(scr_w - curr_x, curr_y)
                cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = curr_x, curr_y

                stab_buf.append((curr_x, curr_y))
                if len(stab_buf) > stab_thresh:
                    stab_buf.pop(0)
            
            # Single left click           
            if fingers[1] == 1 and fingers[2] == 1:
                if len(stab_buf) == stab_thresh and all(
                    np.linalg.norm(np.array(pos) - np.array(stab_buf[0])) < stab_rad
                    for pos in stab_buf
                ):
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()
                    stab_buf.clear()
            
            # Double left click
            if fingers[1] == 1 and fingers[4] == 1 and all(f == 0 for f in [fingers[0], fingers[2], fingers[3]]):
                autopy.mouse.click()
                autopy.mouse.click()
            
            # Single right click
            if fingers[0] == 1 and all(f == 0 for f in fingers[1:]):
                if len(stab_buf) == stab_thresh and all(
                    np.linalg.norm(np.array(pos) - np.array(stab_buf[0])) < stab_rad
                    for pos in stab_buf
                ):
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)
                    stab_buf.clear()
            
            # Hold left click and move
            if fingers[0] == 0 and all(f == 1 for f in fingers[1:]):
                if not hold:
                    autopy.mouse.toggle(down=True)
                    hold = True

                x3 = np.interp(x1, (frameR, w - frameR), (0, scr_w))
                y3 = np.interp(y1, (frameR, h - frameR), (0, scr_h))
                curr_x = prev_x + (x3 - prev_x) / smooth
                curr_y = prev_y + (y3 - prev_y) / smooth
                autopy.mouse.move(scr_w - curr_x, curr_y)
                cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = curr_x, curr_y
            else:
                if hold:
                    autopy.mouse.toggle(down=False)
                    hold = False

            # Scroll down
            if all(f == 0 for f in fingers):
                pyautogui.scroll(scroll_down_speed)
            
            # Scroll up
            if all(f == 1 for f in fingers):
                pyautogui.scroll(scroll_up_speed)

            # Speech to text
            if fingers[2] == 1 and all(f == 0 for f in [fingers[0], fingers[1], fingers[3], fingers[4]]):
                text = speech_to_text()
                if text:
                    pyautogui.write(text, interval=interval)


        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        cv2.imshow("Archand", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()