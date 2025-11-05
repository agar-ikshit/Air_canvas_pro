import cv2
import mediapipe as mp
import math

class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.results = None
        self.last_lm_list = []  #last valid landmarks

   
  
   
    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    
   
    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            try:
                hand = self.results.multi_hand_landmarks[hand_no]
                h, w, c = img.shape
                for id, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, cx, cy))
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                self.last_lm_list = lm_list
            except IndexError:
                pass
        return lm_list

    
  
   
    def fingers_up(self, lm_list):
        fingers = []
        if len(lm_list) == 0:
            return fingers

        if lm_list[self.tipIds[0]][1] < lm_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lm_list[self.tipIds[id]][2] < lm_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def how_many_fingers_up(self, lm_list=None):
        if lm_list is None:
            lm_list = self.last_lm_list
        fingers = self.fingers_up(lm_list)
        return sum(fingers)

    

  
    def find_distance(self, p1, p2, img=None, draw=True, r=10, t=3):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)

        if draw and img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def get_index_finger_position(self, lm_list=None):
        if lm_list is None:
            lm_list = self.last_lm_list
        if len(lm_list) > 8:
            return (lm_list[8][1], lm_list[8][2])
        return None
