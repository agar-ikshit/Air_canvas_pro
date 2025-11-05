import sys
import os
import cv2
import time
import subprocess
from handtracking.HandTracking import HandTracker
from utils.settings import WINDOW_WIDTH, WINDOW_HEIGHT, WHITE, GREEN, RED, FONT

PYTHON_CMD = "python" if os.name == "nt" else "python3"

#number of fingers 
GAME_MAP = {
    1: "games/ConnectDots.py",
    2: "games/CatchDroplets.py",
    3: "games/ShapeDrawing.py",
    4: "games/BalloonPop.py",
    5: "games/ColorMatch.py",
}

HOLD_DURATION = 2.0  

def draw_menu(img, selected_fingers=None, hold_progress=0):
   
    h, w, _ = img.shape
    box_width = w // 3
    box_height = 80
    start_y = 150

    cv2.putText(img, "Hand Therapy Game Menu", (50, 60), FONT, 1, WHITE, 2)
    cv2.putText(img, "Raise 1–5 fingers to choose a game", (50, 100), FONT, 0.7, (200, 200, 200), 1)

    for i, (finger_count, script_path) in enumerate(GAME_MAP.items()):
        y = start_y + i * (box_height + 20)
        game_name = script_path.split("/")[-1].replace(".py", "")
        color = (100, 100, 100)

        #highlight selected game
        if selected_fingers == finger_count:
            color = (0, 255, 0)
            cv2.rectangle(img, (50, y - 10), (w - 50, y + box_height), color, 3)
        else:
            cv2.rectangle(img, (50, y - 10), (w - 50, y + box_height), color, 1)

        cv2.putText(img, f"{finger_count} Finger(s): {game_name}", (70, y + 50), FONT, 0.8, WHITE, 2)

    #loading bar
    if selected_fingers in GAME_MAP and hold_progress > 0:
        progress_width = int((w - 100) * min(hold_progress / HOLD_DURATION, 1))
        cv2.rectangle(img, (50, h - 50), (50 + progress_width, h - 20), GREEN, -1)
        cv2.rectangle(img, (50, h - 50), (w - 50, h - 20), WHITE, 2)
        cv2.putText(img, f"Holding... {int(hold_progress)}s", (60, h - 60), FONT, 0.7, WHITE, 2)

def run_main_menu():
    cap = cv2.VideoCapture(0)
    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)

    tracker = HandTracker(max_hands=1)
    prev_fingers = -1
    hold_start = None
    confirmed_game = None

    print("🎮 Gesture-based game menu started!")
    print("👉 Raise 1–5 fingers to choose a game:")
    for k, v in GAME_MAP.items():
        print(f"  {k} finger(s): {v.split('/')[-1].replace('.py','')}")

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Camera read failed")
            break

        frame = cv2.flip(frame, 1)
        img = tracker.find_hands(frame, draw=True)
        lm_list = tracker.find_position(img, draw=False)

        
        count = tracker.how_many_fingers_up(lm_list) if lm_list else 0

        #menu
        hold_progress = 0
        if count in GAME_MAP:
            if count == prev_fingers:
                if hold_start is None:
                    hold_start = time.time()
                else:
                    hold_progress = time.time() - hold_start
                    if hold_progress >= HOLD_DURATION:
                        confirmed_game = GAME_MAP[count]
                        break
            else:
                hold_start = time.time()
        else:
            hold_start = None

        draw_menu(img, selected_fingers=count, hold_progress=hold_progress)
        prev_fingers = count

        cv2.imshow("Hand Therapy Game Menu", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or Q
            break

    cap.release()
    cv2.destroyAllWindows()

    if confirmed_game:
        game_name = confirmed_game.split("/")[-1].replace(".py", "")
        print(f"Launching: {game_name}")
        subprocess.run([PYTHON_CMD, confirmed_game])

if __name__ == "__main__":
    run_main_menu()
