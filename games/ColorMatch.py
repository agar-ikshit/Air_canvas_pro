import cv2
import numpy as np
import random
import time
import sys, os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from handtracking.HandTracking import HandTracker
from utils.scoring import ScoreTracker
from utils.ui_helper import draw_text
from utils.settings import *


FINGER_COLORS = [RED, GREEN, BLUE, YELLOW, PURPLE]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

class ColorDot:
    def __init__(self, x, y, color, radius=40):
        self.x = x
        self.y = y
        self.color = color
        self.radius = radius
        self.selected = False

    def draw(self, img):
        cv2.circle(img, (self.x, self.y), self.radius, self.color, -1)

def check_touch(finger_pos, dot):
    if finger_pos is None:
        return False
    return np.linalg.norm(np.array(finger_pos) - np.array((dot.x, dot.y))) <= dot.radius

def save_scores_to_json(score_tracker, reaction_times=None, game_name="SequenceColorMatch"):
   
    save_path = os.path.join(os.path.dirname(__file__), "utils", "scores.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = score_tracker.get_summary()
    data["game"] = game_name
    if reaction_times:
        data["avg_reaction_time"] = round(np.mean(reaction_times), 3)

    if os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                all_scores = json.load(f)
        except json.JSONDecodeError:
            all_scores = []
    else:
        all_scores = []

    all_scores.append(data)
    with open(save_path, "w") as f:
        json.dump(all_scores, f, indent=4)
    print(f"💾 Score saved to {save_path}")

def game_over_screen(final_score):
    img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)
    draw_text(img, "GAME OVER", (180, 250), (0, 255, 255), 1.2, 3)
    draw_text(img, f"Final Score: {final_score}", (250, 320), (0, 255, 0), 1.0, 2)
    draw_text(img, "Returning to Main Menu...", (220, 400), WHITE, 0.8, 2)
    cv2.imshow("Game Over", img)
    cv2.waitKey(2500)
    cv2.destroyAllWindows()

def run_sequence_color_match(level_limit=3, sequence_length=5):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)

    tracker = HandTracker(max_hands=1)
    score = ScoreTracker("Player1")
    reaction_times = [] 

    current_level = 1
    print("Sequence Color Match — Touch the colors in the correct order.")
    print("Press ESC or Q to quit.")

    while current_level <= level_limit:
    
        sequence = random.choices(FINGER_COLORS, k=sequence_length)
        dots = []

        margin = 150
        y_pos = WINDOW_HEIGHT // 2
        total_dots = sequence_length * 2  
        for i in range(total_dots):
            x = margin + i * (WINDOW_WIDTH - 2 * margin) // total_dots
            if i % 2 == 0:
                color = sequence[i // 2]  
            else:
                color = random.choice([c for c in FINGER_COLORS if c != sequence[i // 2]]) 
            dots.append(ColorDot(x, y_pos, color))

        current_index = 0
        level_start_time = time.time() 
        move_start_time = time.time()   

        while current_index < sequence_length:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            img = tracker.find_hands(frame, draw=True)
            lm_list = tracker.find_position(img, draw=False)

            for dot in dots:
                if not dot.selected:
                    dot.draw(img)

            if lm_list:
                fingers = tracker.fingers_up(lm_list)
                index_tip = tuple(lm_list[8][1:3])
                required_color = sequence[current_index]

                for i, finger_state in enumerate(fingers):
                    if finger_state == 1:
                        for dot in dots:
                            if not dot.selected and dot.color == required_color and check_touch(index_tip, dot):
                                dot.selected = True
                                score.add_points(10)
                                current_index += 1

                                reaction_time = time.time() - move_start_time
                                reaction_times.append(reaction_time)
                                move_start_time = time.time() 
                                break
                        break 

            draw_text(img, f"Level {current_level}/{level_limit}", (30, 50), GREEN)
            draw_text(img, f"Score: {score.score}", (30, 90), BLUE)
            draw_text(img, f"Next Dot:", (30, 130), YELLOW)

            if current_index < sequence_length:
                next_color = sequence[current_index]
                cv2.circle(img, (150, 160), 30, next_color, -1) 

            if reaction_times:
                draw_text(img, f"Last Reaction Time: {reaction_times[-1]:.2f}s", (30, 180), PURPLE, 0.7)

            cv2.imshow("Sequence Color Match", img)
            key = cv2.waitKey(30) & 0xFF
            if key in [27, ord('q')]:
                current_level = level_limit + 1
                break

        print(f"Level {current_level} completed!")
        current_level += 1

    cap.release()
    cv2.destroyAllWindows()
    score.save_score("SequenceColorMatch")
    save_scores_to_json(score, reaction_times=reaction_times, game_name="SequenceColorMatch")
    game_over_screen(score.score)
    print("Final Summary:", score.get_summary())
    if reaction_times:
        print("Average Reaction Time: {:.2f} sec".format(np.mean(reaction_times)))
    return score.score
        

if __name__ == "__main__":
    run_sequence_color_match()
