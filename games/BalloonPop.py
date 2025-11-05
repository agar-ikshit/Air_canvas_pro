import cv2
import numpy as np
import random
import time
import sys, os, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtracking.HandTracking import HandTracker
from utils.scoring import ScoreTracker
from utils.ui_helper import draw_text
from utils.settings import *

#faster with every level
class Balloon:
    def __init__(self, x, y, color, radius=30, speed=5):
        self.x = x
        self.y = y
        self.color = color
        self.radius = radius
        self.speed = speed
        self.popped = False

    def move(self):
        self.y -= self.speed

    def draw(self, img):
        if not self.popped:
            cv2.circle(img, (self.x, self.y), self.radius, self.color, -1)

def check_pop(balloon, finger_pos):
    dist = np.linalg.norm(np.array([balloon.x, balloon.y]) - np.array(finger_pos))
    return dist < balloon.radius

def game_over_screen(final_score):
    img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)
    draw_text(img, "GAME OVER", (200, 250), (0, 255, 255), 1.2, 3)
    draw_text(img, f"Final Score: {final_score}", (250, 320), (0, 255, 0), 1.0, 2)
    draw_text(img, "Returning to Main Menu...", (220, 400), WHITE, 0.8, 2)
    cv2.imshow("Game Over", img)
    cv2.waitKey(2500)
    cv2.destroyAllWindows()

def save_scores_to_json(score_tracker):
    save_path = os.path.join(os.path.dirname(__file__), "utils", "scores.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = score_tracker.get_summary()

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
    print(f"Score saved to {save_path}")

def run_balloon_pop():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)

    tracker = HandTracker(max_hands=1)
    score = ScoreTracker("Player1")

    level = 1
    max_level = 3
    level_duration = 10  

    print("Balloon Pop — Pop balloons with your index finger.")
    print("Press ESC to quit.")

    while level <= max_level:
        balloons = []
        frame_count = 0
        start_time = time.time()
        balloon_spawn_interval = max(20 - level * 5, 5)  #level difficulty
        speed_range = (3 + level * 2, 5 + level * 3)

        while True:
            success, frame = cap.read()
            if not success:
                print("❌ Camera read error")
                break

            frame = cv2.flip(frame, 1)
            img = tracker.find_hands(frame, draw=True)
            lm_list = tracker.find_position(img, draw=False)

        
            if frame_count % balloon_spawn_interval == 0:
                margin = 100
                x = random.randint(WINDOW_WIDTH//2 - margin, WINDOW_WIDTH//2 + margin)
                color = random.choice([RED, GREEN, BLUE, YELLOW, PURPLE])
                speed = random.randint(speed_range[0], speed_range[1])
                balloons.append(Balloon(x, WINDOW_HEIGHT + 30, color, radius=30, speed=speed))

            frame_count += 1

            finger_pos = None
            if lm_list:
                fingers_state = tracker.fingers_up(lm_list)
                if sum(fingers_state) >= 1:
                    finger_pos = tuple(lm_list[8][1:3])
                    cv2.circle(img, finger_pos, 8, WHITE, -1)

            #draw balloons
            for balloon in balloons:
                balloon.move()
                balloon.draw(img)

                if finger_pos and not balloon.popped:
                    if check_pop(balloon, finger_pos):
                        balloon.popped = True
                        score.add_points(10)

            # Remove off-screen balloons
            balloons = [b for b in balloons if b.y + b.radius > 0 and not b.popped]

            
            elapsed = int(time.time() - start_time)
            remaining = max(level_duration - elapsed, 0)
            draw_text(img, f"Score: {score.score}", (30, 50), GREEN)
            draw_text(img, f"Level: {level}", (WINDOW_WIDTH - 220, 50), BLUE)
            draw_text(img, f"Time: {remaining}s", (WINDOW_WIDTH - 210, 90), BLUE)

            cv2.imshow("🎈 Balloon Pop", img)
            key = cv2.waitKey(30) & 0xFF
            if key in [27, ord('q')]:
                cap.release()
                cv2.destroyAllWindows()
                score.save_score()
                save_scores_to_json(score)
                game_over_screen(score.score)
                return score.score

            if remaining <= 0:
                break

        print(f"Level {level} finished! Score: {score.score}")
        level += 1

    cap.release()
    cv2.destroyAllWindows()
    score.save_score("BalloonPop")
    save_scores_to_json(score)
    game_over_screen(score.score)
    print("Final Summary:", score.get_summary())
    return score.score

if __name__ == "__main__":
    run_balloon_pop()
