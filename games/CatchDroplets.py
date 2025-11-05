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


class Droplet:
    def __init__(self, x, y, color, radius=15, speed=5):
        self.x = x
        self.y = y
        self.color = color
        self.radius = radius
        self.speed = speed
        self.caught = False

    def move(self):
        self.y += self.speed

    def draw(self, img):
        if not self.caught:
            cv2.circle(img, (self.x, self.y), self.radius, self.color, -1)

def check_catch(droplet, bucket_x, bucket_y, bucket_w, bucket_h):
    return (bucket_x < droplet.x < bucket_x + bucket_w) and (bucket_y < droplet.y < bucket_y + bucket_h)

def draw_bucket(img, x, y, w, h):
    
    
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), -1)  
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 6)  
    
    cv2.line(img, (x, y), (x + w, y), (255, 255, 255), 10)  
    cv2.putText(img, "BUCKET", (x + 20, y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)  
    cv2.putText(img, "BUCKET", (x + 20, y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) 

def game_over_screen(final_score):
    img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)
    draw_text(img, "💧 GAME OVER 💧", (180, 250), (0, 255, 255), 1.2, 3)
    draw_text(img, f"Final Score: {final_score}", (250, 320), (0, 255, 0), 1.0, 2)
    draw_text(img, "Returning to Main Menu...", (220, 400), WHITE, 0.8, 2)
    cv2.imshow("Game Over", img)
    cv2.waitKey(2500)
    cv2.destroyAllWindows()

def save_scores_to_json(score_tracker, game_name="CatchDroplets"):
    save_path = os.path.join(os.path.dirname(__file__), "utils", "scores.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = score_tracker.get_summary()
    data["game"] = game_name

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

def run_catch_droplets(level_limit=3, level_time=10):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)

    tracker = HandTracker(max_hands=1)
    score = ScoreTracker("Player1")

    ret, test_frame = cap.read()
    if ret:
        actual_height, actual_width = test_frame.shape[:2]
        print(f"📐 Actual camera resolution: {actual_width}x{actual_height}")
    else:
        actual_height = WINDOW_HEIGHT
        actual_width = WINDOW_WIDTH

    bucket_w, bucket_h = 150, 60  
    droplet_spawn_interval = 25
    frame_count = 0

    current_level = 1
    droplets = []

    print("Catch the Droplets — Move the bucket with your index finger.")
    print("Press ESC or Q to quit.")

    while current_level <= level_limit:
        level_start_time = time.time()
        while time.time() - level_start_time < level_time:
            success, frame = cap.read()
            if not success:
                print("Camera read error")
                break

            frame = cv2.flip(frame, 1)
            img = frame.copy()
            
            frame_height, frame_width = img.shape[:2]
          
            bucket_y = frame_height - bucket_h - 30  
            bucket_x = frame_width // 2 - bucket_w // 2  
            
            img = tracker.find_hands(img, draw=True)
            lm_list = tracker.find_position(img, draw=False)
            frame_count += 1

            if lm_list and len(lm_list) > 8:
                finger_x = lm_list[8][1]  
                bucket_x = int(np.clip(finger_x - bucket_w // 2, 0, frame_width - bucket_w))
                cv2.circle(img, (finger_x, lm_list[8][2]), 10, (0, 255, 0), -1)

            if frame_count % droplet_spawn_interval == 0:
                center_margin = 200
                x = random.randint(frame_width//2 - center_margin, frame_width//2 + center_margin)
                color = random.choice([(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)])
                speed = random.randint(3 + current_level, 6 + current_level)
                droplets.append(Droplet(x, -10, color, radius=15, speed=speed))

            for drop in droplets:
                drop.move()
                drop.draw(img)
                if check_catch(drop, bucket_x, bucket_y, bucket_w, bucket_h):
                    drop.caught = True
                    score.add_points(5)
                    cv2.circle(img, (drop.x, drop.y), 30, (0, 255, 0), 3)

            droplets = [d for d in droplets if not d.caught and d.y < frame_height + 20]

          
            draw_bucket(img, bucket_x, bucket_y, bucket_w, bucket_h)
            
            cv2.putText(img, f"Bucket: X={bucket_x} Y={bucket_y}", 
                       (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.putText(img, f"Frame: {frame_width}x{frame_height}", 
                       (20, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

          
            elapsed_time = int(time.time() - level_start_time)
            remaining = max(level_time - elapsed_time, 0)
            
            draw_text(img, f"Score: {score.score}", (20, 40), (0, 255, 0))
            draw_text(img, f"Level {current_level}/{level_limit}", (20, 80), (255, 0, 0))
            draw_text(img, f"Time: {remaining}s", (frame_width - 180, 40), (0, 0, 255))

            cv2.imshow("💧 Catch the Droplets", img)
            key = cv2.waitKey(30) & 0xFF
            if key in [27, ord('q')]:
                current_level = level_limit + 1
                break

        current_level += 1

    cap.release()
    cv2.destroyAllWindows()
    score.save_score("CatchDroplets")
    save_scores_to_json(score, game_name="CatchDroplets")

    game_over_screen(score.score)
    print("Final Summary:", score.get_summary())
    return score.score

if __name__ == "__main__":
    run_catch_droplets()