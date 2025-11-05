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


def calculate_jitter(points, threshold=10):
    if len(points) < 2:
        return 0
    diffs = [np.linalg.norm(np.array(points[i]) - np.array(points[i - 1])) for i in range(1, len(points))]
    jitter_values = [d for d in diffs if d < threshold]  # small moves are considered jitter
    if not jitter_values:
        return 0
    return np.mean(jitter_values)

def generate_shape(level):
    cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
    offset = 150

    if level == 1:  # Triangle
        return [(cx, cy - offset),
                (cx - offset, cy + offset),
                (cx + offset, cy + offset)]
    elif level == 2:  # Circle
        positions = []
        num_dots = 10
        for i in range(num_dots):
            angle = 2 * np.pi * i / num_dots
            x = int(cx + offset * np.cos(angle))
            y = int(cy + offset * np.sin(angle))
            positions.append((x, y))
        return positions
    elif level == 3:  # Square
        return [(cx - offset, cy - offset),
                (cx - offset, cy + offset),
                (cx + offset, cy + offset),
                (cx + offset, cy - offset)]
    elif level == 4:  # Star
        return [(cx, cy - offset), (cx + 50, cy - 50), (cx + offset, cy),
                (cx + 50, cy + 50), (cx, cy + offset),
                (cx - 50, cy + 50), (cx - offset, cy), (cx - 50, cy - 50)]
    else:  
        return [(cx + random.randint(-offset, offset),
                 cy + random.randint(-offset, offset)) for _ in range(3 + level)]

def interpolate_points(points, steps_per_edge=20):
    interpolated = []
    for i in range(len(points)):
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % len(points)])
        for t in np.linspace(0, 1, steps_per_edge):
            interpolated.append(tuple((p1 + (p2 - p1) * t).astype(int)))
    return interpolated

def calculate_accuracy(drawn_points, ideal_points, max_dist=50):
    if not drawn_points:
        return 0
    total_score = 0
    for dp in drawn_points:
        min_dist = min(np.linalg.norm(np.array(dp) - np.array(ip)) for ip in ideal_points)
        total_score += max(0, 1 - min_dist / max_dist)
    accuracy = (total_score / len(drawn_points)) * 100
    return round(accuracy, 2)

def resample_points(points, step=5):
    if not points:
        return []
    resampled = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(np.array(p) - np.array(resampled[-1])) >= step:
            resampled.append(p)
    return resampled

def draw_accuracy_meter(img, accuracy):
    x, y, w, h = 30, 120, 300, 25
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), -1)
    color = (0, 255, 0) if accuracy > 80 else (0, 255, 255) if accuracy > 50 else (0, 0, 255)
    fill = int((accuracy / 100) * w)
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), WHITE, 2)
    draw_text(img, f"Accuracy: {accuracy:.1f}%", (x + 10, y - 10), WHITE, 0.6, 2)

def save_scores_to_json(score_tracker, reaction_times=None):
    path = os.path.join(os.path.dirname(__file__), "utils", "scores.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = score_tracker.get_summary()
    if reaction_times:
        avg_time = round(np.mean(reaction_times), 2)
        data["avg_reaction_time_sec"] = avg_time

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                all_scores = json.load(f)
        except json.JSONDecodeError:
            all_scores = []
    else:
        all_scores = []
    all_scores.append(data)
    with open(path, "w") as f:
        json.dump(all_scores, f, indent=4)
    print(f"Score saved to {path}")

def game_over_screen(final_score):
    img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)
    draw_text(img, "🎮 GAME OVER 🎮", (200, 250), (0, 255, 255), 1.2, 3)
    draw_text(img, f"Final Score: {final_score}", (250, 320), (0, 255, 0), 1.0, 2)
    draw_text(img, "Returning to Main Menu...", (220, 400), WHITE, 0.8, 2)
    cv2.imshow("Game Over", img)
    cv2.waitKey(2500)
    cv2.destroyAllWindows()

def run_shape_drawing(level_limit=4):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)

    tracker = HandTracker(max_hands=1)
    score = ScoreTracker("Player1")
    current_level = 1
    drawn_path = []
    last_pos = None
    accuracy = 0
    points = generate_shape(current_level)
    ideal_path = interpolate_points(points)
    reaction_times = []
    level_jitter = []

    print("Shape Drawing Game — Draw with your index finger.")
    print("Raise 1 finger to draw, 2 fingers to move freely.")
    print("Press SPACE after each shape to check accuracy or ESC to quit.")

    level_start_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        img = tracker.find_hands(frame, draw=True)
        lm_list = tracker.find_position(img, draw=False)

        for i, p in enumerate(points):
            cv2.circle(img, p, 10, YELLOW, -1)
            draw_text(img, str(i + 1), (p[0] - 10, p[1] - 25), WHITE)
        for i in range(len(points)):
            cv2.line(img, points[i], points[(i + 1) % len(points)], BLUE, 2)

        drawing_enabled = False

        if lm_list:
            fingers_state = tracker.fingers_up(lm_list)
            fingers = sum(fingers_state)
            index_tip = tuple(lm_list[8][1:3])
            cv2.circle(img, index_tip, 6, PURPLE, -1)

            if fingers == 1:  
                drawing_enabled = True
                if last_pos is None:
                    last_pos = index_tip
                    drawn_path.append(index_tip)
                else:
                    cv2.line(img, last_pos, index_tip, GREEN, 3)
                    drawn_path.append(index_tip)
                    last_pos = index_tip
            else:  
                drawing_enabled = False
                last_pos = None

        for i in range(1, len(drawn_path)):
            cv2.line(img, drawn_path[i - 1], drawn_path[i], GREEN, 2)

        if len(drawn_path) > 0 and len(drawn_path) % 20 == 0:
            accuracy = calculate_accuracy(drawn_path, ideal_path)

        current_jitter = calculate_jitter(drawn_path[-10:]) if len(drawn_path) >= 2 else 0

        draw_accuracy_meter(img, accuracy)
        draw_text(img, f"Level {current_level}/{level_limit}", (30, 50), GREEN)
        draw_text(img, f"Score: {score.score}", (30, 90), BLUE)
        draw_text(img, f"Jitter: {current_jitter:.2f}px", (30, 160), ORANGE)

        cv2.imshow(" Shape Drawing", img)
        key = cv2.waitKey(30) & 0xFF

        if key in [32, ord('n')]:  
            resampled_path = resample_points(drawn_path, step=5)
            accuracy = calculate_accuracy(resampled_path, ideal_path)
            score.add_points(int(accuracy))

            reaction_times.append(time.time() - level_start_time)
            level_jitter.append(calculate_jitter(drawn_path))

            print(f"✅ Level {current_level} accuracy: {accuracy}%, Jitter: {level_jitter[-1]:.2f}px")
            current_level += 1
            if current_level > level_limit:
                break

            points = generate_shape(current_level)
            ideal_path = interpolate_points(points)
            drawn_path = []
            last_pos = None
            accuracy = 0
            level_start_time = time.time()

        elif key in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
    score.save_score("ShapeDrawing")
    
    save_scores_to_json(score, reaction_times=reaction_times)
    game_over_screen(score.score)
    print("Final Summary:", score.get_summary())
    if reaction_times:
        print(f"Average Reaction Time per Level: {np.mean(reaction_times):.2f} sec")
    if level_jitter:
        print(f"Average Jitter per Level: {np.mean(level_jitter):.2f}px")
    return score.score

if __name__ == "__main__":
    run_shape_drawing()
