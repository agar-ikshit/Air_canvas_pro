import cv2
import numpy as np
import random
import sys, os, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtracking.HandTracking import HandTracker
from utils.scoring import ScoreTracker
from utils.ui_helper import draw_text
from utils.settings import *


def generate_shape(level):

    cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
    offset_x = 150
    offset_y = 100

    if level == 1:
        return [(cx - offset_x, cy), (cx, cy), (cx + offset_x, cy)]

    elif level == 2:
        return [
            (cx - offset_x, cy + offset_y),
            (cx - offset_x // 2, cy - offset_y),
            (cx, cy + offset_y),
            (cx + offset_x // 2, cy - offset_y),
            (cx + offset_x, cy + offset_y),
        ]

    elif level == 3:
        points = []
        for i in range(7):
            x = cx - 180 + i * 60
            y = cy - 100 if i % 2 == 0 else cy + 100
            points.append((x, y))
        return points

    else:
        return [(cx + random.randint(-150, 150), cy + random.randint(-150, 150)) for _ in range(3 + level)]


def interpolate_path(points, step=5):
    dense_path = []
    for i in range(len(points)):
        start = np.array(points[i])
        end = np.array(points[(i + 1) % len(points)])  
        vec = end - start
        dist = np.linalg.norm(vec)
        num_steps = max(int(dist / step), 1)
        for j in range(num_steps):
            dense_path.append(tuple((start + vec * (j / num_steps)).astype(int)))
    return dense_path


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
    bar_x, bar_y = 30, 120
    bar_width, bar_height = 300, 25

    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (80, 80, 80), -1)
    color = (0, 255, 0) if accuracy > 80 else (0, 255, 255) if accuracy > 50 else (0, 0, 255)
    fill_width = int((accuracy / 100) * bar_width)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), WHITE, 2)
    draw_text(img, f"Accuracy: {accuracy:.1f}%", (bar_x + 10, bar_y - 10), WHITE, 0.6, 2)


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


def game_over_screen(final_score):
    img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)
    draw_text(img, "🎮 GAME OVER 🎮", (200, 250), (0, 255, 255), 1.2, 3)
    draw_text(img, f"Final Score: {final_score}", (250, 320), (0, 255, 0), 1.0, 2)
    draw_text(img, "Returning to Main Menu...", (220, 400), WHITE, 0.8, 2)
    cv2.imshow("Game Over", img)
    cv2.waitKey(2500)
    cv2.destroyAllWindows()


def run_connect_dots():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)

    tracker = HandTracker(max_hands=1)
    score = ScoreTracker("Player1")

    level_limit = 3
    drawn_path = []
    current_level = 1
    points = generate_shape(current_level)
    accuracy = 0
    last_pos = None

    print("Connect the Dots — Raise 1 finger to draw, 2 fingers to move freely.")
    print("Press SPACE after each shape to check accuracy or ESC to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        img = tracker.find_hands(frame, draw=True)
        lm_list = tracker.find_position(img, draw=False)

        # Draw guide shape
        for i, p in enumerate(points):
            cv2.circle(img, p, 12, YELLOW, -1)
            if i > 0:
                cv2.line(img, points[i - 1], points[i], (100, 100, 255), 2)
            draw_text(img, str(i + 1), (p[0] - 10, p[1] - 30), WHITE)

        drawing_enabled = False

        if lm_list:
            fingers_state = tracker.fingers_up(lm_list)
            fingers = sum(fingers_state)
            index_tip = tuple(lm_list[8][1:3])
            cv2.circle(img, index_tip, 6, PURPLE, -1)

            if fingers == 1:
                drawing_enabled = True
            else:
                drawing_enabled = False
                last_pos = None

            if drawing_enabled:
                if last_pos is not None:
                    cv2.line(img, last_pos, index_tip, GREEN, 3)
                    drawn_path.append(index_tip)
                last_pos = index_tip

        if len(drawn_path) > 1:
            for i in range(1, len(drawn_path)):
                cv2.line(img, drawn_path[i - 1], drawn_path[i], GREEN, 2)

        if len(drawn_path) > 0 and len(drawn_path) % 20 == 0:
            dense_points = interpolate_path(points)
            accuracy = calculate_accuracy(drawn_path, dense_points)

        draw_accuracy_meter(img, accuracy)
        draw_text(img, f"Level {current_level}/{level_limit}", (30, 50), GREEN)
        draw_text(img, f"Score: {score.score}", (30, 90), BLUE)

        cv2.imshow("Connect the Dots (Drawing)", img)
        key = cv2.waitKey(30) & 0xFF

        if key in [ord('n'), 32]:
            if drawn_path:
                    resampled_path = resample_points(drawn_path, step=5)
                    dense_points = interpolate_path(points)
                    accuracy = calculate_accuracy(resampled_path, dense_points)
                    score.add_points(int(accuracy))
                    print(f"Level {current_level} accuracy: {accuracy}%")
                
                # Move to next level
            current_level += 1
            if current_level > level_limit:
                break

                # Prepare next level
            points = generate_shape(current_level)
            drawn_path = []
            last_pos = None
            accuracy = 0

                # Show "Next Level" screen for 1 second
            img[:] = 0
            draw_text(img, f"Starting Level {current_level}...", (200, 300), GREEN, 1.0, 2)
            cv2.imshow("✏️ Connect the Dots (Drawing)", img)
            cv2.waitKey(1000)

        elif key in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
    score.save_score("ConnectDots")
    save_scores_to_json(score)
    game_over_screen(score.score)
    print("Final Summary:", score.get_summary())
    return score.score


if __name__ == "__main__":
    run_connect_dots()
