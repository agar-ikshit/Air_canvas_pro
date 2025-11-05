import time
import json
import os

class ScoreTracker:
    def __init__(self, player_name="Player1"):
        self.score = 0
        self.start_time = time.time()
        self.level = 1
        self.player_name = player_name
        self.save_path = os.path.join("utils", "scores.json")

    def add_points(self, points):
        self.score += points

    def deduct_points(self, points):
        self.score = max(0, self.score - points)

    def get_time_elapsed(self):
        return round(time.time() - self.start_time, 2)

    def reset(self):
        self.score = 0
        self.start_time = time.time()
        self.level = 1

    def get_summary(self):
        return {
            "player": self.player_name,
            "score": self.score,
            "time_elapsed": self.get_time_elapsed(),
            "level": self.level
        }

    def save_score(self, game_name,avg_time=None):

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    
        if not os.path.exists(self.save_path):
            data = []
        else:
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []

        data.append({
            "game": game_name,
            "player": self.player_name,
            "score": self.score,
            "level": self.level,
            "avg_reaction_time": avg_time,
            "time_elapsed": self.get_time_elapsed()
        })

       
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=4)

        print("✅ Score saved successfully!")
