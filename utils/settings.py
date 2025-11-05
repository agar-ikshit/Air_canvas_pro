import cv2
import os

WINDOW_WIDTH = 1380
WINDOW_HEIGHT = 820

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

BUCKET_IMG_PATH = os.path.join(ASSETS_DIR, "bucket.png")

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 2


TARGET_RADIUS = 30
CONNECT_TOLERANCE = 25  
