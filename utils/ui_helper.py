
import cv2
from utils.settings import *

def draw_text(img, text, pos=(50, 50), color=WHITE, scale=FONT_SCALE, thickness=THICKNESS):
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)

def draw_button(img, text, top_left, bottom_right, color=BLUE, text_color=WHITE):
    cv2.rectangle(img, top_left, bottom_right, color, -1)
    text_pos = (top_left[0] + 20, top_left[1] + 40)
    draw_text(img, text, pos=text_pos, color=text_color)

def draw_circle_target(img, center, radius=TARGET_RADIUS, color=YELLOW):
    cv2.circle(img, center, radius, color, 3)
    cv2.circle(img, center, 5, color, -1)
