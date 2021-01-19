import pygame
from keymap import *
def get_dir():
    pressed = pygame.key.get_pressed()
    if pressed[KEYMAP[RIGHT]]:
        return 0.0
    elif pressed[KEYMAP[UP]]:
        return 270.0
    elif pressed[KEYMAP[LEFT]]:
        return 180.0
    elif pressed[KEYMAP[DOWN]]:
        return 90.0
    return None

def update_fps(fps):
    pressed = pygame.key.get_pressed()
    if pressed[KEYMAP[FPS_2X]]:
        fps += 2
    elif pressed[KEYMAP[FPS_05X]]:
        fps -= 2
    return fps