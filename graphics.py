import pygame
import functools
from consts import * 
import game_logic

green = pygame.Color(0, 255, 0)
red = pygame.Color(255, 0, 0)
black = pygame.Color(0, 0, 0)
screen_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
pygame.font.init()
font = pygame.font.SysFont('Comic Sans MS', 30)


def refresh(rects, scene):
    rect = functools.reduce(lambda r1, r2: r1.union(r2), rects)
    scene.fill(black, rect=rect)

def draw_body(body, scene, reward):
    f = lambda t: pygame.Rect(t[0], t[1], TILE_SIZE, TILE_SIZE)

    rects = list(map(f, body))
    for r in rects:
        scene.fill(green, rect=r)
    return rects

def draw_score(score, scene):
    text = font.render(f"score: {score}", False, green)
    dirty_rec = scene.blit(text, SCORE_OFFSETS)
    return dirty_rec

def draw_food(food, scene):
    fx, fy = food
    rect = pygame.Rect(fx, fy, TILE_SIZE, TILE_SIZE)
    scene.fill(green, rect=rect)
    return rect
def draw_hunger(hunger, scene):
    text = font.render(f"hunger: {hunger}", False, green)
    dirty_rec = scene.blit(text, (WIDTH - SCORE_OFFSETS[0] - (text.get_width() // 20) * 20 , SCORE_OFFSETS[1]))
    return dirty_rec

def render_state(state, scene, reward, to_refresh=None):
    if to_refresh is not None:
        refresh(to_refresh, scene)
    rects = draw_body(state[BODY], scene, reward)
    rects.append(draw_score(state[SCORE], scene))
    rects.append(draw_food(state[FOOD], scene))
    rects.append(draw_hunger(state[HUNGER], scene))
    return rects