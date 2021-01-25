TILE_SIZE = 20
TERRAIN_DIMS = (30, 30)
HEIGHT = TERRAIN_DIMS[0] * TILE_SIZE
WIDTH = TERRAIN_DIMS[1] * TILE_SIZE
#The directions the snake can move to
DIRS = [0.0, 90.0, 180, 270.0]
SNAKE_SIZE = 3
#How many turns the snake will live before being
#killed by the game
#(necessary to prevent it from looping)
MAX_HUNGER = 2000
HIST_SIZE = 5
#State indexes
HEAD = 0
BODY = 1
DIR = 2
FOOD = 3
SCORE = 4
HAS_FOOD = 5
ALIVE = 6
HUNGER = 7
HIST = 8
#event indexes
TURN = 0
#Graphical constants
SCORE_OFFSETS = (10, 10)
FPS_OFFSETS = (10, HEIGHT - 50)
DEFAULT_FPS = 300
