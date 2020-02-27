from pynput.keyboard import Key, Controller
import numpy as np


# def square(x):
#     return np.sqare(x)


# def euclidian_distance(old_coods, new_coords):
#     x1, y1 = old_coords
#     x, y   = new_coords

#     np.sqrt(square(y1-x1) + square(y-x)) = distance

    
# coords = [123, 456.8]

def direction(old_coords, new_coords):
    p, q = old_coords
    x, y = new_coords

    if x-p > 0:
        x_axis = 'Left'

    elif x-p < 0:
        x_axis = 'Right'

    if y-q > 0:
        y_axis = 'Up'

    if y-q < 0:
        y_axis = 'Down'

    print(x_axis, y_axis)


keyboard = Controller()
def player_at_coords(coords):
    keyboard.type(str(coords))
