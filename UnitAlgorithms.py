from UnitPhysics import Particle
import numpy as np
from typing import Any
from random import randint, random
from PIL import Image

RAND_STRENGTH = 1

def new_vector_field_state(particle: Particle) -> list[Any]:
    if (
        particle.simulation_area.vector_field is None
        or particle.simulation_area.vector_field.size == 0
    ):
        return [None, None]

    convertedPosX = -round(particle.position[1] - particle.simulation_area.height // 2)
    convertedPosY = round(particle.position[0] - particle.simulation_area.width // 2)

    new_velocity = (
        particle.simulation_area.vector_field[convertedPosX][convertedPosY][0],
        particle.simulation_area.vector_field[convertedPosX][convertedPosY][1] 
    )

    new_position = (
        particle.position[0] + new_velocity[0],
        particle.position[1] + new_velocity[1]
    )

    return [new_position, new_velocity]

img = Image.open("watershed.png")
pixels = img.load()

def follow_contour_line(particle: Particle) -> list[Any]:
    concentration = particle.simulation_area.concentration
    if concentration is None or concentration.size == 0:
        return [None, None]

    col = int(
        (particle.position[0] - particle.simulation_area.left_edge)
    )
    row = int(
        (particle.simulation_area.top_edge - particle.position[1])
    )
    
    curr_conc = concentration[row, col]
    next_col = col
    next_row = row
    if col + 2 < np.shape(concentration)[1]:
        next_col += 2

    d_x = concentration[row, next_col] - \
        curr_conc

    if row - 2 >= 0:
        next_row -= 2

    d_y = concentration[next_row, col] - \
        curr_conc
    if d_x == 0.0 and d_y == 0.0:
        particle.reached_source = True
        return [None, None]
    img = Image.open("watershed.png")
    pixels = np.array(img)
    print(particle.position, pixels[row, col], curr_conc, concentration[row, next_col], d_x, concentration[next_row, col], d_y)

    new_pos = [
        particle.position[0] + d_y + \
            randint(-RAND_STRENGTH, RAND_STRENGTH) * random(),
        particle.position[1] - d_x + \
            randint(-RAND_STRENGTH, RAND_STRENGTH) * random()
    ]

    return [new_pos, None]


img = Image.open("watershed8.png")
pixels = np.array(img)
def concentration_gradient_state(particle: Particle) -> list[Any]:
    concentration = particle.simulation_area.concentration
    boundary = particle.simulation_area.boundary
    if concentration is None or concentration.size == 0:
        return [None, None]
    if boundary is None or boundary.size == 0:
        return [None, None]

    col = int(
        (particle.position[0] - particle.simulation_area.left_edge)
    )
    row = int(
        (particle.simulation_area.top_edge - particle.position[1])
    )
    
    curr_conc = concentration[row, col]
    next_col = col
    next_row = row
    if col + 30 < np.shape(concentration)[1]:
        next_col += 30

    d_x = concentration[row, next_col] - \
        curr_conc

    if row - 30 >= 0:
        next_row -= 30

    if boundary[next_row, col]:
        next_row += 60

    d_y = concentration[next_row, col] - \
        curr_conc

    if d_x == 0.0 and d_y == 0.0:
        particle.reached_source = True
        return [None, None]

    length = ((d_x ** 2) + (d_y ** 2))**(1/2)
    d_x /= length
    d_y /= length

    new_pos = [
        particle.position[0] + d_x + \
            randint(-RAND_STRENGTH, RAND_STRENGTH) * random(),
        particle.position[1] + d_y + \
            randint(-RAND_STRENGTH, RAND_STRENGTH) * random()
    ]

    return [new_pos, None]
