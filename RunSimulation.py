# run_demo.py
 
from UnitPhysics import SimulationArea, Particle
from UnitAlgorithms import new_vector_field_state, follow_contour_line, \
    concentration_gradient_state

from EnvironmentGeneration import *
import numpy as np
from PIL import Image

vector_array = np.load('field.npy')

img = Image.open("watershed8.png")
img_arr = np.array(img)
boundary_array = gen_collision_arr(img_arr)
if boundary_array is None:
    np.load('collision.npy')
concentration_array = gen_concentration_arr(img_arr)
if concentration_array is None:
    np.load('concentration.npy')

simulation_area = SimulationArea(
    *img.size,
    vector_field=vector_array,
    boundary=boundary_array,
    concentration=concentration_array,
    cell_grid=(10,7)
)
simulation_area.set_title("Diffusion Demo")


particles = []
particle = Particle(
    simulation_area,
    position=(0, 0)
)
#particle2 = Particle(
    #simulation_area,
    #position=(10, 30)
#)
#particle3 = Particle(
    #simulation_area,
    #position=(-20, -30)
#)
particle2 = Particle(
    simulation_area,
    position=(-20, -320)
)
particle3 = Particle(
    simulation_area,
    position=(240, -145)
)
particle4 = Particle(
    simulation_area,
    position=(0, 300)
)

particle.assign_colour("blue")
particle2.assign_colour("red")
particle3.assign_colour("green")
particle4.assign_colour("yellow")
particles.append(particle)
particles.append(particle2)
particles.append(particle3)
particles.append(particle4)

import time
# Animation loop
while True:
    for cell in simulation_area.get_sequence_of_cells():
        for particle in cell:
            particle.move(
                *concentration_gradient_state(particle)
            )
            for other_particle in cell:
                if particle.check_collision(other_particle):
                    particle.collide(other_particle)
                elif other_particle in particle.in_collision_with:
                        particle.remove_collision(other_particle)
    simulation_area.update()
    time.sleep(0.023)
