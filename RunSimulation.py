# run_demo.py
 
from UnitPhysics import SimulationArea, Particle
from EnvironmentGeneration import gen_concentration_arr, gen_collision_arr
import numpy as np
from PIL import Image

n_particles = 500
vector_array = np.load('field.npy')

img = Image.open("watershed.png")
img_arr = np.array(img)
boundary_array = gen_collision_arr or np.load('collision.npy')
concentration_array = gen_concentration_arr or np.load('concentration.npy')

simulation_area = SimulationArea(1500, 852, vector_field=vector_array, boundary = boundary_array, concentration = concentration_array, cell_grid=(10, 7))
simulation_area.set_title("Diffusion Demo")


particles = []
for _ in range(n_particles):
    particle = Particle(simulation_area)
    if particle.get_position()[0] < 0:
        particle.assign_colour("#1A6B72")
    else:
        particle.assign_colour("#FDB33B")
    particles.append(particle)

# Animation loop
while True:
    for cell in simulation_area.get_sequence_of_cells():
        for particle in cell:
            particle.move()
            for other_particle in cell:
                if particle.check_collision(other_particle):
                    particle.collide(other_particle)
                elif other_particle in particle.in_collision_with:
                        particle.remove_collision(other_particle)

    simulation_area.update()
