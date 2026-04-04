# run_demo.py
 
from collections import deque

from UnitPhysics import SimulationArea, Particle

n_particles = 50

simulation_area = SimulationArea(1000, 700, cell_grid=(10, 7))
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