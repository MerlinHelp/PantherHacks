import numpy as np
import random
import turtle


class SimulationArea:
    def __init__(
        self,
        width: float | int,
        height: float | int,
        boundary: np.ndarray | None = None,
        vector_field: np.ndarray | None = None,
        concentration: np.ndarray | None = None,
        colour: str = "white",
        cell_grid: tuple[int, int] = (1, 1),
    ):
        self.width = width
        self.height = height
        # Define the edges of the simulation area as integers
        # (we'll need them as integers later)
        self.right_edge: int = int(width // 2)
        self.left_edge: int = -self.right_edge
        self.top_edge: int = int(height // 2)
        self.bottom_edge: int = -self.top_edge
        # Create a turtle screen object
        self.window = turtle.Screen()
        self.window.setup(width, height)
        self.window.tracer(0)
        self.window.bgcolor(colour)
        self.window.bgpic("watershed8.png")

        # Numpy Arrays determining physics of Simulation Area

        self.boundary = boundary
        self.vector_field = vector_field
        self.concentration = concentration

        # Define the cell grid
        self.cell_grid = cell_grid
        self.cell_width = width / cell_grid[0]
        self.cell_height = height / cell_grid[1]
        self.cells = {
            (i, j): [] for i in range(cell_grid[0]) for j in range(cell_grid[1])
        }

    def update(self):
        self.window.update()

    def set_title(self, title):
        self.window.title(title)

    @staticmethod
    def run_mainloop():
        turtle.mainloop()

    def get_sequence_of_cells(self):
        return self.cells.values()


class Particle:
    MAX_VELOCITY = 4

    def __init__(
        self,
        simulation_area: SimulationArea,
        position=None,
        velocity=None,
        colour="black",
        size_factor=0.5,
    ):
        self.simulation_area = simulation_area
        self.particle = turtle.Turtle()
        self.particle.shape("circle")
        self.particle.color(colour)
        self.particle.penup()
        self.particle.shapesize(size_factor)
        if position is None:
            position = [
                random.randint(
                    self.simulation_area.left_edge // 2,
                    self.simulation_area.right_edge // 2,
                ),
                random.randint(
                    self.simulation_area.bottom_edge // 2,
                    self.simulation_area.top_edge // 2,
                ),
            ]
        self.position = position
        if velocity is None:
            velocity = [
                random.randint(-self.MAX_VELOCITY * 0, self.MAX_VELOCITY * 0) / 10,
                random.randint(-self.MAX_VELOCITY * 0, self.MAX_VELOCITY * 0) / 10,
            ]
        self.velocity = velocity

        self.in_collision_with = []
        self.cell = None
        self.assign_to_cell()
        self.reached_source = False

    def assign_to_cell(self):
        # Convert the x and y coordinates to cell indices i and j
        cell_i = int(
            (self.position[0] - self.simulation_area.left_edge)
            // self.simulation_area.cell_width
        )
        cell_j = int(
            (self.position[1] - self.simulation_area.bottom_edge)
            // self.simulation_area.cell_height
        )
        # Ensure the cell indices are within the grid
        if cell_i < 0:
            cell_i = 0
        elif cell_i >= self.simulation_area.cell_grid[0]:
            cell_i = self.simulation_area.cell_grid[0] - 1
        if cell_j < 0:
            cell_j = 0
        elif cell_j >= self.simulation_area.cell_grid[1]:
            cell_j = self.simulation_area.cell_grid[1] - 1

        # Add the particle if it's not already in the
        # cell (first iteration)
        if not self.cell:
            self.cell = (cell_i, cell_j)
            self.simulation_area.cells[self.cell].append(self)
        elif self.cell != (cell_i, cell_j):
            # Remove the particle from the current cell
            self.simulation_area.cells[self.cell].remove(self)
            # Update the cell assignment
            self.cell = (cell_i, cell_j)
            # Add the particle to the new cell
            self.simulation_area.cells[self.cell].append(self)

    def move(
        self,
        new_pos: list[float] | None = None,
        new_velocity: list[float] | None = None,
    ):
        if new_pos is not None:
            self.position = new_pos
        if new_velocity is not None:
            self.velocity = new_velocity

        # Bounce off the edges
        if (
            self.position[0] < self.simulation_area.left_edge
            or self.position[0] > self.simulation_area.right_edge
        ):
            self.velocity[0] *= -1
        if (
            self.position[1] < self.simulation_area.bottom_edge
            or self.position[1] > self.simulation_area.top_edge
        ):
            self.velocity[1] *= -1

        """
        if (self.simulation_area.boundary[convertedPosX][convertedPosY] ):
            self.velocity[0] *= -1
            self.velocity[1] *= -1
        """

        self.particle.setposition(*self.position)
        self.assign_to_cell()

    def check_collision(self, other_particle):
        return (
            self.particle.distance(other_particle.particle)
            < 20 * self.particle.shapesize()[0]
        )

    def collide(self, other_particle):
        """
        This assumes that the particles are of equal mass and
        that the collision is a head-on elastic collision.
        """
        if other_particle is self:
            return
        if self not in other_particle.in_collision_with:
            self.velocity, other_particle.velocity = (
                other_particle.velocity,
                self.velocity,
            )
            # Keep a record of collision in other_particle.
            # No need to keep it in self
            other_particle.in_collision_with.append(self)

    def remove_collision(self, other_particle):
        self.in_collision_with.remove(other_particle)

    def assign_colour(self, colour):
        self.particle.color(colour)

    def get_position(self):
        return self.position
