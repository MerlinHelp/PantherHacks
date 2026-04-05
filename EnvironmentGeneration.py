from __future__ import annotations
import turtle
import numpy as np
vec2d = turtle.Vec2D

class WaterShed:
    def __init__(
        self,
        width: int | float,
        height: int | float,
        bgpic: str="",
        color: str=""
    ):
        self.width = width
        self.height = height
        self.calculate_edges()
        self.window = turtle.Screen()
        if bgpic:
            self.window.bgpic(bgpic)
        self.color = color
        self.window.setup(int(width), int(height))

    def calculate_edges(self):
        self.left_x = int(-self.width // 2)
        self.right_x = -self.left_x
        self.top_y = int(self.height // 2)
        self.bot_y = -self.top_y

    def update(self):
        self.window.update()

    def run(self):
        self.window.mainloop()

class WaterShedFactory:
    def __init__(
        self,
        width: int | float,
        height: int | float,
    ):
        self.width = width
        self.height = height

    def watershed_from_img(
        self,
        width=None,
        height=None,
        bgpic=""
    ):
        if width is None:
            width = self.width
        if height is None:
            height = self.height

        watershed = WaterShed(width, height, bgpic)

        return watershed

def gen_collision_arr(pixels):
    return (pixels == (0, 0, 0, 255)).all(axis=-1)

def gen_concentration_arr(pixels):
    return np.clip(
        (
            (pixels[:, :, 0] - (pixels[:, :, 1] + pixels[:, :, 2]) / 2.0)
                * (pixels[:, :, 3] / 255)
        ),
        0,
        255
    )
