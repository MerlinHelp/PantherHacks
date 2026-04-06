from __future__ import annotations
import numpy as np

def gen_collision_arr(pixels) -> np.ndarray:
    return (pixels == (0, 0, 0, 255)).all(axis=-1)

def gen_red_concentration_arr(pixels) -> np.ndarray:
    R = pixels[:, :, 0].astype(float)
    G = pixels[:, :, 1].astype(float)
    B = pixels[:, :, 2].astype(float)
    A = pixels[:, :, 3].astype(float)

    return np.clip((R - (G + B)/2.0) * (A / 255.0), 0, 255)

def gen_blue_concentration_arr(pixels) -> np.ndarray:
    R = pixels[:, :, 0].astype(float)
    G = pixels[:, :, 1].astype(float)
    B = pixels[:, :, 2].astype(float)
    A = pixels[:, :, 3].astype(float)

    return np.clip((B - (R + G)/2.0) * (A / 255.0), 0, 255)

def gen_concentration_arr(pixels) -> np.ndarray:
    R = pixels[:, :, 0].astype(float)
    G = pixels[:, :, 1].astype(float)
    B = pixels[:, :, 2].astype(float)
    A = pixels[:, :, 3].astype(float)

    R_WEIGHT = -30
    G_WEIGHT = 90
    B_WEIGHT = 90
    A_WEIGHT = 100

    return np.clip(
        (
            R/255.0 * R_WEIGHT +
            G/255.0 * G_WEIGHT +
            B/255.0 * B_WEIGHT +
            255.0/A * A_WEIGHT
        ),
        0,
        255
    )
