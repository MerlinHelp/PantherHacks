"""
Fluidics vector field: Poisson solve with wall absorption and placed sources.

Run: ``python GenerateVectorFieldFluid.py --image map.png --out field.npy``

Black (or near-black) pixels are solid walls with Dirichlet ``phi = 0``. You place
**source** cells on traversable pixels; the steady Poisson equation
``∇²φ = -ρ`` is solved on the grid (ρ > 0 at sources), then velocity is
``v = -∇φ``.

Conventions
-----------
- Output array shape ``(height, width, 2)``, dtype float64:
  - ``[..., 0]`` = vector_x (positive = right)
  - ``[..., 1]`` = vector_y (positive = down)
- Wall cells are set to ``(0, 0)``.

Interactive controls (matplotlib window)
----------------------------------------
- Left click: toggle a flow source on/off at that traversable pixel.
- Enter / Return: solve and close the figure.
- z: remove last placed source
- c: clear all sources
"""

from __future__ import annotations

import argparse
import sys
from typing import Set, Tuple

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


def load_image_and_free_mask(
    image_path: str, threshold: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an image and build a traversable mask.

    Parameters
    ----------
    image_path : str
        Path to an image file (RGB, RGBA, or grayscale).
    threshold : int
        Pixels with luminance <= threshold (0–255) are obstacles.

    Returns
    -------
    rgb : ndarray, shape (H, W, 3), float64 in [0, 1]
        For display.
    free : ndarray, shape (H, W), bool
        True where traversable.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lum_255 = lum * 255.0
    free = lum_255 > float(threshold)
    return arr, free


def _pixel_free(free: np.ndarray, x: float, y: float) -> bool:
    """Return True if (x, y) in image coordinates lies on a traversable pixel."""
    h, w = free.shape
    j = int(round(x))
    i = int(round(y))
    if i < 0 or i >= h or j < 0 or j >= w:
        return False
    return bool(free[i, j])


def solve_poisson_sor(
    free: np.ndarray,
    rho: np.ndarray,
    max_iters: int = 80_000,
    tol: float = 1e-5,
    omega: float = 1.85,
) -> np.ndarray:
    """
    Solve ∇²φ = -ρ on traversable cells, φ = 0 on walls (5-point stencil, h = 1).

    Wall neighbors contribute 0 to the neighbor sum; denominator is always 4.
    """
    h, w = free.shape
    phi = np.zeros((h, w), dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)

    for _ in range(max_iters):
        delta_max = 0.0
        for i in range(h):
            for j in range(w):
                if not free[i, j]:
                    phi[i, j] = 0.0
                    continue
                s = 0.0
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= h or nj < 0 or nj >= w or not free[ni, nj]:
                        s += 0.0
                    else:
                        s += phi[ni, nj]
                new_phi = (s + rho[i, j]) / 4.0
                old = phi[i, j]
                blended = (1.0 - omega) * old + omega * new_phi
                d = abs(blended - old)
                if d > delta_max:
                    delta_max = d
                phi[i, j] = blended
        if delta_max < tol:
            break
    return phi


def velocity_from_phi(phi: np.ndarray, free: np.ndarray) -> np.ndarray:
    """
    v = -∇φ with one-sided differences where a neighbor is a wall (φ = 0 there).
    Returns (H, W, 2).
    """
    pw = np.where(free, phi, 0.0)
    h, w = free.shape

    pl = np.zeros_like(pw)
    pr = np.zeros_like(pw)
    pu = np.zeros_like(pw)
    pd = np.zeros_like(pw)
    pl[:, 1:] = pw[:, :-1]
    pr[:, :-1] = pw[:, 1:]
    pu[1:, :] = pw[:-1, :]
    pd[:-1, :] = pw[1:, :]

    left_f = np.zeros_like(free, dtype=bool)
    left_f[:, 1:] = free[:, :-1]
    right_f = np.zeros_like(free, dtype=bool)
    right_f[:, :-1] = free[:, 1:]
    up_f = np.zeros_like(free, dtype=bool)
    up_f[1:, :] = free[:-1, :]
    down_f = np.zeros_like(free, dtype=bool)
    down_f[:-1, :] = free[1:, :]

    ddx = np.zeros_like(pw)
    both_x = left_f & right_f & free
    only_l = (~right_f) & left_f & free
    only_r = (~left_f) & right_f & free
    ddx[both_x] = (pr[both_x] - pl[both_x]) * 0.5
    ddx[only_l] = pw[only_l] - pl[only_l]
    ddx[only_r] = pr[only_r] - pw[only_r]

    ddy = np.zeros_like(pw)
    both_y = up_f & down_f & free
    only_u = (~down_f) & up_f & free
    only_d = (~up_f) & down_f & free
    ddy[both_y] = (pd[both_y] - pu[both_y]) * 0.5
    ddy[only_u] = pw[only_u] - pu[only_u]
    ddy[only_d] = pd[only_d] - pw[only_d]

    vx = -ddx
    vy = -ddy
    vx = np.where(free, vx, 0.0)
    vy = np.where(free, vy, 0.0)
    return np.stack([vx, vy], axis=-1)


def build_vector_field_from_fluidics(
    free: np.ndarray,
    source_cells: Set[Tuple[int, int]],
    source_strength: float = 1.0,
    max_iters: int = 80_000,
    tol: float = 1e-5,
    sor_omega: float = 1.85,
    unit_vectors: bool = True,
) -> np.ndarray:
    """
    Run Poisson solve and gradient to produce (H, W, 2) velocity field.
    """
    if not source_cells:
        raise ValueError("At least one source cell is required.")

    h, w = free.shape
    rho = np.zeros((h, w), dtype=np.float64)
    for i, j in source_cells:
        if 0 <= i < h and 0 <= j < w and free[i, j]:
            rho[i, j] += float(source_strength)

    phi = solve_poisson_sor(free, rho, max_iters=max_iters, tol=tol, omega=sor_omega)
    field = velocity_from_phi(phi, free)

    if unit_vectors:
        mag = np.linalg.norm(field, axis=-1, keepdims=True)
        mag = np.maximum(mag, 1e-12)
        unit = field / mag
        field = np.where(free[..., np.newaxis], unit, 0.0)

    return field.astype(np.float64)


def collect_sources_interactive(
    rgb: np.ndarray,
    free: np.ndarray,
) -> Set[Tuple[int, int]]:
    """
    Show matplotlib UI; return set of (row, col) source indices when user presses Enter.
    """
    height, width = free.shape
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, origin="upper", extent=(0, width, height, 0), aspect="auto")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(
        "Left click: toggle source | Enter: solve | z: undo | c: clear all"
    )

    state: dict = {
        "order": [],
        "set": set(),
        "done": False,
        "scatter": None,
    }

    def redraw_sources():
        if state["scatter"] is not None:
            state["scatter"].remove()
            state["scatter"] = None
        if state["set"]:
            xs = [j for (i, j) in state["order"]]
            ys = [i for (i, j) in state["order"]]
            state["scatter"] = ax.scatter(
                xs,
                ys,
                s=120,
                c="lime",
                marker="o",
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
            )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return
        j = int(round(event.xdata))
        i = int(round(event.ydata))
        if not _pixel_free(free, float(j), float(i)):
            return
        key = (i, j)
        if key in state["set"]:
            state["set"].discard(key)
            state["order"] = [p for p in state["order"] if p != key]
        else:
            state["set"].add(key)
            state["order"].append(key)
        redraw_sources()

    def on_key(event):
        if event.key in ("enter", "return"):
            if not state["set"]:
                return
            state["done"] = True
            plt.close(fig)
        elif event.key == "z":
            if state["order"]:
                last = state["order"].pop()
                state["set"].discard(last)
                redraw_sources()
        elif event.key == "c":
            state["order"].clear()
            state["set"].clear()
            redraw_sources()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    if not state["done"]:
        raise RuntimeError("Window closed before finishing (press Enter to compute).")

    if not state["set"]:
        raise ValueError("No sources placed.")

    return set(state["set"])


def generate_vector_field(
    image_path: str,
    threshold: int = 16,
    unit_vectors: bool = True,
    source_strength: float = 1.0,
    max_iters: int = 80_000,
    tol: float = 1e-5,
    sor_omega: float = 1.85,
) -> np.ndarray:
    """
    Load image, open GUI to place sources, solve Poisson, return (H, W, 2) field.
    """
    rgb, free = load_image_and_free_mask(image_path, threshold=threshold)
    sources = collect_sources_interactive(rgb, free)
    return build_vector_field_from_fluidics(
        free,
        sources,
        source_strength=source_strength,
        max_iters=max_iters,
        tol=tol,
        sor_omega=sor_omega,
        unit_vectors=unit_vectors,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Place fluid sources on a collision image; solve Poisson; "
            "save a (H,W,2) vector field (v = -grad phi)."
        )
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--out",
        default="vector_field.npy",
        help="Output .npy path (default: vector_field.npy)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=16,
        help="Luminance threshold (0-255); pixels <= this are obstacles (default: 16)",
    )
    parser.add_argument(
        "--no-unit",
        action="store_true",
        help="Do not normalize to unit vectors on traversable cells",
    )
    parser.add_argument(
        "--source-strength",
        type=float,
        default=1.0,
        help="Per-source rho value in Poisson RHS (default: 1.0)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=80_000,
        help="Max SOR iterations (default: 80000)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="SOR stopping tolerance on max cell change (default: 1e-5)",
    )
    parser.add_argument(
        "--sor-omega",
        type=float,
        default=1.85,
        help="SOR relaxation weight in (1,2) (default: 1.85)",
    )
    args = parser.parse_args()

    try:
        field = generate_vector_field(
            args.image,
            threshold=args.threshold,
            unit_vectors=not args.no_unit,
            source_strength=args.source_strength,
            max_iters=args.max_iters,
            tol=args.tol,
            sor_omega=args.sor_omega,
        )
    except (ValueError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1

    np.save(args.out, field)
    print(f"Saved shape {field.shape} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
