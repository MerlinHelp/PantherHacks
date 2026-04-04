"""
Generate a smooth 2D vector field from a collision image and user-drawn arrows.

Conventions
-----------
- Image: black (or near-black) pixels are obstacles; brighter pixels are traversable.
- Output array shape ``(height, width, 2)``, dtype float64:
  - ``[..., 0]`` = vector_x (positive = right)
  - ``[..., 1]`` = vector_y (positive = down)
- Obstacle cells are set to ``(0, 0)``.

Interactive controls (matplotlib window)
----------------------------------------
- Click-drag: draw a flow arrow (both endpoints must lie on traversable pixels).
- Enter / Return: build the field from all arrows and close the figure.
- z: undo last arrow
- c: clear all arrows
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import ndimage
from scipy.interpolate import RBFInterpolator, griddata


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


def sample_arrow_segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    free: np.ndarray,
    sample_spacing: float = 3.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Sample unit direction along segment; skip if endpoints not free.

    Returns (xs, ys, vxs, vys) or None if invalid.
    """
    if not (_pixel_free(free, x0, y0) and _pixel_free(free, x1, y1)):
        return None
    dx, dy = x1 - x0, y1 - y0
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return None
    ux, uy = dx / length, dy / length
    n = max(1, int(np.ceil(length / sample_spacing)))
    t = np.linspace(0.0, 1.0, n)
    xs = x0 + t * dx
    ys = y0 + t * dy
    vxs = np.full(n, ux, dtype=np.float64)
    vys = np.full(n, uy, dtype=np.float64)
    return xs, ys, vxs, vys


def _rbf_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    height: int,
    width: int,
    kernels: Tuple[str, ...] = ("thin_plate_spline", "linear", "cubic"),
) -> np.ndarray:
    """Interpolate scattered values to (height, width) grid."""
    points = np.column_stack([xs, ys])
    xi = np.arange(width, dtype=np.float64)
    yi = np.arange(height, dtype=np.float64)
    X, Y = np.meshgrid(xi, yi, indexing="xy")
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])
    for kernel in kernels:
        try:
            rbf = RBFInterpolator(points, values, kernel=kernel)
            out = rbf(grid_pts).reshape(height, width)
            return out
        except (np.linalg.LinAlgError, ValueError):
            continue
    out = griddata(
        points,
        values,
        (X, Y),
        method="linear",
        fill_value=0.0,
    )
    if np.any(np.isnan(out)):
        out = np.nan_to_num(out, nan=0.0)
    return out.astype(np.float64)


def build_vector_field_from_samples(
    xs: np.ndarray,
    ys: np.ndarray,
    vxs: np.ndarray,
    vys: np.ndarray,
    free: np.ndarray,
    blur_sigma: float = 1.5,
    unit_vectors: bool = True,
) -> np.ndarray:
    """
    Build (H, W, 2) vector field from scattered (x, y, vx, vy) samples.
    """
    height, width = free.shape
    if xs.size == 0:
        raise ValueError("No arrow samples to interpolate.")

    vx_grid = _rbf_grid(xs, ys, vxs, height, width)
    vy_grid = _rbf_grid(xs, ys, vys, height, width)

    field = np.stack([vx_grid, vy_grid], axis=-1)

    if blur_sigma > 0:
        for c in range(2):
            blurred = ndimage.gaussian_filter(
                field[..., c], sigma=blur_sigma, mode="nearest"
            )
            field[..., c] = blurred * free.astype(np.float64)

    field[~free] = 0.0

    if unit_vectors:
        mag = np.linalg.norm(field, axis=-1, keepdims=True)
        mag = np.maximum(mag, 1e-12)
        unit = field / mag
        field = np.where(free[..., np.newaxis], unit, 0.0)

    return field.astype(np.float64)


def collect_arrows_interactive(
    rgb: np.ndarray,
    free: np.ndarray,
    sample_spacing: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Show matplotlib UI; return concatenated sample arrays when user presses Enter.
    """
    height, width = free.shape
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, origin="upper", extent=(0, width, height, 0), aspect="auto")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title("Draw arrows (drag). Enter: finish | z: undo | c: clear")

    drag_start: Optional[Tuple[float, float]] = None
    preview_line: Optional[Line2D] = None
    state: dict = {"segments": [], "done": False, "_artists": []}

    def redraw_arrows():
        for artist in state["_artists"]:
            artist.remove()
        state["_artists"] = []
        for (sx, sy, ex, ey) in state["segments"]:
            ln, = ax.plot([sx, ex], [sy, ey], "c-", lw=2)
            ann = ax.annotate(
                "",
                xy=(ex, ey),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=2),
            )
            state["_artists"].extend([ln, ann])
        fig.canvas.draw_idle()

    def on_press(event):
        nonlocal drag_start, preview_line
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return
        drag_start = (float(event.xdata), float(event.ydata))
        if preview_line is not None:
            preview_line.remove()
            preview_line = None
        preview_line, = ax.plot(
            [drag_start[0], drag_start[0]],
            [drag_start[1], drag_start[1]],
            "y--",
            lw=1,
        )
        fig.canvas.draw_idle()

    def on_motion(event):
        nonlocal preview_line
        if drag_start is None or preview_line is None:
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        preview_line.set_data(
            [drag_start[0], event.xdata], [drag_start[1], event.ydata]
        )
        fig.canvas.draw_idle()

    def on_release(event):
        nonlocal drag_start, preview_line
        if drag_start is None:
            return
        if event.button != 1:
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            if preview_line is not None:
                preview_line.remove()
                preview_line = None
            drag_start = None
            fig.canvas.draw_idle()
            return
        x0, y0 = drag_start
        x1, y1 = float(event.xdata), float(event.ydata)
        if preview_line is not None:
            preview_line.remove()
            preview_line = None
        drag_start = None
        if np.hypot(x1 - x0, y1 - y0) < 3.0:
            fig.canvas.draw_idle()
            return
        seg = sample_arrow_segment(x0, y0, x1, y1, free, sample_spacing)
        if seg is None:
            fig.canvas.draw_idle()
            return
        state["segments"].append((x0, y0, x1, y1))
        redraw_arrows()

    def on_key(event):
        if event.key in ("enter", "return"):
            state["done"] = True
            plt.close(fig)
        elif event.key == "z":
            if state["segments"]:
                state["segments"].pop()
                redraw_arrows()
        elif event.key == "c":
            state["segments"].clear()
            redraw_arrows()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    if not state["done"]:
        raise RuntimeError("Window closed before finishing (press Enter to compute).")

    if not state["segments"]:
        raise ValueError("No valid arrows were drawn.")

    xs_list, ys_list, vxs_list, vys_list = [], [], [], []
    for (x0, y0, x1, y1) in state["segments"]:
        seg = sample_arrow_segment(x0, y0, x1, y1, free, sample_spacing)
        if seg is None:
            continue
        xa, ya, vxa, vya = seg
        xs_list.append(xa)
        ys_list.append(ya)
        vxs_list.append(vxa)
        vys_list.append(vya)

    if not xs_list:
        raise ValueError("No valid arrow samples after redraw.")

    return (
        np.concatenate(xs_list),
        np.concatenate(ys_list),
        np.concatenate(vxs_list),
        np.concatenate(vys_list),
    )


def generate_vector_field(
    image_path: str,
    threshold: int = 16,
    blur_sigma: float = 1.5,
    unit_vectors: bool = True,
    sample_spacing: float = 3.0,
) -> np.ndarray:
    """
    Load image, open GUI to draw arrows, return (H, W, 2) vector field.
    """
    rgb, free = load_image_and_free_mask(image_path, threshold=threshold)
    xs, ys, vxs, vys = collect_arrows_interactive(
        rgb, free, sample_spacing=sample_spacing
    )
    return build_vector_field_from_samples(
        xs, ys, vxs, vys, free, blur_sigma=blur_sigma, unit_vectors=unit_vectors
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw flow arrows on a collision image; save a (H,W,2) vector field."
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
        "--blur-sigma",
        type=float,
        default=1.5,
        help="Gaussian blur sigma on vx/vy (0 to disable, default: 1.5)",
    )
    parser.add_argument(
        "--no-unit",
        action="store_true",
        help="Do not normalize to unit vectors on traversable cells",
    )
    parser.add_argument(
        "--sample-spacing",
        type=float,
        default=3.0,
        help="Pixels between samples along each arrow (default: 3)",
    )
    args = parser.parse_args()

    try:
        field = generate_vector_field(
            args.image,
            threshold=args.threshold,
            blur_sigma=args.blur_sigma,
            unit_vectors=not args.no_unit,
            sample_spacing=args.sample_spacing,
        )
    except (ValueError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1

    np.save(args.out, field)
    print(f"Saved shape {field.shape} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
