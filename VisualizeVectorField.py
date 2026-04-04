"""
Render a saved vector field (H, W, 2) as a static image.

Expects the same layout as :mod:`GenerateVectorField`: index ``[i, j, 0]`` is x-velocity
at row ``i``, column ``j`` (x right, y down).

Examples
--------
.. code-block:: bash

   python VisualizeVectorField.py --field field.npy --image watershed.png --out flow.png
   python VisualizeVectorField.py --field field.npy --out flow.png --no-stream --quiver-step 12
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from GenerateVectorField import load_image_and_free_mask


def load_field(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(
            f"Expected array shape (H, W, 2), got {arr.shape} from {path!r}"
        )
    return np.asarray(arr, dtype=np.float64)


def visualize_vector_field(
    field: np.ndarray,
    out_path: str,
    image_path: str | None = None,
    threshold: int = 16,
    dpi: float = 150,
    stream_density: float = 1.2,
    quiver_step: int = 18,
    show_stream: bool = True,
    show_quiver: bool = True,
) -> None:
    """
    Save a figure: optional map underlay, streamlines, optional subsampled quiver.
    """
    height, width = field.shape[:2]
    u = field[:, :, 0].copy()
    v = field[:, :, 1].copy()
    speed = np.sqrt(u * u + v * v)

    if image_path is not None:
        rgb, free = load_image_and_free_mask(image_path, threshold=threshold)
        if rgb.shape[0] != height or rgb.shape[1] != width:
            raise ValueError(
                f"Image size {rgb.shape[1]}x{rgb.shape[0]} does not match "
                f"field size {width}x{height}"
            )
    else:
        rgb = None
        free = np.ones((height, width), dtype=bool)

    # Fade obstacles / zero-motion cells slightly for readability
    u = np.where(free, u, 0.0)
    v = np.where(free, v, 0.0)

    fig_w = max(8, width / 80)
    fig_h = max(8, height / 80)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    if rgb is not None:
        ax.imshow(
            rgb,
            origin="upper",
            extent=(0, width, height, 0),
            aspect="auto",
            zorder=0,
        )
    else:
        ax.set_facecolor("#1a1a1e")
        ax.imshow(
            np.zeros((height, width, 3)),
            origin="upper",
            extent=(0, width, height, 0),
            aspect="auto",
            zorder=0,
        )

    x = np.arange(width, dtype=np.float64)
    y = np.arange(height, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")

    if show_stream and np.any(speed > 1e-9):
        strm = ax.streamplot(
            X,
            Y,
            u,
            v,
            density=stream_density,
            color=speed,
            cmap="plasma",
            linewidth=np.clip(speed * 2.0, 0.4, 2.5),
            arrowsize=1.2,
            zorder=2,
        )
        if strm.lines is not None:
            cbar = fig.colorbar(strm.lines, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label("Speed")

    if show_quiver and quiver_step > 0:
        step = quiver_step
        qy = np.arange(0, height, step)
        qx = np.arange(0, width, step)
        QY, QX = np.meshgrid(qy, qx, indexing="ij")
        Uq = u[QY, QX]
        Vq = v[QY, QX]
        ax.quiver(
            QX,
            QY,
            Uq,
            Vq,
            color="white",
            alpha=0.55,
            angles="xy",
            scale_units="xy",
            scale=1.0 / max(step * 0.35, 1e-6),
            width=0.0025,
            headwidth=3,
            headlength=4,
            zorder=3,
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Vector field")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Visualize a (H,W,2) vector field .npy file.")
    p.add_argument("--field", required=True, help="Path to .npy (shape H, W, 2)")
    p.add_argument(
        "--image",
        default=None,
        help="Optional background map (must match field dimensions)",
    )
    p.add_argument("--out", default="vector_field_viz.png", help="Output image path")
    p.add_argument(
        "--threshold",
        type=int,
        default=16,
        help="Obstacle luminance threshold when using --image (default: 16)",
    )
    p.add_argument("--dpi", type=float, default=150, help="Figure DPI (default: 150)")
    p.add_argument(
        "--stream-density",
        type=float,
        default=1.2,
        help="Matplotlib streamplot density (default: 1.2)",
    )
    p.add_argument(
        "--quiver-step",
        type=int,
        default=18,
        help="Pixel stride for quiver grid (0 = skip quiver; default: 18)",
    )
    p.add_argument("--no-stream", action="store_true", help="Hide streamlines")
    p.add_argument("--no-quiver", action="store_true", help="Hide quiver arrows")
    args = p.parse_args()

    try:
        field = load_field(args.field)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    quiver_step = 0 if args.no_quiver else args.quiver_step
    visualize_vector_field(
        field,
        args.out,
        image_path=args.image,
        threshold=args.threshold,
        dpi=args.dpi,
        stream_density=args.stream_density,
        quiver_step=quiver_step,
        show_stream=not args.no_stream,
        show_quiver=not args.no_quiver,
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
