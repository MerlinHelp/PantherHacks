"""
Compatibility entry point: re-exports the **fluidics** workflow (Poisson + sources).

- Fluid simulation: ``GenerateVectorFieldFluid.py`` (same API as this module).
- Arrow + RBF workflow: ``GenerateVectorFieldArrows.py``.

Running ``python GenerateVectorField.py`` is equivalent to ``python GenerateVectorFieldFluid.py``.
"""

from __future__ import annotations

from GenerateVectorFieldFluid import (
    build_vector_field_from_fluidics,
    collect_sources_interactive,
    generate_vector_field,
    load_image_and_free_mask,
    main,
    solve_poisson_sor,
    velocity_from_phi,
)

__all__ = [
    "build_vector_field_from_fluidics",
    "collect_sources_interactive",
    "generate_vector_field",
    "load_image_and_free_mask",
    "main",
    "solve_poisson_sor",
    "velocity_from_phi",
]

if __name__ == "__main__":
    raise SystemExit(main())
