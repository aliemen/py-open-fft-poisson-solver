from __future__ import annotations

from importlib import metadata as _metadata

from .hockney import (
    GridSpec,
    auto_bbox,
    convolve_open_poisson_hockney,
    efield_from_potential,
    gather_cic_vector,
    gradient_fd,
    greens_function_hockney,
    make_gridspec_from_particles,
    scatter_cic,
    solve_open_poisson_hockney,
)

__author__ = "Alexander Liemen"
__date__ = "2026-03-18"

try:
    __version__ = _metadata.version("pyHockneySolver")
except _metadata.PackageNotFoundError:  # pragma: no cover
    # Fallback for editable installs / direct source execution without installation metadata.
    __version__ = "0.0.0"

__license__ = "GPL-3.0-or-later"

__all__ = [
    "GridSpec",
    "auto_bbox",
    "make_gridspec_from_particles",
    "scatter_cic",
    "gather_cic_vector",
    "greens_function_hockney",
    "convolve_open_poisson_hockney",
    "gradient_fd",
    "efield_from_potential",
    "solve_open_poisson_hockney",
]

