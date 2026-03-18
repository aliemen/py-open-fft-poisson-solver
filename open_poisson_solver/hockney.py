from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    origin: np.ndarray  # (3,)
    spacing: np.ndarray  # (3,)
    shape: tuple[int, int, int]  # (Nx,Ny,Nz)


def auto_bbox(particles_xyz: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    mins = np.min(particles_xyz, axis=0)
    maxs = np.max(particles_xyz, axis=0)
    span = np.maximum(maxs - mins, 1e-30)
    mins = mins - padding * span
    maxs = maxs + padding * span
    return mins, maxs


def make_gridspec_from_particles(
    particles_xyz: np.ndarray,
    grid_shape: tuple[int, int, int],
    padding: float,
    bbox: tuple[float, float, float, float, float, float] | None,
) -> GridSpec:
    if bbox is None:
        mins, maxs = auto_bbox(particles_xyz, padding=padding)
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        mins = np.array([xmin, ymin, zmin], dtype=float)
        maxs = np.array([xmax, ymax, zmax], dtype=float)

    shape = tuple(int(n) for n in grid_shape)
    if any(n < 2 for n in shape):
        raise ValueError(f"grid_shape must be >=2 in each dim, got {shape}")

    # Cell-centered grid spanning [mins, maxs] with Nx cells; spacing is domain_len/Nx.
    lens = maxs - mins
    spacing = lens / np.array(shape, dtype=float)
    origin = mins
    return GridSpec(origin=origin, spacing=spacing, shape=shape)


def _as_positions(particles_xyz: np.ndarray, *, name: str = "particles_xyz") -> np.ndarray:
    pos = np.asarray(particles_xyz, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3), got {pos.shape}")
    return pos


def _cic_indices_and_weights(
    positions_xyz: np.ndarray,
    grid: GridSpec,
    *,
    require_inside: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return i0 indices and CIC weights for cell-centered grids.

    positions_xyz: (N,3)
    Returns (i0, wx0, wy0, wz0, mask) where i0 is int (N,3) and weights are float (N,).
    If require_inside is True, raises if any particle is outside the valid [0, N-2] neighborhood.
    """
    pos = _as_positions(positions_xyz, name="positions_xyz")
    Nx, Ny, Nz = grid.shape

    inv_h = 1.0 / grid.spacing
    s = (pos - grid.origin) * inv_h - 0.5

    i0 = np.floor(s).astype(np.int64)
    frac = s - i0

    inside = (
        (i0[:, 0] >= 0)
        & (i0[:, 0] < Nx - 1)
        & (i0[:, 1] >= 0)
        & (i0[:, 1] < Ny - 1)
        & (i0[:, 2] >= 0)
        & (i0[:, 2] < Nz - 1)
    )
    if require_inside and not np.all(inside):
        bad = np.nonzero(~inside)[0]
        raise ValueError(
            f"{bad.size} particles are outside the grid domain for CIC "
            f"(need neighbors within [0..N-1]); first bad index={int(bad[0])}"
        )

    fx, fy, fz = (frac[:, 0], frac[:, 1], frac[:, 2])
    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    wz0 = 1.0 - fz
    return i0, wx0, wy0, wz0, inside


def scatter_cic(
    particles_xyz: np.ndarray,
    *,
    grid: GridSpec,
    charge_per_particle: float | None = None,
    particle_charges: np.ndarray | None = None,
) -> np.ndarray:
    """
    Deposit charges onto a cell-centered grid using CIC (trilinear) shape.

    Exactly one of:
      - charge_per_particle (scalar)
      - particle_charges (shape (N,))

    Returns rho on the grid as charge density, so that sum(rho * dV) == total charge.

    Outside-domain policy: raises if any particle cannot deposit to its 8 neighbors.
    """
    pos = _as_positions(particles_xyz)
    N = pos.shape[0]

    has_scalar = charge_per_particle is not None
    has_array = particle_charges is not None
    if has_scalar == has_array:
        raise ValueError("Provide exactly one of charge_per_particle or particle_charges")

    if particle_charges is not None:
        q = np.asarray(particle_charges, dtype=float)
        if q.shape != (N,):
            raise ValueError(f"particle_charges must have shape ({N},), got {q.shape}")
    else:
        q = np.full((N,), float(charge_per_particle), dtype=float)

    Nx, Ny, Nz = grid.shape
    rho = np.zeros((Nx, Ny, Nz), dtype=float)

    i0, wx0, wy0, wz0, inside = _cic_indices_and_weights(pos, grid, require_inside=True)

    fx = 1.0 - wx0
    fy = 1.0 - wy0
    fz = 1.0 - wz0
    wx1, wy1, wz1 = fx, fy, fz

    dV = float(np.prod(grid.spacing))
    q_over_dV = q / dV

    ix = i0[:, 0]
    iy = i0[:, 1]
    iz = i0[:, 2]

    np.add.at(rho, (ix, iy, iz), q_over_dV * (wx0 * wy0 * wz0))
    np.add.at(rho, (ix + 1, iy, iz), q_over_dV * (wx1 * wy0 * wz0))
    np.add.at(rho, (ix, iy + 1, iz), q_over_dV * (wx0 * wy1 * wz0))
    np.add.at(rho, (ix, iy, iz + 1), q_over_dV * (wx0 * wy0 * wz1))
    np.add.at(rho, (ix + 1, iy + 1, iz), q_over_dV * (wx1 * wy1 * wz0))
    np.add.at(rho, (ix + 1, iy, iz + 1), q_over_dV * (wx1 * wy0 * wz1))
    np.add.at(rho, (ix, iy + 1, iz + 1), q_over_dV * (wx0 * wy1 * wz1))
    np.add.at(rho, (ix + 1, iy + 1, iz + 1), q_over_dV * (wx1 * wy1 * wz1))

    return rho


def cic_deposit(particles_xyz: np.ndarray, charge_per_particle: float, grid: GridSpec) -> np.ndarray:
    """Backward-compatible alias for scalar-charge scatter."""
    return scatter_cic(particles_xyz, grid=grid, charge_per_particle=charge_per_particle)


def gather_cic_vector(
    positions_xyz: np.ndarray,
    vec_grid: np.ndarray,
    grid: GridSpec,
) -> np.ndarray:
    """
    Gather a vector field defined on cell centers back to particle positions using CIC.

    vec_grid is shape (Nx,Ny,Nz,3).
    """
    pos = _as_positions(positions_xyz, name="positions_xyz")

    if vec_grid.ndim != 4 or vec_grid.shape[3] != 3:
        raise ValueError(f"vec_grid must have shape (Nx,Ny,Nz,3), got {vec_grid.shape}")

    Nx, Ny, Nz, _ = vec_grid.shape

    i0, wx0, wy0, wz0, inside = _cic_indices_and_weights(pos, grid, require_inside=True)

    fx = 1.0 - wx0
    fy = 1.0 - wy0
    fz = 1.0 - wz0
    wx1, wy1, wz1 = fx, fy, fz

    ix = i0[:, 0]
    iy = i0[:, 1]
    iz = i0[:, 2]

    out = np.zeros((pos.shape[0], 3), dtype=float)
    out += vec_grid[ix, iy, iz] * (wx0 * wy0 * wz0)[:, None]
    out += vec_grid[ix + 1, iy, iz] * (wx1 * wy0 * wz0)[:, None]
    out += vec_grid[ix, iy + 1, iz] * (wx0 * wy1 * wz0)[:, None]
    out += vec_grid[ix, iy, iz + 1] * (wx0 * wy0 * wz1)[:, None]
    out += vec_grid[ix + 1, iy + 1, iz] * (wx1 * wy1 * wz0)[:, None]
    out += vec_grid[ix + 1, iy, iz + 1] * (wx1 * wy0 * wz1)[:, None]
    out += vec_grid[ix, iy + 1, iz + 1] * (wx0 * wy1 * wz1)[:, None]
    out += vec_grid[ix + 1, iy + 1, iz + 1] * (wx1 * wy1 * wz1)[:, None]

    return out


def cic_gather_vector(positions_xyz: np.ndarray, vec_grid: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Backward-compatible alias for CIC gather."""
    return gather_cic_vector(positions_xyz, vec_grid, grid)


def gradient_fd(phi: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """
    Centered finite-difference gradient on a cell-centered grid.

    Returns grad(phi) with shape (Nx,Ny,Nz,3).
    Boundary cells use one-sided differences.
    """
    if phi.ndim != 3:
        raise ValueError(f"phi must be 3D, got shape {phi.shape}")
    hx, hy, hz = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

    gx = np.empty_like(phi)
    gy = np.empty_like(phi)
    gz = np.empty_like(phi)

    gx[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * hx)
    gy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * hy)
    gz[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * hz)

    gx[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / hx
    gx[-1, :, :] = (phi[-1, :, :] - phi[-2, :, :]) / hx
    gy[:, 0, :] = (phi[:, 1, :] - phi[:, 0, :]) / hy
    gy[:, -1, :] = (phi[:, -1, :] - phi[:, -2, :]) / hy
    gz[:, :, 0] = (phi[:, :, 1] - phi[:, :, 0]) / hz
    gz[:, :, -1] = (phi[:, :, -1] - phi[:, :, -2]) / hz

    return np.stack([gx, gy, gz], axis=3)


def greens_function_hockney(
    shape2: tuple[int, int, int],
    physical_shape: tuple[int, int, int],
    spacing: np.ndarray,
    *,
    regularize_origin: bool = True,
) -> np.ndarray:
    """
    Build Hockney free-space Green's function on the doubled grid.

    Mirrors IPPL's approach: for each dimension d, use squared folded index distance:
      r_d^2 = (i)^2 if i < N_d else (2*N_d - i)^2
    then r = sqrt(sum_d (r_d^2 * h_d^2)), and G = -1/(4*pi*r),
    with a special value at the origin to avoid singularity.
    """
    Nx2, Ny2, Nz2 = shape2
    Nx, Ny, Nz = physical_shape
    if (Nx2, Ny2, Nz2) != (2 * Nx, 2 * Ny, 2 * Nz):
        raise ValueError("shape2 must be exactly doubled physical_shape")

    ix = np.arange(Nx2, dtype=float)
    iy = np.arange(Ny2, dtype=float)
    iz = np.arange(Nz2, dtype=float)

    dx = np.where(ix < Nx, ix, 2 * Nx - ix)
    dy = np.where(iy < Ny, iy, 2 * Ny - iy)
    dz = np.where(iz < Nz, iz, 2 * Nz - iz)

    hrsq = spacing * spacing

    r2 = (
        (dx[:, None, None] ** 2) * hrsq[0]
        + (dy[None, :, None] ** 2) * hrsq[1]
        + (dz[None, None, :] ** 2) * hrsq[2]
    )
    r = np.sqrt(r2, dtype=float)

    G = -1.0 / (4.0 * np.pi * np.where(r > 0.0, r, 1.0))
    if regularize_origin:
        G[0, 0, 0] = -1.0 / (4.0 * np.pi)
    return G


def convolve_open_poisson_hockney(
    rho: np.ndarray,
    spacing: np.ndarray,
    *,
    eps0: float | None = None,
) -> np.ndarray:
    """
    Solve laplace(phi) = -rho on an open domain using Hockney doubled-grid convolution.

    rho is the physical-grid charge density (Nx,Ny,Nz). Returns phi on the physical grid.
    """
    if rho.ndim != 3:
        raise ValueError(f"rho must be 3D, got shape {rho.shape}")
    Nx, Ny, Nz = rho.shape
    shape2 = (2 * Nx, 2 * Ny, 2 * Nz)

    rho2 = np.zeros(shape2, dtype=float)
    rho2[:Nx, :Ny, :Nz] = rho

    G2 = greens_function_hockney(shape2, (Nx, Ny, Nz), spacing)
    if eps0 is not None:
        G2 = G2 / float(eps0)

    rho2k = np.fft.rfftn(rho2)
    G2k = np.fft.rfftn(G2)

    # IPPL does: rho_hat = FFT(rho); phi_hat = -(rho_hat * G_hat); phi = IFFT(phi_hat)
    phi2 = np.fft.irfftn(-(rho2k * G2k), s=shape2)

    # Match IPPL normalization for Hockney: multiply by prod_d (2*N_d*h_d)
    norm = (2.0 * Nx * spacing[0]) * (2.0 * Ny * spacing[1]) * (2.0 * Nz * spacing[2])
    phi2 *= norm

    return phi2[:Nx, :Ny, :Nz].copy()


def efield_from_potential(phi: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Compute E = -grad(phi) on the grid."""
    return -gradient_fd(phi, spacing)


def solve_open_poisson_hockney(
    particles_xyz: np.ndarray,
    *,
    charge_per_particle: float | None = None,
    particle_charges: np.ndarray | None = None,
    grid_shape: tuple[int, int, int],
    padding: float = 0.2,
    bbox: tuple[float, float, float, float, float, float] | None = None,
    eps0: float | None = None,
) -> dict[str, Any]:
    """
    End-to-end pipeline:
      1) CIC scatter -> rho_grid
      2) Hockney FFT open Poisson -> phi_grid
      3) E_grid = -grad(phi_grid) via finite differences
      4) CIC gather E_grid -> E_particles
    """
    pos = _as_positions(particles_xyz)
    grid = make_gridspec_from_particles(pos, grid_shape=grid_shape, padding=padding, bbox=bbox)

    rho = scatter_cic(
        pos,
        grid=grid,
        charge_per_particle=charge_per_particle,
        particle_charges=particle_charges,
    )
    phi = convolve_open_poisson_hockney(rho, grid.spacing, eps0=eps0)

    E_grid = efield_from_potential(phi, grid.spacing)
    E_particles = gather_cic_vector(pos, E_grid, grid=grid)

    return {
        "rho_grid": rho,
        "phi_grid": phi,
        "E_grid": E_grid,
        "E_particles": E_particles,
        "origin": grid.origin.copy(),
        "spacing": grid.spacing.copy(),
        "shape": grid.shape,
    }

