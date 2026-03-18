from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    origin: np.ndarray  # (3,)
    spacing: np.ndarray  # (3,)
    shape: tuple[int, int, int]  # (Nx,Ny,Nz)


def _auto_bbox(particles_xyz: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    mins = np.min(particles_xyz, axis=0)
    maxs = np.max(particles_xyz, axis=0)
    span = np.maximum(maxs - mins, 1e-30)
    mins = mins - padding * span
    maxs = maxs + padding * span
    return mins, maxs


def _make_gridspec(
    particles_xyz: np.ndarray,
    grid_shape: tuple[int, int, int],
    padding: float,
    bbox: tuple[float, float, float, float, float, float] | None,
) -> GridSpec:
    if bbox is None:
        mins, maxs = _auto_bbox(particles_xyz, padding=padding)
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


def cic_deposit(
    particles_xyz: np.ndarray,
    charge_per_particle: float,
    grid: GridSpec,
) -> np.ndarray:
    """
    Deposit charges onto a cell-centered grid using CIC (trilinear) shape.

    Returns rho on the grid as "charge per cell volume" (i.e. charge density),
    so that sum(rho * dV) == total charge deposited (up to particles outside domain).
    """
    pos = np.asarray(particles_xyz, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"particles_xyz must have shape (N,3), got {pos.shape}")

    Nx, Ny, Nz = grid.shape
    rho = np.zeros((Nx, Ny, Nz), dtype=float)

    inv_h = 1.0 / grid.spacing
    # Continuous cell index, where cell centers are at (i+0.5)*h from origin.
    s = (pos - grid.origin) * inv_h - 0.5

    i0 = np.floor(s).astype(np.int64)
    frac = s - i0  # in [0,1) for points inside

    # Skip particles that cannot deposit to 8 neighbors inside bounds.
    mask = (
        (i0[:, 0] >= 0)
        & (i0[:, 0] < Nx - 1)
        & (i0[:, 1] >= 0)
        & (i0[:, 1] < Ny - 1)
        & (i0[:, 2] >= 0)
        & (i0[:, 2] < Nz - 1)
    )
    if not np.any(mask):
        return rho

    i0 = i0[mask]
    fx, fy, fz = (frac[mask, 0], frac[mask, 1], frac[mask, 2])

    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    wz0 = 1.0 - fz
    wx1 = fx
    wy1 = fy
    wz1 = fz

    q = float(charge_per_particle)
    # Convert charge to charge density contribution per cell by dividing by cell volume.
    dV = float(np.prod(grid.spacing))
    q_over_dV = q / dV

    ix = i0[:, 0]
    iy = i0[:, 1]
    iz = i0[:, 2]

    # Accumulate 8 corners.
    np.add.at(rho, (ix, iy, iz), q_over_dV * (wx0 * wy0 * wz0))
    np.add.at(rho, (ix + 1, iy, iz), q_over_dV * (wx1 * wy0 * wz0))
    np.add.at(rho, (ix, iy + 1, iz), q_over_dV * (wx0 * wy1 * wz0))
    np.add.at(rho, (ix, iy, iz + 1), q_over_dV * (wx0 * wy0 * wz1))
    np.add.at(rho, (ix + 1, iy + 1, iz), q_over_dV * (wx1 * wy1 * wz0))
    np.add.at(rho, (ix + 1, iy, iz + 1), q_over_dV * (wx1 * wy0 * wz1))
    np.add.at(rho, (ix, iy + 1, iz + 1), q_over_dV * (wx0 * wy1 * wz1))
    np.add.at(rho, (ix + 1, iy + 1, iz + 1), q_over_dV * (wx1 * wy1 * wz1))

    return rho


def cic_gather_vector(
    positions_xyz: np.ndarray,
    vec_grid: np.ndarray,
    grid: GridSpec,
) -> np.ndarray:
    """
    Gather a vector field defined on cell centers back to particle positions using CIC.

    vec_grid is shape (Nx,Ny,Nz,3).
    """
    pos = np.asarray(positions_xyz, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions_xyz must have shape (N,3), got {pos.shape}")

    if vec_grid.ndim != 4 or vec_grid.shape[3] != 3:
        raise ValueError(f"vec_grid must have shape (Nx,Ny,Nz,3), got {vec_grid.shape}")

    Nx, Ny, Nz, _ = vec_grid.shape

    inv_h = 1.0 / grid.spacing
    s = (pos - grid.origin) * inv_h - 0.5

    i0 = np.floor(s).astype(np.int64)
    frac = s - i0

    out = np.zeros((pos.shape[0], 3), dtype=float)

    mask = (
        (i0[:, 0] >= 0)
        & (i0[:, 0] < Nx - 1)
        & (i0[:, 1] >= 0)
        & (i0[:, 1] < Ny - 1)
        & (i0[:, 2] >= 0)
        & (i0[:, 2] < Nz - 1)
    )
    if not np.any(mask):
        return out

    idx = np.nonzero(mask)[0]
    i0m = i0[mask]
    fx, fy, fz = (frac[mask, 0], frac[mask, 1], frac[mask, 2])

    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    wz0 = 1.0 - fz
    wx1 = fx
    wy1 = fy
    wz1 = fz

    ix = i0m[:, 0]
    iy = i0m[:, 1]
    iz = i0m[:, 2]

    def acc(w: np.ndarray, i: np.ndarray, j: np.ndarray, k: np.ndarray) -> None:
        out[idx] += vec_grid[i, j, k] * w[:, None]

    acc(wx0 * wy0 * wz0, ix, iy, iz)
    acc(wx1 * wy0 * wz0, ix + 1, iy, iz)
    acc(wx0 * wy1 * wz0, ix, iy + 1, iz)
    acc(wx0 * wy0 * wz1, ix, iy, iz + 1)
    acc(wx1 * wy1 * wz0, ix + 1, iy + 1, iz)
    acc(wx1 * wy0 * wz1, ix + 1, iy, iz + 1)
    acc(wx0 * wy1 * wz1, ix, iy + 1, iz + 1)
    acc(wx1 * wy1 * wz1, ix + 1, iy + 1, iz + 1)

    return out


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


def solve_open_poisson_hockney(
    particles_xyz: np.ndarray,
    *,
    charge_per_particle: float,
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
    pos = np.asarray(particles_xyz, dtype=float)
    grid = _make_gridspec(pos, grid_shape=grid_shape, padding=padding, bbox=bbox)

    rho = cic_deposit(pos, charge_per_particle=charge_per_particle, grid=grid)
    phi = convolve_open_poisson_hockney(rho, grid.spacing, eps0=eps0)

    grad_phi = gradient_fd(phi, grid.spacing)
    E_grid = -grad_phi
    E_particles = cic_gather_vector(pos, E_grid, grid=grid)

    return {
        "rho_grid": rho,
        "phi_grid": phi,
        "E_grid": E_grid,
        "E_particles": E_particles,
        "origin": grid.origin.copy(),
        "spacing": grid.spacing.copy(),
        "shape": grid.shape,
    }

