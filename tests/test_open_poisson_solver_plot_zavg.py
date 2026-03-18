import numpy as np


def main() -> None:
    import matplotlib.pyplot as plt

    from open_poisson_solver import solve_open_poisson_hockney

    rng = np.random.default_rng(1)

    # Normally distributed bunch.
    N = 200_000
    sigma = np.array([1e-3, 1e-3, 2e-3])
    particles = rng.normal(size=(N, 3)) * sigma[None, :]

    out = solve_open_poisson_hockney(
        particles,
        charge_per_particle=1.0,
        grid_shape=(64, 64, 96),
        padding=0.3,
    )

    phi = out["phi_grid"]
    origin = out["origin"]
    spacing = out["spacing"]

    # Average potential over z.
    phi_xy = phi.mean(axis=2)  # (Nx,Ny)

    Nx, Ny, _ = phi.shape
    x = origin[0] + (np.arange(Nx) + 0.5) * spacing[0]
    y = origin[1] + (np.arange(Ny) + 0.5) * spacing[1]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(
        phi_xy.T,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        aspect="auto",
    )
    ax.set_title("Open Poisson potential (z-averaged)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("phi")

    outpath = "tests/phi_zavg.png"
    fig.savefig(outpath, dpi=150)
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()

