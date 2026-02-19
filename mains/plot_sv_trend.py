# General.
import numpy as np
import os, glob, h5py
import matplotlib.pyplot as plt
import argparse

# My implementation.
import funcs.igb2ngsolve as igb2ng
from funcs.rb import PODBasis


def main():
    parser = argparse.ArgumentParser(description='Plot spectral content')
    parser.add_argument('--eigvals_path',
                        required=False,
                        type=str,
                        help='path of the eigenvalues')
    parser.add_argument('--data_path',
                        required=True,
                        type=str,
                        help='path of the hdf5 input data')
    parser.add_argument('--output_path',
                        required=True,
                        type=str,
                        help='path of the output files')
    args = parser.parse_args()

    if (not os.path.exists(args.output_path)):
        os.mkdir(args.output_path)

    # Eigenvalues have been already computed
    if (args.eigvals_path is not None and os.path.exists(args.eigvals_path)):
        all_evals = np.load(args.eigvals_path, allow_pickle=True).item()
    # Compute eigenvalues
    else:
        # Load h5 file.
        f = h5py.File(args.data_path)

        # Load mesh.
        xyz, e2n, etags, surfdata = igb2ng.read_carp_mesh(f["mesh"])

        # Load SpaceMatrix.
        SpaceMatrix = igb2ng.load_sparse_csr(
            f["riesz_reps_hz8_sz8"]["sp_matrix"])

        all_evals = {}
        for val in np.arange(100, 301, 25):

            # Load snapshots
            W = f["snapshots"][f"snapshots{val}"][()]

            # Dataset size
            n_samples = W.shape[1]

            # Compute eigenvalues
            evals = PODBasis(SpaceMatrix, W, 3)[0]
            all_evals[val] = evals

        # Save eigvals
        np.save(args.output_path + "/eigvals.npy", all_evals, allow_pickle=True)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for val in np.arange(100, 301, 25):
        # Plot eigenvalue decay
        ax1.plot(np.arange(len(all_evals[val])),
                 all_evals[val],
                 label=rf"$N_{{\text{{samples}}}} = {val}$",
                 linewidth=2,
                 alpha=0.8)

    # Improve visualization
    ax1.set_yscale("log")
    ax1.set_xlabel("Sample index", fontsize=18)
    ax1.set_ylabel("Eigenvalue", fontsize=18)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", frameon=True, shadow=True, fontsize=18)

    fig1.tight_layout()
    fig1.savefig(args.output_path + "/all_curves.png",
                 dpi=300,
                 bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(8, 6))

    vals = np.arange(100, 301, 25)
    evals_list = [all_evals[val] for val in vals]

    # Find the minimum length across all eigenvalue arrays
    min_len = min(len(evals) for evals in evals_list)

    # Truncate all arrays to the same length
    evals_truncated = np.array([evals[:min_len] for evals in evals_list])

    # Compute statistics
    mean_evals = np.mean(evals_truncated, axis=0)
    std_evals = np.std(evals_truncated, axis=0)

    x = np.arange(min_len)

    # Plot mean
    ax2.plot(x, mean_evals, linewidth=3, label="Mean eigenvalue decay")

    # Plot shaded variability (±1 std)
    ax2.fill_between(x,
                     mean_evals - std_evals,
                     mean_evals + std_evals,
                     alpha=0.3,
                     label=r"$\pm\sigma$")

    # Styling
    ax2.set_yscale("log")
    ax2.set_xlabel("Sample index", fontsize=18)
    ax2.set_ylabel("Eigenvalue", fontsize=18)
    ax2.tick_params(axis="both", which="major", labelsize=16)
    # ax2.tick_params(axis="y", which="minor", labelsize=30)
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", frameon=True, shadow=True, fontsize=18)

    fig2.tight_layout()
    fig2.savefig(args.output_path + "mean_plus_variance.png",
                 dpi=300,
                 bbox_inches="tight")


if __name__ == "__main__":
    main()
