import numpy as np
import argparse, h5py

from funcs.vtk_stuff import write_binary_grid_vtk_hexmesh
import funcs.igb2ngsolve as igb2ng


def main():
    parser = argparse.ArgumentParser(
        description='Visualize reconstruction and ground truth')
    parser.add_argument('--fem_idx',
                        default=0,
                        type=int,
                        help='idx of the fem solution')
    parser.add_argument('--fold_idx',
                        required=True,
                        type=int,
                        help='idx of the fold')
    parser.add_argument('--num_random_runs',
                        required=True,
                        type=int,
                        help='number of random runs')
    parser.add_argument('--num_snapshots',
                        required=True,
                        type=int,
                        help='number of snapshots')
    parser.add_argument('--num_reconstructions',
                        required=True,
                        type=int,
                        help='number of reconstructions')
    parser.add_argument('--permutation',
                        type=str,
                        default="P",
                        help='permutation or not [P,I]')
    parser.add_argument('--run_type',
                        required=True,
                        type=str,
                        help='type of run [seq, par]')
    parser.add_argument('--rb_alg',
                        required=True,
                        type=str,
                        help='name of RB algorithm')
    parser.add_argument('--ss_alg',
                        required=True,
                        type=str,
                        help='name of SS algorithm')
    parser.add_argument('--H', type=str, default="", help='minibatch size')
    parser.add_argument('--num_ss_components',
                        required=True,
                        type=int,
                        help='number of components detected by SS [2,3]')
    parser.add_argument('--data_basepath',
                        required=True,
                        type=str,
                        help='basepath of PBDW result data')
    parser.add_argument('--data_path',
                        required=True,
                        type=str,
                        help='path of the hdf5 input data')
    parser.add_argument('--SNR_type', required=True, type=str, help='SNR type')
    parser.add_argument('--SNR_value',
                        required=True,
                        type=float,
                        help='SNR value')
    parser.add_argument('--xi', required=True, type=float, help='xi value')

    args = parser.parse_args()

    if args.num_ss_components == 2:
        coverage = "sparsez"
        access = "riesz_reps_hz1_sz8"
    elif args.num_ss_components == 3:
        coverage = ""
        access = "riesz_reps_hz8_sz8"
    else:
        raise RuntimeError(f"{args.num_ss_components} must be [2,3]")

    # paths
    base_filename = f"{args.data_basepath}/mech" + \
        f"_{args.fold_idx}fold" + \
        f"_train{args.num_snapshots}" + \
        f"_test{args.num_reconstructions}" + \
        f"_runs{args.num_random_runs}" + \
        f"_{args.rb_alg}{args.ss_alg}{args.H}{coverage}" + \
        f"_{args.SNR_type}{args.SNR_value}_xi{args.xi}" + \
        f"_{args.run_type}run_{args.permutation}"

    ss_filename = f"{args.data_basepath}/mech" + \
        f"_{args.fold_idx}fold" + \
        f"_train{args.num_snapshots}" + \
        f"{args.ss_alg}{args.H}{coverage}{args.rb_alg}"

    # Load h5 file.
    f = h5py.File(args.data_path)

    # Load mesh.
    xyz, e2n, etags, surfdata = igb2ng.read_carp_mesh(f["mesh"])

    # Ground truth ( first random run )
    u_true = np.load(base_filename + "_true.npy")[args.fem_idx]
    noise = np.load(base_filename + "_noise.npy")[args.fem_idx *
                                                  args.num_random_runs]
    ndofs = u_true.shape[0]
    nrows = ndofs // 3

    # linear algebra
    chosen = np.load(ss_filename + "chosen.npy")
    quots = chosen // args.num_ss_components
    rests = chosen % args.num_ss_components
    isx = rests == 0
    isy = rests == 1
    isz = rests == 2
    Mx, My, Mz = isx.sum(), isy.sum(), isz.sum()

    # Load Riesz reprensenters.
    R1d = f[access]["rieszreps"][()]
    xyz_midpoints = f[access]["xyz_midpoints_voxels"][()]
    h = f[access]["h_step_size"][()]

    # Load SpaceMatrix.
    SpaceMatrix = igb2ng.load_sparse_csr(f[access]["sp_matrix"])
    num_voxels = R1d.shape[0]

    # Voxelization
    x_values = np.zeros(num_voxels, dtype=np.float64)
    y_values = np.zeros(num_voxels, dtype=np.float64)
    mag_values = np.zeros(num_voxels, dtype=np.float64)
    x_values[quots[
        isx]] = R1d[quots[isx]] @ SpaceMatrix @ u_true[:nrows] + noise[:Mx]
    y_values[quots[isy]] = R1d[
        quots[isy]] @ SpaceMatrix @ u_true[nrows:2 * nrows] + noise[Mx:Mx + My]

    nan_indices_x = np.setdiff1d(np.arange(0, num_voxels, 1), quots[isx])
    nan_indices_y = np.setdiff1d(np.arange(0, num_voxels, 1), quots[isy])
    nan_indices_z = np.setdiff1d(np.arange(0, num_voxels, 1), quots[isz])

    if args.num_ss_components == 3:
        z_values = np.zeros(num_voxels, dtype=np.float64)
        z_values[quots[isz]] = R1d[
            quots[isz]] @ SpaceMatrix @ u_true[2 * nrows:] + noise[Mx + My:]
        mag_values[:] = np.sqrt(x_values**2 + y_values**2 + z_values**2)

        x_values[nan_indices_x] = np.nan
        y_values[nan_indices_y] = np.nan
        z_values[nan_indices_z] = np.nan
        mag_values[np.intersect1d(np.intersect1d(nan_indices_x, nan_indices_y),
                                  nan_indices_z)] = np.nan
        write_binary_grid_vtk_hexmesh(
            filename=f"{base_filename}_vox{args.fem_idx}.vtk",
            voxel_centers=xyz_midpoints,
            hx=h[0],
            hy=h[1],
            hz=h[2],
            x_values=x_values,
            y_values=y_values,
            mag_values=mag_values,
            z_values=z_values)
    else:
        mag_values[:] = np.sqrt(x_values**2 + y_values**2)

        x_values[nan_indices_x] = np.nan
        y_values[nan_indices_y] = np.nan
        mag_values[np.intersect1d(nan_indices_x, nan_indices_y)] = np.nan
        write_binary_grid_vtk_hexmesh(
            filename=f"{base_filename}_vox{args.fem_idx}.vtk",
            voxel_centers=xyz_midpoints,
            hx=h[0],
            hy=h[1],
            hz=h[2],
            x_values=x_values,
            y_values=y_values,
            mag_values=mag_values)


if __name__ == "__main__":
    main()
