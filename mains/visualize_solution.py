import numpy as np
import argparse, h5py

from funcs.vtk_stuff import write_binary_mesh_vtk
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
    elif args.num_ss_components == 3:
        coverage = ""
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

    rb_filename = f"{args.data_basepath}/mech" + \
        f"_{args.fold_idx}fold" + \
        f"_train{args.num_snapshots}{args.rb_alg}"

    # Load h5 file.
    f = h5py.File(args.data_path)

    # Load mesh.
    xyz, e2n, etags, surfdata = igb2ng.read_carp_mesh(f["mesh"])

    # Ground truth
    u_true = np.load(base_filename +
                     "_true.npy")[args.fem_idx]  # first random run

    # Reconstruction
    Q = np.load(ss_filename + ".npy")
    chosen = np.load(ss_filename + "chosen.npy")
    V = np.load(rb_filename + ".npy")
    zeta_star = np.load(base_filename + "_zeta.npy")[args.fem_idx *
                                                     args.num_random_runs]
    eta_star = np.load(base_filename + "_eta.npy")[args.fem_idx *
                                                   args.num_random_runs]

    ndofs = u_true.shape[0]
    nrows = ndofs // 3
    rests = chosen % args.num_ss_components
    isx = rests == 0
    isy = rests == 1
    isz = rests == 2
    Mx, My, Mz = isx.sum(), isy.sum(), isz.sum()

    eta_rec = np.empty((ndofs), dtype=np.float64)
    zeta_rec = np.empty((ndofs), dtype=np.float64)
    u_rec = np.empty((ndofs), dtype=np.float64)
    eta_rec[:nrows] = Q[isx].T @ eta_star[:Mx]
    eta_rec[nrows:2 * nrows] = Q[isy].T @ eta_star[Mx:Mx + My]
    eta_rec[2 * nrows:] = Q[isz].T @ eta_star[Mx + My:]
    zeta_rec[:] = V.T @ zeta_star
    u_rec[:] = eta_rec + zeta_rec  # PBDW reconstruction

    # Save
    nodal_solutions = [('SS_rec', eta_rec, True), ('RB_rec', zeta_rec, True),
                       ('u_rec', u_rec, True), ('u_true', u_true, True),
                       ('abs_diff', abs(u_true - u_rec), True)]

    # Write the VTK file
    write_binary_mesh_vtk(f"{base_filename}_rec{args.fem_idx}.vtk",
                          xyz,
                          e2n,
                          nodal_solutions=nodal_solutions)


if __name__ == "__main__":
    main()
