# General.
import numpy as np
import os, json5
import argparse, h5py
from pathlib import Path
import ngsolve as ng
from sklearn.model_selection import ShuffleSplit

# My implementation.
from funcs.pbdw_runner import PBDWRunner
import funcs.igb2ngsolve as igb2ng


def main():
    parser = argparse.ArgumentParser(description='Run PBDW reconstruction')
    parser.add_argument('--json',
                        required=True,
                        type=str,
                        help='JSON file path')
    args = parser.parse_args()

    # Possible search paths
    json_paths = [
        Path(args.json).expanduser(),
        Path.home() / 'PBDW_method' / args.json
    ]

    # Find first existing file
    for path in json_paths:
        if path.exists():
            with open(path, 'r') as file:
                json_data = json5.load(file)
            break
    else:  # runs if loop didn't break
        raise RuntimeError(
            f"JSON file not found in: {', '.join(str(p) for p in json_paths)}")

    # Create output directory.
    path_result = json_data['general']['path_output_folder']
    if (not os.path.exists(path_result)):
        print(f'creating directory: {path_result}')
        os.makedirs(path_result)

    # Read params.
    RB = json_data['general']['RB']
    SS = json_data['general']['SS']
    noise_type = json_data['noise']['type']
    noise_SNR = json_data['noise'][noise_type]
    xi = json_data['noise']['xi']
    num_ss_components = json_data['SS'][SS]["sensor_dim"]
    num_splits = json_data['general']['num_splits']
    fraction_test = json_data['general']['fraction_test']
    shuffle_split = ShuffleSplit(
        n_splits=num_splits,
        test_size=fraction_test,
        random_state=json_data['general']['kfold_seed'])

    # Load h5 file.
    f = h5py.File(json_data['general']['path_data'])

    # Load mesh.
    xyz, e2n, etags, surfdata = igb2ng.read_carp_mesh(f["mesh"])
    surfnames = ['Top', 'Outside', 'Inside']
    mesh = igb2ng.carp_mesh_to_ngmesh(xyz, e2n, etags, surfdata, surfnames)

    # Load FEM solutions.
    solsNP = f["snapshots"]["snapshots150_original"][()]

    if num_ss_components == 2:
        access = "riesz_reps_hz1_sz8"
    elif num_ss_components == 3:
        access = "riesz_reps_hz8_sz8"
    else:
        raise RuntimeError(f"{num_ss_components} must be [2,3]")

    # Load Riesz representers.
    rieszRepsNP = f[access]["rieszreps"][()]

    # Load space matrix.
    SpaceMatrix = igb2ng.load_sparse_csr(f[access]["sp_matrix"])

    # FE space.
    fes = ng.VectorH1(mesh, order=1)

    # Dump JSON file.
    with open(path_result + f'/mech_{num_splits}fold' +
              f'_testfract{fraction_test}' + f'_{noise_type}{noise_SNR}' +
              f'_xi{xi}_params.json',
              'w',
              encoding='utf-8') as json_file:
        json5.dump(json_data, json_file, ensure_ascii=False, indent=4)

    # Apply PBDW.
    test_data = json_data['general'].get('test_data')

    if test_data is not None:
        # Single split: train on solsNP, test on external data
        split_iter = [(0, next(shuffle_split.split(solsNP))[0], None)]
    else:
        # K-fold cross-validation on solsNP
        split_iter = [
            (k, train_idx, test_idx)
            for k, (train_idx,
                    test_idx) in enumerate(shuffle_split.split(solsNP))
        ]

    for k, train_idx, test_idx in split_iter:
        train_data = solsNP[train_idx]

        if test_data is not None:
            eval_data = f["snapshots"][test_data]
        else:
            eval_data = solsNP[test_idx]

        runner = PBDWRunner(mesh, fes, train_data, eval_data, rieszRepsNP,
                            SpaceMatrix, RB, SS, json_data, k,
                            (xyz, e2n, etags))
        runner.run()


if __name__ == "__main__":
    main()
