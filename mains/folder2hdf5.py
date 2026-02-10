import numpy as np
import h5py, argparse
from pathlib import Path


def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}  shape={obj.shape} dtype={obj.dtype}")
    else:
        print(name)


def store_file(h5, file_path, root):

    rel = file_path.relative_to(root)
    parts = list(rel.parts)

    suffix = file_path.suffix.lower()

    # Keep extensions ONLY inside mesh/
    if parts[0] != "mesh":
        parts[-1] = Path(parts[-1]).stem  # strip final suffix

    h5_path = "/".join(parts)

    # Ensure parent groups exist
    parent = "/".join(h5_path.split("/")[:-1])
    if parent:
        h5.require_group(parent)

    # ======================
    # NPY
    # ======================
    if suffix == ".npy":

        arr = np.load(file_path)

        h5.create_dataset(h5_path,
                          data=arr,
                          compression="gzip",
                          compression_opts=4,
                          chunks=True)

    # ======================
    # NPZ → group
    # ======================
    elif suffix == ".npz":

        grp = h5.require_group(h5_path)
        data = np.load(file_path)

        for k in data.files:
            grp.create_dataset(k,
                               data=data[k],
                               compression="gzip",
                               compression_opts=4,
                               chunks=True)

    # ======================
    # VTX (only OUTSIDE mesh)
    # ======================
    elif suffix == ".vtx" and parts[0] != "mesh":

        indices = np.loadtxt(file_path, dtype=np.int64)

        h5.create_dataset(h5_path,
                          data=indices,
                          compression="gzip",
                          compression_opts=4)

    # ======================
    # Everything else (mesh binaries + mesh vtx)
    # ======================
    else:

        with open(file_path, "rb") as f:
            raw = f.read()

        h5.create_dataset(h5_path, data=np.void(raw))


def main():
    parser = argparse.ArgumentParser(
        description='Convert a folder into a hdf5 file')
    parser.add_argument('--folder-path',
                        required=True,
                        type=str,
                        help='full path of the target folder')
    parser.add_argument('--output-path',
                        required=True,
                        type=str,
                        help='full path of the output hdf5 file')
    args = parser.parse_args()

    OUT = Path(args.output_path)
    ROOT = Path(args.folder_path)

    with h5py.File(OUT, "w") as h5:

        for path in ROOT.rglob("*"):
            if path.is_file():
                print("Storing:", path)
                store_file(h5, path, ROOT)

    print("\nSaved to:", OUT)

    with h5py.File(OUT, "r") as f:
        f.visititems(walk)


if __name__ == "__main__":
    main()
