import argparse
import numpy as np


def extract_values(H, ss_filename):
    """
    Placeholder for real parsing logic.

    Must return a dict with:
      H, N_over_3, M, t
    """

    # Example dummy values
    return {
        "H": H,
        "N_over_3": np.load(ss_filename + ".npy").shape[1],
        "M": np.load(ss_filename + "chosen.npy").shape[0],
        "t": float(np.load(ss_filename + "time.npy")),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build Table 2 PBDW result files")

    parser.add_argument('--fold_idx',
                        required=True,
                        type=int,
                        help='idx of the fold')
    parser.add_argument('--num_snapshots',
                        required=True,
                        type=int,
                        help='number of snapshots')
    parser.add_argument('--data_basepath',
                        required=True,
                        type=str,
                        help='basepath of data')
    parser.add_argument('--ss_alg',
                        required=True,
                        type=str,
                        help='name of SS algorithm')
    parser.add_argument('--H_list',
                        required=True,
                        type=int,
                        nargs="+",
                        help='minibatch sizes')
    parser.add_argument('--num_ss_components',
                        required=True,
                        type=int,
                        help='number of components detected by SS [2,3]')
    parser.add_argument("--out", default="table2.txt", help="Output TXT file")

    args = parser.parse_args()

    if args.num_ss_components == 2:
        coverage = "sparsez"
    elif args.num_ss_components == 3:
        coverage = ""
    else:
        raise RuntimeError(f"{args.num_ss_components} must be [2,3]")

    rows = []
    for H in args.H_list:
        rows.append(
            extract_values(
                H, f"{args.data_basepath}/mech" + f"_{args.fold_idx}fold" +
                f"_train{args.num_snapshots}" + f"_{args.ss_alg}{H}{coverage}"))

    headers = ["H", "N/3", "M", "t [s]"]

    # Compute column widths
    columns = [
        [str(r["H"]) for r in rows],
        [str(r["N_over_3"]) for r in rows],
        [str(r["M"]) for r in rows],
        [f"{r['t']:.2e}" for r in rows],
    ]

    widths = [
        max(len(h), max(len(c) for c in col))
        for h, col in zip(headers, columns)
    ]

    def fmt_row(values):
        return " | ".join(v.rjust(w) for v, w in zip(values, widths))

    sep = "-+-".join("-" * w for w in widths)

    with open(args.out, "w") as f:
        f.write(fmt_row(headers) + "\n")
        f.write(sep + "\n")

        for r in rows:
            f.write(
                fmt_row([
                    str(r["H"]),
                    str(r["N_over_3"]),
                    str(r["M"]),
                    f"{r['t']:.2e}",
                ]) + "\n")

    print(f"Table written to {args.out}")


if __name__ == "__main__":
    main()
