import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(description='Plot Kfold cross-validation')
    parser.add_argument('--num_folds',
                        required=True,
                        type=int,
                        help='number of cross-validation folds')
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
    parser.add_argument('--data_basepath',
                        required=True,
                        type=str,
                        help='basepath of data')
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
    parser.add_argument('--SNR_type',
                        required=True,
                        type=str,
                        help='type of SNR [NONE, LD, SNR]')
    parser.add_argument('--SNR_value',
                        required=True,
                        type=float,
                        help='value of SNR')
    parser.add_argument('--xi', required=True, type=float, help='value of xi')
    args = parser.parse_args()

    meanerrs = [np.empty(num_recon) for i in range(args.num_folds)]
    norms = ['L^2', 'H^1', 'H^1_0', 'L^\infty']

    if args.num_ss_components == 2:
        coverage = "sparsez"
    elif args.num_ss_components == 3:
        coverage = ""
    else:
        raise RuntimeError(f"{args.num_ss_components} must be [2,3]")

    for j, norm in enumerate(norms):
        fig, ax = plt.subplots(figsize=(8, 6))
        for idx in range(num_folds):
            meanerrs[idx] = np.load(
                f"{args.data_basepath}/mech_{idx}fold" +
                f"_train{args.num_snapshots}" +
                f"_test{args.num_reconstructions}" +
                f"_runs{args.num_random_runs}" +
                f"_{args.rb_alg}{args.ss_alg}{args.H}{coverage}" +
                f"_{args.SNR_type}{args.SNR_value}_xi{args.xi}" +
                f"_{args.run_type}_{args.permutation}" + f"_meanerrors.npy")[:,
                                                                             j]
        box = ax.boxplot(meanerrs, patch_artist=True)  #, showmeans=True)

        # Label each box
        ax.set_xticks(range(1, num_folds + 1))
        ax.set_xticklabels([rf'$Fold \, {i+1}$' for i in range(num_folds)],
                           fontsize=14)
        ax.tick_params(axis='y', labelsize=14)

        # Styling
        ax.set_ylabel(rf'$\mathrm{{err}}_{{{norm}}}$', fontsize=16)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(f"{data_basepath}/mech" + f"_train{args.num_snaphots}" +
                    f"_test{args.num_reconstructions}" +
                    f"_runs{args.num_random_runs}" +
                    f"_{args.rb_alg}{args.ss_alg}{args.H}{coverage}" +
                    f"_{args.SNR_type}{args.SNR_value}_xi{args.xi}" +
                    f"_{norm}kfolds.png",
                    dpi=300,
                    bbox_inches='tight')


if __name__ == "__main__":
    main()
