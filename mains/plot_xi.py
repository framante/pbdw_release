import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse


def main():
    parser = argparse.ArgumentParser(description='Plot xi trend for SNR')
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
    parser.add_argument('--SNR_values',
                        required=True,
                        nargs='+',
                        type=float,
                        help='List of SNR values')
    parser.add_argument('--xi_values',
                        required=True,
                        nargs='+',
                        type=float,
                        help='List of xi values')
    args = parser.parse_args()

    meanerrs = np.empty((len(xi_values), num_reconstructions), dtype=np.float64)
    norms = ['L^2(\Omega)', 'H^1(\Omega)', 'H^1_0(\Omega)', 'L^\infty(\Omega)']
    SNR_values = np.array(args.SNR_values)
    xi_values = np.array(args.xi_values)

    if args.num_ss_components == 2:
        coverage = "sparsez"
    elif args.num_ss_components == 3:
        coverage = ""
    else:
        raise RuntimeError(f"{args.num_ss_components} must be [2,3]")

    for j, norm in enumerate(norms):
        plt.figure(figsize=(8, 6))

        meanerrs[:] = 0.
        for snr in SNR_values:
            for i, xi in enumerate(xi_values):
                meanerrs[i] = np.load(
                    f"{args.data_basepath}/mech" + f"_{args.fold_idx}fold" +
                    f"_train{args.num_snapshots}" +
                    f"_test{args.num_reconstructions}" +
                    f"_runs{args.num_random_runs}" +
                    f"_{args.rb_alg}{args.ss_alg}{args.H}{coverage}" +
                    f"_SNR{snr}_xi{xi}" +
                    f"_{args.run_type}_{args.permutation}" +
                    f"_meanerrors.npy")[:, j]

            mean = meanerrs.mean(axis=1)
            std = meanerrs.std(axis=1)
            line, = plt.plot(xi_values,
                             mean,
                             marker='o',
                             label=rf'$\widetilde{{SNR}}={snr}$')
            if band:
                plt.fill_between(xi_values,
                                 mean - std,
                                 mean + std,
                                 alpha=0.3,
                                 color=line.get_color())
        plt.xscale('log')
        plt.yscale('log')

        # Set x-axis ticks at specified xi values
        ax = plt.gca()
        ax.set_xticks(xi_values[::2])
        ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%.0e'))
        ax.get_yaxis().set_major_formatter(ticker.FormatStrFormatter('%.0e'))
        ax.tick_params(axis='both', labelsize=14)

        # Expand x-axis range slightly
        x_min = xi_values[1] / 10
        x_max = xi_values[-1] * 10
        plt.xlim(x_min, x_max)

        plt.xlabel(r'$\xi$', fontsize=16)
        plt.ylabel(rf'$\mathrm{{err}}_{{{norm}}}$', fontsize=16)
        # plt.title(f'L2 relative Error vs xi')
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"{args.data_basepath}/mech_{fold_idx}fold" +
                    f"_train{args.num_snapshots}" +
                    f"_test{args.num_reconstructions}" +
                    f"_runs{args.num_random_runs}" +
                    f"_{args.rb_alg}{args.ss_alg}{args.H}{coverage}" +
                    f"_{norm}_SNR_xi_trend.png",
                    dpi=300,
                    bbox_inches='tight')


if __name__ == "__main__":
    main()
