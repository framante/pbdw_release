import numpy as np
import timeit as tt
import ngsolve as ng
import multiprocessing as mp
import os
from functools import partial

import funcs.rb as rb
import funcs.actions as actions
from funcs.wOMP import wOMP
from funcs.vtk_stuff import write_binary_mesh_vtk
import funcs.shmem as shmem
import funcs.seqmem as seqmem


class PBDWRunner:
    """
    Orchestrates a full PBDW run (offline + online + output).
    """

    def __init__(self,
                 mesh,
                 fes,
                 W,
                 utest_mat,
                 R1d,
                 SpaceMatrix,
                 rb_alg,
                 ss_alg,
                 json_data,
                 k_split,
                 carpmesh=None):
        """
        Initialize a PBDWRunner instance.

        Stores all inputs required for a full PBDW run, extracts problem
        dimensions, initializes output paths, and sets up internal state.
        No computation is performed at this stage.
        """

        # ---- inputs (same as pbdw_algorithm) ----
        self.mesh = mesh
        self.fes = fes
        self.W = W
        self.utest_mat = utest_mat
        self.R1d = R1d
        self.SpaceMatrix = SpaceMatrix
        self.rb_alg = rb_alg
        self.ss_alg = ss_alg
        self.json_data = json_data
        self.k_split = k_split
        self.carpmesh = carpmesh

        # ---- derived ----
        self.num_train = W.shape[0]
        self.num_test = utest_mat.shape[0]
        self.dim_m = mesh.dim
        self.ndofs = fes.ndof
        self.nrows = SpaceMatrix.shape[0]
        self.num_norms = 5

        self.path = (json_data['general']['path_output_folder'] +
                     f"/mech_{k_split}fold_train{self.num_train}")
        self.permutation = self.json_data['algebra']['permutation']
        self.run_type = self.json_data['algebra']['run_type']
        self.dim_s = self.json_data['SS'][ss_alg]['sensor_dim']
        self.xi = self.json_data['noise']['xi']
        self.sparse_attr = 'sparsez' if self.dim_s == 2 else ''
        self.H = self.json_data['SS'][self.ss_alg].get('H', '')

        # ---- internal state ----
        self.V = self.Q = self.Qg = self.lhs = None
        self.Pmat = self.Ttmat = None
        self.R3d = self.Q3d = self.W3dinv = None
        self.RBdata = self.RBtime = None
        self.SSdata = self.SStime = self.assemble_time = None
        self.isx = self.isy = self.isz = None
        self.quots = self.rests = self.chosen = None
        self.R3dx = self.R3dy = self.R3dz = None
        self.Fx = self.Fy = self.Fz = None
        self.perm = self.invperm = None
        self.M = self.N = self.Mx = self.My = self.Mz = None
        self.results = None

    def run(self):
        """
        Execute the full PBDW pipeline.

        Runs offline RB and SS construction, assembles the linear system,
        performs online reconstruction (parallel or sequential), saves all
        results to disk, and releases allocated memory.
        """

        self._build_rb()
        self._build_ss()
        self._assemble_system()
        self._run_online()
        self._save_results()
        self._cleanup()

    def _build_rb(self):
        """
        Build or load the Reduced Basis (RB).

        Loads a precomputed RB if available, otherwise constructs it using
        the selected RB algorithm (e.g. POD or Weak Greedy), and stores
        timing and diagnostic data.
        """

        rb_path = self.path + f"{self.rb_alg}"

        if os.path.exists(rb_path + ".npy"):
            print(f"load {self.rb_alg} basis...")
            self.V = np.load(rb_path + f".npy")
            self.RBtime = np.load(rb_path + f"time.npy")
            self.RBdata = np.load(rb_path + f"data.npy")
        else:
            print(f"construct {self.rb_alg} basis...")
            if self.rb_alg == "POD":
                _, self.V, self.N, self.RBtime, _ = rb.PODBasis(
                    self.SpaceMatrix, self.W, self.dim_m)
                self.RBdata = np.empty(0)
            elif self.rb_alg == "WG":
                self.V, self.N, self.RBtime, self.RBdata = rb.WeakGreedy(
                    self.SpaceMatrix,
                    self.W,
                    self.json_data['RB'][self.rb_alg],
                    self.dim_m,
                )
            else:
                raise RuntimeError(f"{self.rb_alg} not implemented")

            np.save(rb_path + f".npy", self.V)
            np.save(rb_path + f"time.npy", self.RBtime)
            np.save(rb_path + f"data.npy", self.RBdata)

        self.N = self.V.shape[0]

    def _build_ss(self):
        """
        Build or load the Sensor Selection (SS) operator.

        Loads a precomputed sensor selection if available, otherwise computes
        it using the selected algorithm (e.g. wOMP or random), including
        selected sensor indices and timing data.
        """

        ss_path = self.path + f"{self.ss_alg}{self.H}{self.sparse_attr}{self.rb_alg}"

        if os.path.exists(ss_path + ".npy"):
            print(f"load {self.ss_alg} basis...")
            self.Q = np.load(ss_path + ".npy")
            self.chosen = np.load(ss_path + "chosen.npy")
            self.SSdata = np.load(ss_path + "data.npy")
            self.SStime = np.load(ss_path + "time.npy")
        else:
            print(f"construct {self.ss_alg} basis...")
            if self.ss_alg == "wOMP":
                self.Q, self.chosen, self.SStime, self.SSdata = wOMP(
                    self.V, self.R1d, self.SpaceMatrix, self.ndofs, self.path,
                    self.rb_alg, self.dim_m, self.nrows, self.N,
                    self.json_data['SS'][self.ss_alg])
            else:
                raise RuntimeError(f"{self.ss_alg} not implemented")

            np.save(ss_path + ".npy", self.Q)
            np.save(ss_path + "chosen.npy", self.chosen)
            np.save(ss_path + "data.npy", self.SSdata)
            np.save(ss_path + "time.npy", self.SStime)

        self.M = len(self.chosen)

    def _preassemble(self):
        """
        Splits sensors by spatial component, constructs Riesz representer
        blocks, applies stabilization.
        """

        self.rests = self.chosen % self.dim_s
        self.quots = self.chosen // self.dim_s

        if self.permutation == "P":
            self.isx = self.rests == 0
            self.isy = self.rests == 1
            self.isz = self.rests == 2

            self.Mx = self.isx.sum()
            self.My = self.isy.sum()
            self.Mz = self.isz.sum()

            self.perm = np.concatenate(
                (np.where(self.isx)[0], np.where(self.isy)[0],
                 np.where(self.isz)[0]))
            self.invperm = np.argsort(self.perm)

            self.R3dx = self.R1d[self.quots[self.isx]]
            self.R3dy = self.R1d[self.quots[self.isy]]
            self.R3dz = self.R1d[self.quots[self.isz]]
            self.Qg = self.Q[self.perm]

    def _assemble_system(self):
        """
        Assemble the PBDW system.
        """

        start = tt.default_timer()

        self._preassemble()

        self._assemble_saddle_point_system()

        self.assemble_time = tt.default_timer() - start
        print("linear system assembly:", self.assemble_time, "s")

    def _assemble_saddle_point_system(self):
        """
        Assemble the PBDW saddle-point matrix.

        assembles the left-hand-side system matrix used in the online phase.
        """

        # Construct saddle-point matrix [[ \xi M I_M + T^\top T, T^\top T P^\top],
        #                                [ P,                    0             ]].
        self.lhs = np.zeros((self.N + self.M, self.N + self.M))

        if self.permutation == "P":
            Qx = self.Qg[:self.Mx]
            Qy = self.Qg[self.Mx:self.Mx + self.My]
            Qz = self.Qg[self.Mx + self.My:]

            Vx = self.V[:, :self.nrows]
            Vy = self.V[:, self.nrows:2 * self.nrows]
            Vz = self.V[:, 2 * self.nrows:]

            self.Fx = Qx @ self.SpaceMatrix.dot(self.R3dx.T)
            self.Fy = Qy @ self.SpaceMatrix.dot(self.R3dy.T)
            self.Fz = Qz @ self.SpaceMatrix.dot(self.R3dz.T)

            Gx = Qx @ self.SpaceMatrix.dot(Vx.T)
            Gy = Qy @ self.SpaceMatrix.dot(Vy.T)
            Gz = Qz @ self.SpaceMatrix.dot(Vz.T)

            # upper left: diag( M \xi I_{M_x} + F_x F_x^\top
            #                   M \xi I_{M_y} + F_y F_y^\top
            #                   M \xi I_{M_z} + F_z F_z^\top )
            self.lhs[:self.Mx, :self.Mx] = self.Fx @ self.Fx.T
            self.lhs[self.Mx:self.Mx + self.My,
                     self.Mx:self.Mx + self.My] = self.Fy @ self.Fy.T
            self.lhs[self.Mx + self.My:self.M,
                     self.Mx + self.My:self.M] = self.Fz @ self.Fz.T
            self.lhs[:self.M, :self.M].flat[::self.M + 1] += self.M * self.xi

            # upper right: [ F_x F_x^\top G_x
            #                F_y F_y^\top G_y
            #                F_z F_z^\top G_z ]
            self.lhs[:self.Mx, self.M:] = self.Fx @ self.Fx.T @ Gx
            self.lhs[self.Mx:self.Mx + self.My,
                     self.M:] = self.Fy @ self.Fy.T @ Gy
            self.lhs[self.Mx + self.My:self.M,
                     self.M:] = self.Fz @ self.Fz.T @ Gz

            # lower left: [G_x^\top & G_y^\top & G_z^\top]
            self.lhs[self.M:, :self.Mx] = Gx.T
            self.lhs[self.M:, self.Mx:self.Mx + self.My] = Gy.T
            self.lhs[self.M:, self.Mx + self.My:self.M] = Gz.T

        else:
            from scipy.sparse import block_diag
            IPmat = block_diag([self.SpaceMatrix] * 3).tocsr()

            self.R3d = np.zeros((self.M, self.ndofs), dtype=np.float64)
            self.R3d[np.arange(self.M)[:,
                                       None], self.rests[:, None] * self.nrows +
                     np.arange(self.nrows)] = self.R1d[self.quots]

            self.Q3d = np.zeros((self.M, self.ndofs), dtype=np.float64)
            self.Q3d[np.arange(self.M)[:,
                                       None], self.rests[:, None] * self.nrows +
                     np.arange(self.nrows)] = self.Q

            # T^\top = \tilde{Q} I_\Chi (\tilde{R}_{3d}^f)^\top
            self.Ttmat = self.Q3d @ IPmat.dot(self.R3d.T)
            #      P = V I_\Chi \tilde{Q}^\top
            self.Pmat = self.V @ IPmat.dot(self.Q3d.T)

            # upper left: M \xi P_{ord} I_M + T^\top T P_{ord}
            self.lhs[:self.M, :self.M] = self.Ttmat @ self.Ttmat.T
            self.lhs[:self.M, :self.M].flat[::self.M + 1] += self.M * self.xi

            # upper right: T^\top T P^\top
            self.lhs[:self.M, self.M:] = self.Ttmat @ self.Ttmat.T @ self.Pmat.T

            # lower left: P
            self.lhs[self.M:, :self.M] = self.Pmat

    def _run_online(self):
        """
        Run the online reconstruction phase.

        Selects the appropriate noise strategy and dispatches either
        parallel or sequential reconstruction for all test snapshots.
        """

        noise_type = self.json_data['noise']['type']

        strategies = {
            "LD": actions.StrategyMax,
            "SNR": actions.StrategyStd,
            "NONE": actions.StrategyNone
        }
        strategy = strategies[noise_type]

        if self.run_type == 'par':
            self._run_parallel(strategy)
        elif self.run_type == 'seq':
            self._run_sequential(strategy)
        else:
            raise RuntimeError(f"{self.run_type} not implemented")

    def _run_parallel(self, strategy):
        """
        Run online reconstruction in parallel using shared memory.

        Allocates shared memory for large read-only arrays, spawns worker
        processes, collects reconstruction results, and cleans up shared
        memory resources.
        """

        shm = {
            "V": shmem.create_shared_array(self.V),
            "lhs": shmem.create_shared_array(self.lhs)
        }
        if self.permutation == "P":
            shm["Q"] = shmem.create_shared_array(self.Qg)
            shm["R3dx"] = shmem.create_shared_array(self.R3dx)
            shm["R3dy"] = shmem.create_shared_array(self.R3dy)
            shm["R3dz"] = shmem.create_shared_array(self.R3dz)
            shm["Fx"] = shmem.create_shared_array(self.Fx)
            shm["Fy"] = shmem.create_shared_array(self.Fy)
            shm["Fz"] = shmem.create_shared_array(self.Fz)
        else:
            shm["Ttmat"] = shmem.create_shared_array(self.Ttmat)
            shm["R3d"] = shmem.create_shared_array(self.R3d)
            shm["Q3d"] = shmem.create_shared_array(self.Q3d)

        shm_meta = {
            k: {
                "name": v.name,
                "shape": getattr(self, k).shape,
                "dtype": getattr(self, k).dtype
            }
            for k, v in shm.items()
        }

        process_func = partial(shmem.reconstruct_Aretz_par,
                               Mx=self.Mx,
                               My=self.My,
                               M=self.M,
                               N=self.N,
                               num_norms=self.num_norms,
                               SpaceMatrix=self.SpaceMatrix,
                               fes=self.fes,
                               mesh=self.mesh,
                               nrows=self.nrows,
                               strategy=strategy,
                               json_noise=self.json_data['noise'],
                               json_algebra=self.json_data['algebra'],
                               shm_meta=shm_meta)

        self.results = shmem.process_in_parallel(process_func, self.utest_mat,
                                                 mp.cpu_count())

        shmem.cleanup(shm.values())

    def _run_sequential(self, strategy):
        """
        Run online reconstruction sequentially.

        Executes the reconstruction routine for each test snapshot in a
        single process. Primarily used for debugging or non-parallel runs.
        """
        self.results = [
            seqmem.reconstruct_Aretz_seq(
                u, self.Mx, self.My, self.M, self.N, self.num_norms,
                self.SpaceMatrix, self.fes, self.mesh, self.nrows, strategy,
                self.json_data['noise'], self.json_data['algebra'], self.Ttmat,
                self.R3d, self.Q3d, self.V, self.Qg, self.lhs, self.R3dx,
                self.R3dy, self.R3dz, self.Fx, self.Fy, self.Fz)
            for u in self.utest_mat
        ]

    def _save_results(self):
        """
        Save reconstruction results and diagnostics to disk.

        Aggregates errors and statistics, saves debug arrays, optionally
        exports VTK visualizations, and writes summary and history reports.
        """

        json_data = self.json_data
        export = json_data['general']['export']
        num_rnd_runs = json_data['noise']['num_random_runs']
        noise_type = json_data['noise']['type']
        noise_SNR = json_data['noise'][noise_type]

        path = (self.path + f"_test{self.num_test}_runs{num_rnd_runs}_" +
                f"{self.rb_alg}{self.ss_alg}" + f"{self.H}{self.sparse_attr}_" +
                f"{noise_type}{noise_SNR}_xi{self.xi}_" +
                f"{self.run_type}run_{self.permutation}_")

        # ---------------- Allocation (identical to original) ----------------
        errors_table = [[
            'problem', 'L^2 error', 'H^1 error', 'H^1_0 error', 'L_infty error',
            'online time'
        ]]

        errors_mean = np.empty((self.num_test, self.num_norms))
        errors_std = np.empty((self.num_test, self.num_norms))
        sigma_debug = np.empty(self.num_test)

        errors = np.empty((num_rnd_runs * self.num_test, self.num_norms))
        obs_debug = np.empty((self.num_test, self.M))
        true_debug = np.empty((self.num_test, self.ndofs))
        rhs_debug = np.empty((num_rnd_runs * self.num_test, self.M))

        zeta_debug = np.empty((num_rnd_runs * self.num_test, self.N))
        eta_debug = np.empty((num_rnd_runs * self.num_test, self.M))
        noise_debug = np.empty((num_rnd_runs * self.num_test, self.M))

        # ---------------- Fill arrays ----------------
        for test_idx, res in enumerate(self.results):
            (obs, sigma, rhs_rr, zeta_rr, eta_rr, noise_rr, errors_rr, u_true,
             u_rec, zeta_rec, eta_rec) = res

            obs_debug[test_idx] = obs
            sigma_debug[test_idx] = sigma
            true_debug[test_idx] = u_true

            errors_mean[test_idx] = errors_rr.mean(axis=0)
            errors_std[test_idx] = errors_rr.std(axis=0)

            sl = slice(test_idx * num_rnd_runs, (test_idx + 1) * num_rnd_runs)
            errors[sl] = errors_rr
            rhs_debug[sl] = rhs_rr
            zeta_debug[sl] = zeta_rr
            eta_debug[sl] = eta_rr
            noise_debug[sl] = noise_rr

            # table row
            row = [f"{test_idx}"]
            row += [
                f"{errors_mean[test_idx][k]:.4e}±{errors_std[test_idx][k]:.4e}"
                for k in range(self.num_norms)
            ]
            errors_table.append(row)

            # ---------------- VTK export ----------------
            if export and self.carpmesh is not None:
                xyz, e2n, tags = self.carpmesh
                nodal_solutions = [
                    ('SS_rec', eta_rec, True),
                    ('RB_rec', zeta_rec, True),
                    ('u_rec', u_rec, True),
                    ('u_true', u_true, True),
                    ('abs_diff', abs(u_rec - u_true), True),
                ]
                write_binary_mesh_vtk(path + f"rec{test_idx}.vtk",
                                      xyz,
                                      e2n,
                                      nodal_solutions=nodal_solutions)

        # ---------------- Save arrays ----------------
        np.save(path + 'sigma.npy', sigma_debug)
        np.save(path + 'true.npy', true_debug)
        np.save(path + 'meanerrors.npy', errors_mean)
        np.save(path + 'stderrors.npy', errors_std)
        np.save(path + 'randerrors.npy', errors)
        np.save(path + 'zeta.npy', zeta_debug)
        np.save(path + 'eta.npy', eta_debug)
        np.save(path + 'rhs.npy', rhs_debug)
        np.save(path + 'noise.npy', noise_debug)
        np.save(path + 'obs.npy', obs_debug)
        np.save(path + 'lhs.npy', self.lhs)
        if self.permutation == "P":
            np.save(path + 'perm.npy', self.perm)
            np.save(path + 'invperm.npy', self.invperm)

        # ---------------- Summary & history ----------------

        path_sumfile = path + 'summary.out'

        with open(path_sumfile, "w") as f:
            self._write_stability_section(f)
            self._write_summary_section(f)
            self._write_timings_section(f, noise_type, noise_SNR)
            self._write_error_statistics_section(
                f, errors.reshape(num_rnd_runs, self.num_test, self.num_norms))

        self._plot_error_distributions(
            errors.reshape(num_rnd_runs, self.num_test, self.num_norms), path)

    def _cleanup(self):
        """
        Explicitly release large arrays to reduce memory footprint.

        Clears references to large NumPy arrays after completion.
        Shared memory is cleaned up elsewhere and is not handled here.
        """

        # Large dense arrays
        for attr in ('W', 'R1d', 'SpaceMatrix', 'V', 'Q', 'lhs', 'Qg', 'R3dx',
                     'R3dy', 'R3dz', 'isx', 'isy', 'isz', 'Fx'
                     'Fy', 'Fz', 'Gx'
                     'Gy', 'Gz', 'Wx'
                     'Wy', 'Wz', 'quots', 'rests', 'chosen', 'invperm', 'perm',
                     'Ttmat', 'Pmat', 'R3d', 'Q3d', 'W3dinv', 'results'):
            if hasattr(self, attr):
                setattr(self, attr, None)

    def _write_table(self, f, header, rows, col_align=None):
        """
        Write a nicely formatted table to an open file handle.

        Parameters
        ----------
        f : file handle
            Open file (append mode).
        header : list[str]
            Column names.
        rows : list[list]
            Table rows.
        col_align : list[str], optional
            'l', 'c', or 'r' per column.
        """
        data = [header] + rows
        ncols = len(header)

        if col_align is None:
            col_align = ['l'] * ncols

        widths = [
            max(len(str(row[j])) for row in data) + 2 for j in range(ncols)
        ]

        def fmt_row(row):
            cells = []
            for j, item in enumerate(row):
                if col_align[j] == 'r':
                    cells.append(f"{str(item):>{widths[j]}}")
                elif col_align[j] == 'c':
                    cells.append(f"{str(item):^{widths[j]}}")
                else:
                    cells.append(f"{str(item):<{widths[j]}}")
            return "|" + "|".join(cells) + "|"

        sep = "+" + "+".join("-" * w for w in widths) + "+"

        f.write(sep + "\n")
        f.write(fmt_row(header) + "\n")
        f.write(sep + "\n")
        for row in rows:
            f.write(fmt_row(row) + "\n")
        f.write(sep + "\n\n")

    def _write_section(self, f, title):
        """
        Write a visually separated section header.
        """
        f.write("\n")
        f.write("=" * (len(title) + 4) + "\n")
        f.write(f"  {title}\n")
        f.write("=" * (len(title) + 4) + "\n\n")

    def _compute_error_stats(self, errors):
        """
        Compute mean, within-field std, and across-field std.

        Parameters
        ----------
        errors : ndarray, shape (num_runs, num_tests, num_norms)

        Returns
        -------
        stats : dict
        """
        mean = errors.mean(axis=(0, 1))
        within_std = errors.std(axis=0).mean(axis=0)
        across_std = errors.mean(axis=0).std(axis=0)

        return {
            "mean": mean,
            "within_std": within_std,
            "across_std": across_std,
        }

    def _plot_error_distributions(self, errors, path):
        """
        Plot and save boxplots of error distributions.
        """
        import matplotlib.pyplot as plt

        labels = ["L2", "H1", "H1_0", "Linf", "Online time"]

        data = [errors[:, :, k].ravel() for k in range(errors.shape[2])]

        plt.figure(figsize=(10, 4))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("Error value")
        plt.title("Error distributions over all test fields and noise runs")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path + "error_distributions.png", dpi=150)
        plt.close()

    def _write_stability_section(self, f):
        """
        Write stability information for the PBDW system.
        """
        self._write_section(f, "Stability")

        stable = self.M >= self.N
        status = "stable" if stable else "NOT stable"

        f.write(f"{self.rb_alg}-{self.ss_alg} is {status} "
                f"(M = {self.M}, N = {self.N})\n\n")

    def _write_summary_section(self, f):
        """
        Write problem size and discretization summary.
        """
        self._write_section(f, "Summary")

        self._write_table(
            f,
            ["#train", "#test", "RB dim (N)", "SS dim (M)", "K*dim_s", "ndofs"],
            [[
                self.num_train, self.num_test, self.N, self.M,
                self.R1d.shape[0] * self.dim_s, self.ndofs
            ]],
            col_align=['r'] * 6)

    def _write_timings_section(self, f, noise_type, noise_SNR):
        """
        Write timing information and noise configuration.
        """
        json_data = self.json_data

        self._write_section(f, "Timings and configuration")

        self._write_table(f, [
            f"{self.rb_alg} time", f"{self.ss_alg} time",
            f"{json_data['general']['norm_rieszReps']} RRs time",
            "offline time", noise_type, "xi"
        ], [[
            f"{self.RBtime:.4e} s", f"{self.SStime:.4e} s",
            f"{json_data['general']['time_rieszReps']} s",
            f"{self.RBtime + self.SStime + self.assemble_time:.4e} s",
            f"{noise_SNR}", f"{self.xi}"
        ]],
                          col_align=['r'] * 6)

    def _write_error_statistics_section(self, f, errors):
        """
        Write aggregated error statistics.

        Parameters
        ----------
        errors : ndarray, shape (num_runs, num_tests, num_norms)
            Error values for all noise realizations and test fields.
        """
        self._write_section(f, "Error statistics")

        stats = self._compute_error_stats(errors)
        names = ["L2", "H1", "H1_0", "Linf", "Online time"]

        rows = []
        for i, name in enumerate(names):
            rows.append([
                name,
                f"{stats['mean'][i]:.3e}",
                f"{stats['within_std'][i]:.3e}",
                f"{stats['across_std'][i]:.3e}",
            ])

        self._write_table(
            f, ["Norm", "Mean", "Within-field std", "Across-field std"],
            rows,
            col_align=['l', 'r', 'r', 'r'])
