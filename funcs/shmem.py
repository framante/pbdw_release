import numpy as np
import timeit as tt
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory as sm
from tqdm import tqdm
from funcs.err_computation import compute_errors


def create_shared_array(array):
    shm = sm.SharedMemory(create=True, size=max(array.nbytes, 1))
    arr = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    np.copyto(arr, array)
    return shm


def cleanup(objects):
    for obj in objects:
        shm = sm.SharedMemory(name=obj.name)
        shm.close()
        shm.unlink()


def reconstruct_Aretz_par(utest_vec, Mx, My, M, N, num_norms, SpaceMatrix, fes,
                          mesh, nrows, strategy, json_noise, json_algebra,
                          shm_meta):
    # ============================================================
    # Helpers
    # ============================================================

    def attach(name, shape, dtype):
        shm = sm.SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return shm, arr

    # ============================================================
    # Attach arrays
    # ============================================================

    permutation = json_algebra['permutation'] == "P"

    shm_objs = {}
    arrays = {}
    keys_init = ("V", "lhs")

    if permutation:
        keys = keys_init + ("Q", "R3dx", "R3dy", "R3dz", "Fx", "Fy", "Fz")
    else:
        keys = keys_init + ("Ttmat", "R3d", "Q3d")

    for k in keys:
        shm_objs[k], arrays[k] = attach(**shm_meta[k])

    # unpack
    V = arrays["V"]
    lhs = arrays["lhs"]

    R3dcmp = [arrays.get("R3dx"), arrays.get("R3dy"), arrays.get("R3dz")]
    Fcmp = [arrays.get("Fx"), arrays.get("Fy"), arrays.get("Fz")]
    Q3dcmp = ([Q[:Mx], Q[Mx:Mx + My], Q[Mx + My:]] if
              (Q := arrays.get("Q")) is not None else None)

    Ttmat = arrays.get("Ttmat")
    R3d = arrays.get("R3d")
    Q3d = arrays.get("Q3d")

    # ============================================================
    # Noise setup
    # ============================================================

    seed = json_noise['seed']
    num_rnd_runs = json_noise['num_random_runs']
    xi = json_noise['xi']

    np.random.seed(seed)

    # ============================================================
    # Storage
    # ============================================================

    errors_rnd_runs = np.empty((num_rnd_runs, num_norms))
    rhs_rnd_runs = np.empty((num_rnd_runs, M))
    zeta_rnd_runs = np.empty((num_rnd_runs, N))
    eta_rnd_runs = np.empty((num_rnd_runs, M))
    noise_rnd_runs = np.empty((num_rnd_runs, M))

    obs = np.zeros(M)
    rhs = np.zeros(N + M)
    noise = np.zeros(M)

    ndofs = fes.ndof

    # ============================================================
    # Precomputation
    # ============================================================

    start_pre = tt.default_timer()

    if permutation:
        M_list = [slice(0, Mx), slice(Mx, Mx + My), slice(Mx + My, M)]
        dofs_list = [
            slice(0, nrows),
            slice(nrows, 2 * nrows),
            slice(2 * nrows, 3 * nrows)
        ]

        for i, (off, sl) in enumerate(zip(dofs_list, M_list)):
            obs[sl] = R3dcmp[i] @ SpaceMatrix.dot(utest_vec[off])
    else:
        from scipy.sparse import block_diag
        IPmat = block_diag([SpaceMatrix] * 3).tocsr()
        obs[:] = R3d @ IPmat.dot(utest_vec)

    sigma = strategy().execute(obs, json_noise)

    end_pre = tt.default_timer()

    # ============================================================
    # Online loop
    # ============================================================

    for r in range(num_rnd_runs):
        start_on = tt.default_timer()

        noise[:] = np.random.normal(0, sigma, M)
        rhs[:] = 0.
        # noise = noise[perm] # for reproducibility

        # ---------------- RHS ----------------
        if permutation:
            for i, sl in enumerate(M_list):
                rhs[sl] = Fcmp[i] @ (obs[sl] + noise[sl])
        else:
            rhs[:M] = Ttmat @ (obs + noise)

        # ---------------- Solve --------------
        x = np.linalg.solve(lhs, rhs)

        # ---------------- ETA ----------------
        eta = x[:M]
        zeta = x[M:]

        eta_rec = np.zeros(ndofs)
        zeta_rec = V.T @ zeta

        # ---------------- Reconstruction ----------------

        if permutation:
            for i, (off, sl) in enumerate(zip(dofs_list, M_list)):
                eta_rec[off] = Q3dcmp[i].T @ eta[sl]
        else:
            eta_rec[:] = Q3d.T @ eta

        u_rec = eta_rec + zeta_rec

        # ---------------- Store ----------------
        end_on = tt.default_timer()
        errors_rnd_runs[r, -1] = (end_pre - start_pre) + (end_on - start_on)
        errors_rnd_runs[r, :-1] = compute_errors(u_rec, utest_vec, fes, mesh)

        rhs_rnd_runs[r] = rhs[:M]
        zeta_rnd_runs[r] = zeta
        eta_rnd_runs[r] = eta
        noise_rnd_runs[r] = noise

    return (obs, sigma, rhs_rnd_runs, zeta_rnd_runs, eta_rnd_runs,
            noise_rnd_runs, errors_rnd_runs, utest_vec, u_rec, zeta_rec,
            eta_rec)


def process_in_parallel(process_func,
                        items_to_process,
                        n_processes,
                        batch_size=1000):
    # Use fewer processes if we have a small number of items
    actual_processes = min(n_processes, len(items_to_process), os.cpu_count())

    if actual_processes < n_processes:
        print(f"Using {actual_processes} processes instead of {n_processes}")
    print(f"Using {actual_processes} processes instead of {n_processes}")

    # Process in batches
    total_batches = (len(items_to_process) + batch_size - 1) // batch_size
    total_processed = 0

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(items_to_process))
        current_batch = items_to_process[start_idx:end_idx]

        # Process this batch in parallel
        with ProcessPoolExecutor() as pool:
            batch_results = list(
                tqdm(
                    pool.map(
                        process_func,
                        current_batch,  #pool.map to maintain order
                        chunksize=max(
                            1,
                            len(current_batch) // actual_processes // 10)),
                    total=len(current_batch),
                    desc=f"Batch {batch_idx+1}/{total_batches}"))

    return batch_results
