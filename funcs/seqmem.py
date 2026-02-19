import numpy as np
import timeit as tt
from funcs.err_computation import compute_errors


def reconstruct_Aretz_seq(utest_vec, Mx, My, M, N, num_norms, SpaceMatrix, fes,
                          mesh, nrows, strategy, json_noise, json_algebra,
                          Ttmat, R3d, Q3d, V, Q, lhs, R3dx, R3dy, R3dz, Fx, Fy,
                          Fz):

    seed = json_noise['seed']
    num_rnd_runs = json_noise['num_random_runs']
    permutation = json_algebra['permutation'] == "P"

    np.random.seed(seed)
    ndofs = fes.ndof

    errors_rnd_runs = np.empty((num_rnd_runs, num_norms))
    rhs_rnd_runs = np.empty((num_rnd_runs, M))
    zeta_rnd_runs = np.empty((num_rnd_runs, N))
    eta_rnd_runs = np.empty((num_rnd_runs, M))
    noise_rnd_runs = np.empty((num_rnd_runs, M))

    obs = np.zeros(M)
    rhs = np.zeros(M + N)
    noise = np.zeros(M)

    start_pre = tt.default_timer()

    if permutation:
        Qx, Qy, Qz = Q[:Mx], Q[Mx:Mx + My], Q[Mx + My:]

        obs[:Mx] = R3dx @ SpaceMatrix.dot(utest_vec[:nrows])
        obs[Mx:Mx + My] = R3dy @ SpaceMatrix.dot(utest_vec[nrows:2 * nrows])
        obs[Mx + My:] = R3dz @ SpaceMatrix.dot(utest_vec[2 * nrows:])
    else:
        from scipy.sparse import block_diag
        IPmat = block_diag([SpaceMatrix, SpaceMatrix, SpaceMatrix]).tocsr()
        obs[:] = R3d @ IPmat.dot(utest_vec)

    sigma = strategy().execute(obs, json_noise)

    end_pre = tt.default_timer()

    for r in range(num_rnd_runs):
        start_on = tt.default_timer()

        noise[:] = np.random.normal(0, sigma, M)
        # noise = noise[perm] # for reproducibility
        if permutation:
            rhs[:Mx] = Fx @ (obs[:Mx] + noise[:Mx])
            rhs[Mx:Mx + My] = Fy @ (obs[Mx:Mx + My] + noise[Mx:Mx + My])
            rhs[Mx + My:M] = Fz @ (obs[Mx + My:M] + noise[Mx + My:M])
        else:
            rhs[:M] = Ttmat @ (obs + noise)

        x = np.linalg.solve(lhs, rhs)

        eta = x[:M]
        zeta = x[M:]

        eta_rec = np.zeros(ndofs)
        zeta_rec = V.T @ zeta
        if permutation:
            eta_rec[:nrows] = Qx.T @ eta[:Mx]
            eta_rec[nrows:2 * nrows] = Qy.T @ eta[Mx:Mx + My]
            eta_rec[2 * nrows:] = Qz.T @ eta[Mx + My:]
        else:
            eta_rec[:] = Q3d.T @ eta

        u_rec = eta_rec + zeta_rec

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
