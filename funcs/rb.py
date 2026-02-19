# Linear Algebra
import numpy as np
from scipy.sparse import block_diag
from scipy.linalg import eigh

import timeit as tt
import funcs.orthogonalization as orth


# -------------------------------------------------------------------------------
def truncation_s(s):
    epod = np.cumsum(s**2) / (s**2).sum()
    rank = min(np.where(1.0 - epod < 1e-6)[0]) + 1
    return rank


# -------------------------------------------------------------------------------
def PODBasis(SpaceMatrix, W, dim, N=None):
    """
    Construct the RB via Proper Orthogonal Decomposition (POD).

    :param numpy.ndarray SpaceMatrix: the space matrix.
    :param numpy.ndarray W: the snapshots.
    :param int dim: the dimension of the problem.
    :param int N: truncation.

    :return: the RB, the its dimension and the computational time.
    :rtype: numpy.ndarray, int, float.
    """
    start = tt.default_timer()
    num_snap, ndofs = W.shape

    IPmat = SpaceMatrix
    if dim == 3:
        IPmat = block_diag([SpaceMatrix, SpaceMatrix, SpaceMatrix]).tocsr()

    Cmat = W @ (IPmat @ W.T)
    evals, evecs = eigh(Cmat)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    evals = np.sqrt(evals)
    if N == None:
        pod_sz = truncation_s(evals)
    else:
        pod_sz = N

    podBasis = (W.T @ ((1. / evals[:pod_sz]) * evecs[:, :pod_sz])).T

    orth.orthogonalizeNGSolve(podBasis, SpaceMatrix, 0)

    end = tt.default_timer()
    comp_time = end - start

    return evals, podBasis, pod_sz, comp_time, idx
