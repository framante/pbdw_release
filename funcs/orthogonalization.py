import numpy as np


def orthogonalizeNGSolve_scalar(basis, SpaceMatrix, chosen, dim_s, offset=0):
    """
    In-place orthonormalize the basis with respect to a defined inner product using NGSsolve algorithm.

    :param numpy.ndarray basis: the input basis.
    :param numpy.ndarray SpaceMatrix: the space matrix.
    :param list chosen: list of components the vectors refer to.
    :param int dim_s: the number of components of the sensors.
    :param int offset: the index of the first vector to be orthogonalized.
    """
    chosen_rests = chosen % dim_s
    nrows = SpaceMatrix.shape[0]
    for comp in range(dim_s):
        mask = (chosen_rests == comp)
        indices = np.where(mask)[0]
        mask_idx = (indices >= offset)
        if mask.any() and mask_idx.any():
            offset_comp = np.where(mask_idx)[0][0]
            dim_comp = mask.sum()
            remove_offset = dim_comp - offset_comp
            tmp = np.empty(nrows, dtype=np.float64)
            Rfactor = np.zeros((dim_comp, remove_offset), dtype=np.float64)
            for i, idx in enumerate(indices):
                tmp[:] = SpaceMatrix.dot(basis[idx])
                norm = np.sqrt(tmp @ basis[idx])
                # Rfactor[i,i] = norm
                selection = np.array(
                    [*range(np.maximum(offset_comp, i + 1), dim_comp)],
                    dtype=np.int64)
                basis[idx] /= norm
                Rfactor[i, selection -
                        offset_comp] = basis[indices[selection]] @ tmp / norm
                for j in selection:
                    basis[indices[j]] -= Rfactor[i,
                                                 j - offset_comp] * basis[idx]


def orthogonalizeNGSolve(basis, SpaceMatrix, offset):
    """
    In-place orthonormalize the basis with respect to a defined inner product using NGSsolve algorithm.

    :param numpy.ndarray basis: the input basis.
    :param numpy.ndarray SpaceMatrix: the space matrix.
    :param int offset: offset where to start from.
    """
    nrows = SpaceMatrix.shape[0]
    ndofs = nrows * 3
    dim_basis = basis.shape[0]
    dim = 3
    tmp = np.empty(ndofs, dtype=np.float64)
    remove_offset = dim_basis - offset
    Rfactor = np.zeros((dim_basis, remove_offset), dtype=np.float64)
    for i in range(dim_basis):
        tmp[:] = np.array([
            SpaceMatrix.dot(basis[i][k * nrows:(k + 1) * nrows])
            for k in range(dim)
        ]).reshape(ndofs)
        norm = np.sqrt(tmp @ basis[i])
        # Rfactor[i,i] = norm
        select_offset = np.array([*range(np.maximum(offset, i + 1), dim_basis)],
                                 dtype=np.int64)
        basis[i] /= norm
        Rfactor[i, select_offset - offset] = basis[select_offset] @ tmp / norm
        for j in select_offset:
            basis[j] -= Rfactor[i, j - offset] * basis[i]


def orthogonalizeGramSchmidt(basis, SpaceMatrix, offset):
    """
    In-place orthonormalize the basis with respect to a defined inner product using Gram Schmidt orthonormalization.

    :param numpy.ndarray basis: the input basis.
    :param numpy.ndarray SpaceMatrix: the space matrix.
    :param int offset: offset where to start.
    """
    nrows = SpaceMatrix.shape[0]
    ndofs = nrows * 3
    tmp = np.empty(ndofs, dtype=np.float64)
    dim = 3
    for k in range(offset, basis.shape[0]):
        tmp[:] = basis[k]
        for j in range(0, k):
            fact = np.array([
                SpaceMatrix.dot(basis[j][i * nrows:(i + 1) * nrows])
                for i in range(dim)
            ]).reshape(ndofs)
            num = tmp @ fact
            den = basis[j] @ fact
            basis[k] -= (num / den) * basis[j]
        basis[k] /= np.sqrt(basis[k] @ np.array([
            SpaceMatrix.dot(basis[k][i * nrows:(i + 1) * nrows])
            for i in range(dim)
        ]).reshape(ndofs))
