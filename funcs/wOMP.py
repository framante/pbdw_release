"""
Improved Worst-case Orthogonal Matching Pursuit (wOMP) implementation.

This module provides an improved implementation of the wOMP algorithm for sensor
selection in reduced-order modeling, with emphasis on robustness, efficiency,
and maintainability through modular design.

The wOMP algorithm constructs an update space by selecting observation functionals
that maximize the worst-case error, orthogonalizing them, and solving an eigenvalue
problem to assess stability.

Author: Francesco Carlo Mantegazza
Date: April 2025
"""

# Linear Algebra
import numpy as np
import scipy as sp
import timeit as tt
import funcs.orthogonalization as orth


def format_table_row(data, widths, is_header=False):
    """Format a single row with proper spacing and borders"""
    row = "│"
    for i, item in enumerate(data):
        content = str(item)
        if len(content) > widths[i] - 2:  # -2 for padding
            content = content[:widths[i] - 5] + "..."
        row += f" {content:{widths[i]-2}} │"
    return row


def format_table_separator(widths, style="-"):
    """Create a separator line with the given style"""
    parts = []
    for width in widths:
        parts.append(style * width)
    if style == "-":
        return "├" + "┼".join(parts) + "┤"
    elif style == "=":
        return "┌" + "┬".join(parts) + "┐"
    elif style == "_":  # Bottom border
        return "└" + "┴".join(parts) + "┘"
    else:
        return "├" + "┼".join(parts) + "┤"  # Default case


class WOMPState:
    """
    Container for the wOMP algorithm state.

    This class encapsulates all the state variables needed during the execution
    of the wOMP algorithm, including parameters, intermediate matrices and vectors,
    and results.
    """

    def __init__(self, V, R1d, SpaceMatrix, ndofs, json_data, N, nrows, dim_m,
                 Q, chosen):
        """
        Initialize the wOMP algorithm state.

        Parameters:
        -----------
        V : numpy.ndarray
            The reduced basis.
        R1d : numpy.ndarray
            The Riesz representers in 1d.
        SpaceMatrix : numpy.ndarray or scipy.sparse matrix
            The inner product matrix.
        ndofs : int
            The number of degrees of freedom.
        json_data : dict
            Dictionary containing algorithm parameters.
        N : int
            The dimension of the reduced space.
        nrows : int
            The number of rows in SpaceMatrix.
        dim_m : int
            The mesh dimension.
        """
        # Parameters from json_data
        self.beta_0 = json_data['beta_0']
        self.dim_s = json_data['sensor_dim']
        self.H = json_data['H']
        self.seed = json_data['seed']
        self.threshold = json_data['threshold']

        # Algorithm state
        self.K = self.dim_s * R1d.shape[0]  # maximal number of sensors
        self.Q = np.empty((0, nrows), dtype=np.float64)
        self.chosen = list()
        self.remaining = np.array([*range(self.K)])
        self.preselection = ''
        if chosen is not None and Q is not None:
            self.Q = Q
            self.chosen.extend(chosen)
            self.remaining = np.setdiff1d(self.remaining, self.chosen)
            self.preselection = 'p-'
        self.iter = 0
        self.beta = 0
        self.beta_old = 0

        # Pre-computed values for error calculation
        self.num_pre = SpaceMatrix.dot(R1d.T)
        self.den = np.einsum('ij,ji->i', R1d, self.num_pre)

        # Reference data
        self.V = V
        self.R1d = R1d
        self.SpaceMatrix = SpaceMatrix
        self.ndofs = ndofs
        self.nrows = nrows
        self.dim_m = dim_m
        self.N = N

        # Initialize algorithm internal state
        self.PPT = np.zeros((self.N, self.N), dtype=np.float64)
        self.IQQIx, self.IQQIy, self.IQQIz = self._initialize_IQQI()
        self.AVx = SpaceMatrix.dot(self.IQQIx)
        self.AVy = SpaceMatrix.dot(self.IQQIy)
        self.AVz = SpaceMatrix.dot(self.IQQIz)
        self.q_beta_perp = self._initialize_q_beta_perp()

        # Iteration timings and metrics
        self.start_time = tt.default_timer()
        self.time_comp1 = 0
        self.time_orth = 0
        self.time_eig = 0
        self.time_comp2 = 0
        self.time_it = 0
        self.error = None
        self.err_orth = None
        self.check = None

    def _initialize_q_beta_perp(self):
        """
        Initialize q_beta_perp with a random vector.

        Parameters:
        -----------
        V : numpy.ndarray
            The reduced basis.
        N : int
            The dimension of the reduced space.
        nrows : int
            The number of rows in SpaceMatrix.

        Returns:
        --------
        q_beta_perp : numpy.ndarray
            The initialized vector.
        """
        np.random.seed(self.seed)
        first_guess = np.random.randint(0,
                                        self.N)  # random number in [0,..,N-1]

        q_beta_perp = np.zeros(self.ndofs, dtype=np.float64)
        # Initialize with a random vector from V
        q_beta_perp[:] = self.V[first_guess] / np.sqrt(
            self.V[first_guess][:self.nrows] @ self.AVx[:, first_guess] +
            self.V[first_guess][self.nrows:2 *
                                self.nrows] @ self.AVy[:, first_guess] +
            self.V[first_guess][2 * self.nrows:] @ self.AVz[:, first_guess])
        return q_beta_perp

    def _initialize_IQQI(self):
        """
        Initialize the IQQI matrices used for projection updates.

        Parameters:
        -----------
        V : numpy.ndarray
            The reduced basis.
        nrows : int
            The number of rows in SpaceMatrix.
        N : int
            The dimension of the reduced space.

        Returns:
        --------
        IQQIx, IQQIy, IQQIz : numpy.ndarray
            The initialized matrices.
        """
        IQQIx = np.empty((self.nrows, self.N), dtype=np.float64)
        IQQIy = np.empty((self.nrows, self.N), dtype=np.float64)
        IQQIz = np.empty((self.nrows, self.N), dtype=np.float64)

        IQQIx[:] = self.V[:, :self.nrows].T
        IQQIy[:] = self.V[:, self.nrows:2 * self.nrows].T
        IQQIz[:] = self.V[:, 2 * self.nrows:].T

        return IQQIx, IQQIy, IQQIz


class WOMPLogger:
    """
    Logger for the wOMP algorithm.

    Handles writing of iteration data to log files and storing of iteration data.
    """

    def __init__(self, path, rb_alg, state):
        """
        Initialize the logger.

        Parameters:
        -----------
        path : str
            Path to write log files to.
        rb_alg : str
            Name of the reduced basis algorithm.
        state : WOMPState
            The wOMP algorithm state.
        """
        sparse_attr = 'sparsez' if state.dim_s == 2 else ''
        self.file = open(
            path +
            f'{state.preselection}wOMP{state.H}{sparse_attr}{rb_alg}.log', 'w')

        # Write header
        header = [
            'iter', 'chosen', 're-orth?', 'max_error', 'max_err orth', 'beta',
            'time_pre', 'time_orth', 'time_eig', 'time_post', 'time_it'
        ]

        self.widths = [6, 40, 10, 15, 15, 15, 15, 15, 15, 15, 15]
        # Write table header
        self.file.write(format_table_separator(self.widths, "=") + "\n")
        self.file.write(
            format_table_row(header, self.widths, is_header=True) + "\n")
        self.file.write(format_table_separator(self.widths) + "\n")
        self.file.flush()

        #self._write_row(header)

        # Store data for return
        self.data = [header]

    def log_iteration(self, state):
        """
        Log information about the current iteration.

        Parameters:
        -----------
        state : WOMPState
            The wOMP algorithm state.
        """
        keys = list(state.chosen)[-state.H:]
        row = [
            f'{state.iter}', f'{",".join(str(x) for x in keys)}',
            f'{not state.check}', f'{state.error[0]:.6e}',
            f'{max(state.err_orth):.6e}', f'{state.beta:.6e}',
            f'{state.time_comp1:.6e}', f'{state.time_orth:.6e}',
            f'{state.time_eig:.6e}', f'{state.time_comp2:.6e}',
            f'{state.time_it:.6e}'
        ]
        self._write_row(row)
        self.data.append(row)

    def log_first_iteration(self, state):
        """
        Log information about the current iteration.

        Parameters:
        -----------
        state : WOMPState
            The wOMP algorithm state.
        """
        row = [
            f'{state.iter}', '-', '-', '-', '-', f'{state.beta:.6e}', '-', '-',
            '-', '-', '-'
        ]
        self._write_row(row)
        self.data.append(row)

    def _write_row(self, row):
        """
        Write a row to the log file.

        Parameters:
        -----------
        row : list
            The row to write.
        """
        self.file.write(format_table_row(row, self.widths) + "\n")
        self.file.flush()

    def close(self):
        """Close the log file."""
        self.file.write(format_table_separator(self.widths, '_') + "\n")
        self.file.close()


def select_candidates(state, H):
    """
    Select H candidates that maximize the error.

    Parameters:
    -----------
    state : WOMPState
        The wOMP algorithm state.
    H : int
        The number of candidates to select.

    Returns:
    --------
    keys : numpy.ndarray
        The indices of the selected candidates.
    error : numpy.ndarray
        The error values for the selected candidates.
    keys_rests : numpy.ndarray
        The sensor component indices.
    keys_quots : numpy.ndarray
        The sensor number indices.
    time_comp : float
        The computation time.
    """
    start = tt.default_timer()

    # Compute errors for remaining candidates
    testerror = np.zeros(state.K, dtype=np.float64)
    remaining_rests = state.remaining % state.dim_s
    remaining_quots = state.remaining // state.dim_s

    # Calculate error for each candidate
    testerror[state.remaining] = np.abs(
        np.einsum(
            'ij,ji->i',
            state.q_beta_perp.reshape(state.dim_m,
                                      state.nrows)[remaining_rests],
            state.num_pre[:, remaining_quots])) / state.den[remaining_quots]

    # Pick batch_size sensors maximizing the error
    keys = np.argsort(testerror)[::-1][:H]
    keys_rests = keys % state.dim_s  # sensor component
    keys_quots = keys // state.dim_s  # sensor number in 1d
    error = testerror[keys]

    end = tt.default_timer()
    time_comp = end - start

    return keys, error, keys_rests, keys_quots, time_comp


def orthogonalize_batch(state, keys, keys_rests, keys_quots, H):
    """
    Orthogonalize the selected vectors using modified Gram-Schmidt.

    Parameters:
    -----------
    state : WOMPState
        The wOMP algorithm state.
    keys : numpy.ndarray
        The indices of the selected vectors.
    keys_rests : numpy.ndarray
        The sensor component indices.
    keys_quots : numpy.ndarray
        The sensor number indices.
    H : int
        The number of vectors to orthogonalize.

    Returns:
    --------
    tauKs : numpy.ndarray
        The orthogonalized vectors.
    err_orth : numpy.ndarray
        The orthogonalization error for each vector.
    check : bool
        Whether the orthogonalization was successful.
    time_orth : float
        The orthogonalization time.
    """
    start = tt.default_timer()

    tauKs = np.empty((H, state.nrows), dtype=np.float64)
    err_orth = np.zeros(H, dtype=np.float64)

    # Process each vector
    for index in range(H):
        tauK = np.copy(state.R1d[keys_quots[index]])
        flag = ((np.array(state.chosen) % state.dim_s) == keys_rests[index])

        if np.any(flag):
            # First orthogonalization pass
            Q_flag = state.Q[flag]
            SpaceMatrix_tauK = state.SpaceMatrix.dot(tauK)
            projections = Q_flag @ SpaceMatrix_tauK
            tauK = tauK - Q_flag.T @ projections

            # Second orthogonalization pass for stability
            SpaceMatrix_tauK = state.SpaceMatrix.dot(tauK)
            projections = Q_flag @ SpaceMatrix_tauK
            tauK = tauK - Q_flag.T @ projections

        # Normalize
        SpaceMatrix_tauK = state.SpaceMatrix.dot(tauK)
        norm = np.sqrt(max(tauK @ SpaceMatrix_tauK, 1e-15))
        if norm > 1e-14:
            tauK = tauK / norm
        else:
            # Handle nearly zero vectors
            tauK = np.zeros_like(tauK)

        # Compute orthogonalization error
        if np.any(flag):
            Q_flag = state.Q[flag]
            SpaceMatrix_tauK = state.SpaceMatrix.dot(tauK)
            err_orth[index] = np.linalg.norm(
                Q_flag.T @ (Q_flag @ SpaceMatrix_tauK))

        state.Q = np.append(state.Q, [tauK], axis=0)
        tauKs[index] = tauK
        state.chosen.append(keys[index])

    # Check orthonormality
    check = np.allclose(err_orth, np.zeros(H, dtype=np.float64), atol=1e-10)

    # If orthogonalization was not successful, re-orthogonalize
    if not check:
        orth.orthogonalizeNGSolve_scalar(state.Q, state.SpaceMatrix,
                                         np.array(state.chosen), state.dim_s,
                                         state.Q.shape[0] - H)
        tauKs[:] = state.Q[-H:]

    end = tt.default_timer()
    time_orth = end - start

    return tauKs, err_orth, check, time_orth


def update_eigenvalue_problem(state, keys_rests, tauKs):
    """
    Update and solve the eigenvalue problem.

    Parameters:
    -----------
    state : WOMPState
        The wOMP algorithm state.
    keys_rests : numpy.ndarray
        The sensor component indices.
    tauKs : numpy.ndarray
        The orthogonalized vectors.

    Returns:
    --------
    time_eig : float
        The computation time.
    """
    start = tt.default_timer()

    # Update PPT matrix
    termx = tauKs[keys_rests == 0] @ state.AVx if np.any(
        keys_rests == 0) else np.empty((0, state.N))
    termy = tauKs[keys_rests == 1] @ state.AVy if np.any(
        keys_rests == 1) else np.empty((0, state.N))
    termz = tauKs[keys_rests == 2] @ state.AVz if np.any(
        keys_rests == 2) else np.empty((0, state.N))

    # Update PPT matrix
    if termx.size > 0:
        state.PPT += termx.T @ termx
    if termy.size > 0:
        state.PPT += termy.T @ termy
    if termz.size > 0:
        state.PPT += termz.T @ termz

    # Solve eigenvalue problem
    eigval, eigvec = sp.linalg.eigh(state.PPT, subset_by_index=[0, 0])

    # Update state variables
    state.beta_old = state.beta
    state.beta = np.sqrt(np.abs(eigval[0]))

    # Update projections
    if termx.size > 0:
        state.IQQIx -= tauKs[keys_rests == 0].T @ termx
    if termy.size > 0:
        state.IQQIy -= tauKs[keys_rests == 1].T @ termy
    if termz.size > 0:
        state.IQQIz -= tauKs[keys_rests == 2].T @ termz

    # Update q_beta_perp
    state.q_beta_perp[:state.nrows] = state.IQQIx @ eigvec[:, 0]
    state.q_beta_perp[state.nrows:2 * state.nrows] = state.IQQIy @ eigvec[:, 0]
    state.q_beta_perp[2 * state.nrows:] = state.IQQIz @ eigvec[:, 0]

    # Normalize
    q_components = [
        state.SpaceMatrix.dot(state.q_beta_perp[i * state.nrows:(i + 1) *
                                                state.nrows])
        for i in range(state.dim_m)
    ]
    norm_vector = np.array(q_components).reshape(state.ndofs)
    norm = np.sqrt(max(state.q_beta_perp @ norm_vector, 1e-15))
    state.q_beta_perp /= norm

    end = tt.default_timer()
    time_eig = end - start

    return time_eig


def update_state(state, keys):
    """
    Update algorithm state.

    Parameters:
    -----------
    state : WOMPState
        The wOMP algorithm state.
    keys : numpy.ndarray
        The indices of the selected vectors.

    Returns:
    --------
    time_comp2 : float
        The computation time.
    """
    start = tt.default_timer()

    # Remove selected indices from remaining set
    state.remaining = np.setdiff1d(state.remaining, keys)

    # Increment iteration counter
    state.iter += 1

    end = tt.default_timer()
    time_comp2 = end - start

    return time_comp2


def finalize_results(state, logger):
    """
    Prepare and return final results.

    Parameters:
    -----------
    state : WOMPState
        The wOMP algorithm state.
    logger : WOMPLogger
        The logger.

    Returns:
    --------
    Q : numpy.ndarray
        The orthonormal basis for U_M.
    chosen : numpy.ndarray
        The indices of the chosen observation functionals.
    comp_time : float
        The total computation time.
    data : numpy.ndarray
        The logged data.
    """
    end = tt.default_timer()
    comp_time = end - state.start_time

    return state.Q, np.array(state.chosen), comp_time, np.array(logger.data)


def process_batch(state, H):
    """
    Process one batch iteration of the wOMP algorithm.

    Parameters:
    -----------
    state : WOMPState
        The wOMP algorithm state.
    H : int
        The batch size.
    """
    start_it = tt.default_timer()

    # Step 1: Select candidates
    keys, error, keys_rests, keys_quots, time_comp1 = select_candidates(
        state, H)

    # Step 2: Orthogonalize the selected vectors
    tauKs, err_orth, check, time_orth = orthogonalize_batch(
        state, keys, keys_rests, keys_quots, H)

    # Step 3: Update eigenvalue problem
    time_eig = update_eigenvalue_problem(state, keys_rests, tauKs)

    # Step 4: Update state
    time_comp2 = update_state(state, keys)

    # Record timing and metrics
    end_it = tt.default_timer()
    state.time_it = end_it - start_it
    state.time_comp1 = time_comp1
    state.time_orth = time_orth
    state.time_eig = time_eig
    state.time_comp2 = time_comp2
    state.error = error
    state.err_orth = err_orth
    state.check = check


def initialize_wOMP(V, R1d, SpaceMatrix, ndofs, json_data, N, nrows, dim_m, Q,
                    chosen):
    """
    Initialize the wOMP algorithm state.

    Parameters:
    -----------
    V : numpy.ndarray
        The reduced basis.
    R1d : numpy.ndarray
        The Riesz representers in 1d.
    SpaceMatrix : numpy.ndarray or scipy.sparse matrix
        The inner product matrix.
    ndofs : int
        The number of degrees of freedom.
    json_data : dict
        Dictionary containing algorithm parameters.
    N : int
        The dimension of the reduced space.
    nrows : int
        The number of rows in SpaceMatrix.
    dim_m : int
        The mesh dimension.

    Returns:
    --------
    state : WOMPState
        The initialized wOMP algorithm state.
    """
    return WOMPState(V, R1d, SpaceMatrix, ndofs, json_data, N, nrows, dim_m, Q,
                     chosen)


def wOMP(V,
         R1d,
         SpaceMatrix,
         ndofs,
         path,
         rb_alg,
         dim_m,
         nrows,
         N,
         json_data,
         Q=None,
         chosen=None):
    """
    Improved implementation of the worst-case orthogonal matching pursuit (wOMP) algorithm.

    Constructs the update space U_M according to the wOMP algorithm with improved
    numerical stability, error handling, and code structure.

    Parameters:
    -----------
    V : numpy.ndarray
        The reduced basis.
    R1d : numpy.ndarray
        The Riesz representers in 1d.
    SpaceMatrix : numpy.ndarray or scipy.sparse matrix
        The inner product matrix.
    ndofs : int
        The number of degrees of freedom.
    path : str
        Output folder path.
    rb_alg : str
        RB method name.
    dim_m : int
        The mesh dimension.
    nrows : int
        The number of rows of the space matrix.
    N : int
        The dimension of \mathcal{Z}_N.
    json_data : dict
        Data for the algorithm.

    Returns:
    --------
    Q : numpy.ndarray
        The constructed orthonormal basis for U_M.
    chosen : numpy.ndarray
        The indices of the chosen observation functionals in the library.
    comp_time : float
        The computational time.
    data : numpy.ndarray
        The data info.
    """
    # Initialize parameters and state
    state = initialize_wOMP(V, R1d, SpaceMatrix, ndofs, json_data, N, nrows,
                            dim_m, Q, chosen)
    # Setup output logging
    logger = WOMPLogger(path, rb_alg, state)

    if chosen is not None and Q is not None:
        update_eigenvalue_problem(state, chosen % state.dim_s, state.Q)
        # Log the iteration results
        logger.log_first_iteration(state)

    # Main iteration loop
    while state.beta < state.beta_0 and len(state.remaining) > 0:
        # Adjust batch size if needed
        H = min(state.H, len(state.remaining))

        # Process one batch iteration
        process_batch(state, H)

        # Log the iteration results
        logger.log_iteration(state)

        if abs(state.beta - state.beta_old) / state.beta < state.threshold:
            break

    # Finalize and return results
    logger.close()
    return finalize_results(state, logger)
