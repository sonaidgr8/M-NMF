import time
import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition.nmf import _initialize_nmf
from numpy.linalg import norm
from os import path

eps = np.finfo(np.float).eps
RESULT_DICT = {'Q': None, 'U': None, 'H': None, 'i': 0}

def refine_factor_matrix(X):
    """ This function takes matrix as input and checks for underflow and
    replaces entries with lowest possible float-value eps to correct underflow.
    In case of nan entries, it replaces them by corresponding rowmean. """
    X[X < eps] = eps
    row_mean = np.nanmean(X, axis=1)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(row_mean, inds[0])
    return X

def sparse_to_matrix(X) :
    if issparse(X):
        X = X.toarray()
    else:
        X = np.array(X)
    return X

def initialize_factor_matrices(S, init, dtype, logger, config):
    """ This function initializes factor matrices based on the choice of initialization method
    either 'random' or 'nndsvd', random seed can be set based on user input. """
    if config.FIXED_SEED == 'Y':
        np.random.seed(int(config.SEED_VALUE))
    logger.debug('Initializing factor matrices')
    if init == 'random':
        M = np.array(np.random.rand(int(config.N), int(config.L_COMPONENTS)), dtype=dtype)
        U = np.array(np.random.rand(int(config.L_COMPONENTS), int(config.N)), dtype=dtype)
    elif init == 'nndsvd':
        M, U = _initialize_nmf(S, int(config.L_COMPONENTS), 'nndsvd')
    else:
        raise('Unknown init option ("%s")' % init)
    U = sparse_to_matrix(U)
    M = sparse_to_matrix(M)
    H = np.array(np.random.rand(int(config.N), int(config.K)), dtype=dtype)
    C = np.array(np.random.rand(int(config.K), int(config.L_COMPONENTS)), dtype=dtype)
    logger.debug('Initialization completed')
    return M, U.T, C, H

def __LS_updateM_L2(S, M, U, alpha, lmbda):
    """ Multiplicative update equation for M """
    UtU = np.dot(U.T, U)
    numerator = alpha * np.dot(S, U)
    denominator = alpha * np.dot(M, UtU) + lmbda * M
    denominator[denominator == 0] = eps
    M = M * (numerator / denominator)
    M = refine_factor_matrix(M)
    return M

def __LS_updateC_L2(H, U, C, beta, lmbda) :
    """ Multiplicative update equation for C """
    UtU = np.dot(U.T, U)
    numerator = beta * np.dot(H.T, U)
    denominator = beta * np.dot(C, UtU) + lmbda * C
    denominator[denominator == 0] = eps
    C = C * (numerator / denominator)
    C = refine_factor_matrix(C)
    return C

def __LS_updateU_L2(S, M, U, H, C, alpha, beta, lmbda):
    nominator = np.zeros((U.shape), dtype=U.dtype)
    dnominator = np.zeros((U.shape), dtype=U.dtype)
    MtM = np.dot(M.T, M)
    CtC = np.dot(C.T, C)
    nominator += alpha * np.dot(S.T, M)
    dnominator += alpha * np.dot(U, MtM) + lmbda * U
    nominator += beta * np.dot(H, C)
    dnominator += beta * np.dot(U, CtC)
    dnominator[dnominator == 0] = eps
    U = U * (nominator / dnominator)
    U = refine_factor_matrix(U)
    return U

def __LS_updateH_L2(H, U, C, S, B, beta, gamma, zeta, lmbda):
    HtH = np.dot(H.T, H)
    HHtH = np.dot(H, HtH)
    BH = np.dot(B, H)
    small_delta_1 = (2 * gamma * BH) * (2 * gamma * BH) + (16 * zeta * HHtH) * (
    2 * gamma * np.dot(S, H) + 2 * beta * np.dot(U, C.T) + (4 * zeta - 2 * beta) * H)
    H = H * np.sqrt((-2 * gamma * BH + np.sqrt(small_delta_1)) / (8 * zeta * HHtH + lmbda * 2 * H))
    # small_delta_2 = (np.add((2 * gamma * BH), (8 * zeta * HHtH))) ** 2
    # H = H * np.sqrt( (-2 * gamma * BH + np.sqrt(small_delta_2)) / (8 * zeta * HHtH + lmbda * 2 * H))
    H = refine_factor_matrix(H)
    return H

def __LS_compute_fit(S, M, U, H, C, B, alpha, beta, gamma, zeta, lmbda):
    MUt = np.dot(M, U.T)
    fitSMUt = norm(S - MUt) ** 2
    UCt = np.dot(U, C.T)
    fitHUCt = norm(H - UCt) ** 2
    HtH = np.dot(H.T, H)
    fitHtHI = norm(HtH - np.eye(H.shape[1])) ** 2
    z = - np.trace(np.dot(np.dot(H.T, S), H)) + np.trace(np.dot(np.dot(H.T, B), H))
    l2_reg = norm(U) ** 2 + norm(C) ** 2 + norm(M) ** 2 + norm(H) ** 2
    return (alpha * fitSMUt + beta * fitHUCt + zeta * fitHtHI +  gamma * z + lmbda * l2_reg)

def factorize(config, S, B, Y, Y_train, train_ids, val_ids, test_ids, logger):
    # ---------- Get the parameter values-----------------------------------------
    alpha = float(config.ALPHA)
    beta = float(config.BETA)
    gamma = float(config.GAMMA)
    lmbda = float(config.LAMBDA)
    zeta = float(config.ZETA)
    init = config.INIT
    maxIter = int(config.MAX_ITER)
    costF = config.COST_F
    stop_index = int(config.STOP_INDEX)
    early_stopping = int(config.EARLY_STOPPING)
    if costF == 'LS':
        conv = float(config.CONV_LS)
    n = np.shape(S)[0]
    config.N = n
    q, _ = Y.shape
    config.Q = q
    dtype = np.float32
    mult = 1
    # Creation of class prior vector from training data
    train_label_dist = np.sum(Y_train.T, axis=0) / np.sum(Y_train)
    logger.debug("Train label distribution : [%s]" % (str(train_label_dist)))

    # Creation of penalty matrix from label matrix and training data
    W = np.copy(Y.T)
    unlabelled_ids = np.logical_not(train_ids)
    n_unlabelled = np.count_nonzero(unlabelled_ids)
    W[unlabelled_ids, :] = np.zeros((n_unlabelled, q))
    W[train_ids, :] = np.ones((n - n_unlabelled, q))
    W = W.T

    # ---------- Initialize factor matrices-----------------------------------------
    M, U, C, H = initialize_factor_matrices(S, init, dtype, logger, config)

    #  ------ compute factorization -------------------------------------------
    fit = 0
    exectimes = []
    best_svm_result, best_lr_result = RESULT_DICT, RESULT_DICT
    max_svm_accu, max_lr_accu = -1, -1
    conv_list = list()
    Q = U  # We are not learning any Q, so for convenience assign Q as U
    for iter in range(maxIter):
        tic = time.time()
        fitold = fit
        if costF == 'LS':
            M = __LS_updateM_L2(S, M, U, alpha, lmbda)
            if beta != 0:
                C = __LS_updateC_L2(H, U, C, beta, lmbda)
                H = __LS_updateH_L2(H, U, C, S, B, beta, gamma, zeta, lmbda)
            U = __LS_updateU_L2(S, M, U, H, C, alpha, beta, lmbda)
            fit = __LS_compute_fit(S, M, U, H, C, B, alpha, beta, gamma, zeta, lmbda)

            if ((iter % (config.STEP * mult)) == 0):
                from main_algo import get_perf_metrics
                # Can be set test_ids during testing or val_ids during hyper-param search using validation
                val_lr_accu = get_perf_metrics(config, U, Q, Y.T, train_ids, test_ids, 'lr')
                if val_lr_accu["micro_f1"] > max_lr_accu:
                    best_lr_result = {'Q': Q, 'U': U, 'H': H, 'i': iter}
                    max_lr_accu = val_lr_accu["micro_f1"]
                    stop_index = 0
                else:
                    stop_index = stop_index + 1

        if iter <= 1 :
            fitchange = abs(fitold - fit)
        else :
            fitchange = abs(fitold - fit) / fitold
        # conv_list.append(fit)
        toc = time.time()
        exectimes.append(toc - tic)
        logger.debug('::: [%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (iter, fit, fitchange, exectimes[-1]))

        if stop_index > early_stopping:
            logger.debug("Early stopping")
            break

    return best_lr_result
