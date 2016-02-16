"""make data sets acording to four modles"""
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle
from scipy.linalg import block_diag
import logging
import multiprocessing


def make_block_diag(blk_size, size, rho=0.25):
    """make block diagonal covariance matrix

    :param blk_size:
    :param size:
    :param rho:
    """
    #  from scipy.linalg import block_diag
    blk = np.ones((blk_size, blk_size)) * rho
    np.fill_diagonal(blk, 1)
    cov = block_diag(blk)
    n_blks = size // blk_size
    for i in range(n_blks):
        cov = block_diag(cov, blk)
    return cov[:size, :size]


def test_make_block_diag():
    """test make block diagonal matrix"""
    G, D, rho = 2, 5, 0.25
    M = make_block_diag(G, D, rho)
    A = np.eye(D)
    A[0, 1], A[1, 0] = rho, rho
    A[2, 3], A[3, 2] = rho, rho
    assert M.shape == (D, D)
    #  print(M.diagonal())
    np.testing.assert_allclose(M.diagonal(), np.ones(D), rtol=1e-7, atol=1e-5)
    np.testing.assert_allclose(M, A, rtol=1e-5, atol=1e-1)


def make_dataset(n_samples_1, n_samples_2, n_features,
                 n_informative, class_sep,
                 shuffle=True, random_state=None):
    """Generate dataset for class 0 and class 1

    :param n_samples_1:
    :param n_samples_2:
    :param n_features:
    :param n_informative:
    :param class_sep:
    :param shuffle:
    :param random_state:
    """
    generator = check_random_state(random_state)
    n_samples = n_samples_1 + n_samples_2
    # Initialize X and y
    # print("n_samples: {}, n_features: {}".format(n_samples, n_features))
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int)
    X = generator.randn(n_samples, n_features)
    # class 0 and class 1 have different means.
    X[:n_samples_1, :n_informative] -= class_sep
    X[-n_samples_2:, :n_informative] += class_sep
    # assign class labels to y
    y[:n_samples_1] = 0
    y[-n_samples_2:] = 1
    if shuffle:
        # Randomly permute samples.
        X, y = util_shuffle(X, y, random_state=generator)
        # Randomly permute features
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
    return X, y


def make_dataset_2(
        n_samples_1, n_samples_2, n_features, n_informative,
        class_sep, scale=2.25, type=1, G=20, rho=0.25,
        shuffle=True, random_state=None):
    """make_dataset_2

    :param n_samples_1:
    :param n_samples_2:
    :param n_features:
    :param n_informative:
    :param class_sep:
    :param scale:
    :param type:
    :param G:
    :param rho:
    :param shuffle:
    :param random_state:
    """
    logger = logging.getLogger(__name__)
    cpname = multiprocessing.current_process().name
    logger.debug("{} is currently doing make_dataset_2".format(cpname))
    generator = check_random_state(random_state)
    n_samples = n_samples_1 + n_samples_2
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int)
    D = n_features
    #  print("Get into make_dataset_2")
    if type == 1:
        mu1 = np.zeros(D)
        mu1[:n_informative] = class_sep
        cov1 = np.eye(D)
        #  print("Start generate multivariate normal in make_dataset_2")
        #  X1 = generator.multivariate_normal(mu1, cov1, size=n_samples_1)
        X1 = np.random.multivariate_normal(mu1, cov1, size=n_samples_1)

        #  print("In make_dataset_2")
        mu2 = np.zeros(D)
        mu2[:n_informative] = class_sep * -1
        cov2 = np.eye(D)
        X2 = generator.multivariate_normal(mu2, cov2, size=n_samples_2)
    elif type == 2:
        mu1 = np.zeros(D)
        mu1[:n_informative] = class_sep
        cov1 = np.eye(D)
        X1 = generator.multivariate_normal(mu1, cov1, size=n_samples_1)

        mu2 = np.zeros(D)
        mu2[:n_informative] = class_sep * -1
        cov2 = np.eye(D) * scale
        X2 = generator.multivariate_normal(mu2, cov2, size=n_samples_2)
    elif type == 3:
        mu1 = np.ones(D) * class_sep
        cov1 = make_block_diag(G, D, rho=rho)
        X1 = generator.multivariate_normal(mu1, cov1, size=n_samples_1)

        mu2 = np.ones(D) * class_sep * -1
        cov2 = make_block_diag(G, D, rho=rho)
        X2 = generator.multivariate_normal(mu2, cov2, size=n_samples_2)
    elif type == 4:
        mu1 = np.ones(D) * class_sep
        cov1 = make_block_diag(G, D, rho=rho)
        X1 = generator.multivariate_normal(mu1, cov1, size=n_samples_1)

        mu2 = np.ones(D) * class_sep * -1
        cov2 = make_block_diag(G, D, rho=rho) * scale
        X2 = generator.multivariate_normal(mu2, cov2, size=n_samples_2)
    X = np.vstack((X1, X2))

    # assign class labels to y
    y[:n_samples_1] = 0
    y[-n_samples_2:] = 1
    if shuffle:
        # Randomly permute samples.
        X, y = util_shuffle(X, y, random_state=generator)
        # Randomly permute features
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
    print("Getting out of make_dataset_2")
    return X, y


def test_make_dataset():
    """test of make_dataset"""
    n1, n2 = 10, 10
    n_feat, n_info = 3, 2
    class_sep = 1
    X, y = make_dataset(n1, n2, n_feat, n_info, class_sep, random_state=0)
    assert X.shape == (n1 + n2, n_feat)
    assert y.shape == (n1 + n2,)


def test_make_dataset_2():
    """test of make_dataset_2"""
    n1, n2 = 10, 15
    n_feat, n_info = 3, 2
    class_sep = 1
    #  type = 1
    for type in (1, 2, 3, 4):
        X, y = make_dataset_2(
            n1, n2, n_feat, n_info, class_sep,
            type=type, random_state=0)
        assert X.shape == (n1 + n2, n_feat)
        assert y.shape == (n1 + n2,)
        assert y.sum() == n2
