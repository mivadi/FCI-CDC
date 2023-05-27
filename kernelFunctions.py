import numpy as np
from scipy import linalg as LA
import statistics


def kernelWidth(distance, N):
    """
    Return kernel width which is median of the of the pairwise distances.
        :param distance: pairwise distances between the data points
        :param N: size of data set
    """

    # get off-diagonal distances: delete the diagonal matrix for the possible distances
    all_dist = list(np.reshape(np.array([np.delete(row, i) for i, row in enumerate(distance)]), N*(N-1)))

    # kernel width is median of the pairwise distances
    kernel_width = statistics.median(all_dist)

    return kernel_width


def kernel(X, N, coef=1, kernel_width=0, linear_coef=0, bias=0):
    """
    Returns the kernel matrix, see:
    Kernel-based Conditional Independence Test and Application in Causal Discovery
     - Zhang et al, 2012
        :param X: data of variable X
        :param N: size of data set
        :other coeffients: see equation 6.63 at page 307 in Pattern Recognition
                        and Machine Learning - Bishop (defaults are coef=1,
                        kernel_width=0, linear_coef=0, bias=0)
    """

    # euclidean norm
    K = X.shape[1]
    sq_distance = 0
    for k in range(K):
        sq_distance += ( np.expand_dims(X[:,k], axis=1) @ np.ones((1,N)) - np.ones((N, 1)) @ np.expand_dims(X[:,k], axis=0) )**2
    distance = np.sqrt(sq_distance)

    # compute kernel width
    if kernel_width<=0:
        kernel_width = kernelWidth(distance, N)
        if kernel_width <= 0:
            kernel_width = 1

    # compute kernel
    kernel = coef * np.exp( - kernel_width * distance**2 / 2 ) + bias

    # add linear part if linear coefficient is non-zero
    if linear_coef!=0:
        kernel += linear_coef * X @ X.T

    return kernel


def centralize(K):
    """
    Returns centralized kernel of kernel K.
        :param K: kernel matrix, NxN numpy array
    """

    N = K.shape[0]

    # centralize
    centralizer = np.eye(N) - 1/N

    return centralizer @ K @ centralizer


def symmetric(K):
    """
    Returns symmetric version of kernel K to avoid small computational differences
    which makes the kernel matrix non-symmetric.
        :param K: kernel matrix, NxN numpy array
    """
    return (K + K.T) / 2


def centralizedKernel(X, N, kernel_width=0):
    """
    Returns the centralized kernel matrix of variable X, see:
    Kernel-based Conditional Independence Test and Application in Causal Discovery - Zhang et al, 2012
        :param X: data of variable X
        :param N: size of data set
        :param kernel_width: coeffient for computing kernel matrix (float>0)
    """

    # define centralizer
    centralizer = np.eye(N) - 1/N

    # centralize kernel matrix
    centralized_kernel = centralizer @ kernel(X, N, kernel_width) @ centralizer

    # make sure it is symmetric
    centralized_kernel = (centralized_kernel + centralized_kernel.T) / 2

    return centralized_kernel


def linearGram(X, Y, N, lambda_=2):
    """
    Returns the Gram matrix G with linear kernel of P(X|Y).
        :param X: kernel of X
        :param Y: kernel of Y (condition)
        :param N: size of data set
        :param lambda_: regularization parameter
    """
    inv_Y = LA.inv(Y + lambda_ * np.eye(N))
    linear_gram = Y @ inv_Y @ X @ inv_Y @ Y
    return linear_gram


def gaussianGram(gram):
    """
    Compute Gaussian Gram matrix:
        :param gram: NxN Gram matrix
    """

    N = gram.shape[0]
    diag = np.diagonal(gram)

    distance = np.expand_dims(diag,axis=1) @ np.ones((1,N))
    distance += np.ones((N,1)) @ np.expand_dims(diag,axis=0)
    distance -= 2*gram

    # Gaussian kernel width based on median of pairwise distances
    width_square = kernelWidth(distance, N)

    if width_square <= 0:
        width_square = 1

    ggram = np.exp(- distance / width_square / 2)

    return ggram


################ Eigen value decomposition ################


def eig(matrix, vals_only=False):
    """
    Returns eigenvalues (and eigenvectors) of the matrix.
        :param matrix: NxN numpy array
        :param vals_only: boolean to indicate if only eigenvalues are required (default is False)
    """
    if vals_only:
        eigenvalues = LA.eigvals(matrix)
        eigenvalues = recoverEigenOrder(eigenvalues.real)
        return eigenvalues
    else:
        eigenvalues, eigenvectors = LA.eig(matrix)
        eigenvalues,  eigenvectors = recoverEigenOrder(eigenvalues.real, eigenvectors.real)
        return eigenvalues, eigenvectors


def recoverEigenOrder(eigenvalues, eigenvectors=None):
    """
    Returns eigenvalues which are 'recovered', i.e. which are non-zero and ordered.
    In case the eigenvectors are required, the eigenvectors in the corresponding
    order will be returned.
        :param eigenvalues: numpy array of length N
        :param eigenvectors: numpy array of length NxN (default is None)
    """
    new_eigenvalues = np.zeros(eigenvalues.shape)
    if eigenvectors is not None:
        new_eigenvectors = np.zeros(eigenvectors.shape)

    i = 0
    max_eigenvalue = max(eigenvalues)
    while i<len(new_eigenvalues):

        # get next max value
        max_eigenvalue = np.amax(eigenvalues)

        # make sure that we have positive values
        if max_eigenvalue <= 0: break

        # get indices of max value
        index_max =  np.where(eigenvalues == max_eigenvalue)[0]

        for j in index_max:

            # set them to the next in the ordered eigenvalues and -vectors
            new_eigenvalues[i] = max_eigenvalue
            if eigenvectors is not None:
                new_eigenvectors[:, i] = eigenvectors[:, j]

            # set the max to zero
            eigenvalues[j] = 0.0
            i+=1

    if eigenvectors is not None:
        return new_eigenvalues, new_eigenvectors
    else:
        return new_eigenvalues


################ help functions ################


def is_pos_def(x):
    """
    Returns boolean whether x is positive semi-definite matrix.
        :param x: data of variable
    """
    return np.all(np.linalg.eigvals(x) >= 0)


def is_almost_pos_def(x):
    """
    Returns boolean whether x is almost positive semi-definite matrix.
        :param x: data of variable
    """
    threshold = 1e-10
    # check if real number is bigger than -threshold and if complex number is
    # in (-threshold, threshold)
    if np.min(np.real(np.linalg.eigvals(x))) > -threshold and np.min(np.imag(np.linalg.eigvals(x))) > -threshold and np.max(np.imag(np.linalg.eigvals(x))) < threshold:
        return True
    else:
        return False
