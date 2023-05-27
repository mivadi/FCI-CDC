import statistics
from causalGraph import CausalGraph
from kernelFunctions import *


def independenceTest(graph, X, Y, condition=[], alpha=0.05, oracle=None, nr_samples=50000):
    """
    Returns boolean if X and Y are independent given condition.
        :param graph: CausalGraph object
        :param X: variable
        :param Y: variable
        :param condition: list of variables (default is empty list [] )
        :param alpha: threshold / significance level for independence test
        :param oracle: dictionary (keys:variables) of dictionary (keys:variables) for
                        separating sets -> {v1:{v2:[ss1, ss2]}} means v1 and v2 are
                        separated by set ss1 and separated by set ss2 (default is None)
        :param nr_samples: number of samples that have to be generated for independence test
    """

    # find independence making use of the oracle
    if oracle != None:
        independent = False
        # oracle has only asymmetric information
        if X in oracle.keys() and Y in oracle[X].keys():
            for sep_set in oracle[X][Y]:
                if set(sep_set) == set(condition):
                    independent =  True
                    break
        elif Y in oracle.keys() and X in oracle[Y].keys():
            for sep_set in oracle[Y][X]:
                if set(sep_set) == set(condition):
                    independent =  True
                    break

    # find independence from data with an independence test
    else:

        # find the data of the variables
        N = graph.data_size
        dX = graph.getData([X])
        dY = graph.getData([Y])

        # if no condition is given, we use the unconditional independence test
        if len(condition) == 0:
            p_val = unconditionalIndependenceTest(dX, dY, nr_samples)

        # otherwise we use the conditional independence test
        else:
            dcondition = graph.getData(condition)
            p_val = conditionalIndependenceTest(dX, dY, dcondition, nr_samples=50000)

        # if the p-value is smaller than a value alpha, we will reject the null-hypothesis
        # and except the alternative hypothesis, i.e. the variables are dependent
        if p_val > alpha:
            independent = True

        # otherwise we cannot reject the null-hypothesis
        else:
            independent = False

    return independent


def unconditionalIndependenceTest(X, Y, nr_samples=50000):
    """
    Returns a p-value which refers to if variables X and Y are independent.
        :param X: data of variable X
        :param Y: data of variable Y
        :param nr_samples: number of samples that have to be generated for independence test
    """
    N = X.shape[0]
    width = statistics.median(np.abs(X - Y))[0]
    if width==0:
        width = 1
    width = 2 / (width**2)

    # compute centralized kernel matrices
    kernel_X = centralizedKernel(X, N, width)
    kernel_Y = centralizedKernel(Y, N, width)

    # run the kernel unconditional independence test
    p_value = KUIT(kernel_X, kernel_Y, nr_samples)

    return p_value


def conditionalIndependenceTest(X, Y, Z, nr_samples=50000):
    """
    Returns a p-value which refers to if variables X and Y are independent given condition Z.
        :param X: data of variable X
        :param Y: data of variable Y
        :param Z: data of variables we condition on
        :param nr_samples: number of samples that have to be generated for independence test
    """

    N = X.shape[0]

    # compute width using distance between X and Y
    width = statistics.median(np.abs(X - Y))[0]
    width = 2 / (width**2)

    # compute centralized kernel matrices to find kernel based statistic
    XZ = np.append(X, Z/2, 1) # division by 2 -> following other implementations
    kernel_XZ = centralizedKernel(XZ, N, width)
    kernel_Y = centralizedKernel(Y, N, width)
    kernel_Z = centralizedKernel(Z, N, width)

    # run the kernel conditional independence test
    p_value = KCIT(kernel_XZ, kernel_Y, kernel_Z, nr_samples)

    return p_value


def KUIT(kernel1, kernel2, nr_samples=50000):
    """
    Returns a p-value under null hypothesis that variable 1 and variable 2 are independent.
        :param kernel1: centralized kernel of variable 1, NxN numpy array
        :param kernel2: centralized kernel of variable 2, NxN numpy array
        :param nr_samples: number of samples that have to be generated for independence test
    """

    # compute statistics
    kernel_based_statistic = kernelBasedStatistic(kernel1, kernel2)
    sample_based_statistic = sampleBasedStatistic(kernel1, kernel2, nr_samples)

    # compute p-value
    p_value = np.sum(sample_based_statistic>kernel_based_statistic) / nr_samples

    return p_value


def KCIT(kernel_XZ, kernel_Y, kernel_Z, zhang=True, nr_samples=50000):
    """
    Returns a p-value under null hypothesis that X and Y are independent given condition Z.
        :param kernel_XZ: centralized kernel of (X,Z), NxN numpy array
        :param kernel_Y: centralized kernel of Y, NxN numpy array
        :param kernel_Z: centralized kernel of Z, NxN numpy array
        :param zhang: boolean (True if we use Zhangs (2012?) method, otherwise we run faster version)
        :param nr_samples: number of samples that have to be generated for independence test
    """

    # small regularization parameter (Learning with kernels - Scholkopf and Smola 2002)
    regularization = 2

    # compute centralized kernel given the condition
    RZ = regularization * LA.inv(kernel_Z + (regularization*np.eye(kernel_Z.shape[0])))
    kernel_XZ_given_Z = symmetric(RZ @ kernel_XZ @ RZ)
    kernel_Y_given_Z = symmetric(RZ @ kernel_Y @ RZ)

    # compute kernel based statistic
    kernel_based_statistic = kernelBasedStatistic(kernel_XZ_given_Z, kernel_Y_given_Z)

    # compute sample based statistic sampleBasedStatistic
    if zhang:
        sample_based_statistic = sampleBasedConditionalStatistic(kernel_XZ_given_Z, kernel_Y_given_Z, nr_samples)
    else:
        sample_based_statistic = sampleBasedStatistic(kernel_XZ_given_Z, kernel_Y_given_Z, nr_samples)

    # compute p-value
    p_value = np.sum(sample_based_statistic>kernel_based_statistic) / nr_samples

    return p_value


def kernelBasedStatistic(kernel_X, kernel_Y):
    """
    Returns statistic computed by 1/N times the trace of the product of the kernels.
        :param kernel_X: centralized kernel of X, NxN numpy array
        :param kernel_Y: centralized kernel of Y, NxN numpy array
    """

    N = kernel_X.shape[0]

    # compute statistic under null hypothesis that X and Y are independent
    kernel_based_statistic = np.trace(kernel_X @ kernel_Y) / N

    return kernel_based_statistic


def sampleBasedStatistic(kernel_X, kernel_Y, nr_samples=50000):
    """
    Returns sample based statistics.
        :param kernel_X: centralized kernel of X, NxN numpy array
        :param kernel_Y: centralized kernel of Y, NxN numpy array
        :param nr_samples: number of samples that have to be generated for independence test
    """

    N = kernel_X.shape[0]

    # get eigenvalues
    eigenvalues_X = eig(kernel_X, True)
    eigenvalues_Y = eig(kernel_Y, True)

    # number of useful eigenvalues
    threshold = 10**-5
    nr_eig_X = min(50, np.where(eigenvalues_X < eigenvalues_X[0]*threshold)[0][0])
    nr_eig_Y = min(50, np.where(eigenvalues_Y < eigenvalues_Y[0]*threshold)[0][0])
    size_eig_prod = nr_eig_X * nr_eig_Y

    # compute eigenproduct
    eig_prod = np.reshape(np.reshape(eigenvalues_X[:nr_eig_X], (nr_eig_X, 1)) @ np.reshape(eigenvalues_Y[:nr_eig_Y], (1,nr_eig_Y)), size_eig_prod)

    # squared sample from Normal distribution with mean 0 and variance 1
    sample = np.random.normal(0, 1, (nr_samples, size_eig_prod))**2

    # get sample based statistic
    sample_based_statistic = np.sum(eig_prod * sample, axis=1)/ N**2

    return sample_based_statistic


def sampleBasedConditionalStatistic(kernel_XZ_given_Z, kernel_Y_given_Z, nr_samples=50000):
    """
    Returns sample based conditional statistic.
        :param kernel_XZ_given_Z: centralized kernel of XZ given Z, NxN numpy array
        :param kernel_Y_given_Z: centralized kernel of Y given Z, NxN numpy array
        :param nr_samples: number of samples that have to be generated for independence test
    """

    N = kernel_XZ_given_Z.shape[0]

    # get eigenvalues
    eigenvalues_XZ, eigenvectors_XZ = eig(kernel_XZ_given_Z)
    eigenvalues_Y, eigenvectors_Y = eig(kernel_Y_given_Z)

    # number of useful eigenvalues
    threshold = 10**-5
    nr_eig_XZ = np.where(eigenvalues_XZ < threshold)[0][0]
    nr_eig_Y = np.where(eigenvalues_Y < threshold)[0][0]

    # get feature mappings
    psi = eigenvectors_XZ[:,:nr_eig_XZ] @ (np.diag(eigenvalues_XZ[:nr_eig_XZ])**(0.5))
    phi = eigenvectors_Y[:,:nr_eig_Y] @ (np.diag(eigenvalues_Y[:nr_eig_Y])**(0.5))

    w = []
    size_w = nr_eig_XZ * nr_eig_Y
    for t in range(N):

        # compute M_t
        M_t = np.reshape(psi[t, :], (nr_eig_XZ, 1)) @ np.reshape(phi[t, :], (1, nr_eig_Y))

        # stacking the vectors together
        M_t = np.reshape(M_t, size_w)

        # and add them to w
        w.append(M_t)

    # shape of w is (N, size_w) since each vector w_t is stacked matrix M_t, t=1,2,...,N
    # and M_t is of size (size_w, size_w)

    # compute ww^T
    w = np.array(w)
    if size_w > N:
        ww = w @ w.T # [N x size_w] @ [size_w x N] = NxN
        size_w = N
    else:
        ww = w.T @ w # size_w x size_w
    ww = symmetric(ww)

    # compute eigenvalues of ww^T
    eigenvalues_ww = eig(ww, True)

    # Squared sample from Normal Distribution with mean 0 and variance 2
    sample = np.random.normal(0, 1, (nr_samples, size_w))**2

    # get sample based statistic
    sample_based_statistic = np.sum(eigenvalues_ww * sample, axis=1) / N

    return sample_based_statistic
