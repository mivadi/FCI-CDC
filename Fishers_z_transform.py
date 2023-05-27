import numpy as np
from scipy import stats


def condIndFisherZ(x, y, S, C, n, alpha=0.05):
    """
    copy from implemntation in R : "pcalg::gaussCItest"
    ## Purpose: Return boolean result on conditional independence using
    ## Fisher's z-transform
    ## ----------------------------------------------------------------------
    ## Arguments:
    ## - x,y,S: Are x,y cond. indep. given S?
    ## - C: Correlation matrix among nodes
    ## - n: Samples used to estimate correlation matrix
    ## - cutoff: Cutoff for significance level for individual
    ##           partial correlation tests
    ## ----------------------------------------------------------------------
    ## Author: Markus Kalisch, Date: 26 Jan 2006, 17:32
    """

    r = pcorOrder(x, y, S, C)

    T = np.sqrt(n-len(S)-3) * 0.5 * np.log1p(2 * r / (1-r))

    p_value = 2*(1-stats.norm.cdf(abs(T)))

    # test for independence : so if the p-value is smaller than alpha,
    # then we accept the alternative hypothesis (we keep the edge= dependence)
    # if the p-value bigger than alpha, we failed to reject the null hypothesis
    # we delete the edge since the variable are independent with a type 1/2 error
    if np.isnan(p_value):
        return 0, 'nan'
    else:
        return p_value, not p_value<=alpha


def pcorOrder(i, j, k, C, cut_at = 0.9999999):
    ## Purpose: Compute partial correlation
    ## ----------------------------------------------------------------------
    ## Arguments:
    ## - i,j,k: Partial correlation of i and j given k
    ## - C: Correlation matrix among nodes
    ## ----------------------------------------------------------------------
    ## Author: Markus Kalisch, Date: 26 Jan 2006; Martin Maechler
    if len(k)==0:
        r = C.loc[i,j]
    elif len(k)==1:
        r = (C.loc[i, j] - C.loc[i, k[0]] * C.loc[j, k[0]]) / np.sqrt((1 - C.loc[j, k[0]]**2) * (1 - C.loc[i, k[0]]**2))
    else:
        index = [i,j] + list(k)
        PM = np.linalg.pinv(C.loc[index,index].to_numpy())
        r = - PM[0, 1] / np.sqrt(PM[0, 0] * PM[1, 1])
    if np.isnan(r):
        return 0
    else:
        return min(cut_at, max(-cut_at, r))

