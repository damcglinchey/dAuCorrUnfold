import numpy as np
import scipy as sp
from scipy.special import gammaln
import unfold_input as ui
import corr_funcs as cf
# np.set_printoptions(precision=3)
# np.set_printoptions(suppress=True) # suppress exponents on small numbers

def lnpoisson(x, mu):
    '''
    Multivariate log likelihood for Poisson(x|mu)
    '''
    if np.any(x < 0.) or np.any(mu <= 0.):
        return -np.inf
    return np.sum(x * np.log(mu) - gammaln(x + 1.0) - mu)


def lngauss(x, mu, prec):
    '''
    Log likelihood for multivariate Gaussian N(x | mu, prec)
    where prec is the inverse covariance matrix.
    Neglecting terms that don't depend on x.
    '''
    diff = x - mu
    ln_det_sigma = np.sum(np.log(1. / np.diag(prec)))
    ln_prefactors = -0.5 * (x.shape[0] * np.log(2 * np.pi) + ln_det_sigma)
    return ln_prefactors - 0.5 * np.dot(diff, np.dot(prec, diff))
    # return -0.5 * np.dot(diff, np.dot(prec, diff))


def mvn(data, pred):
    '''
    Multivariate normal (Gaussian) log likelihood. 
    data: column 0 contains points, column 1 contains errors. 
    pred: predicted values
    Currently assumes data points are independent (diagonal covariance).
    '''
    # icov_data = np.diag(1. / (data[:, 1] ** 2))
    # Scale everything to by the error to avoid machine precision problems
    # in lngauss()
    icov_data = np.diag(np.full(len(data[:, 0]), 1.))
    return lngauss(data[:, 0] / data[:, 1], pred / data[:, 1], icov_data)



def lncorr(data, cn):
    '''
    Log Likelihood calculation for correlation function given 
    Correlation coefficients
    data: column 0 contains delta phi values
          column 1 contains correlation function values
          column 2 contains errors
    cn: correlation coefficients
    '''
    return mvn(data[:, 1:], cf.corr(data[:, 0], cn))


def lndata(vn, data, xlim):
    '''
    Summed likelihood from the data
    '''
    if np.any(vn < xlim[:, 0]) or np.any(vn > xlim[:, 1]):
        return -np.inf

    vn = vn.reshape((ui.nn, ui.nv)).T
    cn = ui.calccn(vn, len(data))
    llsum = np.sum(np.array([lncorr(data[i], cn[i, :]) for i in range(len(data))]))
    return llsum


