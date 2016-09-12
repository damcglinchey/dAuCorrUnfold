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


def lndata(vn, data, xlim, nv, nn):
    '''
    Summed likelihood from the data
    '''
    if np.any(vn < xlim[:, 0]) or np.any(vn > xlim[:, 1]):
        return -np.inf

    vn = vn.reshape((nn, nv)).T
    cn = ui.calccn(vn, len(data))
    llsum = np.sum(np.array([lncorr(data[i], cn[i, :]) for i in range(len(data))]))
    return llsum





# (Log) posterior pdf functions for MCMC samplers
# Designed for linear systems Ax = b where
# A = transfer matrix
# x = vector of parameters to be found
# b = vector of measured data points
# Prior is p(x|prior pars)
# Likelihood is p(b|A, x, prior pars)
# Posterior is \propto prior*likelihood
# -----------------------------------------------


def logp_ept_dca(x, matlist, datalist, w, x_prior, alpha, xlim, L, scf, moddcalist):
    '''
    Intended for use with electron pt model (data) as first element of 
    matlist (datalist), and electron DCA model/data as remaining elements.
    '''
    if np.any(x < xlim[:, 0]) or np.any(x > xlim[:, 1]):
        return -np.inf

    # Require the charm & bottom sum to stay the same
    # c, b = ui.idx['c'], ui.idx['b']
    # cf = np.absolute(np.sum(x[c]) - np.sum(x_prior[c])) / np.sum(x_prior[c])
    # bf = np.absolute(np.sum(x[b]) - np.sum(x_prior[b])) / np.sum(x_prior[b])
    # if cf > 0.01 or bf > 0.01:
    #   return -np.inf

    # lp_ept = w[0] * mvn(x, matlist[0], datalist[0])
    lp_ept = w[0] * mvn_asymm(x, matlist[0], datalist[0])
    # lp_dca = w[1] * dca_shape(x, matlist[1:], datalist[1:])
    lp_dca = w[1] * dca_shape_mod(x, matlist[1:], datalist[1:], scf, moddcalist)
    lp_reg = l2reg(x, x_prior, alpha, L)

    # print(' lp_ept: {}'.format(lp_ept))
    # print(' lp_dca: {}'.format(lp_dca))
    # print(' lp_reg: {}'.format(lp_reg))
    return lp_ept + lp_dca + lp_reg




def mvn_asymm(x, A, d):
    '''
    Multivariate normal (Gaussian) log likelihood. 
    x: trial solution
    A: concatenated A_c, A_b matrices mapping x -> prediction
    d: data,
       column 0 contains points, 
       column 1 contains upper errors. 
       column 2 contains lower errors. 
    Currently assumes data points are independent (diagonal covariance).
    '''
    c, b = ui.idx['c'], ui.idx['b']
    p = np.dot(A[:, c], x[c]) + np.dot(A[:, b], x[b])
    n = len(d[:, 0])
    err = np.array([d[i,1] if p[i] >= d[i,0] else d[i,2] for i in range(n)])
    # icov_data = np.diag(1. / (err ** 2))
    icov_data = np.diag(np.full(n, 1.))
    return lngauss(d[:, 0] / err, p / err, icov_data)




