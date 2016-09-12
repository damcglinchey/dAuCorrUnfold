import numpy as np


def corr(xv, cn):
    '''
    The correlation function given correlation coefficients
    xv: array of delta phi values the correlation will be calculated at
    cn: array of correlation coefficients
    '''
    val = np.ones(xv.size)
    for idx, c in enumerate(cn):
        val += 2 * c * np.cos((idx+1)*xv) 

    return val

def ci(x, c, order):
    '''
    The correlation function for a given order
    '''
    return 1 + 2 * c * np.cos(order * x)