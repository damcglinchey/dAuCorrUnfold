import numpy as np
from scipy.stats import norm
from ROOT import TFile, TH1, TH2D, TCanvas, gStyle
from h2np import h2a, binctrs


# pT bin edges
ptbins = np.array([0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00])

# pT bin width
ptw = np.diff(ptbins)

# pT bin center
ptx = ptbins[:-1] + ptw / 2

# number of pT bins
npt = len(ptx)



def corrdata(corr, file='correlations.root'):
    '''
    Get dphi correlation functions from file and return as array
    '''

    f = TFile(file)
    hcor = f.Get(corr)

    x = binctrs(hcor)
    y = h2a(hcor)
    e = h2a(hcor, 'e')

    cor = np.vstack((x, y, e)).T

    return cor