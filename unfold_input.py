import numpy as np
from scipy.stats import norm
from ROOT import TFile, TH1, TH2D, TCanvas, gStyle
from h2np import h2a, binctrs





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