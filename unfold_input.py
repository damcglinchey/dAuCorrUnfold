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

# various useful indecies for vn results
# This needs to match the layout used below!!
idx = {'bbcs': np.array((0, npt+2, 2*(npt+2))),
       'fvtxs': np.array((1, npt+3, 2*(npt+2)+1)),
       'cnt': np.array((np.arange(2,npt+2), np.arange(npt+4, 2*(npt+2)), np.arange(2*(npt+2)+2, 3*(npt+2)))).flatten(),
       'v1': np.arange(0, npt+2),
       'v2': np.arange(npt+2, 2*(npt+2)),
       'v3': np.arange(2*(npt+2), 3*(npt+2)),
       'cntv1': np.arange(2, npt+2),
       'cntv2': np.arange(npt+4, 2*(npt+2)),
       'cntv3': np.arange(2*(npt+2)+2, 3*(npt+2))}

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


def getdata(energy, file='correlations.root'):
    '''
    Get data and also return list of labels
    datalist:
     [0]          := BBCS--FVTXS
     [1:ui.npt+1] := CNT--BBCS in pT bins
     [ui.npt+1:]  := CNT--FVTXS in pT bins
    '''
    print('\nGetting data from {}'.format(file))

    # Get the BBCS -- FVTXS correlation for 0-5% central
    datalist = [corrdata('dphi_corr_dAu{}_BBCSFVTXS_c0'.format(energy), file)]
    labellist = ['BBCS--FVTXS 0-5%']

    # Get the CNT -- BBCS correlation for 0-5% central pT bins
    [datalist.append(corrdata('dphi_corr_dAu{}_CNTBBCS_c0_pt{}'.format(energy, i), file)) for i in range(npt)]
    [labellist.append('CNT--BBCS 0-5% {:.2f}<pT<{:.2f}'.format(ptbins[i], ptbins[i+1])) for i in range(npt)]

    # Get the CNT -- FVTXS correlation for 0-5% central pT bins
    [datalist.append(corrdata('dphi_corr_dAu{}_CNTFVTXS_c0_pt{}'.format(energy, i), file)) for i in range(npt)]
    [labellist.append('CNT--FVTXS 0-5% {:.2f}<pT<{:.2f}'.format(ptbins[i], ptbins[i+1])) for i in range(npt)]

    print('  len(datalist) : {}'.format(len(datalist)))
    print('  len(labellist): {}'.format(len(labellist)))

    return datalist, labellist

def vnini():
    '''
    Setup the initial vn guesses
    vn_ini (and all parameter lists) nx3 matrix
    [:, 0] := v_1
    [:, 1] := v_2
    [:, 2] := v_3
    [0, :]           := v_n BBCS
    [1, :]           := v_n FVTXS
    [2:ui.ncpt+2, :] := v_n CNT in pT bins
    '''
    # vn_ini = np.vstack((np.full(2 + npt, 0.2),
    # np.full(2 + npt, 0.1),
    # np.full(2 + npt, 0.02))).T

    csvi = 'dAu200/csv/4/pq.csv'
    vn_ini = np.loadtxt(csvi, delimiter=',')[:, 0] # Previous step
    vn_ini = vn_ini.reshape((3, 9)).T # Need to automate this ...

    return vn_ini

def calccn(vn, ncor):
    '''
    Calculate C_n's given the input vn matrix
    '''
    cn = np.zeros((ncor, vn.shape[1]))
    cn[0, :] = vn[0, :] * vn[1, :]
    cn[1:npt + 1, :] = vn[2:npt+2,:] * vn[0, :]
    cn[npt + 1:, :] = vn[2:npt+2,:] * vn[1, :]

    # This is a bit of a cheat, but force the C_1's to be negative
    r = cn[:, 0] > 0
    cn[r, 0] = -1 * cn[r, 0]

    return cn


