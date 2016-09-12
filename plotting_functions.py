import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch


def plot_corr(corr, fcorr, corrname='', energy=200, figname="corr.pdf"):
    '''
    Plot two particle correlation vs delta phi
    '''

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_ylabel(r'$C(\Delta\phi)$')
    ax.set_xlabel(r'$\Delta\phi$')
    ax.set_xlim(-1. * np.pi / 2., 3.*np.pi / 2.)
    ax.axhline(y=1., linestyle='-', color='black')

    # Correlation function
    ax.errorbar(corr[:, 0], corr[:, 1], yerr=corr[:, 2],
                ls='*', lw=2, marker='o', ms=6, color='black',
                label=corrname)

    # Correlation fit
    if fcorr is not None:
        ax.plot(corr[:, 0], fcorr[:, 1], 
                ls='--', color='blue', label=r'$1+2C_1\cos(\Delta\phi)$')
        ax.plot(corr[:, 0], fcorr[:, 2], 
                ls='--', color='red', label=r'$1+2C_2\cos(2\Delta\phi)$')
        ax.plot(corr[:, 0], fcorr[:, 3], 
                ls='--', color='darkgreen', label=r'$1+2C_3\cos(3\Delta\phi)$')
        ax.plot(corr[:, 0], fcorr[:, 0], 
                ls='--', color='black', label='Total')

    # Labels
    ax.legend(fontsize='10', loc=2)
    en = r'd+Au $\sqrt{s_{NN}}=$' + '{}'.format(energy) + ' GeV'
    ax.text(0.55, 0.1, en,
            fontsize=12, transform=ax.transAxes)

    fig.savefig(figname)
    plt.close(fig)

    return