import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch


def plot_corr(corr, corrname='', energy=200, figname="corr.pdf"):
    '''
    Plot two particle correlation vs delta phi
    '''

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_ylabel(r'$C(\Delta\phi)$')
    ax.set_xlabel(r'$\Delta\phi$')
    ax.set_xlim(-1. * np.pi / 2., 3.*np.pi / 2.)
    ax.axhline(y=1., linestyle='--', color='black')

    # Correlation function
    ax.errorbar(corr[:, 0], corr[:, 1], yerr=corr[:, 2],
                ls='*', lw=2, marker='o', ms=6, color='black',
                label=corrname)

    # Labels
    ax.legend(fontsize='10', loc=2)
    en = r'd+Au $\sqrt{s_{NN}}=$' + '{}'.format(energy) + ' GeV'
    ax.text(0.55, 0.1, en,
            fontsize=12, transform=ax.transAxes)

    fig.savefig(figname)
    plt.close(fig)

    return