import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import unfold_input as ui


def plot_corr(corr, fcorr, 
              corrname='', energy=200, ll=None,
              figname="corr.pdf"):
    '''
    Plot two particle correlation vs delta phi
    '''
    print('plot_corr()')

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
                ls='--', color='black', label=r'$1+\sum_{n=1}^{4}2C_n\cos(n\Delta\phi)$')

    # Labels
    ax.legend(fontsize='10', loc=2)
    en = r'd+Au $\sqrt{s_{NN}}=$' + '{}'.format(energy) + ' GeV'
    ax.text(0.55, 0.1, en,
            fontsize=12, transform=ax.transAxes)
    if ll is not None:
        ax.text(0.06, 0.65, 'LL={:.2f}'.format(ll),
                fontsize=12, transform=ax.transAxes)

    fig.savefig(figname)
    plt.close(fig)
    return


def plot_vnpt(vn, order, energy, figname='vn_pt.pdf'):
    '''
    Plot the v_n vs pT
    '''
    print('plot_vnpt()')

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_ylabel(r'$v_' + '{}'.format(order) + r'$')
    ax.set_xlabel(r'$p_T$ [GeV/c]')
    ax.set_xlim(0, ui.ptbins[-1])
    if order == 2:
        ax.set_ylim(0, 0.24)
    if order == 3:
        ax.set_ylim(0, 0.075)
    ax.axhline(y=0., linestyle='-', color='black')

    # Correlation function
    ax.errorbar(ui.ptx, vn[:, 0], yerr=[vn[:, 2], vn[:, 1]],
                ls='*', lw=2, marker='o', ms=6, color='black', ecolor='black')

    en = r'd+Au $\sqrt{s_{NN}}=$' + '{}'.format(energy) + ' GeV'
    ax.text(0.07, 0.9, en,
            fontsize=12, transform=ax.transAxes)
    ax.text(0.18, 0.85, r'$y<|0.35|$',
            fontsize=12, transform=ax.transAxes)


    fig.savefig(figname)
    plt.close(fig)
    return



def plot_lnprob(lnp, figname='lnprob.pdf'):
    print("plot_lnprob()")
    fig, ax = plt.subplots()
    ax.set_xlabel('log likelihood distribution')
    ax.set_ylabel('samples')
    ax.hist(
        lnp, 100, color='k', facecolor='lightyellow', histtype='stepfilled')
    ax.text(0.05, 0.95, r'mean, std dev: {:.3g}, {:.3g}'
            .format(np.mean(lnp), np.std(lnp)), transform=ax.transAxes)
    fig.savefig(figname)
    plt.close(fig)


def plot_lnp_steps(sampler, nburnin, figname='lnprob_vs_step.pdf'):
    print("plot_lnp_steps()")
    nwalkers = sampler.chain.shape[0]
    fig, ax = plt.subplots()
    ax.set_xlabel('step')
    ax.set_ylabel(r'$\langle \ln(L) \rangle_{chains}$')
    ax.set_title(r'$\langle \ln(L) \rangle_{chains}$ vs. sample step')
    ax.plot(np.sum(sampler.lnprobability / nwalkers, axis=0), color='k')
    ax.text(0.05, 0.95,
            '{} chains after {} burn-in steps'.format(nwalkers, nburnin),
            transform=ax.transAxes)
    fig.savefig(figname)
    plt.close(fig)


def plot_post_marg(chain, xlim, figname='posterior.pdf'):
    # Draw posterior marginal distributions
    print("plot_post_marg()")
    nr, nc = 5, 6
    fig, axes = plt.subplots(nr, nc)
    for row in range(nr):
        for col in range(nc):
            i = nc * row + col
            a = axes[row, col]
            if i >= chain.shape[1]: 
                a.tick_params(axis='x', top='off', bottom='off', labelsize=0)               
                a.tick_params(axis='y', left='off', right='off', labelsize=0)               
                continue
            a.hist(chain[:, i], 1000,
                   color='k', facecolor='steelblue', histtype='stepfilled')
            # a.hist(chain[:, i], 1000, range=(xlim[i, 0], xlim[i, 1]),
            #        color='k', facecolor='steelblue', histtype='stepfilled')
            # a.hist(chain[:, i], 1000, range=(-0.2, 0.2),
            #        color='k', facecolor='lightyellow', histtype='stepfilled')
            a.tick_params(axis='x', top='off', labelsize=4)
            a.tick_params(axis='y', left='off', right='off', labelsize=0)
            a.xaxis.get_offset_text().set_size(4)
            a.xaxis.get_major_formatter().set_powerlimits((0, 1))
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)


