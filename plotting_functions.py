import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import unfold_input as ui
import triangle

# array of colors used for plotting each order of c_n/v_n
colors = {'all': 'black', 
          'v1': 'dodgerblue', 
          'v2': 'crimson', 
          'v3': 'forestgreen',
          'cnt': 'crimson',
          'bbcs': 'forestgreen',
          'fvtxs': 'dodgerblue',
          'cor': 'darkorange'}


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
                ls='--', color=colors['v1'], label=r'$1+2C_1\cos(\Delta\phi)$')
        ax.plot(corr[:, 0], fcorr[:, 2], 
                ls='--', color=colors['v2'], label=r'$1+2C_2\cos(2\Delta\phi)$')
        ax.plot(corr[:, 0], fcorr[:, 3], 
                ls='--', color=colors['v3'], label=r'$1+2C_3\cos(3\Delta\phi)$')
        ax.plot(corr[:, 0], fcorr[:, 0], 
                ls='--', color=colors['all'], label=r'$1+\sum_{n=1}^{4}2C_n\cos(n\Delta\phi)$')

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
        ax.set_ylim(-0.02, 0.075)
    ax.axhline(y=0., linestyle='-', color='black')

    # Correlation function
    if order == 2:
        c = colors['v2']
    elif order == 3:
        c = colors['v3']
    else:
        c = 'black'
    ax.errorbar(ui.ptx, vn[:, 0], yerr=[vn[:, 2], vn[:, 1]],
                ls='*', lw=2, marker='o', ms=6, 
                color=c, ecolor=c)

    en = r'd+Au $\sqrt{s_{NN}}=$' + '{}'.format(energy) + ' GeV'
    ax.text(0.07, 0.9, en,
            fontsize=12, transform=ax.transAxes)
    ax.text(0.18, 0.85, r'$y<|0.35|$',
            fontsize=12, transform=ax.transAxes)


    fig.savefig(figname)
    plt.close(fig)
    return


def plot_v2v3pt(v2, v3, energy, figname='v2v3_pt.pdf'):
    '''
    Plot the v_2 and v_3 vs pT
    '''
    print('plot_vnpt()')

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_ylabel(r'$v_n$')
    ax.set_xlabel(r'$p_T$ [GeV/c]')
    ax.set_xlim(0, ui.ptbins[-1])
    ax.set_ylim(-0.02, 0.24)
    ax.axhline(y=0., linestyle='-', color='black')

    # Correlation function
    ax.errorbar(ui.ptx, v2[:, 0], yerr=[v2[:, 2], v2[:, 1]],
                ls='*', lw=2, marker='o', ms=6, 
                color=colors['v2'], ecolor=colors['v2'],
                label=r'$v_2$')

    ax.errorbar(ui.ptx, v3[:, 0], yerr=[v3[:, 2], v3[:, 1]],
                ls='*', lw=2, marker='o', ms=6, 
                color=colors['v3'], ecolor=colors['v3'],
                label=r'$v_3$')

    en = r'd+Au $\sqrt{s_{NN}}=$' + '{}'.format(energy) + ' GeV'
    ax.text(0.07, 0.75, en,
            fontsize=12, transform=ax.transAxes)
    ax.text(0.18, 0.70, r'$y<|0.35|$',
            fontsize=12, transform=ax.transAxes)

    ax.legend(fontsize='10', loc=2)

    fig.savefig(figname)
    plt.close(fig)
    return


def plot_vnpt_prob(chain, order, energy, figname='posterior.pdf'):
    '''
    Draw posterior marginal distributions
    '''
    print("plot_vnpt_prob()")

    if order == 2:
        c = colors['v2']
    elif order == 3:
        c = colors['v3']
    else:
        c = 'black'

    nr, nc = 2, 4
    fig, axes = plt.subplots(nr, nc)
    for row in range(nr):
        for col in range(nc):
            i = nc * row + col
            a = axes[row, col]
            if i >= chain.shape[1]: 
                a.tick_params(axis='x', top='off', bottom='off', labelsize=0)               
                a.tick_params(axis='y', left='off', right='off', labelsize=0)               
                continue
            a.set_xlabel(r'$v_' + '{}'.format(order) + r'$', fontsize=6)
            a.set_ylabel(r'$P(v_' + '{}'.format(order) + r')$', fontsize=6)

            a.hist(chain[:, i], 100,
                   color='k', facecolor=c, histtype='stepfilled')

            # Get quantiles
            q = np.percentile(chain[:, i], [16, 50, 84], axis=0)
            a.axvline(q[0], ls='--', lw=1, color='k')
            a.axvline(q[1], ls='-', lw=1, color='k')
            a.axvline(q[2], ls='--', lw=1, color='k')

            en = r'd+Au $\sqrt{s_{NN}}=$' + '{}'.format(energy) + ' GeV'
            a.text(0.07, 0.95, en,
                   fontsize=6, transform=a.transAxes)
            tpt = r'$' + '{:.2f}'.format(ui.ptbins[i]) + r'<p_T<' + '{:.2f}'.format(ui.ptbins[i+1]) + r'$'
            a.text(0.07, 0.85, tpt,
                   fontsize=6, transform=a.transAxes)
            a.tick_params(axis='x', top='off', labelsize=4)
            a.tick_params(axis='y', left='off', right='off', labelsize=0)
            a.xaxis.get_offset_text().set_size(4)
            # a.xaxis.get_major_formatter().set_powerlimits((0, 1))
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)


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
            #color
            c = 'lightyellow'
            if i in ui.idx['bbcs']:
                c = colors['bbcs']
            if i in ui.idx['fvtxs']:
                c = colors['fvtxs']
            if i in ui.idx['cnt']:
                c = colors['cnt']
            a.hist(chain[:, i], 1000,
                   color='k', facecolor=c, histtype='stepfilled')
            # a.hist(chain[:, i], 1000, range=(xlim[i, 0], xlim[i, 1]),
            #        color='k', facecolor='steelblue', histtype='stepfilled')
            a.tick_params(axis='x', top='off', labelsize=4)
            a.tick_params(axis='y', left='off', right='off', labelsize=0)
            a.xaxis.get_offset_text().set_size(4)
            a.xaxis.get_major_formatter().set_powerlimits((0, 1))
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)


def plot_post_marg_triangle(chain, figname='posterior-triangle.pdf'):
    print("plot_post_marg_triangle()")
    fig = triangle.corner(chain, quantiles=[0.16, 0.5, 0.84],)
    fig.savefig(figname, bbox_inches='tight')
    return


def plot_triangle_vn(d, figname='posterior-triangle-vn.pdf', plotcor=True):
    '''
    Plot the 2D correlations between vn's
    '''
    print("plot_triangle_vn()")
    K = d.shape[1]
    # factor = 2.0           # size of one side of one panel
    factor = 1.0           # size of one side of one panel
    lbdim = 2.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim    
    frac = factor / dim

    # make the figure
    fig = plt.figure(figsize=(dim, dim))

    # full figure axes
    ax_full = fig.add_axes([0, 0, 1, 1], )
    ax_full.axis('off')

    # Axis labels
    fig.text(0.16, 0.01, r'$v_1$', fontsize=100, color=colors['v1'])
    fig.text(0.49, 0.01, r'$v_2$', fontsize=100, color=colors['v2'])
    fig.text(0.82, 0.01, r'$v_3$', fontsize=100, color=colors['v3'])

    fig.text(0.005, 0.82, r'$v_1$', fontsize=100, color=colors['v1'])
    fig.text(0.005, 0.49, r'$v_2$', fontsize=100, color=colors['v2'])
    fig.text(0.005, 0.16, r'$v_3$', fontsize=100, color=colors['v3'])


    ax_full.arrow(0.05, (1.0 - trdim/dim),
                  0, -1*(1.0 - trdim/dim - lbdim/dim)/3.,
                  length_includes_head=True,
                  shape='full',
                  fc=colors['v1'], ec=colors['v1'],
                  # head_width=20, head_length=5, linewidth=10,
                  transform=ax_full.transAxes)
    ax_full.arrow(0.05, (1.0 - trdim/dim) - (1.0 - trdim/dim - lbdim/dim)/3.,
                  0, -1*(1.0 - trdim/dim - lbdim/dim)/3.,
                  length_includes_head=True,
                  shape='full',
                  fc=colors['v2'], ec=colors['v2'],
                  # head_width=20, head_length=5, linewidth=10,
                  transform=ax_full.transAxes)
    ax_full.arrow(0.05, (1.0 - trdim/dim) - 2*(1.0 - trdim/dim - lbdim/dim)/3.,
                  0, -1*(1.0 - trdim/dim - lbdim/dim)/3.,
                  length_includes_head=True,
                  shape='full',
                  fc=colors['v3'], ec=colors['v3'],
                  # head_width=20, head_length=5, linewidth=10,
                  transform=ax_full.transAxes)

    ax_full.arrow(lbdim/dim, 0.05,
                  1*(1.0 - trdim/dim - lbdim/dim)/3., 0,
                  length_includes_head=True,
                  shape='full',
                  fc=colors['v1'], ec=colors['v1'],
                  # head_width=20, head_length=5, linewidth=10,
                  transform=ax_full.transAxes)
    ax_full.arrow(lbdim/dim + 1*(1.0 - trdim/dim - lbdim/dim)/3., 0.05,
                  1*(1.0 - trdim/dim - lbdim/dim)/3., 0,
                  length_includes_head=True,
                  shape='full',
                  fc=colors['v2'], ec=colors['v2'],
                  # head_width=20, head_length=5, linewidth=10,
                  transform=ax_full.transAxes)
    ax_full.arrow(lbdim/dim + 2*(1.0 - trdim/dim - lbdim/dim)/3., 0.05,
                  1*(1.0 - trdim/dim - lbdim/dim)/3., 0,
                  length_includes_head=True,
                  shape='full',
                  fc=colors['v3'], ec=colors['v3'],
                  # head_width=20, head_length=5, linewidth=10,
                  transform=ax_full.transAxes)


    # Individual correlations and marginals
    axes = []
    for i in range(K):
        for j in range(K):
            # Calcualte the bottom left corner of the desired axis
            # Plots start in the top left
            x1 = lbdim/dim + i * (frac + whspace/dim)
            y1 = 1.0 - trdim/dim - (frac + whspace/dim) * (j + 1)
            # print("({}, {}): [{}, {}, {}, {}]".format(i, j, x1, y1, 
            #       x1+frac, y1+frac))

            c = 'darkorange'
            if j in ui.idx['v1'] and i in ui.idx['v1']:
                c = colors['v1']
            if j in ui.idx['v2'] and i in ui.idx['v2']:
                c = colors['v2']
            if j in ui.idx['v3'] and i in ui.idx['v3']:
                c = colors['v3']

            if i > j:
                if plotcor:
                    ax = fig.add_axes([x1, y1, frac, frac])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                    mi = np.mean(d[:, i])
                    mj = np.mean(d[:, j])
                    N = d.shape[0]

                    num = np.sum(d[:, i] * d[:, j]) - N * mi * mj
    
                    denomi = np.sum(d[:, i]*d[:, i]) - N * mi**2
    
                    denomj = np.sum(d[:, j]*d[:, j]) - N * mj**2
    
                    denom = np.sqrt(denomi) * np.sqrt(denomj)
        
                    p = num / denom

                    ax.text(0.5, 0.5, '{:.2f}'.format(p),
                            fontsize=np.abs(p)**0.5 * 35,
                            color=c, weight='bold',
                            ha='center', va='center')

                continue

            ax = fig.add_axes([x1, y1, frac, frac])
            ax.set_xticklabels([])
            ax.set_yticklabels([])


            if i == j:
                ax.hist(d[:, i], 100, 
                        color='k', fc=c, histtype='stepfilled')
            else:
                rg = [[d[:, i].min(), d[:, i].max()], 
                         [d[:, j].min(), d[:, j].max()]]

                H, X, Y = np.histogram2d(d[:, i], d[:, j], bins=20,
                                         range=rg)

                # compute the bin centers
                X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

                # Compute the density levels.
                levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
                Hflat = H.flatten()
                inds = np.argsort(Hflat)[::-1]
                Hflat = Hflat[inds]
                sm = np.cumsum(Hflat)
                sm /= sm[-1]
                V = np.empty(len(levels))
                for k, v0 in enumerate(levels):
                    try:
                        V[k] = Hflat[sm <= v0][-1]
                    except:
                        V[k] = Hflat[0]
            
                H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
                H2[2:-2, 2:-2] = H
                H2[2:-2, 1] = H[:, 0]
                H2[2:-2, -2] = H[:, -1]
                H2[1, 2:-2] = H[0]
                H2[-2, 2:-2] = H[-1]
                H2[1, 1] = H[0, 0]
                H2[1, -2] = H[0, -1]
                H2[-2, 1] = H[-1, 0]
                H2[-2, -2] = H[-1, -1]
                X2 = np.concatenate([
                    X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
                    X1,
                    X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
                ])
                Y2 = np.concatenate([
                    Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
                    Y1,
                    Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
                ])
                # Filled contour
                contourf_kwargs = dict()
                contourf_kwargs["colors"] = contourf_kwargs.get("colors", c)
                contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                                     False)
                ax.contourf(X2, Y2, H2.T, np.concatenate([[H.max()], V, [0]]),
                            **contourf_kwargs)                



    fig.savefig(figname, transparent=True)

    

