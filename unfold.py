import os
import sys
import numpy as np
import emcee
import unfold_input as ui
import plotting_functions as pf
import corr_funcs as cf
import lnpmodels as lnp


def unfold(step=1,
           energy=20,
           rootfile='correlations.root',
           outdir='dAu20',
           nwalkers=500,
           nburnin=500,
           nsteps=500):
    '''
    Perform the unfolding
    energy   := Collision energy [200, 62, 39, 20]
    rootfile := Root file containing correlation data
    outdir   := Output directory
    nwalkers := The number of MCMC samplers.
    nburnin  := The number of steps run while not keeping results.
    nsteps   := The number of recorded steps taken by the walkers.
    '''

    #--------------------------------------------------------------------------
    # Setup/configuration
    #--------------------------------------------------------------------------

    # Output locations
    pdfdir = '{}/pdfs/{}/'.format(outdir, step + 1)
    csvdir = '{}/csv/{}/'.format(outdir, step + 1)

    if not os.path.isdir(pdfdir):
        os.makedirs(pdfdir)
    if not os.path.isdir(csvdir):
        os.makedirs(csvdir)
    
    
    # Print running conditions
    print("--------------------------------------------")
    print(" step          : {}".format(step))
    print(" energy        : {}".format(energy))
    print(" rootfile      : {}".format(rootfile))
    print(" nwalkers      : {}".format(nwalkers))
    print(" nburnin       : {}".format(nburnin))
    print(" nsteps        : {}".format(nsteps))
    print(" pdfdir        : {}".format(pdfdir))
    print(" csvdir        : {}".format(csvdir))
    print("--------------------------------------------")


    # Get the Correlation data
    datalist, labellist = ui.getdata(energy, rootfile)

    # Get the initial parameter guesses
    if step == 0:
        vn_ini = ui.vnini()
    else:
        csvi = '{}/csv/{}/pq.csv'.format(outdir, step)
        vn_ini = np.loadtxt(csvi, delimiter=',')[:, 0] # Previous step
        vn_ini = vn_ini.reshape((3, 9)).T # Need to automate this ...
    print(vn_ini.shape)
    print(vn_ini)

    # Calculate the Cn's from initial vn's
    cn_ini = ui.calccn(vn_ini, len(datalist))
    print(cn_ini.shape)
    print(cn_ini)


    nv = vn_ini.shape[0]
    nn = vn_ini.shape[1]

    vn_ini = vn_ini.T.flatten()
    ndim = len(vn_ini)
    print('ndim = {}'.format(ndim))



    #--------------------------------------------------------------------------
    # Run sampler
    #--------------------------------------------------------------------------

    # Set parameter limits - put in array with shape (ndim,2)
    parlimits = np.vstack((np.full(ndim, -1), np.full(ndim, 1))).T

    
    # Ensemble of starting points for the walkers - shape (nwalkers, ndim)
    print("Initializing {} {}-dim walkers...".format(nwalkers, ndim))
    x0 = vn_ini * (1 + 0.1 * np.random.randn(nwalkers, ndim))


    # testing
    llsum = lnp.lndata(vn_ini, datalist, parlimits, nv, nn)
    print('llsum = {}'.format(llsum))

    # Function returning values \propto posterior probability and arg tuple
    fcn, args = None, None
    
    print("Setting up sampler...")
    fcn = lnp.lndata
    args = (datalist, parlimits, nv, nn)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, fcn, args=args, threads=2)
    
    print("Burning in for {} steps...".format(nburnin))
    pos, prob, state = sampler.run_mcmc(x0, nburnin)
    sampler.reset()
    print("Running sampler for {} steps...".format(nsteps))
    sampler.run_mcmc(pos, nsteps)
    acc_frac = np.mean(sampler.acceptance_fraction)
    print("Mean acceptance fraction: {0:.3f}".format(acc_frac))
    
    # Initial shape of sampler.chain is (nwalkers, nsteps, ndim).
    # Reshape to (nwalkers*nsteps, ndim).
    # Posterior quantiles: list of ndim (16,50,84) percentile tuples
    print("posterior quantiles")
    samples = sampler.chain.reshape((-1, ndim))
    pq = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    pq = np.array(pq)
    print(pq)

    # # Calculate the sample covariance matrix
    # # https://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance
    # print("Calculating the covariance matrix")
    # cov = np.zeros([ndim, ndim])
    # N = samples.shape[0]
    # for j in range(ndim):
    #     for k in range(ndim):
    #         cov[j, k] = 1./(N-1) * np.sum((samples[:, j] - pq[j, 0]) * (samples[:, k] - pq[k, 0]))

    # print(cov[0:7, 0:7])

    # # Compare the quantiles of pq calculated above to the 
    # # diagonal of the covariance matrix
    # print("pq[:, 1]:\n{}".format(pq[:, 1]))
    # print("pq[:, 2]:\n{}".format(pq[:, 2]))
    # print("diag(cov):\n{}".format(np.sqrt(np.diag(cov))))

    #--------------------------------------------------------------------------
    # Collect the results
    #--------------------------------------------------------------------------

    # calculate cn's
    vn_final = pq[:, 0].reshape((nn, nv)).T
    cn_final = ui.calccn(vn_final, len(datalist))
    ll_final = [lnp.lncorr(datalist[i], cn_final[i, :]) for i in range(len(datalist))]
    fcorr_final = []
    for idx, d in enumerate(datalist):
        fcorr_final.append(np.vstack((cf.corr(d[:, 0], cn_final[idx]),
                                      cf.ci(d[:, 0], cn_final[idx][0], 1),
                                      cf.ci(d[:, 0], cn_final[idx][1], 2),
                                      cf.ci(d[:, 0], cn_final[idx][2], 3))).T)
                  
    v2_pt = pq[4+ui.npt:2*(2+ui.npt), :]
    v3_pt = pq[2*(2+ui.npt)+2:, :]


    # for plotting after the first iteration
    if step > 0:
        parlimits = np.vstack((np.full(ndim, -0.1), np.full(ndim, 0.2))).T


    #--------------------------------------------------------------------------
    # Print
    #--------------------------------------------------------------------------

    # Plot unfold diagnostics
    pf.plot_lnprob(sampler.flatlnprobability, pdfdir + 'lnprob.pdf')
    pf.plot_lnp_steps(sampler, nburnin, pdfdir + 'lnprob-vs-step.pdf')
    pf.plot_post_marg(samples, parlimits, pdfdir + 'posterior.pdf')

    # Plot correlation functions
    [pf.plot_corr(datalist[i], fcorr_final[i], labellist[i], energy, ll_final[i], pdfdir + 'corr_{}.pdf'.format(i)) for i in range(len(datalist))]

    # Plot vn vs pT
    pf.plot_vnpt(v2_pt, 2, energy, figname=pdfdir + 'v2_pt.pdf')
    pf.plot_vnpt(v3_pt, 3, energy, figname=pdfdir + 'v3_pt.pdf')

    # Write out the unfolded vn values
    np.savetxt("{}pq.csv".format(csvdir), pq, delimiter=",")

    #--------------------------------------------------------------------------
    # Done
    #--------------------------------------------------------------------------
    print('\nDone!\n')
    return
if __name__ == '__main__':
    # np.set_printoptions(precision=3)

    # Generate csv files and plots with default settings
    unfold()
