import os
import sys
import numpy as np
import emcee
import unfold_input as ui
import plotting_functions as pf
import corr_funcs as cf
import lnpmodels as lnp


def unfold(step=5,
           energy=200,
           rootfile='correlations.root',
           outdir='CNTBBCSFVTXS/dAu200',
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
    print(vn_ini.shape)
    print(vn_ini)




    #--------------------------------------------------------------------------
    # Run sampler
    #--------------------------------------------------------------------------

    # Set parameter limits - put in array with shape (ndim,2)
    parlimits = np.vstack((np.full(ui.ndim, -1), np.full(ui.ndim, 1))).T

    
    # Ensemble of starting points for the walkers - shape (nwalkers, ui.ndim)
    print("Initializing {} {}-dim walkers...".format(nwalkers, ui.ndim))
    x0 = vn_ini * (1 + 0.1 * np.random.randn(nwalkers, ui.ndim))


    # testing
    llsum = lnp.lndata(vn_ini, datalist, parlimits)
    print('llsum = {}'.format(llsum))

    # Function returning values \propto posterior probability and arg tuple
    fcn, args = None, None
    
    print("Setting up sampler...")
    fcn = lnp.lndata
    args = (datalist, parlimits)
    
    sampler = emcee.EnsembleSampler(nwalkers, ui.ndim, fcn, args=args, threads=2)
    
    print("Burning in for {} steps...".format(nburnin))
    pos, prob, state = sampler.run_mcmc(x0, nburnin)
    sampler.reset()
    print("Running sampler for {} steps...".format(nsteps))
    sampler.run_mcmc(pos, nsteps)
    acc_frac = np.mean(sampler.acceptance_fraction)
    print("Mean acceptance fraction: {0:.3f}".format(acc_frac))
    
    # Initial shape of sampler.chain is (nwalkers, nsteps, ui.ndim).
    # Reshape to (nwalkers*nsteps, ui.ndim).
    # Posterior quantiles: list of ui.ndim (16,50,84) percentile tuples
    print("posterior quantiles")
    samples = sampler.chain.reshape((-1, ui.ndim))
    pq = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    pq = np.array(pq)
    print(pq)

    # # Calculate the sample covariance matrix
    # # https://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance
    # print("Calculating the covariance matrix")
    # cov = np.zeros([ui.ndim, ui.ndim])
    # N = samples.shape[0]
    # for j in range(ui.ndim):
    #     for k in range(ui.ndim):
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
    vn_final = pq[:, 0].reshape((ui.nn, ui.nv)).T
    cn_final = ui.calccn(vn_final, len(datalist))
    ll_final = [lnp.lncorr(datalist[i], cn_final[i, :]) for i in range(len(datalist))]
    fcorr_final = []
    for idx, d in enumerate(datalist):
        fcorr_final.append(np.vstack((cf.corr(d[:, 0], cn_final[idx]),
                                      cf.ci(d[:, 0], cn_final[idx][0], 1),
                                      cf.ci(d[:, 0], cn_final[idx][1], 2),
                                      cf.ci(d[:, 0], cn_final[idx][2], 3))).T)
                  
    v2_pt = pq[ui.idx['cntv2'], :]
    v3_pt = pq[ui.idx['cntv3'], :]



    #--------------------------------------------------------------------------
    # Print
    #--------------------------------------------------------------------------

    # Plot unfold diagnostics
    pf.plot_lnprob(sampler.flatlnprobability, pdfdir + 'lnprob.pdf')
    pf.plot_lnp_steps(sampler, nburnin, pdfdir + 'lnprob-vs-step.pdf')
    pf.plot_post_marg(samples, parlimits, pdfdir + 'posterior.pdf')
    # pf.plot_post_marg_triangle(samples, pdfdir + 'posterior-triangle.pdf')

    pf.plot_vnpt_prob(samples[:, ui.idx['cntv2']], 2, energy,
                      pdfdir + 'v2pt-prob.pdf')
    pf.plot_vnpt_prob(samples[:, ui.idx['cntv3']], 3, energy,
                      pdfdir + 'v3pt-prob.pdf')

    # Plot correlation functions
    [pf.plot_corr(datalist[i], fcorr_final[i], labellist[i], energy, ll_final[i], pdfdir + 'corr_{}.pdf'.format(i)) for i in range(len(datalist))]

    # Plot vn vs pT
    pf.plot_vnpt(v2_pt, 2, energy, figname=pdfdir + 'v2_pt.pdf')
    pf.plot_vnpt(v3_pt, 3, energy, figname=pdfdir + 'v3_pt.pdf')
    pf.plot_v2v3pt(v2_pt, v3_pt, energy, figname=pdfdir + 'v2v3_pt.pdf')
    # Write out the unfolded vn values
    np.savetxt("{}pq.csv".format(csvdir), pq, delimiter=",")

    # Plot triangle plot last ..
    if step > 0:
        pf.plot_triangle_vn(samples, pdfdir + 'posterior-triangle-vn.pdf')

    #--------------------------------------------------------------------------
    # Done
    #--------------------------------------------------------------------------
    print('\nDone!\n')
    return
if __name__ == '__main__':
    # np.set_printoptions(precision=3)

    # Generate csv files and plots with default settings
    unfold()
