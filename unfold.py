import os
import sys
import numpy as np
import emcee
import unfold_input as ui
import plotting_functions as pf
import corr_funcs as cf
import lnpmodels as lnp


def unfold(energy=200,
           rootfile='correlations.root',
           outdir='test',
           nwalkers=500,
           nburnin=100,
           nsteps=100):
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
    pdfdir = '{}/pdfs/'.format(outdir)
    csvdir = '{}/csv/'.format(outdir)

    if not os.path.isdir(pdfdir):
        os.makedirs(pdfdir)
    if not os.path.isdir(csvdir):
        os.makedirs(csvdir)
    
    
    # Print running conditions
    print("--------------------------------------------")
    print(" energy        : {}".format(energy))
    print(" rootfile      : {}".format(rootfile))
    print(" nwalkers      : {}".format(nwalkers))
    print(" nburnin       : {}".format(nburnin))
    print(" nsteps        : {}".format(nsteps))
    print(" pdfdir        : {}".format(pdfdir))
    print(" csvdir        : {}".format(csvdir))
    print("--------------------------------------------")


    # Get the Correlation data
    corr_BBCSFVTXS = ui.corrdata('dphi_corr_dAu{}_BBCSFVTXS_c0'.format(energy))
    corr_CNTBBCS = [ui.corrdata('dphi_corr_dAu{}_CNTBBCS_c0_pt{}'.format(energy, i)) for i in range(ui.npt)]
    corr_CNTFVTXS = [ui.corrdata('dphi_corr_dAu{}_CNTFVTXS_c0_pt{}'.format(energy, i)) for i in range(ui.npt)]


    # Wrap everything into a single data list
    # data list:
    #  [0]          := BBCS--FVTXS
    #  [1:ui.npt+1] := CNT--BBCS in pT bins
    #  [ui.npt+1:]  := CNT--FVTXS in pT bins
    datalist = [corr_BBCSFVTXS]
    [datalist.append(m) for m in corr_CNTBBCS]                         
    [datalist.append(m) for m in corr_CNTFVTXS]                         
    print(len(datalist))

    # Setup the initial vn guesses
    # vn_ini (and all parameter lists) nx3 matrix
    # [:, 0] := v_1
    # [:, 1] := v_2
    # [:, 2] := v_3
    # [0, :]           := v_n BBCS
    # [1, :]           := v_n FVTXS
    # [2:ui.ncpt+2, :] := v_n CNT in pT bins
    vn_ini = np.vstack((np.full(2 + ui.npt, 0.2),
    np.full(2 + ui.npt, 0.1),
    np.full(2 + ui.npt, 0.02))).T
    print(vn_ini.shape)
    print(vn_ini)

    nv = vn_ini.shape[0]
    nn = vn_ini.shape[1]

    vn_ini = vn_ini.flatten()
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


if __name__ == '__main__':
    # np.set_printoptions(precision=3)

    # Generate csv files and plots with default settings
    unfold()
