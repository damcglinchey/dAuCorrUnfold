import numpy as np
import unfold_input as ui
import plotting_functions as pf
import corr_funcs as cf
import lnpmodels as lnp


# get correlation data from file
corr_FVTXNFVTXS = ui.corrdata('dphi_corr_dAu200_FVTXNFVTXS_c0')
print('FVTXN--FVTXS correlation data:{}'.format(corr_FVTXNFVTXS))

# trail cn values
cn_FVTXNFVTXS = np.array([-0.0031, 0.0018, 0.0])
print('Cn values:{}'.format(cn_FVTXNFVTXS))

# get correlation functions given cn values
fcorr_FVTXNFVTXS = cf.corr(corr_FVTXNFVTXS[:, 0], cn_FVTXNFVTXS)

fc1_FVTXNFVTXS = cf.ci(corr_FVTXNFVTXS[:, 0], cn_FVTXNFVTXS[0], 1)
fc2_FVTXNFVTXS = cf.ci(corr_FVTXNFVTXS[:, 0], cn_FVTXNFVTXS[1], 2)
fc3_FVTXNFVTXS = cf.ci(corr_FVTXNFVTXS[:, 0], cn_FVTXNFVTXS[2], 3)

fcn_FVTXNFVTXS = np.vstack((fcorr_FVTXNFVTXS, 
                            fc1_FVTXNFVTXS, 
                            fc2_FVTXNFVTXS,
                            fc3_FVTXNFVTXS)).T
print fcorr_FVTXNFVTXS
print fcn_FVTXNFVTXS

# Get likelihood values
ll = lnp.lncorr(corr_FVTXNFVTXS, cn_FVTXNFVTXS)
print('ll:{}'.format(ll))

pf.plot_corr(corr_FVTXNFVTXS, fcn_FVTXNFVTXS, 
             corrname="FVTXN--FVTXS", 
             energy=200,
             ll=ll,
             figname="test.pdf")


