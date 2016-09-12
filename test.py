import numpy as np
import unfold_input as ui
import plotting_functions as pf
import corr_funcs as cf


corr_FVTXNFVTXS = ui.corrdata('dphi_corr_dAu200_FVTXNFVTXS_c0')
corr_CNTFVTXS = ui.corrdata('dphi_corr_dAu200_CNTFVTXS_c0')
corr_CNTFVTXN = ui.corrdata('dphi_corr_dAu200_CNTFVTXN_c0')

cn_FVTXNFVTXS = np.array([-0.0031, 0.0018, 0.0])

fcorr_FVTXNFVTXS = cf.corr(corr_FVTXNFVTXS[:, 0], cn_FVTXNFVTXS)

fc1_FVTXNFVTXS = cf.ci(corr_FVTXNFVTXS[:, 0], cn_FVTXNFVTXS[0], 1)
fc2_FVTXNFVTXS = cf.ci(corr_FVTXNFVTXS[:, 0], cn_FVTXNFVTXS[1], 2)
fc3_FVTXNFVTXS = cf.ci(corr_FVTXNFVTXS[:, 0], cn_FVTXNFVTXS[2], 3)

fcn_FVTXNFVTXS = np.vstack((fcorr_FVTXNFVTXS, 
                            fc1_FVTXNFVTXS, 
                            fc2_FVTXNFVTXS,
                            fc3_FVTXNFVTXS)).T

print corr_FVTXNFVTXS
print fcorr_FVTXNFVTXS
print fcn_FVTXNFVTXS

pf.plot_corr(corr_FVTXNFVTXS, fcn_FVTXNFVTXS, "FVTXN--FVTXS", 200,
             figname="test.pdf")


