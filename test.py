import numpy as np
import unfold_input as ui
import plotting_functions as pf


corr_FVTXNFVTX = ui.corrdata('dphi_corr_dAu200_FVTXNFVTXS_c0')

print corr_FVTXNFVTX

pf.plot_corr(corr_FVTXNFVTX, "FVTXN--FVTXS", 200,
             figname="test.pdf")