## Import needed libraries
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import pylab   as plt

# hSBM specific libraries
from non_hsbm import non_hsbm
import graph_tool.all as gt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

file = open('SBM_full_simple_model.pickle', 'rb')
model = pickle.load(file)

# Generate a random graph by sampling from the stochastic block model.
lstate = model.state
adj_matrix = gt.adjacency(g=lstate.get_bg(),weight=lstate.get_ers()).T
synth_graph = gt.generate_sbm(b=lstate.b.a, probs=adj_matrix, 
                            out_degs=lstate.g.degree_property_map("total").a,
                            in_degs=lstate.g.degree_property_map("total").a,
                            directed=False)


synth_nest_state = None
mdl = np.inf
# fit SBM on synthetic graph
for i_n_init in range(10):
    synth_nest_state_tmp = gt.minimize_blockmodel_dl(synth_graph,
                                                state_args=dict(deg_corr=True,
                                                    **{'clabel': lstate.g.vp['kind'], 'pclabel': lstate.g.vp['kind']}),
                                                multilevel_mcmc_args=dict(
                                                    verbose=False))
    
    print(synth_nest_state_tmp)

    mdl_tmp = synth_nest_state_tmp.entropy()
    if mdl_tmp < mdl:
        mdl = 1.0*mdl_tmp
        synth_nest_state = synth_nest_state_tmp.copy()


entropy=[synth_nest_state.multiflip_mcmc_sweep(beta=np.inf, niter=10) for i in range(10)] # In this case we are runing the sweep 10 time with 1000 batches each
print("Entropy after multiflip mcmc sweep: {0}".format(synth_nest_state.entropy()))


# Save the state object to a pickle file
with open('synth_SBM_full_simple_model.pickle', 'wb') as f:
    pickle.dump(synth_nest_state, f)


# Load the model from the specified path
synth_file = open('synth_SBM_full_simple_model.pickle', 'rb')
synth_model = pickle.load(synth_file)


# Visualize the results
# synth_model.draw(subsample_edges=6000,layout='bipartite',bip_aspect=1, hvertex_size=8, hedge_pen_width=1.9, output_size=(600, 600), 
# output="synth_SBM_full_simple_model.svg")