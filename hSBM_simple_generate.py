## Import needed libraries
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import pylab   as plt

# hSBM specific libraries
from sbmtm import sbmtm
import graph_tool.all as gt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

file = open('hSBM_no_deg_core_full_simple_model.pickle', 'rb')
model = pickle.load(file)


# The hierarchical levels themselves are represented by individual BlockState() instances
levels = model.state.get_levels()
for s in levels:
    print(s)
    if s.get_N() == 1:
        break


# Generate a random graph by sampling from the stochastic block model.
nest_state = model.state
lstate = nest_state.get_levels()[0]
adj_matrix = gt.adjacency(g=lstate.get_bg(),weight=lstate.get_ers()).T
synth_graph = gt.generate_sbm(b=lstate.b.a, probs=adj_matrix, 
                            directed=False)

synth_nest_state = None
mdl = np.inf
# fit SBM on synthetic graph
for i_n_init in range(10):
    base_type = gt.BlockState
    synth_nest_state_tmp = gt.minimize_nested_blockmodel_dl(synth_graph,
                                                state_args=dict(deg_corr=False,
                                                    base_type=base_type,
                                                    **{'clabel': lstate.g.vp['kind'], 'pclabel': lstate.g.vp['kind']}),
                                                multilevel_mcmc_args=dict(
                                                    verbose=False))
    L = 0
    for s in synth_nest_state_tmp.levels:
        L += 1
        if s.get_nonempty_B() == 2:
            break
    synth_nest_state_tmp = synth_nest_state_tmp.copy(bs=synth_nest_state_tmp.get_bs()[:L] + [np.zeros(1)])
    print(synth_nest_state_tmp)

    mdl_tmp = synth_nest_state_tmp.entropy()
    if mdl_tmp < mdl:
        mdl = 1.0*mdl_tmp
        synth_nest_state = synth_nest_state_tmp.copy()


entropy=[synth_nest_state.multiflip_mcmc_sweep(beta=np.inf, niter=10) for i in range(10)] # In this case we are runing the sweep 10 time with 1000 batches each
print("Entropy after multiflip mcmc sweep: {0}".format(synth_nest_state.entropy()))


# Save the state object to a pickle file
with open('synth_hSBM_no_deg_core_full_simple_model.pickle', 'wb') as f:
    pickle.dump(synth_nest_state, f)


# Load the model from the specified path
synth_file = open('synth_hSBM_no_deg_core_full_simple_model.pickle', 'rb')
synth_model = pickle.load(synth_file)


# Visualize the results
synth_model.draw(subsample_edges=6000,layout='bipartite',bip_aspect=1, hvertex_size=8, hedge_pen_width=1.9, output_size=(600, 600), 
output="synth_hSBM_no_deg_core_full_simple_model.svg")

