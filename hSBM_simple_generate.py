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

file = open('hSBM_simple_model.pickle', 'rb')
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
                            out_degs=lstate.g.degree_property_map("total").a,
                            in_degs=lstate.g.degree_property_map("total").a,
                            directed=False)


# Once a random graph has been generated, create new state object which has the same block structure as the original state object, 
# but applied to a different graph
synth_nest_state = nest_state.copy(g=synth_graph)


# Save the state object to a pickle file
with open('synth_hSBM_simple_model.pickle', 'wb') as f:
    pickle.dump(synth_nest_state, f)


# Load the model from the specified path
synth_file = open('synth_hSBM_simple_model.pickle', 'rb')
synth_model = pickle.load(synth_file)


# Visualize the results
synth_model.draw(subsample_edges=6000,layout='bipartite',bip_aspect=1, hvertex_size=8, hedge_pen_width=1.9, output_size=(600, 600), 
output="synth_hSBM_bipartite_network.svg")

