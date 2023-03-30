## Import needed libraries
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import pylab   as plt

# hSBM specific libraries
from sbmtm import sbmtm
import graph_tool.all as gt

##############################################################
# 1. Load dataset. Example with Elixauser comorbidity dataset.
##############################################################
admissions = pd.read_csv("elixhauser_comorbidity_icd9_primary_secondary.csv")                                      # Loading dataset
data_input = admissions[['hadm_id','16-24', '25-44', '45-64', 'F', 'M', 'cardiac_arrhythmias','valvular_disease']] # Get features I need, just some for example
data_input.set_index("hadm_id",inplace=True) # I am setting the hadm_id as index. I use this later to define the observation nodes (patients in my case)

# Need to replace the values in the columns for the column name to pass to make_graph function
# make_graph makes a node out of each existing patient feature.
# Since in my df I have binary features for each columnm, the algorithm will understand that there are only two nodes: "1" and "0"
# By replacing the binary flag by the column name I indicate the algorithm that there is 1 node per feature.
# If your df is different this might not be needed.
for col in data_input.columns:
    data_input.replace(col)
    data_input[col].replace({1: col}, inplace=True)
data_input[:3]

###############################
#2. Build the bipartite network
###############################
patients = [h.split()[0] for h in data_input.index.values.astype('str')]              # get all patients identifier
features = [[t for t in patient if t != 0] for patient in data_input.values.tolist()] # get all patient features

model = sbmtm()                                            ## we create an instance of the sbmtm-class
model.make_graph(features,documents=patients,counts=False) ## Create the bipartite network

#################
#3. Run the model
#################
gt.seed_rng(40)                                            # Setting up the random seed for the inference process to ensure reproducibility of the experiments
model.fit(n_init = 10, verbose = False)  # fit the model 10 times and keep the best solution

#########################   
#4. Optimize the solution
#########################

# multiflip mcmc sweep optimizes the results and updates the BlockState in the model object automatically.
entropy=[model.state.multiflip_mcmc_sweep(beta=np.inf, niter=10) for i in range(1000)] # In this case we are runing the sweep 10 time with 1000 batches each
model.save_model('hSBM_demogr_model')
file = open('hSBM_demogr_model.pickle', 'rb')
model = pickle.load(file)
print("Entropy after multiflip mcmc sweep: {0}".format(model.state.entropy()))         # To confirmed that the entropy has imporoved after runing the optimization


#5. Visualize the results. This will display directly if run in a jupyter notebook, if not you can save the graph.
model.state.draw(subsample_edges=6000,layout='bipartite',bip_aspect=1, hvertex_size=8, hedge_pen_width=1.9, output_size=(600, 600), 
output="hSBM_demogr_bipartite_network.svg")

