import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import os
from core.build_struct_eq import *
from core.counter_causal_generator import *
from core.metrics import *

# create folder to save models
os.makedirs('models', exist_ok=True)
# choose the conterfactuals filename, if not specified (default) is "counterfactuals"
cf_file_name = "counterfactuals"

def graph_synthetic():
    """
    define a simple graph to create a synthetic dataset
    the logic is to have a feature X3 that is important to make decisions Y and is caused by other variables X1, X2
    """
    
    G = nx.DiGraph(directed=True)
    # 3 nodes
    G.add_node("X1")
    G.add_node("X2")
    G.add_node("Y")
    G.add_edges_from([('X1', "X2"), ("X1", "Y"), ("X2", "Y")])

    return G


def sample_synthetic_data(size, random=42):
    '''
    generate a synthetic dataframe in line with graph_synthetic and with specific Structural Equations
    X1 = U1**2, X2 = U2, X3 = X1 - X2 + U3, Y = 3 X3 + X2 + UY

    INPUT
    size: int. number of sampled observations
    random: int: random seed for samples
    '''

    # set random seed
    np.random.seed(random)

    # sample U1, U2, U3 and UY
    U1 = np.random.normal(-1, .1, size=size)
    U2 = np.random.normal(5, .1, size=size)
    UY = np.random.normal(0, .1, size=size)

    # build data using the following structural equations
    X2 = U1 + U2
    Y = 3 * X2 - U1 + UY

    return pd.DataFrame({"X1": U1, "X2": X2, "Y": Y})

### START EXPERIMENT

#  generate graph
G = graph_synthetic()
# nx.draw_circular(G, with_labels=True)
# plt.show()

# info about features
constraints_features = {"immutable": [], "higher": []}
categ_features = []

#  build dataset
df = sample_synthetic_data(size=100000)

Y = 1*(df["Y"] > df["Y"].median())
X = df.drop(["Y"], axis=1)

###  Create structural equations (NNs saved in models folder), store residuals (saved in data folder)
struct_eq, nn_causal = create_structural_eqs(X, Y, G, n_nodes_se=100, n_nodes_M=180)

###  Create couterfactuals (saved in data folder)
create_counterfactuals(X, Y, G, struct_eq, nn_causal, constraints_features,
                       numCF=20, output_filename=cf_file_name)

###  Calculate metrics - results will be printed
calculate_metrics(X, Y, G, categ_features, constraints_features, cf_file_name=cf_file_name)

print("----------- metrics on intersection -------------")
calculate_metrics(X, Y, G, categ_features, constraints_features, intersection_only=True, cf_file_name=cf_file_name)