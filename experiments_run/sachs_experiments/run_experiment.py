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
    define a graph to create Sachs dataset
    """
    
    G = nx.DiGraph(directed=True)
    # nodes
    G.add_node('Mek')
    G.add_node('PKC')
    G.add_node('Erk')
    # G.add_node('Plcg')
    # G.add_node('PIP2')
    # G.add_node('PIP3')
    # G.add_node('Akt')
    G.add_node('PKA')
    # G.add_node('P38')
    G.add_node('Raf')
    # G.add_node('Jnk')
    
    # # Morji
    # G.add_edges_from([
    #                  ("PKC", "Mek"), ("PKC", "PKA"), ("PKC", "Akt"),
    #                  ("PKA", "Akt"), ("PKA", 'Mek'),
    #                  ("Mek", "Erk"),
    #                  ("Akt", "Erk")
    #                  ])
    
    # Sachs
    G.add_edges_from([                     
                      ("PKC", "Raf"), ("PKC", "PKA"),
                      ("PKA", "Raf"), ("PKA", "Mek"), ("PKA", "Erk"),
                      ("Raf", "Mek"),
                      ("Mek", "Erk")
                      ])

    return G

### START EXPERIMENT

#  generate graph
G = graph_synthetic()
nx.draw_circular(G, with_labels=True)
plt.show()

# info about features
constraints_features = {"immutable": ['Raf'], "higher": ['PKA'], "lower": ['Mek']} # 'Akt'
categ_features = []

#  build dataset
df = pd.read_csv('data/sachs.csv', '\t')
df = df.dropna()
df = df[list(G.nodes)]

Y = 1*(df["Erk"] > df["Erk"].median())
X = df.drop(["Erk"], axis=1)

###  Create structural equations (NNs saved in models folder), store residuals (saved in data folder)
struct_eq, nn_causal = create_structural_eqs(X, Y, G, n_nodes_se=20,
                                             n_nodes_M=20, activation_se='tanh')

###  Create couterfactuals (saved in data folder)
create_counterfactuals(X, Y, G, struct_eq, nn_causal, constraints_features,
                       numCF=20, output_filename=cf_file_name, bool_distribution_train=True)

###  Calculate metrics - results will be printed
calculate_metrics(X, Y, G, categ_features, constraints_features, cf_file_name=cf_file_name)

print("----------- metrics on intersection -------------")
calculate_metrics(X, Y, G, categ_features, constraints_features,
                  intersection_only=True, cf_file_name=cf_file_name)