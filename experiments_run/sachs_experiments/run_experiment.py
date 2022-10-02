# Copyright 2021 Intesa SanPaolo S.p.A. and Fujitsu Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import os
from core.build_struct_eq import *
from core.counter_causal_generator import *
from core.metrics import *

"""
Script to run the experiments in the Sachs dataset (https://www.bristol.ac.uk/Depts/Economics/Growth/sachs.htm)

Steps:

1. The dataset has been downloaded in the data folder
2. Create graph - this will be plotted 
3. Add info. about the features (constrains and categorical features) in two main list. Currently we use:
    constraints_features = {"immutable": ['Raf'], "higher": ['PKA'], "lower": ['Mek']} # 'Akt'
    categ_features = []
4. CEILS workflow:

    - create model to be explained
    - create structural equations
    - calculate residuals
    - create model in the latent space
    - generate counterfactual explanations using 2 methods: baseline and CEILS
    - evaluate results using a set of metrics: taking into account all the explanations or only the common explanations obtained by both methods.

"""



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


if __name__ == "main":

    ### START EXPERIMENT

    # create folder to save models
    os.makedirs('models', exist_ok=True)
    # choose the conterfactuals filename, if not specified (default) is "counterfactuals"
    cf_file_name = "counterfactuals"

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