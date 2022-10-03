# Copyright 2021 Intesa SanPaolo S.p.A and Fujitsu Limited
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


import os
import networkx as nx
import matplotlib.pyplot as plt
from core.build_struct_eq import *
from core.counter_causal_generator import *
from core.metrics import *
import pandas as pd

"""
Script to run the experiments in the speedData dataset 
link dataset download: https://datahub.io/machine-learning/speed-dating#readme
info columns dataset: https://www.openml.org/d/40536

Steps:

1. The dataset has been downloaded in the data folder
2. Create graph - this will be plotted 
3. Add info. about the features (constrains and categorical features) in two main list. Currently we use:
    constraints_features = {"immutable": ["attractive_o"]}
    categ_features = []
4. CEILS workflow:

    - create model to be explained
    - create structural equations
    - calculate residuals
    - create model in the latent space
    - generate counterfactual explanations using 2 methods: baseline and CEILS
    - evaluate results using a set of metrics: taking into account all the explanations or only the common explanations obtained by both methods.

"""


#  Create graph using only 2 features
def graph_2features():
    '''
    returns a graph for speed date dataset with 3 nodes
    '''

    G = nx.DiGraph(directed=True)
    G.add_node("funny_o")
    G.add_node("attractive_o")
    G.add_node("decision_o")
    G.add_edges_from([('funny_o', "attractive_o"), ("funny_o", "decision_o")])
    G.add_edges_from([("attractive_o", "decision_o")])
    return G


def load_dataset(nodes):
    '''
    read dataset and preprocessing for speed date dataset
    return data only for the nodes
    '''
    
    df = pd.read_csv("data/speed-dating_csv.csv")
    #create quickaccess list with numerical variables labels
    numvars = ['attractive_o', 'funny_o']
    
    #cols = ['attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o', 'ambitous_o',
    # 'shared_interests_o', 
    # 'sports', 'exercise', 'tvsports', 'dining', 'museums', 'art',
    #  'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies',
    # 'concerts', 'music', 'shopping', 'yoga',
    # 'decision_o']
    
    # print(df[cols].corr())
    
    df = df[numvars + ['decision_o']]

    #  all features as float
    df = df.astype("float64")
    df["decision_o"] = df["decision_o"].astype("int32")
    
    # drop na
    df = df.dropna(axis=0)

    return df[nodes]


if __name__ == "__main__":
    ### START EXPERIMENT

    # create folder to save models
    os.makedirs('models', exist_ok=True)


    #  generate graph with 4 features
    G = graph_2features()
    nodes = list(G.nodes)
    nx.draw_circular(G, with_labels=True)
    plt.show()

    # info about features
    constraints_features = {"immutable": ["attractive_o"]}

    #  return dataset with float values and only features included in the graph
    df = load_dataset(nodes)

    Y = df["decision_o"]
    X = df.drop(["decision_o"], axis=1)
    categ_features = []

    ###  Create structural equations (NNs saved in models folder), store residuals (saved in data folder)
    struct_eq, nn_causal = create_structural_eqs(X, Y, G, n_nodes_se=5, n_nodes_M=10)

    ###  Create couterfactuals (saved in data folder)
    create_counterfactuals(X, Y, G, struct_eq, nn_causal, constraints_features, numCF=20)

    ###  Calculate metrics - results will be printed
    calculate_metrics(X, Y, G, categ_features, constraints_features)
    print("----------- metrics on intersection -------------")
    calculate_metrics(X, Y, G, categ_features, constraints_features, intersection_only=True)


