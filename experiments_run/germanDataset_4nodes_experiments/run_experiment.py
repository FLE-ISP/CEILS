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

'''
Script to run the experiments in the German credit dataset (https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

Steps:
1. The dataset has been downloaded in the data folder
2. Create graph - this will be plotted 
3. Add info. about the features (constrains and categorical features) in two main list. Currently we use:
    constraints_features = {"immutable": ["gender"], "higher": ["age"]}
    categ_features = ["gender"]
4. CEILS workflow:
    - create model to be explained
    - create structural equations
    - calculate residuals
    - create model in the latent space
    - generate counterfactual explanations using 2 methods: baseline and CEILS
    - evaluate results using a set of metrics: taking into account all the explanations or only the common explanations obtained by both methods.

'''


# create folder to save models
os.makedirs('models', exist_ok=True)

#  Create graph using only 4 features like: https://arxiv.org/pdf/2002.06278.pdf
def graph_4features():
    '''
    returns a graph for german credit dataset with 4 nodes
    '''

    G = nx.DiGraph(directed=True)
    G.add_node("gender")
    G.add_node("age")
    G.add_node("creditamount")
    G.add_node("classification")
    G.add_edges_from([('gender', "creditamount"), ("age", "creditamount"), ("creditamount", "duration")])
    G.add_edges_from([('gender', "classification"), ("age", "classification"), ("creditamount", "classification"), ("duration", "classification")])
    return G


def load_germandataset(nodes):
    '''
    read dataset and preprocessing for german credit dataset
    return data only for the nodes
    '''
    
    df = pd.read_csv("data/german_data_credit_dataset.csv")
    #create quickaccess list with categorical variables labels
    catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
               'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job',
               'telephone', 'foreignworker']
    #create quickaccess list with numerical variables labels
    numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age',
               'existingcredits', 'peopleliable', 'classification']

    # Binarize the target 0 = 'bad' credit; 1 = 'good' credit
    df.classification.replace([1,2], [1,0], inplace=True)


    #  dic categories Index(['A11', 'A12', 'A13', 'A14'], dtype='object')
    dict_categorical = {}
    for c in catvars:
        dict_categorical[c] = list(df[c].astype("category").cat.categories)
        df[c] = df[c].astype("category").cat.codes

    #  create gender variable 1= female 0 = male

    df.loc[df["statussex"] == 0, "gender"] = 0
    df.loc[df["statussex"] == 1, "gender"] = 1
    df.loc[df["statussex"] == 2, "gender"] = 0
    df.loc[df["statussex"] == 3, "gender"] = 0
    df.loc[df["statussex"] == 4, "gender"] = 1

    #  all features as float
    df = df.astype("float64")
    df["classification"] = df["classification"].astype("int32")
    # save codes
    with open('dict_german.txt', 'w') as f:
        f.write(str(dict_categorical))

    return df[nodes]

### START EXPERIMENT

#  generate graph with 4 features
G = graph_4features()
nodes = list(G.nodes)
nx.draw_circular(G, with_labels=True)
plt.show()

# info about features
constraints_features = {"immutable": ["gender"], "higher": ["age"]}
categ_features = ["gender"]

#  return dataset with float values and only features included in the graph
df = load_germandataset(nodes)

Y = df["classification"]
X = df.drop(["classification"], axis=1)

###  Create structural equations (NNs saved in models folder), store residuals (saved in data folder)
struct_eq, nn_causal = create_structural_eqs(X, Y, G, n_nodes_se=20, n_nodes_M=40)

###  Create couterfactuals (saved in data folder)
create_counterfactuals(X, Y, G, struct_eq, nn_causal, constraints_features, numCF=20)

###  Calculate metrics - results will be printed
calculate_metrics(X, Y, G, categ_features, constraints_features)
print("----------- metrics on intersection -------------")
calculate_metrics(X, Y, G, categ_features, constraints_features, intersection_only=True)
