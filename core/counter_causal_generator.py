"""
Module with function :py:func:`.create_counterfactuals` generating two sets of counterfactual explanations will be generated based on:

- CEILS approach: uses the model in the latent space and a general counterfactual generator ([Alibi](https://github.com/SeldonIO/alibi) in our current implementation)
- Baseline approach: uses the original model and the library [Alibi](https://github.com/SeldonIO/alibi) 
"""

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

import pandas as pd
import tensorflow as tf
import os
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
# from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from alibi import __version__ as alibi_version
if alibi_version>='0.6.0':
    from alibi.explainers import CounterfactualProto
else:
    from alibi.explainers import CounterFactualProto as CounterfactualProto

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1000)


def create_counterfactuals(X, Y, G, struct_eq, nn_causal, constraints_features, numCF,
                           output_filename="counterfactuals", bool_distribution_train=False):
    """Generates counterfactual explanations using a baseline model and CEILS method.

    The explanations will be generated for instances included in X_test.

    Parameters
    ----------
    X : pandas DataFrame
        input features of the dataset
    Y : pandas Series
        target to be predicted
    G : networkx.classes.digraph.DiGraph
        causal graph of the data
    struct_eq: keras.engine.functional.Functional - keras Model
        structural equations (F:U->X)
    nn_causal: keras.engine.functional.Functional - keras Model
        model in the latent space. Final model that uses structural equations and original prediction model: M^:U->Y. M^(u)=M(F(u))
    constraints_features: dict
        dictionary to impose the constraints of the features (i.e. immutable, higher, etc.)
    numCF: int
        number of counterfactual explanations to be generated
    bool_distribution_train: bool
        flag to indicate that a counterfactual explanation is invalid if it's outside the distribution of the train set.

    Returns
    ----------
        None
        
    In the folder data, the counterfactual explanations of both methods (baseline and CEILS) are stored.

    """

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # take all nodes except target >>> classification
    nodes = [n for n in list(G.nodes) if n != Y.name]

    # Standardise data
    scaler = StandardScaler()
    X_train_std = X_train[nodes].std()
    scaler.fit(X_train)
    train_std = scaler.transform(X_train)
    test_std = scaler.transform(X_test)

    # Load models and residual dataset
    U_train = pd.read_csv('data/res_train.csv')
    columns_dataset = U_train.columns
    columns_dataset = [i.split('r_')[-1] for i in columns_dataset]
    mask_order_for_u = [columns_dataset.index(col) for col in nodes]
    U_train = U_train.values
    U_test = pd.read_csv('data/res_test.csv').values
    # Load standard prediction model
    M = tf.keras.models.load_model('models/nn_model.h5')

    ### Counterfactual Prototype

    # Original customer
    X = test_std[1].reshape((1,) + test_std[1].shape)
    shape = X.shape
    # residual customer
    U = U_test[1].reshape((1,) + U_test[1].shape)
    U_shape = U.shape

    # set random seed
    if tf.__version__[0] == '1':
        tf.set_random_seed(1)
    if tf.__version__[0] == '2':
        tf.random.set_seed(1)

    # get Keras session
    sess = tf.compat.v1.keras.backend.get_session()
    # initialise dataframe of caounterfactual examples
    df_examples = pd.DataFrame()
    # select numCF randomly or all
    if(len(X_test) > numCF):
        _, X_cf, _, y_cf = train_test_split(X_test, y_test, test_size=numCF, random_state=42)
        indexes_cf =list(X_cf.index)
        X_test2 = X_test.reset_index()
        indexes_cf2 = list(X_test2[X_test2["index"].isin(indexes_cf)].index)
    else:
        indexes_cf2 = range(0, min(numCF, len(X_test)))

    for num, i in enumerate(indexes_cf2):
        # Factual original
        X = test_std[i].reshape((1,) + test_std[1].shape)
        X_shape = X.shape
        
        # Factual causal constrain
        U = U_test[i].reshape((1,) + U_test[1].shape)
        U_shape = U.shape

        # initialize original explainer, fit and generate counterfactual
        # constraints
        a, b=(train_std.min(axis=0), train_std.max(axis=0))

        # constrain features initialisation
        if "immutable" not in constraints_features.keys():
            constraints_features["immutable"] = []
        if "higher" not in constraints_features.keys():
            constraints_features["higher"] = []
        if "lower" not in constraints_features.keys():
            constraints_features["lower"] = []

        # immutable features
        for featur in constraints_features["immutable"]:
            index_featur = columns_dataset.index(featur)
            a[index_featur] = X[0][index_featur]
            b[index_featur] = X[0][index_featur]

        # only higher features
        for featur in constraints_features["higher"]:
            index_featur = columns_dataset.index(featur)
            a[index_featur] = X[0][index_featur]

        # only lower features
        for featur in constraints_features["lower"]:
            index_featur = columns_dataset.index(featur)
            b[index_featur] = X[0][index_featur]


        cf_ori = CounterfactualProto(M, shape, use_kdtree=True,
                                     theta=100, # L_proto term
                                     kappa=.2, # attack loss term L_pred
                                     beta=.5, # L1 term
                                     gamma=.5, # L_ae term (autoencoder or kd-tree)
                                     max_iterations= 500,
                                     learning_rate_init=1e-3,
                                     feature_range=(a, b),
                                     c_init=1., c_steps=10, sess=sess)
        cf_ori.fit(train_std)
        # initialize explainer, fit and generate counterfactual
        # constrains
        a, b=(U_train.min(axis=0), U_train.max(axis=0))

        for featur in constraints_features["immutable"]:
            index_featur = columns_dataset.index(featur)
            a[index_featur] = U[0][index_featur]
            b[index_featur] = U[0][index_featur]

        # only higher features
        for featur in constraints_features["higher"]:
            index_featur = columns_dataset.index(featur)
            a[index_featur] = U[0][index_featur]

        # only lower features
        for featur in constraints_features["lower"]:
            index_featur = columns_dataset.index(featur)
            b[index_featur] = U[0][index_featur]


        cf_causal = CounterfactualProto(nn_causal, U_shape, use_kdtree=True,
                                        theta=100, # L_proto term
                                        kappa=.2, # attack loss term L_pred
                                        beta=.5, # L1 term
                                        gamma=.5, # L_ae term (autoencoder or kd-tree)
                                        max_iterations= 500,
                                        learning_rate_init=1e-3,
                                        feature_range=(a, b),
                                        c_init=1., c_steps=10, sess=sess)
        cf_causal.fit(U_train)


        # CF original
        explanation_ori = cf_ori.explain(X,
                                         k=1 # number of neighbours for computing a prototype
                                         )
        # CF causal constrain
        explanation_causal = cf_causal.explain(U,
                                               k=1 # number of neighbours for computing a prototype
                                               )
        
        # if the CF are founded in both models
        if explanation_causal.cf != None and explanation_ori.cf != None:
            example = pd.DataFrame(None, index=nodes)
            # factual x0 = F(u0)
            example['Factual_causal'] = scaler.inverse_transform(struct_eq.predict(U)[0])
            # factual x0, it's the same as above but computed for double check (verify SE are correct)
            example['Factual_original'] = scaler.inverse_transform(X[0])
            
            # causal counterfactual u0 -> u0_cf, x0_cf = F(u0_cf)
            example['Counterfactual_causal'] = scaler.inverse_transform(struct_eq.predict(explanation_causal.cf['X'])[0])
            # check if the CF are outside the original distribution
            if ((example['Counterfactual_causal'] < X_train.min(axis=0) - 1e-4) |
                (example['Counterfactual_causal'] > X_train.max(axis=0) + 1e-4)).any() and bool_distribution_train:
                print('cf causal not valid')
                example['Counterfactual_causal'] = None
            # original counterfactual x0 -> x0_cf
            example['Counterfactual_original'] = scaler.inverse_transform(explanation_ori.cf['X'][0])
            if ((example['Counterfactual_original'] < X_train.min(axis=0) - 1e-4) |
                (example['Counterfactual_original'] > X_train.max(axis=0) + 1e-4)).any() and bool_distribution_train:
                print('cf original not valid')
                example['Counterfactual_original'] = None

            # difference between CF in X (feature) space
            example['diff_causal'] = example.loc[:, 'Counterfactual_causal'] - example.loc[:, 'Factual_causal']
            example['diff_original'] = example.loc[:, 'Counterfactual_original'] - example.loc[:, 'Factual_original']
            # difference between CF in U (residual) space. The multiplication
            # with standard deviation let the difference be coherent in the X space
            example['action'] = (explanation_causal.cf['X'] - U)[0][mask_order_for_u]
            example['action_xstd'] = (example['action'] * X_train_std).values

            print('Factual causal y: ', int(nn_causal.predict(U)[0][1] > 0.5),
                  'Counterfactual causal y: ', int(nn_causal.predict(explanation_causal.cf['X'])[0][1] > 0.5),
                  'Factual original y: ', int(M.predict(X)[0][1] > 0.5),
                  'Counterfactual original y: ', int(M.predict(explanation_ori.cf['X'])[0][1] > 0.5)
                  )
            print(example)
        # if one of the CF are not founded
        else:
            example = pd.DataFrame(None, index=nodes)
            # factual x0 = F(u0)
            example['Factual_causal'] = scaler.inverse_transform(struct_eq.predict(U)[0])
            # factual x0, it's the same as above but computed for double check (verify SE are correct)
            example['Factual_original'] = scaler.inverse_transform(X[0])
            
            if explanation_causal.cf is not None:
                # causal counterfactual u0 -> u0_cf, x0_cf = F(u0_cf)
                example['Counterfactual_causal'] = scaler.inverse_transform(struct_eq.predict(explanation_causal.cf['X'])[0])
                # difference between CF in X (feature) space
                example['diff_causal'] = example.loc[:, 'Counterfactual_causal'] - example.loc[:, 'Factual_causal']
                # difference between CF in U (residual) space. The multiplication
                # with standard deviation let the difference be coherent in the X space
                example['action'] = (explanation_causal.cf['X'] - U)[0][mask_order_for_u]
                example['action_xstd'] = (example['action'] * X_train_std).values
                # check if the CF are outside the original distribution
                if ((example['Counterfactual_causal'] < X_train.min(axis=0) - 1e-4) |
                    (example['Counterfactual_causal'] > X_train.max(axis=0) + 1e-4)).any() and bool_distribution_train:
                    print('cf causal not valid')
                example['Counterfactual_causal'] = None
            else:
                print("No counterfactual causal found!")
                example['Counterfactual_causal'] = None

            if explanation_ori.cf is not None:
                # original counterfactual x0 -> x0_cf
                example['Counterfactual_original'] = scaler.inverse_transform(explanation_ori.cf['X'][0])
                # difference between CF in X (feature) space
                example['diff_original'] = example.loc[:, 'Counterfactual_original'] - example.loc[:, 'Factual_original']
                # check if the CF are outside the original distribution
                if ((example['Counterfactual_original'] < X_train.min(axis=0) - 1e-4) |
                    (example['Counterfactual_original'] > X_train.max(axis=0) + 1e-4)).any() and bool_distribution_train:
                    print('cf original not valid')
                example['Counterfactual_original'] = None
            else:
                print("No counterfactual original found!")
                example['Counterfactual_original'] = None
                
            print(example)

        example['id_customer'] = i
        df_examples = df_examples.append(example.reset_index(), ignore_index=False)

        if num % 25 == 0:
            print(f'Progress: {num}/{len(indexes_cf2)}')

    df_examples = df_examples.set_index("id_customer")
    df_examples.to_csv("data/" + output_filename + ".csv")
    print("end generation of counterfactuals")
