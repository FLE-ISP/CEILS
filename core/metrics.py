# Original work copyright 2021 Intesa SanPaolo and Fujitsu Laboratories of Europe
# Riccardo Crupi, Alessandro Castelnovo, Beatriz San Miguel Gonzalez, Daniele Regoli
# This work is based on https://arxiv.org/pdf/2106.07754.pdf

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
try:
    from scipy.stats import median_absolute_deviation as ssMAD
except:
    from scipy.stats import median_abs_deviation as ssMAD
from sklearn.preprocessing import StandardScaler
from core.utils import LabelConverter

import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
#pd.set_option('display.width', 1000)


def calculate_metrics(X, Y, G, categ_features, constraints_features, intersection_only=False, cf_file_name="counterfactuals"):
    """
    This method calculate metrics to evaluate performance of the counterfactuals. Results will be printed.

    :param X: input features as DataFrame
    :param Y: target as Series
    :param G: causal graph of the data - networkx.classes.digraph.DiGraph
    :param categ_features: list of categorical features
    :param constraints_features: dictionary of CF constrains (immutable, higher, lower) respect on factual features
    :param intersection_only: if True computes the metrics only on valid CF between the two methods
    :param cf_file_name: name of the file in which CF are stored
    :return: None
    """

    def calculate_MAD_threshold(X_train):
        """
        Compute a threshold for continuous features over which they are considered as "changed", otherwise they are
        considered fixed. This information is used to compute sparsity in CF.

        :param X_train: input features as DataFrame
        :return: MADs values, threshold values
        """
        X_train = X_train.drop(categ_features, axis=1)
        MADs = ssMAD(X_train)
        # for each feature the min among MAD and limit = q10( |x - median(x)|) such that x!=median(x)
        medians = X_train.median().values
        quantiles = []
        c_index = 0
        for i in medians:
            xx = X_train.iloc[:, c_index]
            # non-identical to the median
            xx = xx[xx != i]
            quant =abs(xx - i).quantile(.1)
            quantiles.append(quant)
            c_index = c_index+1

        limit = np.array(quantiles)

        threshold = np.minimum(MADs, limit)

        return MADs, threshold

    # formula in paper
    def compute_proximity_cont(cf, orig_x):
        """
        Compute the distance in the original space between CF and its factual, scaled by MADs.

        :param cf: list of CFs (matrix N CFs X F features)
        :param orig_x: list of factuals (matrix N instances X F features)
        :return: proximity measure
        """
        dif = abs(cf - orig_x)
        tmp = dif/MADs
        # divide by number of features
        proximity = tmp.sum(axis=1, skipna=False) / orig_x.shape[1]

        return proximity

    def compute_proximity_cat(cf, orig_x):
        """
        Compute the distance in the original space between CF and its factual for categorical feature.

        :param cf: list of CFs (matrix N CFs X F features)
        :param orig_x: list of factuals (matrix N instances X F features)
        :return: proximity cat measure
        """
        prox = np.round(cf[categ_features], 0) != np.round(orig_x[categ_features], 0)
        prox = prox.astype("int")

        if "COD_RATING_PEF" in list(orig_x.columns):
            prox2 = prox.copy()
            prox2.loc[list(cf[cf.iloc[:, 0].isnull()].index)] = np.nan
            print("proximity COD_RATING_PEF: ", np.round(prox2["COD_RATING_PEF"].mean(), 2), "+/-",
                  np.round(prox2["COD_RATING_PEF"].std(), 2))

        prox = prox.sum(axis=1) / len(categ_features)
        prox.loc[list(cf[cf.iloc[:, 0].isnull()].index)] = np.nan

        return prox

    def compute_sparsity(cf, orig_x):
        """
        Compute the sparsity of the difference vector between counterfactual and factual (x_cf - x_0)

        :param cf: list of CFs (matrix N CFs X F features)
        :param orig_x: list of factuals (matrix N instances X F features)
        :return: sparsity measure
        """
        dif = abs(cf - orig_x)
        dif[categ_features] = np.round(dif[categ_features])
        sparsity = (dif  > threshold_changes_action).sum(axis=1)
        sparsity.loc[list(cf[cf.iloc[:, 0].isnull()].index)] = np.nan
        return sparsity

    def causal_plausibility(cf):
        """
        This distance computes the difference in L1 norm difference between cf and its component computed by
         the structural equations
         notice that data must be standardized

        :param cf: dataframe
        :return: median dist, MAD distance: (flaat, float)

        Examples
        --------
        causal_plausibility(pd.DataFrame({'a':[1,2,3,4])})
        """

        # Load columns dataset
        columns_dataset = pd.read_csv('data/res_train.csv', nrows=1).columns
        columns_dataset = [i.split('r_')[-1] for i in columns_dataset]

        # G = graph_4features()

        dist = np.array([.0]*cf.shape[0])
        # Loop for computing |X_v - f (pa(X_v))|
        for node in columns_dataset:
            node_idx = np.where(np.array(columns_dataset) == node)[0][0]
            if len(list(G.predecessors(node))) != 0:
                # parents_idx = [np.where(np.array(columns_dataset) == n)[0][0] for n in G.predecessors(node)]
                if str(node) in categ_features:
                    Mv = tf.keras.models.load_model('models/' + str(node) + '.h5', custom_objects={'LabelConverter': LabelConverter})
                    tf.compat.v1.keras.backend.get_session().run(
                        tf.compat.v1.tables_initializer(name='init_all_tables'))
                else:
                    Mv = tf.keras.models.load_model('models/' + str(node) + '.h5')
                cf_keras = [cf[p].values for p in G.predecessors(node)]
                dist += abs(cf.iloc[:, node_idx]-Mv.predict(cf_keras).reshape(cf.shape[0],))

        return np.nanmedian(dist), ssMAD(dist, nan_policy='omit')

    def compute_residuals(df):
        """
        computes residuals corresponding to observations given as rows in df
        notice that data must be standardized

        :param cf: pandas dataframe observations x features
        :return: pandas dataframe with the same structure of df but with residuals

        """

        # Load graph
        dd = dict()
        for n in df.columns:
            if G.in_degree[n] != 0:
                if str(n) in categ_features:
                    Mv = tf.keras.models.load_model('models/' + str(n) + '.h5', custom_objects={'LabelConverter': LabelConverter})
                    tf.compat.v1.keras.backend.get_session().run(
                        tf.compat.v1.tables_initializer(name='init_all_tables'))
                else:
                    Mv = tf.keras.models.load_model('models/' + str(n) + '.h5')
                input = [df[p].values for p in G.predecessors(n)]
                res = df[n].values - Mv.predict(input).reshape(df.shape[0],)
            else:
                res = df[n].values
            dd[n] = res
        res = pd.DataFrame(dd)
        return res

    def compute_actions_obs(df_factual, df_cf):
        """
        given original observations and counterfactuals, computes the actions as the difference in residuals.
        notice that data must be standardized

        :param df_factual: pandas dataframe containing the original observations (observations x features)
        :param df_cf: pandas dataframe containing the counterfactual (observations x features)

        :return: median, MAD of the action's norm, and actions
        """

        res_orig = compute_residuals(df_factual)
        res_cf = compute_residuals(df_cf)
        # pandas dataframe with the same structure of the originals but with action

        return res_cf - res_orig

    def compute_actions(df_factual, df_cf):
        """
        given original observations and counterfactuals, computes the action norm and MAD.

        :param df_factual: pandas dataframe containing the original observations (observations x features)
        :param df_cf: pandas dataframe containing the counterfactual (observations x features)
        :return:
        """

        df_action = compute_actions_obs(df_factual, df_cf)
        # Norm L1 of the action
        x = df_action.dropna(axis=0)
        norm_df_action = np.sum(abs(x), axis=1)

        return np.nanmedian(norm_df_action), ssMAD(norm_df_action, nan_policy='omit'), df_action


    def compute_actions_diff(df_cf1, df_cf2):
        """
        compare 2 dataframes containing "actions" of counterfactuals observations,
        consider only valid counterfactuals of both df!
        it does so by 2 methods:
            1. MEDIAN(||action1 - action2||), to assess the "distance" among the two set of actions
            2. MEDIAN(||action1|| - ||action2||), to assess whether one set of action is dominating the other
        :param df_cf1, df_cf2: pandas dataframes containing the counterfactual (observations x features) to be compared

        :return: 4 floats, the medians and MADs of the action's norm of both methods
        """

        df = df_cf1 - df_cf2
        # keep only onservations where they both have values
        idx = pd.notnull(df).all(1)
        # Norm L1 of the difference
        x = df.dropna(axis=0)
        norm_ext = np.sum(abs(x), axis=1)
        norm1 = np.sum(abs(df_cf1[idx]), axis=1)
        norm2 = np.sum(abs(df_cf2[idx]), axis=1)
        norm_int = norm1 - norm2

        return np.nanmedian(norm_ext), ssMAD(norm_ext, nan_policy='omit'), np.nanmedian(norm_int), ssMAD(norm_int, nan_policy='omit')

    def feasibility_constraint(df_factual_sca, df_cf_sca):
        """
        given original observations and counterfactuals, computes the % of the valid CF in the latent space.

        :param df_factual: pandas dataframe containing the original observations (observations x features)
        :param df_cf: pandas dataframe containing the counterfactual (observations x features)

        :return: median, MAD of the action's norm, and actions
        """

        res_orig = compute_residuals(df_factual_sca)
        res_cf = compute_residuals(df_cf_sca)
        # pandas dataframe with the same structure of the originals but with action
        df_action = res_cf - res_orig
        filter_na = df_action.isna().any(axis=1)

        res_orig = res_orig[~filter_na]
        res_cf = res_cf[~filter_na]

        # Load res dataset
        U_train = pd.read_csv('data/res_train.csv')
        a, b = (U_train.min(axis=0), U_train.max(axis=0))

        # init val
        val = np.array([True]*len(res_cf))

        # check no inmutable features
        if "immutable" in constraints_features.keys():
            for feature in constraints_features["immutable"]:
                val = val & (abs(res_cf.loc[:, feature].values - res_orig.loc[:, feature]).values <= 1e-4)

        # check if only higher
        if "higher" in constraints_features.keys():
            for feature in constraints_features["higher"]:
                val = val & (res_cf.loc[:, feature].values >= res_orig.loc[:, feature].values - 1e-4)

        # check if only lower
        if "lower" in constraints_features.keys():
            for feature in constraints_features["lower"]:
                val = val & (res_cf.loc[:, feature].values <= res_orig.loc[:, feature].values + 1e-4)

        # # check min and max for features with residuals
        # for feature in list(U_train.columns):
        #     if feature.startswith("r_"):
        #         feature_name = feature[2:]
        #         val = val & (res_cf.loc[:, feature_name] >= a[list(res_cf.columns).index(feature_name)] - 1e-4)
        #         val = val & (res_cf.loc[:, feature_name] <= b[list(res_cf.columns).index(feature_name)] + 1e-4)

        return val.mean()

    # # # # # # Start to calculate metrics
    # # #

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # load counterfactuals
    df_counterfactuals = pd.read_csv("data/" + cf_file_name + ".csv")
    df_counterfactuals = df_counterfactuals.set_index("id_customer")

    names = df_counterfactuals["index"].unique()

    if intersection_only:
        df_counterfactuals = df_counterfactuals[
            ~(df_counterfactuals["Counterfactual_original"].isna()) & ~(df_counterfactuals["Counterfactual_causal"].isna())]

    # Define DataFrame of factuals, CEILS counterfactuals, Alibi Prototype counterfactuals
    df_factual = df_counterfactuals.loc[:, "Factual_original"]
    df_factual = pd.DataFrame(columns=names, data=df_factual.values.reshape(int(df_factual.shape[0]/len(names)), len(names)))
    df_cf_causal = df_counterfactuals.loc[:, "Counterfactual_causal"]
    df_cf_causal = pd.DataFrame(columns=names, data=df_cf_causal.values.reshape(int(df_cf_causal.shape[0]/len(names)), len(names)))
    df_cf_orig = df_counterfactuals.loc[:, "Counterfactual_original"]
    df_cf_orig = pd.DataFrame(columns=names, data=df_cf_orig.values.reshape(int(df_cf_orig.shape[0]/len(names)), len(names)))

    # scale values to compute causal metrics
    scaler = StandardScaler()
    scaler.fit(X_train)

    df_factual_scale = pd.DataFrame(columns=df_factual.columns, data = scaler.transform(df_factual))
    df_cf_orig_scale = pd.DataFrame(columns=df_cf_orig.columns, data = scaler.transform(df_cf_orig))
    df_cf_causal_scale = pd.DataFrame(columns=df_cf_causal.columns, data = scaler.transform(df_cf_causal))

    print("number of valid standard cfs: ", df_cf_orig.dropna(axis=0).shape[0])
    print("number valid CCE: ", df_cf_causal.dropna(axis=0).shape[0])

    # # # COST in the sense of Karimi
    # Perversion (action ex post)
    med, mad, df_action_orig = compute_actions(df_factual_scale , df_cf_orig_scale)
    print('Cost (perversion) - median norm for action original: ', np.round(med, 2), " +- ", np.round(mad, 2))
    # Norm of the action
    med, mad, df_action_caus = compute_actions(df_factual_scale , df_cf_causal_scale)
    print('Cost - median norm for action causal: ', np.round(med, 2), " +- ", np.round(mad, 2))
    # Norm of the diff (action not in the causal sense)
    # Note: similar to proximity of causal!
    # Avoid NAs because turns into 0
    index_not_na = ~df_factual_scale.iloc[:, 0].isna()
    index_not_na = (~df_cf_causal_scale.iloc[:, 0].isna()) & index_not_na
    diff_scale = np.sum(abs(df_factual_scale[index_not_na] - df_cf_causal_scale[index_not_na]), axis=1)
    print('Cost - median norm for diff causal: ', np.round(np.nanmedian(diff_scale), 2),
          " +- ", np.round(ssMAD(diff_scale, nan_policy='omit'), 2))
    # Norm of original diff (similar to proximity original)
    index_not_na = ~df_factual_scale.iloc[:, 0].isna()
    index_not_na = (~df_cf_orig_scale.iloc[:, 0].isna()) & index_not_na
    diff_scale = np.sum(abs(df_factual_scale[index_not_na] - df_cf_orig_scale[index_not_na]), axis=1)
    print('Cost - median norm for diff original: ', np.round(np.nanmedian(diff_scale), 2),
          " +- ", np.round(ssMAD(diff_scale, nan_policy='omit'), 2))

    if intersection_only:
        df_action_caus_obs = compute_actions_obs(df_factual_scale, df_cf_causal_scale)
        df_diff_orig_obs = df_cf_orig_scale - df_factual_scale
        med_ext, mad_ext, med_int, mad_int = compute_actions_diff(df_diff_orig_obs, df_action_caus_obs)
        print('Diff - median of ||diff orig - action causal||: ', np.round(med_ext, 2), " +- ", np.round(mad_ext, 2))
        print('Diff - median of ||diff orig|| - ||action causal||: ', np.round(med_int, 2), " +- ", np.round(mad_int, 2))

    # # # FEASIBILITY constraint validity
    val_cau = feasibility_constraint(df_factual_scale, df_cf_orig_scale)
    print('Feasibility constraint validity for alibi CF: ', val_cau)
    val_cau = feasibility_constraint(df_factual_scale, df_cf_causal_scale)
    print('Feasibility constraint validity for causal CF: ', val_cau)

    # # # CAUSAL PLAUSIBILITY
    # The median norm of the residuals
    med, mad = causal_plausibility(df_cf_orig_scale)
    print('causal plausibility - median distance from SE for action original: ', np.round(med, 2), " +- ", np.round(mad, 2))
    med, mad = causal_plausibility(df_cf_causal_scale)
    print('causal plausibility - median distance from SE for action causal: ', np.round(med, 2), " +- ", np.round(mad, 2))

    # constant values of training dataset
    MADs, threshold_changes = calculate_MAD_threshold(X_train)
    # add threshold =0 for categorical features to computer sparsity of action
    threshold_changes_action = threshold_changes
    for i in categ_features:
        position = list(X_train.columns).index(i)
        if position <= (X_train.shape[1] - len(categ_features)):
            threshold_changes_action = np.insert(threshold_changes_action, position, 0)
        else:
            threshold_changes_action = np.append(threshold_changes_action, 0)

    # # # SPARSITY
    # sparsity  in cf original
    sparsity_orig = compute_sparsity(df_cf_orig, df_factual)
    print("sparsity in cf original", np.round(sparsity_orig.mean(), 2), "+/-", 
          np.round(sparsity_orig.std(), 2))

    # sparsity in cf causal
    sparsity_causal = compute_sparsity(df_cf_causal, df_factual)
    print("sparsity in cf causal", np.round(sparsity_causal.mean(), 2), "+/-",
          np.round(sparsity_causal.std(), 2))

    # sparsity in action
    # action in causal - action calculated with counterfactuals
    df_actions_caus = df_counterfactuals.loc[:, "action_xstd"]
    df_actions_caus = pd.DataFrame(columns=df_counterfactuals["index"].unique(),
                                   data=df_actions_caus.values.reshape(int(df_actions_caus.shape[0] / len(names)),
                                                                       len(names)))
    # Since df_actions_caus is already a difference, to reuse the function compute_sparsity it is defined a 0 dataframe
    # woth the same dimension of df_actions_caus
    df_zeros = df_actions_caus*0
    sparsity_action_caus = compute_sparsity(df_actions_caus, df_zeros)
    print("sparsity in action of causal: ", np.round(sparsity_action_caus.mean(), 2), " +/- ",
          np.round(sparsity_action_caus.std(), 2))

    if not intersection_only:
        # # # VALIDITY of counterfactuals (original space)
        # VALIDITY: found or not found cf - check na in first column: if na then not found
        validity_orig = np.where(df_cf_orig.iloc[:, 0].isnull(), 0, 1).mean()
        print("validity in cf original ", validity_orig)
        validity_caus = np.where(df_cf_causal.iloc[:, 0].isnull(), 0, 1).mean()
        print("validity in cf causal ", validity_caus)

    # # # PROXIMITY
    # PROXIMITY for continuous features remove gender
    prox_cont_orig = compute_proximity_cont(df_cf_orig.drop(categ_features, axis=1), df_factual.drop(categ_features, axis=1))
    print("proximity continuous features in original ", np.round(prox_cont_orig.mean(), 2), "+/-", np.round(prox_cont_orig.std(), 2))

    prox_cont_causal = compute_proximity_cont(df_cf_causal.drop(categ_features, axis=1), df_factual.drop(categ_features, axis=1))
    print("proximity continuous features in causal ", np.round(prox_cont_causal.mean(), 2), "+/-", np.round(prox_cont_causal.std(), 2))

    if len(categ_features) != 0:
        # PROXIMITY for cat feature
        prox_cat_orig  = compute_proximity_cat(df_cf_orig, df_factual)
        print("proximity categorical features in original ", np.round(prox_cat_orig.mean(), 2), "+/-", np.round(prox_cat_orig.std(), 2))

        prox_cat_causal = compute_proximity_cat(df_cf_causal, df_factual)
        print("proximity categorical features in causal ", np.round(prox_cat_causal.mean(), 2), "+/-", np.round(prox_cat_causal.std(), 2))