# Original work copyright 2021 Intesa Sanpaolo S.p.A. and Fujitsu Limited
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

# TODO this module to generate the structural is intended to manage in a
# better way the categorical variables in the datase.
# Use build_struct_eq for the stable version.

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

tf.keras.backend.clear_session()
tf.compat.v1.disable_v2_behavior()
tf.random.set_seed(1)


class LabelConverter(tf.keras.layers.Layer):

    def __init__(self, data_dict, **kwargs):
        super(LabelConverter, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.data_dict = data_dict
        # Implement your StaticHashTable here
        keys = tf.constant([int(x) for x in list(data_dict.keys())],  dtype=tf.int64)
        values = tf.constant([float(data_dict[k]) for k in list(data_dict.keys())])
        table_init = tf.lookup.KeyValueTensorInitializer(keys, values)
        self.table = tf.lookup.StaticHashTable(table_init, -1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'data_dict': self.data_dict
        })
        return config

    def build(self, input_shape):
        self.built = True

    def call(self, tensor_input):
        out = tf.argmax(tensor_input, axis=1)
        # this block is doing the transformation on input dict_cat
        categories_tensor = self.table.lookup(out)
        return categories_tensor


class LimitLayer(tf.keras.layers.Layer):

    def __init__(self, data_dict, **kwargs):
        super(LimitLayer, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.data_dict = data_dict

        values = [float(self.data_dict[k]) for k in list(self.data_dict.keys())]
        values.sort()
        self.values = values
        keys = tf.constant(list(np.arange(len(values))), dtype=tf.int64)
        # Implement your StaticHashTable here
        table_init = tf.lookup.KeyValueTensorInitializer(keys, tf.constant(values))
        self.table = tf.lookup.StaticHashTable(table_init, -1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'data_dict': self.data_dict
        })
        return config

    def build(self, input_shape):
        self.built = True

    def call(self, tensor_input):
        shape = tf.shape(tensor_input)[0]
        # max in scaled value i.e. 1.7
        max_class = tf.constant([self.data_dict[max(list(self.data_dict.keys()))]])
        min_tf = tf.keras.layers.Lambda(lambda x: tf.minimum(x, max_class))(tensor_input)

        # min in scaled value i.e. -1.7
        min_class = tf.constant([self.data_dict[min(list(self.data_dict.keys()))]])
        min_tf = tf.keras.layers.Lambda(lambda x: tf.maximum(x, min_class))(min_tf)

        diffs = tf.keras.layers.Lambda(lambda x: tf.abs(tf.subtract(x, self.values)))(min_tf)
        min_index = tf.argmin(diffs, axis=1)


        out = self.table.lookup(min_index)
        out = tf.reshape(out, shape=[shape, 1])

        return min_tf




def create_structural_eqs_v2(X, Y, G, categ_features, n_nodes_se=40, n_nodes_M=100):
    """
    method to create structural equations

    Parameters:s
    X: input features as DataFrame
    Y: target as Series
    G: causal graph of the data - networkx.classes.digraph.DiGraph
    n_nodes_se: number of nodes in the neural network of SE (structural equation)
    n_nodes_M: number of nodes in the neural network of M (model that estimates y)

    Returns:
    struct_eq, nn_causal -- CANNOT BE saved as .h5 model

    save in files: residuals, nn of the structural equations, model (in models and data folder)

    """

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    X_train_cat = X_train.copy()
    X_test_cat = X_test.copy()
    # take all nodes except target >>> classification
    nodes = [n for n in list(G.nodes) if n != Y.name]

    # Standardise data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train.loc[:, :] = scaler.transform(X_train)
    X_test.loc[:, :] = scaler.transform(X_test)


    # get root nodes
    root_nodes = [n for n, d in G.in_degree() if d == 0]

    # define variables where residuals and residul inputs will be stored
    U_train = X_train[root_nodes].copy()
    U_test = X_test[root_nodes].copy()
    res_inputs = []


    #define tf inputs, one for each node
    node_inputs = {n: keras.Input(shape=(1,), name=n) for n in nodes}

    # define dic to store the final X = F(U) with U = (roots, residuals) for each node
    # fill the root nodes directly with input layers
    X_n = {r: node_inputs[r] for r in root_nodes}

    # auxiliary while-loop variables
    added_nodes = []
    root_nodes_tmp = root_nodes

    while set(root_nodes_tmp) != set(nodes):
        # loop until all nodes are either root or dealt with (root_nodes_tmp
        # contains root nodes and is updated with dealt with nodes)

        for n in nodes:
            parents = list(G.predecessors(n))
            # go on only when:
            # n has parents
            # parents are root_nodes or nodes already dealt with
            # n is not a root node and has not been dealt with yet
            if G.in_degree[n] != 0 and set(parents).issubset(set(root_nodes_tmp)) and not n in root_nodes_tmp:
                print("dealing with ", n, " with parents: ", parents)

                # build the model form parents to n
                if len(parents) == 1:
                    parent = parents[0]
                    inputs = node_inputs[parent]
                    conc = tf.identity(inputs)
                    X_train_p = X_train[parent].values
                    X_test_p = X_test[parent].values
                else:
                    inputs = [node_inputs[p] for p in parents]
                    conc = layers.Concatenate()(inputs)
                    X_train_p = [X_train[p].values for p in parents]
                    X_test_p = [X_test[p].values for p in parents]

                x = layers.Dense(n_nodes_se, activation='relu')(conc)
                x = layers.Dense(n_nodes_se, activation='relu')(x)
                if n in categ_features:
                    num_classes = len(X[n].unique())
                    dict_cat = { n + '_cat': X_train_cat[n].values.astype(int), n + '_scale': X_train[n].values }
                    dict_cat = pd.DataFrame(dict_cat)
                    dict_cat = dict_cat.drop_duplicates()
                    dict_cat = dict_cat.set_index(n + '_cat')[n + '_scale'].to_dict()

                    out = layers.Dense(num_classes, activation="softmax")(x)
                    ff = keras.Model(inputs=inputs, outputs=out, name=n)
                    ff.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
                    hist = ff.fit(X_train_p, X_train_cat[n].values, batch_size=512, epochs=2, verbose=0)

                else:
                    out = layers.Dense(1)(x)
                    ff = keras.Model(inputs=inputs, outputs=out, name=n)
                    ff.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(learning_rate=0.0001))
                    hist = ff.fit(X_train_p, X_train[n].values, batch_size=512, epochs=2, verbose=0)




                # import matplotlib.pyplot as plt
                # plt.plot(hist.history['loss'])
                # plt.show()
                #
                # plt.figure()
                # aaa=ff.predict(X_test_p)
                # plt.plot(X_test[n].values,aaa.reshape(1,-1)[0], '.', alpha=0.2)
                if n in categ_features:
                    score = ff.evaluate(X_train_p, X_train_cat[n].values, verbose=0)
                    print('The TRAIN accuracy for model node ', n, ' is ', score[1])
                    score = ff.evaluate(X_test_p, X_test_cat[n].values, verbose=0)
                    print('The TEST accuracy for model node ', n, ' is ', score[1])

                else:
                    score = ff.evaluate(X_train_p, X_train[n].values, verbose=0)
                    print('The TRAIN score for model node ', n, ' is ', score)
                    score = ff.evaluate(X_test_p, X_test[n].values, verbose=0)
                    print('The TEST score for model node ', n, ' is ', score)

                # save picture
                #dot_img_file = 'model_nn' + node_tmp +'.png'
                #keras.utils.plot_model(nn, to_file=dot_img_file, show_shapes=True)

                # plot model graph
                # keras.utils.plot_model(ff, show_shapes=True)


                # Calculate residuals as the value of the node - the prediction of the model for that node
                if n in categ_features:
                    pred = ff.predict(X_train_p)
                    pred = np.argmax(pred, axis=1)
                    pred = np.vectorize(dict_cat.get)(pred)
                    U_train['r_' + n] = X_train[n].values - pred

                    predt = ff.predict(X_test_p)
                    predt = np.argmax(predt, axis=1)
                    predt = np.vectorize(dict_cat.get)(predt)
                    U_test['r_' + n] = X_test[n].values - predt



                else:
                    pred = ff.predict(X_train_p).reshape(X_train.shape[0],)
                    U_train['r_' + n] = X_train[n].values - pred
                    pred = ff.predict(X_test_p).reshape(X_test.shape[0],)
                    U_test['r_' + n] = X_test[n].values - pred

                # build input for residual of node n
                res = keras.Input(shape=(1,), name="r_" + n)
                res_inputs.append(res)

                # create the reconstructed node as the built model ff + the residual
                if n in categ_features:

                    '''' 
                    keys = tf.constant([x for x in list(dict_cat.keys())], dtype=tf.int64)
                    values = tf.constant([dict_cat[k] for k in list(dict_cat.keys())])
                    table_init = tf.lookup.KeyValueTensorInitializer(keys, values)
                    table = tf.lookup.StaticHashTable(table_init, -1)
                    extra_layer = tf.argmax(out, axis=1)
                    extra_layer = layers.Lambda(table.lookup)(extra_layer)
                    '''
                    extra_layer = LabelConverter(dict_cat)(out)
                    ff = keras.Model(inputs=inputs, outputs=extra_layer, name=n + "_final")
                    tf.compat.v1.keras.backend.get_session().run(
                        tf.compat.v1.tables_initializer(name='init_all_tables'))
                    X_n[n] = layers.Add(name=n + "_reconstructed")([ff([X_n[p] for p in parents]), res])
                    X_n[n]  = LimitLayer(dict_cat)(X_n[n] )


                else:
                    X_n[n] = layers.Add(name=n + "_reconstructed")([ff([X_n[p] for p in parents]), res])

                tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.tables_initializer(name='init_all_tables'))

                # Save nn of the structural equation
                # keras.models.load_model("models/education.h5", custom_objects={'LabelConverter': LabelConverter})
                ff.save('models/'+str(n)+'.h5')

                added_nodes.append(n)

        # Add the node in the roots node, so the graph can be explored in the next dependence level
        root_nodes_tmp = root_nodes_tmp + added_nodes
        added_nodes = []


    # Define the structural equation model
    inputs = [X_n[r] for r in root_nodes] + res_inputs
    # Reorder the inputs and list "nodes" is
    col_name_inputs = [i.name[:-2].split('r_')[-1] for i in inputs]
    inputs = list(np.array(inputs)[[col_name_inputs.index(col) for col in nodes]])
    # concatenate outputs to build a stacked tensor (actually a vector),
    # respecting the order of the original nodes (i.e. same order of X_in)

    X_out = tf.concat([X_n[x] for x in nodes], axis=1, name='X_out')
    struct_eq_tmp = keras.Model(inputs=inputs, outputs=X_out, name="struct_eq_tmp")
    dim_input_se = U_train.shape[1]
    inputs = keras.Input(shape=(dim_input_se,), name="U")

    # x = tf.split(inputs, num_or_sizes_splits=U_train.shape[1], axis=1)
    # out_x = struct_eq_tmp(x)
    x = keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=dim_input_se, axis=1))(inputs)
    out_x = struct_eq_tmp(x)
    struct_eq = keras.Model(inputs=inputs, outputs=out_x, name="struct_eq")


    struct_eq.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())
    # struct_eq.save('models/nn_struct_eq.h5')

    # Save residual dataset
    columns_dataset_u = [i.split('r_')[-1] for i in U_train.columns]
    columns_dataset_u = list(np.array(U_train.columns)[[columns_dataset_u.index(col) for col in nodes]])

    U_train[columns_dataset_u].to_csv('data/res_train.csv', index=False)
    U_test[columns_dataset_u].to_csv('data/res_test.csv', index=False)

    ### Build M, standard ML model
    # model going from features X to target Y
    # the inputs are precisely the node inputs
    # X matrice -> Y
    X_in = keras.Input(shape=(len(nodes)), name='X_in')
    x = layers.Dense(n_nodes_M, activation='relu')(X_in)
    x = layers.Dense(int(n_nodes_M/2), activation='relu')(x)
    #out = layers.Dense(1, activation="sigmoid")(x)
    out = layers.Dense(2, activation='softmax')(x)
    M = keras.Model(inputs=X_in, outputs=out, name="M")
    M.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001))

    M.fit(X_train, y_train, batch_size=1024, epochs=5, verbose=0)
    M.save('models/nn_model.h5')


    ### Build a model from root_nodes + residuals to Y, i.e. Y^ = M(F(U))
    #struct_eq = keras.models.load_model('nn_struct_eq.h5') # lista U -> matrice X



    # matrice U -> Y
    inputs = keras.Input(shape=(U_train.shape[1],), name="U")
    out = M(struct_eq(inputs))
    #out = M(struct_eq.outputs)
    final = keras.Model(inputs=inputs, outputs=out, name="final")

    final.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.Adam())
    # final.summary()
    # dot_img_file = 'final.png'
    #keras.utils.plot_model(final, to_file=dot_img_file, show_shapes=True)

    # final.save('final.h5')

    ### make predictions
    # Load final model
    # final = keras.models.load_model('final.h5')
    pred = final.predict(U_test)[:, 1]
    # Print report
    print(classification_report(y_test, pred > 0.5))

    return struct_eq, final

