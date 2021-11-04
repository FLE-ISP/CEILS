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

import tensorflow as tf

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
