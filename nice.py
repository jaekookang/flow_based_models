'''
NICE modules

# How to use:
# (See `example_moon.ipynb`)
- 1. Import `class NICE` into your training script
- 2. Initiate `NICE` with hyperparameters
    - eg. `model = NICE(inp_dim, n_layer, n_hid, name='NICE')`
- 3. Compile with the loss function
- 4. Training the model

2020-11-18 first created
'''
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils import *
tfk = tf.keras
tfkl = tfk.layers
tfkc = tfk.callbacks
tfd = tfp.distributions
K = tfk.backend


def FCNN(inp_dim, n_hid_layer, n_hid):
    '''Fully-Connected Neural Networks'''
    mlp = tfk.Sequential(name='g')
    for _ in range(n_hid_layer):
        mlp.add(tfkl.Dense(n_hid, activation='relu'))
    mlp.add(tfkl.Dense(inp_dim//2, activation='linear'))
    return mlp

class AdditiveCouplingLayer(tfkl.Layer):
    def __init__(self, inp_dim, shuffle_type, n_hid_layer, n_hid_dim, name):
        super(AdditiveCouplingLayer, self).__init__(name=name)
        self.inp_dim = inp_dim
        self.shuffle_type = shuffle_type
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.g = FCNN(inp_dim, n_hid_layer, n_hid_dim)
        self.idx = tf.Variable(list(range(self.inp_dim)),
                               shape=(self.inp_dim,),
                               trainable=False,
                               name='index',
                               dtype='int64')
        if self.shuffle_type == 'random':
            self.idx.assign(tf.random.shuffle(self.idx))
        elif self.shuffle_type == 'reverse':
            self.idx.assign(tf.reverse(self.idx, axis=[0]))

    def call(self, x):
        # Forward
        x = self.shuffle(x)
        x1, x2 = self.split(x)
        mx1 = self.g(x1)
        # s = self.s(x1)
        # t = self.t(x1)
        # x1, x2 = self.couple([x1, x2, s, t])
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])
        return x

    def shuffle(self, x, isInverse=False):
        if not isInverse:
            # Forward
            idx = self.idx
        else:
            # Inverse
            idx = tf.map_fn(tf.math.invert_permutation,
                            tf.expand_dims(self.idx, 0))
            idx = tf.squeeze(idx)
        x = tf.transpose(x)
        x = tf.gather(x, idx)
        x = tf.transpose(x)
        return x

    def split(self, x):
        dim = x.shape[-1]
        x = tf.reshape(x, [-1, dim//2, 2])
        return [x[:, :, 0], x[:, :, 1]]  # (N,dim//2) (N,dim//2)

    def couple(self, xs, isInverse=False):
        x1, x2, mx1 = xs
        # x1, x2, s, t = xs
        if isInverse:
            return [x1, x2-mx1] # Inverse
            # return [x1, (x2 - t) * tf.math.exp(-s)]
        else:
            return [x1, x2+mx1] # Forward
            # return [x1, x2 * tf.math.exp(s) + t]

    def concat(self, xs):
        xs = [tf.expand_dims(x, 2) for x in xs]  # [(N,392) (N,392)]
        x = tf.concat(xs, 2)  # (N,dim,2)
        return tf.reshape(x, [-1, tf.math.reduce_prod(x.shape[1:])])

    def inverse(self, x):
        x1, x2 = self.split(x)
        mx1 = self.g(x1)
        # s = self.s(x1)
        # t = self.t(x1)
        # x1, x2 = self.couple([x1, x2, s, t], isInverse=True)
        x1, x2 = self.couple([x1, x2, mx1], isInverse=True)
        x = self.concat([x1, x2])
        x = self.shuffle(x, isInverse=True)
        return x


class ScalingLayer(tfkl.Layer):
    def __init__(self, inp_dim):
        super(ScalingLayer, self).__init__()
        self.inp_dim = inp_dim
        self.scaling = self.add_weight(name='scaling',
                                       shape=(1, self.inp_dim),
                                       initializer='glorot_normal',
                                       trainable=True)
    def call(self, x):
        self.add_loss(-tf.math.reduce_sum(self.scaling))
        return tf.math.exp(self.scaling) * x
    
    def inverse(self, x):
        scale = tf.math.exp(-self.scaling)
        return scale * x


class NICE(tfk.Model):
    def __init__(self, inp_dim, shuffle_type, n_couple_layer, n_hid_layer, n_hid_dim, **kwargs):
        super(NICE, self).__init__(**kwargs)
        self.inp_dim = inp_dim
        self.shuffle_type = shuffle_type
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.AffineLayers = []
        for i in range(n_couple_layer):
            layer = AdditiveCouplingLayer(inp_dim, shuffle_type, n_hid_layer, n_hid_dim, name=f'Layer{i}')
            self.AffineLayers.append(layer)
        self.scale = ScalingLayer(inp_dim)
        self.AffineLayers.append(self.scale)
        
    def call(self, x):
        '''Forward mapping: from data to latent'''
        for i in range(self.n_couple_layer):
            x = self.AffineLayers[i](x)
        x = self.scale(x)
        return x
    
    def inverse(self, x):
        '''Inverse mapping: from latent to data'''
        x = self.scale.inverse(x)
        for i in reversed(range(self.n_couple_layer)):
            x = self.AffineLayers[i].inverse(x)
        return x
