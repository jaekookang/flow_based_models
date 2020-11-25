'''
Flow based models

- NICE
- RealNVP

2020-11-18 first created
'''

import tensorflow as tf
from utils import *

tfk = tf.keras
tfkl = tfk.layers
tfkc = tfk.callbacks
K = tfk.backend


def fully_connected(n_dim, n_layer=3, n_hid=512, activaiton='relu'):
    '''Fully connected neural networks'''
    nn = tfk.Sequential(name='neural_net')
    for _ in range(n_layer - 1):
        nn.add(tfkl.Dense(n_hid, activation=activaiton))
    nn.add(tfkl.Dense(n_dim//2, activation='linear'))
    return nn


class AdditiveCouplingLayer(tfkl.Layer):
    '''
    Implementation of Additive Coupling layers in Dinh et al (2015)

    # forward
    y1 = x1
    y2 = x2 + m(x1)
    # inverse
    x1 = y1
    x2 = y2 - m(y1)
    '''

    def __init__(self, inp_dim, n_hid_layer, n_hid_dim, name):
        super(AdditiveCouplingLayer, self).__init__(name=name)
        self.inp_dim = inp_dim
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.m = fully_connected(inp_dim, n_hid_layer, n_hid_dim)
        self.permute = tfkl.Lambda(lambda x: tf.gather(
            x, list(reversed(range(x.shape[-1]))), axis=-1))

    def call(self, x):
        x = self.permute(x)
        x1, x2 = self.split(x)
        mx1 = self.m(x1)
        y1 = x1
        y2 = x2 + mx1
        y = tf.concat([y1, y2], axis=-1)
        return y

    def inverse(self, y):
        y1, y2 = self.split(y)
        my1 = self.m(y1)
        x1 = y1
        x2 = y2 - my1
        x = tf.concat([x1, x2], axis=-1)
        x = self.permute(x)
        return x

    def split(self, x):
        dim = self.inp_dim
        x = tf.reshape(x, [-1, dim])
        return x[:, :dim//2], x[:, dim//2:]


class ScalingLayer(tfkl.Layer):
    def __init__(self, inp_dim):
        super(ScalingLayer, self).__init__(name='ScalingLayer')
        self.inp_dim = inp_dim
        self.scaling = tf.Variable(shape=(inp_dim,),
                                   trainable=True,
                                   initial_value=tfk.initializers.glorot_uniform()((inp_dim,)))

    def call(self, x):
        self.add_loss(-tf.math.reduce_sum(self.scaling))
        return tf.math.exp(self.scaling) * x

    def inverse(self, y):
        return tf.math.exp(-self.scaling) * y


class RealNVP(tfkl.Layer):
    pass


class NICE(tfk.Model):
    def __init__(self, inp_dim, n_couple_layer, n_hid_layer, n_hid_dim, **kwargs):
        super(NICE, self).__init__(**kwargs)
        self.inp_dim = inp_dim
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.AffineLayers = []
        for i in range(n_couple_layer):
            layer = AdditiveCouplingLayer(inp_dim, n_hid_layer, n_hid_dim, name=f'Layer{i}')
            self.AffineLayers.append(layer)
        self.scale = ScalingLayer(inp_dim)
        self.AffineLayers.append(self.scale)

    def call(self, x):
        '''Forward: data (x) --> latent (z); inference'''
        z = x
        for i in range(self.n_couple_layer):
            z = self.AffineLayers[i](z)
        z = self.scale(z)
        return z

    def inverse(self, z):
        '''Inverse: latent (z) --> data (y); sampling'''
        x = self.scale.inverse(z)
        for i in reversed(range(self.n_couple_layer)):
            x = self.AffineLayers[i].inverse(x)
        return x


if __name__ == "__main__":
    inp_dim = 2
    n_couple_layer = 3
    n_hid_layer = 3
    n_hid_dim = 512
    model = NICE(inp_dim, n_couple_layer, n_hid_layer, n_hid_dim)
    x = tfkl.Input(shape=(inp_dim,))
    model(x)
    model.summary()
