
'''
Non-Linear Independent Component Estimination (NICE) code

Modified from
- https://github.com/bojone/flow

2019-09-02 first created
2020-04-15 
'''
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfkl = tfk.layers
tfkc = tfk.callbacks
K = tfk.backend


class AdditiveAffineLayer(tfkl.Layer):
    def __init__(self, inp_dim, shuffle_type, n_couple_layer, n_hid_dim, act_func, name):
        super(AdditiveAffineLayer, self).__init__(name=name)
        self.inp_dim = inp_dim
        self.shuffle_type = shuffle_type
        self.n_couple_layer = n_couple_layer
        self.n_hid_dim = n_hid_dim
        self.act_func = act_func
        self.g = self._g(add_batchnorm=False)
        self.idx = tf.Variable(list(range(self.inp_dim)),
                               shape=(self.inp_dim,),
                               trainable=False,
                               name='index',
                               dtype='int64')
        if self.shuffle_type == 'random':
            self.idx.assign(tf.random.shuffle(self.idx))
        elif self.shuffle_type == 'reverse':
            self.idx.assign(tf.reverse(self.idx, axis=[0]))
        else:
            raise NotImplementedError

    def call(self, x):
        # Forward
        x = self.shuffle(x)
        x1, x2 = self.split(x)
        mx1 = self.g(x1)
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

    def _g(self, add_batchnorm=False):
        mlp = tfk.Sequential(name='DenseLayer')
        for _ in range(self.n_couple_layer):
            mlp.add(tfkl.Dense(self.n_hid_dim, activation=self.act_func))
            if add_batchnorm:
                mlp.add(tfkl.BatchNormalization())
        mlp.add(tfkl.Dense(self.inp_dim//2, activation='linear'))
        return mlp

    def couple(self, xs, isInverse=False):
        x1, x2, mx1 = xs
        if not isInverse:
                # Forward
            return [x1, x2+mx1]
        else:
            # Inverse
            return [x1, x2-mx1]

    def concat(self, xs):
        xs = [tf.expand_dims(x, 2) for x in xs]  # [(N,392) (N,392)]
        x = tf.concat(xs, 2)  # (N,dim,2)
        return tf.reshape(x, [-1, tf.math.reduce_prod(x.shape[1:])])

    def inverse(self, x):
        x1, x2 = self.split(x)
        mx1 = self.g(x1)
        x1, x2 = self.couple([x1, x2, mx1], isInverse=True)
        x = self.concat([x1, x2])
        x = self.shuffle(x, isInverse=True)
        return x


class Scale(tfkl.Layer):
    def __init__(self, inp_dim, **kwargs):
        super(Scale, self).__init__(**kwargs)
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
    def __init__(self, hparams, name='NICE'):
        super(NICE, self).__init__(name=name)
        self.hp = hparams
        self.AffineLayers = []
        for i in range(self.hp.nice.n_couple_layer):
            layer = AdditiveAffineLayer(self.hp.nice.inp_dim,
                                        self.hp.nice.shuffle,
                                        self.hp.nice.n_couple_layer,
                                        self.hp.nice.n_hid_dim,
                                        self.hp.nice.act_func, name=f'layer{i}')
            self.AffineLayers += [layer]

        self.scale = Scale(self.hp.nice.inp_dim, name='ScaleLayer')
        self.AffineLayers += [self.scale]

    def call(self, x):
        act = x
        for i in range(self.hp.nice.n_couple_layer):
            act = self.AffineLayers[i](act)
        act = self.scale(act)
        return act

    def inverse(self, y):
        act = y
        # act = self.scale.inverse()(act)
        act = self.scale.inverse(act)
        for i in reversed(range(self.hp.nice.n_couple_layer)):
            act = self.AffineLayers[i].inverse(act)
        return act

def nice_loss(y_true, y_pred):
    '''Loss function for NICE model'''
    return tf.math.reduce_sum(0.5*y_pred**2)