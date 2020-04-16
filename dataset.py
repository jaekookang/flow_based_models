'''Make dataset

2020-04-09
'''

import random
import json
import dotmap
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def make_dataset(hp):
    '''Make dataset generator (tf.data)
    Args:
    - hp: hyperparameters as DotMap dictionary
        - hp.which_data can be either 'moon', 'mgauss', 'mnist' or 'speech'

    Note:
    - same_input_output: 
        - If True, (input,input) will be generated.
        - If False, (input,target) will be generated.
            - 'moon': target will be 'left' or 'right' shape moon
            - 'mgauss': target will be number of gaussians
            - 'mnist': target will be the number in the image
            - 'speech': target will be acoustics
    - batch_size: number of examples in a mini-batch

    Returns:
    - tf.data.Dataset object (iterator)
    '''
    if hp.which_data == 'moon':
        if hp.moon.same_input_output:
            dataset = make_generator(gen_moonshape, args=(hp.moon.same_input_output, hp.moon.n_data),
                                     output_types=('float32', 'float32'),
                                     output_shapes=(
                                         hp.moon.n_data*2, hp.moon.n_data*2),
                                     n_data=hp.moon.n_data, batch_size=hp.train.batch_size)
        else:
            dataset = make_generator(gen_moonshape, args=(hp.moon.same_input_output, hp.moon.n_data),
                                     output_types=('float32', 'uint8'),
                                     output_shapes=(hp.moon.n_data*2, 2),
                                     n_data=hp.moon.n_data, batch_size=hp.train.batch_size)

    elif hp.which_data == 'mgauss':
        if hp.mgauss.same_input_output:
            dataset = make_generator(gen_multigauss,
                                     args=(hp.mgauss.same_input_output, hp.mgauss.n_data,
                                           hp.mgauss.n_gauss, hp.mgauss.radius, hp.mgauss.sd),
                                     output_types=('float32', 'float32'),
                                     output_shapes=(
                                         hp.mgauss.n_data*2, hp.mgauss.n_data*2),
                                     n_data=hp.mgauss.n_data, batch_size=hp.train.batch_size)
        else:
            dataset = make_generator(gen_multigauss,
                                     args=(hp.mgauss.same_input_output, hp.mgauss.n_data,
                                           hp.mgauss.n_gauss, hp.mgauss.radius, hp.mgauss.sd),
                                     output_types=('float32', 'uint8'),
                                     output_shapes=(2, hp.mgauss.n_gauss),
                                     n_data=hp.mgauss.n_data, batch_size=hp.train.batch_size)
    elif hp.which_data == 'mnist':
        if hp.mnist.same_input_output:
            dataset = make_generator(gen_mnist,
                                     args=(hp.mnist.same_input_output,),
                                     output_types=('float32', 'float32'),
                                     output_shapes=(784, 784),
                                     n_data=hp.mnist.n_data, batch_size=hp.train.batch_size)
        else:
            dataset = make_generator(gen_mnist,
                                     args=(hp.mnist.same_input_output,),
                                     output_types=('float32', 'uint8'),
                                     output_shapes=(784, 10),
                                     n_data=hp.mnist.n_data, batch_size=hp.train.batch_size)
    elif hp.which_data == 'speech':
        if hp.speech.same_input_output:
            dataset = make_generator(gen_speech,
                                     args=(hp.speech.same_input_output,),
                                     output_types=('float32', 'float32'),
                                     output_shapes=(5, 5),
                                     n_data=hp.speech.n_data, batch_size=hp.train.batch_size)
        else:
            dataset = make_generator(gen_speech,
                                     args=(hp.speech.same_input_output,),
                                     output_types=('float32', 'float32'),
                                     output_shapes=(5, 3),
                                     n_data=hp.speech.n_data, batch_size=hp.train.batch_size)
    else:
        raise Exception(
            'Provide dataset among "moon", "mgauss", "mnist" and "speech"')
    return dataset


def make_generator(gen_func, args, output_types, output_shapes, n_data, batch_size):
    '''Make tf.data.Dataset object from python generator'''
    dataset = tf.data.Dataset.from_generator(
        gen_func,
        args=args,
        output_types=output_types,
        output_shapes=output_shapes)
    dataset = dataset.shuffle(n_data).repeat().batch(
        batch_size, drop_remainder=True)
    return dataset


def gen_moonshape(same_input_output, n_data):
    '''
    if same_input_output is False,
        target [1,0] means left and target [0,1] means right skewed shape
    '''
    for _ in range(n_data):
        x2 = tfd.Normal(loc=0., scale=4.)
        x2_samples = x2.sample(n_data)
        x1 = tfd.Normal(loc=.25 * tf.square(x2_samples), scale=1.)
        x1_samples = x1.sample()
        X = tf.stack([x1_samples, x2_samples], axis=0)  # (n_data, 2)
        X = tf.reshape(X, (-1,))  # (n_data*2,) as 1-d array
        if same_input_output:
            yield (X, X)
        else:
            if random.randint(0, 9) % 2:
                yield (X, tf.constant([1, 0], dtype='uint8'))  # left skewed
            else:
                yield (-X, tf.constant([0, 1], dtype='uint8'))  # right skewed


def gen_multigauss(same_input_output, n_data, n_gauss, radius, sd):
    assert n_data % n_gauss == 0, f'Provide `n_data` as multiples of `n_gauss`'
    if same_input_output:
        for _ in range(n_data):
            gauss = list(range(n_gauss))
            random.shuffle(gauss)
            X = np.array([], dtype='float32').reshape((-1, 2))
            for i in gauss:
                th = 2*np.pi/n_gauss*(i+1)
                mean = [radius*np.cos(th), radius*np.sin(th)]
                _X = np.random.multivariate_normal(
                    mean, np.identity(2)*sd, size=n_data//n_gauss)
                X = np.vstack([X, _X])
            X = X.reshape((-1))  # 1-d array
            yield (X, X)
    else:
        for _ in range(n_data):
            gauss = list(range(n_gauss))
            random.shuffle(gauss)
            X = np.array([], dtype='float32').reshape((-1, n_data))
            Y = np.array([], dtype='uint8').reshape((-1, n_gauss))
            for i in gauss:
                th = 2*np.pi/n_gauss*(i+1)
                mean = [radius*np.cos(th), radius*np.sin(th)]
                X = np.random.multivariate_normal(
                    mean, np.identity(2)*sd, size=1)
                Y = np.eye(n_gauss, dtype='uint8')[i]
                X = X.reshape((-1))  # 1-d array
                yield (X, Y)


def gen_mnist(same_input_output):
    mnist = tf.keras.datasets.mnist
    (X, Y), _ = mnist.load_data()
    h, w = X.shape[1:]
    X = X.reshape([-1, h*w]).astype('float32')/255
    if same_input_output:
        for x in X:
            yield (x, x)
    else:
        for x, y in zip(X, Y):
            y = np.eye(10, dtype='uint8')[y]
            yield (x, y)


def gen_speech(same_input_output):
    artic_col = ['T1x', 'T1y', 'T2x', 'T2y', 'T3x', 'T3y', 'T4x', 'T4y',
                 'ULx', 'ULy', 'LLx', 'LLy', 'MNIx', 'MNIy']
    acous_col = ['F1', 'F2', 'F3']
    S = pd.read_pickle('data/artic_acous/JW12.pckl')
    pca = pd.read_pickle('data/artic_acous/pca.pckl')
    X_scaler = pd.read_pickle('data/artic_acous/x_scaler.pckl')
    Y_scaler = pd.read_pickle('data/artic_acous/y_scaler.pckl')

    X = S[artic_col].values.astype('float32')
    X = X_scaler.transform(X)
    X = pca.transform(X)

    if same_input_output:
        for x in X:
            yield (x, x)
    else:
        Y = S[acous_col].values.astype('float32')
        Y = Y_scaler.transform(Y)
        for x, y in zip(X, Y):
            yield (x, y)


if __name__ == '__main__':
    # Load hyperparameters
    with open('hparams.json', 'r') as f:
        hp = dotmap.DotMap(json.load(f))

    # Test
    gen = make_dataset(hp)
    it = iter(gen)
    print(next(it))
    del gen, it
    print('Done')
