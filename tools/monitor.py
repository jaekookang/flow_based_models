'''
Callbacks for train monitoring

2020-04-15
'''
import matplotlib.pyplot as plt
import os
import numpy as np
from time import time, strftime, gmtime
import tensorflow as tf
import tensorflow_probability as tfp
tfk = tf.keras
tfkl = tfk.layers
tfkc = tfk.callbacks
tfd = tfp.distributions

class NBatchLogger(tfkc.Callback):
    '''A Logger that log average performance per `display` steps.'''

    def __init__(self, n_display, max_epoch, save_dir=None, suffix=None):
        self.epoch = 0
        self.display = n_display
        self.max_epoch = max_epoch
        self.logs = {}
        self.save_dir = save_dir
        if self.save_dir is not None:
            assert os.path.exists(self.save_dir), Exception(
                f'Path:{self.save_dir} does not exist!')
            fname = 'train.log'
            if suffix is not None:
                fname = f'train_{suffix}.log'
            self.fid = open(os.path.join(save_dir, fname), 'w')
        self.t0 = time()

    def on_train_begin(self, logs={}):
        logs = logs or self.logs
        txt = f'=== Started at {self.get_time()} ==='
        print(txt)
        self.write_log(txt)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        if self.epoch % self.display == 0:
            txt = f'{self.get_time()} | Epoch: {self.epoch}/{self.max_epoch} | '
            print(txt, end='')
            for i, key in enumerate(logs.keys()):
                if (i+1) == len(logs.keys()):
                    _txt = f'{key}={logs[key]:4f}'
                    print(_txt, end='\n')
                else:
                    _txt = f'{key}={logs[key]:4f} '
                    print(_txt, end='')
                txt = txt + _txt
            self.write_log(txt)
        self.logs = logs

    def on_train_end(self, logs={}):
        logs = logs or self.logs
        t1 = time()
        txt = f'=== Time elapsed: {(t1-self.t0)/60:.4f} min (loss:{logs["loss"]:4f}) ==='
        print(txt)
        self.write_log(txt)

    def get_time(self):
        return strftime('%Y-%m-%d %Hh:%Mm:%Ss', gmtime())

    def write_log(self, txt):
        if self.save_dir is not None:
            self.fid.write(txt+'\n')
            self.fid.flush()


def make_image(init_data, model, title, direction='backward', is_image=False, sharexy=False):
    # direction='backward': data --> latent
    # direction='forward': latent --> data
    import io

    samples = [init_data]
    names = []
    x = init_data

    if direction == 'backward':
        names = ['latent']
        for layer in reversed(model.layers):
            x = layer.inverse(x)
            samples.append(x.numpy())
            names.append(layer.name)
    elif direction == 'forward':
        names = ['sample']
        for layer in model.layers:
            x = layer(x)
            samples.append(x.numpy())
            names.append(layer.name)
    else:
        raise Exception('Must provide either "backward" or "forward"')

    if sharexy:
        f, arr = plt.subplots(1, len(samples),
                              figsize=((4 * (len(samples)))//2, 4//2),
                              sharex=True, sharey=True)
    else:
        f, arr = plt.subplots(1, len(samples),
                              figsize=((4 * (len(samples)))//2, 4//2))

    X0 = samples[0]
    if not is_image:
        for i in range(len(samples)):
            X1 = samples[i]
            idx = np.logical_and(X1[:, 0] < 0, X0[:, 1] < 0)
            arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
            arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
            arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
            arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
            arr[i].set_title(names[i])
    else:
        d = np.sqrt(x.shape[1]).astype('int')
        for i in range(len(samples)):
            X1 = samples[i].reshape(d, d)
            arr[i].imshow(X1, aspect='equal')
            arr[i].set_title(names[i])

    buf = io.BytesIO()
    plt.suptitle(title, fontsize=25, y=1.1)
    plt.tight_layout()
    f.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    buf.close()
    plt.close()
    return image


class TensorBoardImage(tfkc.Callback):
    def __init__(self, hparams, is_image=False):
        super().__init__()
        self.hp = hparams
        self.n_display = self.hp.n_display
        self.n_sample = 400
        if is_image:
            self.n_sample = 1
        self.is_image = is_image
        self.X = tfd.MultivariateNormalDiag(loc=[0.]*self.hp.inp_dim)
        self.x = self.X.sample(self.n_sample).numpy()
        self.writer = tf.summary.create_file_writer(self.hp.logdir)

    def on_epoch_end(self, epoch, logs={}):
        if ((epoch+1) % self.hp.n_display) == 0:
            with self.writer.as_default():
                image = make_image(self.x, self.model, title=epoch+1,
                                   is_image=self.is_image, sharexy=True)
                tf.summary.image('test', image, max_outputs=1, step=1)
