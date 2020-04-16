'''
Training procedure for flow-based models

## Dataset
1. Moonshape
2. Multiple Gaussians
3. MNIST
4. Speech production

## Run
- CPU only mode
```
python train.py --data moon
```

- GPU mode
```
python train.py --data moon --gpu 0  # specify gpu id
```

## Monitor training
- `tensorboard --logdir model/* --port 6006`
- `nohup ./ngrok http 6006 --log=stdout > ngrok.log &`

2020-04-15
'''

import os
import json
import argparse
import pydotplus
from dotmap import DotMap
from dataset import make_dataset
from tools.monitor import NBatchLogger
from tools.utils import safe_mkdir, safe_rmdir, get_datetime
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers
tfkc = tfk.callbacks
plot_model = tfk.utils.plot_model


def build_model(hp):
    if hp.which_model == 'nice':
        from models.nice import NICE
        model = NICE(hp)
        def loss(y_true, y_pred): return tf.math.reduce_sum(0.5*y_pred**2)
        inp_dim = int(hp[hp.which_data].n_data*2)
    # elif hp.which_model == 'realnvp':
    #     from models.realnvp import RealNVP
    #     model = RealNVP(hp)
    # elif hp.which_model == 'inn':
    #     from models.inn import inn
    #     model = INN(hp)
    model.compile(loss=loss, optimizer=hp.train.optimizer)
    model.build(input_shape=(inp_dim,))
    #log_dir = os.path.join(hp.log_dir, get_datetime())
    log_dir = os.path.join(hp.log_dir, hp.which_data+'_'+hp.which_model)
    safe_mkdir(log_dir)
    hp.log_dir = log_dir
    with open(os.path.join(log_dir, 'hparams.json'), 'w') as f:
        json.dump(hp, f, indent=4)
    plot_model(model, to_file=os.path.join(log_dir, 'model.png'),
               show_shapes=True, show_layer_names=True, expand_nested=True)
    return model, hp


def run(args):
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load hyperparameters
    with open('hparams.json', 'r') as f:
        hp = DotMap(json.load(f))

    # Make dataset
    data_gen = make_dataset(hp)

    # Initialize model
    model, hp = build_model(hp)

    # Set callbacks
    checkpoint = tfk.callbacks.ModelCheckpoint(filepath=os.path.join(hp.log_dir,'weight.h5'),
                                               monitor='loss',
                                               verbose=0,
                                               save_best_only=True,
                                               save_weights_only=True)
    tfboard = tfk.callbacks.TensorBoard(log_dir=hp.log_dir, profile_batch=0)

    # Train
    hist = model.fit(data_gen,
                     epochs=hp.train.n_epoch,
                     steps_per_epoch=hp[hp.which_data].n_data//hp.train.batch_size,
                     callbacks=[checkpoint, 
                                NBatchLogger(hp.train.n_display, hp.train.n_epoch, hp.log_dir),
                                tfboard, 
                                # tbimage
                                ],
                     verbose=1)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('-g', '--gpu', type=int, required=False, default=-1,
                        help='Default -1 (cpu only); specify gpu number if any')
    args = parser.parse_args()

    run(args)
