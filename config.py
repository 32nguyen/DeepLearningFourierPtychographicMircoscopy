import os

import json
from errno import EEXIST

#import seaborn as sns
import numpy as np

#sns.set()


class MyDict(dict):

    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DEFAULT_LOG_DIR = 'log'
UNET_WEIGHTS_FILE = 'unet_weights.h5'


def mkdir(mypath):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else:
            raise


def save_weights(models, log_dir=DEFAULT_LOG_DIR, expt_name=None):
    """Save the weights of the models into a file."""
    log_dir = get_log_dir(log_dir, expt_name)
    models.save_weights(os.path.join(log_dir, UNET_WEIGHTS_FILE), overwrite=True)


def get_log_dir(log_dir, expt_name):
    """Compose the log_dir with the experiment name."""
    if log_dir is None:
        raise Exception('log_dir can not be None.')

    if expt_name is not None:
        return os.path.join(log_dir, expt_name)
    return log_dir

def create_expt_dir(params):
    """Create the experiment directory and return it."""
    expt_dir = get_log_dir(params.log_dir, params.expt_name)

    # Create directories if they do not exist
    mkdir(params.log_dir)
    mkdir(expt_dir)

    # Save the parameters
    json.dump(params, open(os.path.join(expt_dir, 'params.json'), 'wb'),
              indent=4, sort_keys=True)

    return expt_dir

params = MyDict({
        # volume input
        'i_in': 80,
        'i_out_scale' : 320,
        'input_generator_channel': 49,
        # Model
        #'filter_size': 16,  # filter size of first layer in 3D UNET
        # Training parameters
        'batch_size': 16,  # The batch size (batch_size, image_size, image_size, image_size)
        'steps_per_epoch': 10,  # ~ number of batches running in 1 single epoch
        'epochs': 1000,  # total number of epochs running
        'val_step': 20,  # number of batches in validation
        'lr': 2e-4,  # The learning rate to train the models
        'beta_1': 0.5,  # The beta_1 value of the Adam optimizer
        # Directories
        #'path_train': '/home/tom/PycharmProjects/3DPhantomCNN/SimulatedPhantom',
        #'path_valid': '/home/tom/PycharmProjects/3DPhantomCNN/SimulatedPhantom_Test',
        #'path_train': '/home/vybui4490/3DPhantomCNN/SimulatedPhantom',
        #'path_valid': '/home/vybui4490/3DPhantomCNN/SimulatedPhantom_Test',
        #'path_history': '/home/vybui4490/3DPhantomCNN/log',
        #'path_train': '/home/tom/PycharmProjects/DataForAll/3DPhantom/IRT_Train',
        #'path_valid': '/home/tom/PycharmProjects/DataForAll/3DPhantom/IRT_Test',
        #'path_train': '/home/tom/PycharmProjects/3DPhantomCNN/DiffPhantom_Train',
        #'path_valid': '/home/tom/PycharmProjects/3DPhantomCNN/DiffPhantom_Test',
        #'path_history': '/home/tom/PycharmProjects/3DPhantomCNN/log',
        # File
        'log_dir': 'log',  # Directory to log
        'expt_name': 'pix2pix',  # The name of the experiment. Saves the logs into a folder with this name
        'save_every': 10,
        })




