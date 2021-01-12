import json
import os

import numpy as np
import seaborn as sns
import tensorflow as tf

from pipelines import KVAETrainer
from utils import reload_config, get_image_config

sns.set_style("whitegrid", {'axes.grid': False})
np.random.seed(1337)


def run():
    """Load and train model
    Create a model object and run the training using the provided config.
    """
    config = get_image_config()
    # To reload a saved model
    config = reload_config(config.FLAGS)

    # Add timestamp to log path
    config.log_dir = os.path.join(config.log_dir, '%s' % config.run_name)

    # Add model name to log path
    config.log_dir = config.log_dir + '_kvae'

    # Create log path
    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)

    # Save hyperparameters
    with open(config.log_dir + '/config.json', 'w') as f:
        json.dump(config.__flags, f)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    with tf.Session() as session:
        trainer = KVAETrainer(session, config)
        trainer.train()
        trainer.imputation_plot()


if __name__ == "__main__":
    run()
