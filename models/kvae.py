import matplotlib

import seaborn as sns
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers import optimize_loss
import numpy as np

from .kalman_filter import KalmanFilter
from utils.data import PymunkData
from utils.nn import (
    simple_sample,
    subpixel_reshape,
    log_likelihood,
    get_activation_fn
)
from tensorflow.contrib.rnn import BasicLSTMCell

np.random.seed(1337)
matplotlib.use('Agg')
sns.set_style("whitegrid", {'axes.grid': False})


class KVAEEncoder(object):

    def __init__(self, config, sess):
        self._config = config

    @property
    def config(self):
        return self._config


class KVAE(object):
    """ This class defines functions to build, train and evaluate Kalman Variational Autoencoders
    """
    def __init__(self, config):
        self.config = config

        # Load the dataset
        self.train_data = PymunkData("../data/{}.npz".format(config.dataset), config)
        self.test_data = PymunkData("../data/{}_test.npz".format(config.dataset), config)

        # Frame sizes
        self.d1 = self.train_data.d1
        self.d2 = self.train_data.d2

        # Initializers for LGSSM variables. A is intialized with identity matrices, B and C randomly from a gaussian
        A = np.array([np.eye(config.dim_z).astype(np.float32) for _ in range(config.K)])
        B = np.array([config.init_kf_matrices * np.random.randn(config.dim_z, config.dim_u).astype(np.float32)
                      for _ in range(config.K)])
        C = np.array([config.init_kf_matrices * np.random.randn(config.dim_a, config.dim_z).astype(np.float32)
                      for _ in range(config.K)])
        # We use isotropic covariance matrices
        Q = config.noise_transition * np.eye(config.dim_z, dtype=np.float32)
        R = config.noise_emission * np.eye(config.dim_a, dtype=np.float32)

        # p(z_1)
        mu = np.zeros((self.config.batch_size, config.dim_z), dtype=np.float32)
        sigma = np.tile(config.init_cov * np.eye(config.dim_z, dtype=np.float32), (self.config.batch_size, 1, 1))

        # Initial variable a_0
        a_0 = np.zeros((config.dim_a,), dtype=np.float32)

        # Collect initial variables
        self.init_vars = dict(A=A, B=B, C=C, Q=Q, R=R, mu=mu, Sigma=sigma, a_0=a_0)

        # Get activation function for hidden layers
        if config.activation.lower() == 'relu':
            self.activation_fn = tf.nn.relu
        elif config.activation.lower() == 'tanh':
            self.activation_fn = tf.nn.tanh
        elif config.activation.lower() == 'elu':
            self.activation_fn = tf.nn.elu
        else:
            self.activation_fn = None

        # Parse num_filters to list of ints
        self.num_filters = [int(f) for f in config.num_filters.split(',')]

        # Init placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, None, self.d1, self.d2], name='x')
        self.ph_steps = tf.placeholder(tf.int32, shape=(), name='n_step')
        self.scale_reconstruction = tf.placeholder(tf.float32, shape=(), name='scale_reconstruction')
        self.mask = tf.placeholder(tf.float32, shape=(None, None), name='mask')
        self.a_prev = tf.placeholder(tf.float32, shape=[None, config.dim_a], name='a_prev')  # For alpha NN plotting

        # Init various
        self.saver = None
        self.kf = None
        self.vae_updates = None
        self.vae_kf_updates = None
        self.all_updates = None
        self.lb_vars = None
        self.model_vars = None
        self.enc_shape = None
        self.n_steps_gen = None
        self.out_gen_det = None
        self.out_gen = None
        self.out_gen_det_impute = None
        self.train_summary = None
        self.test_summary = None

    def encoder(self, x):
        """ Convolutional variational encoder to encode image into a low-dimensional latent code
        If config.conv == False it is a MLP VAE. If config.use_vae == False, it is a normal encoder
        :param x: sequence of images
        :return: a, a_mu, a_var
        """
        with tf.variable_scope('vae/encoder'):
            if self.config.conv:
                x_flat_conv = tf.reshape(x, (-1, self.d1, self.d2, 1))
                enc_hidden = slim.stack(
                    x_flat_conv,
                    slim.conv2d,
                    self.num_filters,
                    kernel_size=self.config.filter_size,
                    stride=2,
                    activation_fn=self.activation_fn,
                    padding='SAME'
                )
                enc_flat = slim.flatten(enc_hidden)
                self.enc_shape = enc_hidden.get_shape().as_list()[1:]

            else:
                x_flat = tf.reshape(x, (-1, self.d1 * self.d2))
                enc_flat = slim.repeat(
                    x_flat,
                    self.config.num_layers,
                    slim.fully_connected,
                    self.config.vae_num_units, self.activation_fn
                )

            a_mu = slim.fully_connected(enc_flat, self.config.dim_a, activation_fn=None)

            if self.config.use_vae:
                a_var = slim.fully_connected(enc_flat, self.config.dim_a, activation_fn=tf.nn.sigmoid)
                a_var = self.config.noise_emission * a_var
                a = simple_sample(a_mu, a_var)
            else:
                a_var = tf.constant(1., dtype=tf.float32, shape=())
                a = a_mu

            a_seq = tf.reshape(a, tf.stack((-1, self.ph_steps, self.config.dim_a)))

        return a_seq, a_mu, a_var

    def decoder(self, a_seq):
        """ Convolutional variational decoder to decode latent code to image reconstruction
        If config.conv == False it is a MLP VAE. If config.use_vae == False it is a normal decoder
        :param a_seq: latent code
        :return: x_hat, x_mu, x_var
        """
        # Create decoder
        if self.config.out_distr == 'bernoulli':
            activation_x_mu = tf.nn.sigmoid
        else:
            activation_x_mu = None

        with tf.variable_scope('vae/decoder'):
            a = tf.reshape(a_seq, (-1, self.config.dim_a))
            if self.config.conv:
                dec_upscale = slim.fully_connected(a, int(np.prod(self.enc_shape)), activation_fn=None)
                dec_upscale = tf.reshape(dec_upscale, [-1] + self.enc_shape)

                dec_hidden = dec_upscale
                for filters in reversed(self.num_filters):
                    dec_hidden = slim.conv2d(dec_hidden, filters * 4, self.config.filter_size,
                                             activation_fn=self.activation_fn)
                    dec_hidden = subpixel_reshape(dec_hidden, 2)
                x_mu = slim.conv2d(dec_hidden, 1, 1, stride=1, activation_fn=activation_x_mu)
                x_var = tf.constant(self.config.noise_pixel_var, dtype=tf.float32, shape=())
            else:
                dec_hidden = slim.repeat(
                    a, self.config.num_layers,
                    slim.fully_connected,
                    self.config.vae_num_units, self.activation_fn
                )

                x_mu = slim.fully_connected(dec_hidden, self.d1 * self.d2, activation_fn=activation_x_mu)
                x_mu = tf.reshape(x_mu, (-1, self.d1, self.d2, 1))
                # x_var is not used for bernoulli outputs. Here we fix the output variance of the Gaussian,
                # we could also learn it globally for each pixel (as we did in the pendulum experiment) or through a
                # neural network.
                x_var = tf.constant(self.config.noise_pixel_var, dtype=tf.float32, shape=())

        if self.config.out_distr == 'bernoulli':
            # For bernoulli we show the probabilities
            x_hat = x_mu
        else:
            x_hat = simple_sample(x_mu, x_var)

        return tf.reshape(x_hat, tf.stack((-1, self.ph_steps, self.d1, self.d2))), x_mu, x_var

    def alpha(self, inputs, state=None, u=None, buffer=None, reuse=None, init_buffer=False, name='alpha'):
        """The dynamics parameter network alpha for mixing transitions in a state space model.
        This function is quite general and supports different architectures (NN, RNN, FIFO queue, learning the inputs)
        Args:
            inputs: tensor to condition mixing vector on
            state: previous state if using RNN network to model alpha
            u: pass-through variable if u is given (learn_u=False)
            buffer: buffer for the FIFO network (used for fifo_size>1)
            reuse: `True` or `None`; if `True`, we go into reuse mode for this scope as
                    well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
            init_buffer: initialize buffer for a_t
            name: name of the scope
        Returns:
            alpha: mixing vector of dimension (batch size, K)
            state: new state
            u: either inferred u from model or pass-through
            buffer: FIFO buffer
        """
        # Increase the number of hidden units if we also learn u (learn_u=True)
        num_units = self.config.alpha_units * 2 if self.config.learn_u else self.config.alpha_units

        # Overwrite input buffer
        if init_buffer:
            buffer = tf.zeros((tf.shape(inputs)[0], self.config.dim_a, self.config.fifo_size), dtype=tf.float32)

        # If K == 1, return inputs
        if self.config.K == 1:
            return tf.ones([self.config.batch_size, self.config.K]), state, u, buffer

        with tf.variable_scope(name, reuse=reuse):
            if self.config.alpha_rnn:
                rnn_cell = BasicLSTMCell(num_units, reuse=reuse)
                output, state = rnn_cell(inputs, state)
            else:
                # Shift buffer
                buffer = tf.concat([buffer[:, :, 1:], tf.expand_dims(inputs, 2)], 2)
                output = slim.repeat(
                    tf.reshape(buffer, (tf.shape(inputs)[0], self.config.dim_a * self.config.fifo_size)),
                    self.config.alpha_layers, slim.fully_connected, num_units,
                    get_activation_fn(self.config.alpha_activation), scope='hidden')

            # Get Alpha as the first part of the output
            alpha = slim.fully_connected(output[:, :self.config.alpha_units],
                                         self.config.K,
                                         activation_fn=tf.nn.softmax,
                                         scope='alpha_var')

            if self.config.learn_u:
                # Get U as the second half of the output
                u = slim.fully_connected(output[:, self.config.alpha_units:],
                                         self.config.dim_u, activation_fn=None, scope='u_var')
        return alpha, state, u, buffer

    def build(self):
        self._build_model()
        self._build_loss()

    def _build_model(self):
        # Encoder q(a|x)
        a_seq, a_mu, a_var = self.encoder(self.x)
        a_vae = a_seq

        # Initial state for the alpha RNN
        dummy_lstm = BasicLSTMCell(self.config.alpha_units * 2 if self.config.learn_u else self.config.alpha_units)
        state_init_rnn = dummy_lstm.zero_state(self.config.batch_size, tf.float32)

        # Initialize Kalman filter (LGSSM)
        self.kf = KalmanFilter(
            dim_z=self.config.dim_z,
            dim_y=self.config.dim_a,
            dim_u=self.config.dim_u,
            dim_k=self.config.K,
            A=self.init_vars['A'],  # state transition function
            B=self.init_vars['B'],  # control matrix
            C=self.init_vars['C'],  # Measurement function
            R=self.init_vars['R'],  # measurement noise
            Q=self.init_vars['Q'],  # process noise
            y=a_seq,  # output
            u=None,
            mask=self.mask,
            mu=self.init_vars['mu'],
            Sigma=self.init_vars['Sigma'],
            y_0=self.init_vars['a_0'],
            alpha=self.alpha,
            state=state_init_rnn
        )

        # Get smoothed posterior over z
        smooth, A, B, C, alpha_plot = self.kf.smooth()

        # Get filtered posterior, used only for imputation plots
        filter, _, _, C_filter, _ = self.kf.filter()

        # Get a from the prior z (for plotting)
        a_mu_pred = tf.matmul(C, tf.expand_dims(smooth[0], 2), transpose_b=True)
        a_mu_pred_seq = tf.reshape(a_mu_pred, tf.stack((-1, self.ph_steps, self.config.dim_a)))
        if self.config.sample_z:
            a_seq = a_mu_pred_seq

        # Decoder p(x|a)
        x_hat, x_mu, x_var = self.decoder(a_seq)

        # Compute variables for generation from the model (for plotting)
        self.n_steps_gen = self.config.n_steps_gen  # We sample for this many iterations

        self.out_gen_det = self.kf.sample_generative_tf(
            smooth,
            self.n_steps_gen,
            deterministic=True,
            init_fixed_steps=self.config.t_init_mask
        )
        self.out_gen = self.kf.sample_generative_tf(
            smooth,
            self.n_steps_gen,
            deterministic=False,
            init_fixed_steps=self.config.t_init_mask
        )
        self.out_gen_det_impute = self.kf.sample_generative_tf(
            smooth,
            self.test_data.timesteps,
            deterministic=True,
            init_fixed_steps=self.config.t_init_mask
        )
        self.out_alpha, _, _, _ = self.alpha(
            self.a_prev,
            state=state_init_rnn, u=None,
            init_buffer=True,
            reuse=True
        )

        # Collect generated model variables
        self.model_vars = dict(
            x_hat=x_hat,
            x_mu=x_mu,
            x_var=x_var,
            a_seq=a_seq,
            a_mu=a_mu,
            a_var=a_var,
            a_vae=a_vae,
            smooth=smooth,
            A=A,
            B=B,
            C=C,
            alpha_plot=alpha_plot,
            a_mu_pred_seq=a_mu_pred_seq,
            filter=filter,
            C_filter=C_filter
        )

    def _build_loss(self):
        # Reshape x for log_likelihood
        x_flat = tf.reshape(self.x, (-1, self.d1 * self.d2))
        x_mu_flat = tf.reshape(self.model_vars['x_mu'], (-1, self.d1 * self.d2))
        mask_flat = tf.reshape(self.mask, (-1,))

        # VAE loss
        elbo_vae, log_px, log_qa = log_likelihood(
            x_mu_flat,
            self.model_vars['x_var'],
            x_flat,
            self.model_vars['a_mu'],
            self.model_vars['a_var'],
            tf.reshape(self.model_vars['a_vae'], (-1, self.config.dim_a)),
            mask_flat,
            self.config)

        # LGSSM loss
        elbo_kf, kf_log_probs, z_smooth = self.kf.get_elbo(
            self.model_vars['smooth'],
            self.model_vars['A'],
            self.model_vars['B'],
            self.model_vars['C']
        )

        # Calc number of batches
        num_batches = self.train_data.sequences // self.config.batch_size

        # Decreasing learning rate
        global_step = tf.contrib.framework.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            self.config.init_lr, global_step,
            self.config.decay_steps * num_batches,
            self.config.decay_rate, staircase=True
        )

        # Combine individual ELBO's
        elbo_tot = self.scale_reconstruction * log_px + elbo_kf - log_qa

        # Collect variables to monitor lb
        self.lb_vars = [elbo_tot, elbo_kf, kf_log_probs, elbo_vae, log_px, log_qa]

        # Get list of vars for gradient computation
        vae_vars = slim.get_variables('vae')
        kf_vars = [self.kf.A, self.kf.B, self.kf.C, self.kf.y_0]
        all_vars = tf.trainable_variables()

        # Define training updates
        self.vae_updates = optimize_loss(
             loss=-elbo_tot,
             global_step=global_step,
             learning_rate=learning_rate,
             optimizer='Adam',
             clip_gradients=self.config.max_grad_norm,
             variables=vae_vars,
             summaries=["gradients", "gradient_norm"],
             name='vae_updates'
        )

        self.vae_kf_updates = optimize_loss(
            loss=-elbo_tot,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer='Adam',
            clip_gradients=self.config.max_grad_norm,
            variables=kf_vars + vae_vars,
            summaries=["gradients", "gradient_norm"],
            name='vae_kf_updates'
        )

        self.all_updates = optimize_loss(
            loss=-elbo_tot,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer='Adam',
            clip_gradients=self.config.max_grad_norm,
            variables=all_vars,
            summaries=["gradients", "gradient_norm"],
            name='all_updates'
        )

        tf.summary.scalar('learningrate', learning_rate)
        tf.summary.scalar('mean_var_qa', tf.reduce_mean(self.model_vars['a_var']))
