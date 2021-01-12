import time
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import numpy as np

from models import KVAE

from utils.nn import (
    norm_rmse
)

from utils.movie import (
    save_frames,
    save_movies_to_frame,
    save_true_generated_frames
)
from utils.plotting import (
    plot_auxiliary,
    plot_alpha_grid,
    plot_ball_trajectories,
    plot_ball_trajectories_comparison,
    plot_ball_and_alpha
)

mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14


class KVAETrainerStdOut:

    def print_elbo(self, elbo_tot, mean_kf_log_probs, elbo_vae, start):
        print("-- TEST, ELBO %.2f, log_probs [%.2f, %.2f, %.2f, %.2f], elbo_vae %.2f, took %.2fs" % (
            np.mean(elbo_tot),
            mean_kf_log_probs[0],
            mean_kf_log_probs[1],
            mean_kf_log_probs[2],
            mean_kf_log_probs[3],
            np.mean(elbo_vae),
            time.time() - start
        ))

    def print_imputation_title(self, mask_type, v):
        if mask_type == 'missing_planning':
            print("--- Imputation planning, t_steps_mask %s" % v)
        elif mask_type == 'missing_random':
            print("--- Imputation random, drop_prob %s" % v)

    def print_imputation_time(self, start):
        print('Imputation plot  took %.2fs' % (time.time()-start))

    def print_hamming_unobs(self, ham_unobs, hamming_baseline):
        print("Hamming distance. x_imputed: %.5f, x_filtered: %.5f, x_gen_det: %.5f, baseline: %.5f. " % (
            ham_unobs['smooth'],
            ham_unobs['filt'],
            ham_unobs['gen'],
            hamming_baseline
        ))

    def print_norm_rmse(self, norm_rmse_a_imputed, norm_rmse_a_gen_det):
        print("Normalized RMSE. a_imputed: %.3f, a_gen_det: %.3f" % (
            norm_rmse_a_imputed,
            norm_rmse_a_gen_det
        ))


class KVAETrainer:

    def __init__(self, session, config):
        self._session = session
        self._config = config
        self._model = KVAE(config)
        self._stdout = KVAETrainerStdOut()
        self._saver = tf.train.Saver()

        self.model.build()
        self.initialize_variables()

    @property
    def session(self):
        return self._session

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    @property
    def saver(self):
        return self._saver

    @property
    def model_vars(self):
        return self._model.model_vars

    @property
    def stdout(self):
        return self._stdout

    def initialize_variables(self):
        """ Initialize variables or load saved model
        :return: self
        """
        # Initialize or reload variables
        if self.config.reload_model != '':
            print("Restoring model in %s" % self.config.reload_model)
            self.saver.restore(self.session, self.config.reload_model)
        else:
            self.session.run(tf.global_variables_initializer())

    def train(self):
        """ Train model given parameters in self.config
        returns
        -------------
        imputation error on test set
        """
        writer = tf.summary.FileWriter(self.config.log_dir, self.session.graph)
        num_batches = self.train_data.sequences // self.config.batch_size

        # This code supports training with missing data (if train_miss_prob > 0.0)
        mask_train = np.ones(
            (num_batches, self.config.batch_size, self.train_data.timesteps),
            dtype=np.float32
        )

        if self.config.train_miss_prob > 0.0:
            # Always use the same mask for each sequence during training
            for j in range(num_batches):
                mask_train[j] = self.mask_impute_random(
                    t_init_mask=self.config.t_init_train_miss,
                    drop_prob=self.config.train_miss_prob
                )

        all_summaries = tf.summary.merge_all()

        for epoch in range(self.config.num_epochs):
            elbo_tot = []
            elbo_kf = []
            kf_log_probs = []
            elbo_vae = []
            log_px = []
            log_qa = []
            time_epoch_start = time.time()

            for i in range(num_batches):
                slc = slice(i * self.config.batch_size, (i + 1) * self.config.batch_size)
                feed_dict = {
                    self.x: self.train_data.images[slc],
                    self.kf.u: self.train_data.controls[slc],
                    self.mask: mask_train[i],
                    self.ph_steps: self.train_data.timesteps,
                    self.scale_reconstruction: self.config.scale_reconstruction
                }

                # Support for different updates schemes. It is beneficial to achieve better convergence not to train
                # alpha from the beginning
                if epoch < self.config.only_vae_epochs:
                    self.session.run(self.model.vae_updates, feed_dict)
                elif epoch < self.config.only_vae_epochs + self.config.kf_update_steps:
                    self.session.run(self.model.vae_kf_updates, feed_dict)
                else:
                    self.session.run(self.model.all_updates, feed_dict)

                # Bookkeeping.
                _elbo_tot, _elbo_kf, _kf_log_probs, _elbo_vae, _log_px, _log_qa = self.session.run(
                    self.model.lb_vars,
                    feed_dict
                )
                elbo_tot.append(_elbo_tot)
                elbo_kf.append(_elbo_kf)
                kf_log_probs.append(_kf_log_probs)
                elbo_vae.append(_elbo_vae)
                log_px.append(_log_px)
                log_qa.append(_log_qa)

            # Write to summary
            summary_train = self.get_summary('train', elbo_tot, elbo_kf, kf_log_probs, elbo_vae, log_px, log_qa)
            writer.add_summary(summary_train, epoch)
            writer.add_summary(self.session.run(all_summaries, feed_dict), epoch)

            if (epoch + 1) % self.config.display_step == 0:
                mean_kf_log_probs = np.mean(kf_log_probs, axis=0)
                print("Epoch %d, ELBO %.2f, log_probs [%.2f, %.2f, %.2f, %.2f], elbo_vae %.2f, took %.2fs" % (
                     epoch,
                     np.mean(elbo_tot),
                     mean_kf_log_probs[0],
                     mean_kf_log_probs[1],
                     mean_kf_log_probs[2],
                     mean_kf_log_probs[3],
                     np.mean(elbo_vae),
                     time.time() - time_epoch_start
                ))

            if (
                epoch == 0 or
                epoch == self.config.num_epochs - 1 or
                (epoch > 0 and (epoch + 1) % self.config.generate_step == 0)
            ):
                # Impute and calculate error
                mask_impute = self.mask_impute_planning(
                    t_init_mask=self.config.t_init_mask,
                    t_steps_mask=self.config.t_steps_mask
                )
                out_res = self.impute(
                    mask_impute,
                    t_init_mask=self.config.t_init_mask,
                    n=epoch
                )

                # Generate sequences for evaluation
                self.generate(n=epoch)

                # Test on previously unseen data
                test_elbo, summary_test = self.test()
                writer.add_summary(summary_test, epoch)

        # Save the last model
        self.saver.save(self.session, self.config.log_dir + '/model.ckpt')
        neg_lower_bound = -np.mean(test_elbo)

        print("Negative lower_bound on the test set: %s" % neg_lower_bound)

        return out_res[0]

    def test(self):
        mask_test = np.ones(
            (self.config.batch_size, self.test_data.timesteps),
            dtype=np.float32
        )

        elbo_tot = []
        elbo_kf = []
        kf_log_probs = []
        elbo_vae = []
        log_px = []
        log_qa = []
        time_test_start = time.time()

        for i in range(self.test_data.sequences // self.config.batch_size):
            slc = slice(i * self.config.batch_size, (i + 1) * self.config.batch_size)
            feed_dict = {
                self.x: self.test_data.images[slc],
                self.kf.u: self.test_data.controls[slc],
                self.mask: mask_test,
                self.ph_steps: self.test_data.timesteps,
                self.scale_reconstruction: 1.0
            }

            # Bookkeeping.
            _elbo_tot, _elbo_kf, _kf_log_probs, _elbo_vae, _log_px, _log_qa  = self.session.run(
                self.model.lb_vars,
                feed_dict
            )
            elbo_tot.append(_elbo_tot)
            elbo_kf.append(_elbo_kf)
            kf_log_probs.append(_kf_log_probs)
            elbo_vae.append(_elbo_vae)
            log_px.append(_log_px)
            log_qa.append(_log_qa)

        # Write to summary
        summary = self.def_summary('test', elbo_tot, elbo_kf, kf_log_probs, elbo_vae, log_px, log_qa)
        mean_kf_log_probs = np.mean(kf_log_probs, axis=0)

        self.stgout.print_elbo(elbo_tot, mean_kf_log_probs, elbo_vae, time_test_start)

        return np.mean(elbo_tot), summary

    def generate(self, idx_batch=0, n=99999):
        """ Sample video deterministic
        """
        # Get initial state z_1
        mask_test = np.ones(
            (self.config.batch_size, self.test_data.timesteps),
            dtype=np.float32
        )
        slc = slice(
            idx_batch * self.config.batch_size,
            (idx_batch + 1) * self.config.batch_size
        )

        smooth_z = self.session.run(
            self.model_vars['smooth'],
            {
                self.x: self.test_data.images[slc],
                self.kf.u: self.test_data.controls[slc],
                self.ph_steps: self.test_data.timesteps,
                self.mask: mask_test
            }
        )

        # Sample deterministic generation
        a_gen_det, _, alpha_gen_det = self.session.run(
            self.out_gen_det,
            {
                self.model_vars['smooth']: smooth_z,
                self.kf.u: np.zeros((self.config.batch_size,self.n_steps_gen, self.config.dim_u)),
                self.ph_steps: self.n_steps_gen
            }
        )

        x_gen_det = self.session.run(
            self.model_vars['x_hat'],
            {
                self.model_vars['a_seq']: a_gen_det,
                self.ph_steps: self.n_steps_gen
            }
        )

        # Save the trajectory of deterministic a (we only plot the first 2 dimensions!) and alpha
        plot_ball_trajectories(
            a_gen_det,
            self.config.log_dir + '/plot_generation_det_%05d.png' % n
        )
        plot_alpha_grid(
            alpha_gen_det,
            self.config.log_dir + '/alpha_generation_det_%05d.png' % n
        )

        # Sample stochastic
        a_gen, _, alpha_gen = self.session.run(self.out_gen, feed_dict)
        x_gen = self.session.run(
            self.model_vars['x_hat'],
            {
                self.model_vars['a_seq']: a_gen,
                self.ph_steps: self.n_steps_gen
            }
        )

        # Save movies
        save_frames(x_gen, self.config.log_dir + '/video_generation_%05d.mp4' % n)
        save_frames(x_gen_det, self.config.log_dir + '/video_generation_det_%05d.mp4' % n)

        # Save stochastic a and alpha
        plot_ball_trajectories(a_gen, self.config.log_dir + '/plot_generation_%05d.png' % n)
        plot_alpha_grid(alpha_gen, self.config.log_dir + '/alpha_generation_%05d.png' % n)

        # Save movie to single frame
        save_movies_to_frame(x_gen_det[:, :20], self.config.log_dir + '/video_generation_image_%05d.png' % n)

        # We can only show the image for alpha when using a simple neural network
        if (
            self.config.dim_a == 2 and
            self.config.fifo_size == 1 and
            self.config.alpha_rnn is False and
            self.config.learn_u is False
        ):
            self.img_alpha_nn(n=n, range_x=(-16, 16), range_y=(-16, 16))

    def impute(self, mask_impute, t_init_mask, idx_batch=0, n=99999, plot=True):
        slc = slice(
            idx_batch * self.config.batch_size,
            (idx_batch + 1) * self.config.batch_size
        )
        feed_dict = {
            self.x: self.test_data.images[slc],
            self.kf.u: self.test_data.controls[slc],
            self.ph_steps: self.test_data.timesteps,
            self.mask: mask_impute
        }

        # Compute reconstructions and imputations (smoothing)
        a_imputed, a_reconstr, x_reconstr, alpha_reconstr, smooth_z, filter_z, C_filter = self.session.run(
            [
                self.model_vars['a_mu_pred_seq'],
                self.model_vars['a_vae'],
                self.model_vars['x_hat'],
                self.model_vars['alpha_plot'],
                self.model_vars['smooth'],
                self.model_vars['filter'],
                self.model_vars['C_filter']
            ],
            feed_dict
        )
        x_imputed = self.session.run(
            self.model_vars['x_hat'],
            {
                self.model_vars['a_seq']: a_imputed,
                self.ph_steps: self.test_data.timesteps
            }
        )
        x_true = feed_dict[self.x]

        # Filtering
        feed_dict = {
            self.model_vars['smooth']: filter_z,
            self.model_vars['C']: C_filter,
            self.ph_steps: self.test_data.timesteps
        }
        a_filtered = self.session.run(
            self.model_vars['a_mu_pred_seq'],
            feed_dict
        )
        x_filtered = self.session.run(
            self.model_vars['x_hat'],
            {
                self.model_vars['a_seq']: a_filtered,
                self.ph_steps: self.test_data.timesteps
            }
        )

        if plot:
            save_frames(x_true, self.config.log_dir + '/video_true.mp4')
            save_frames(x_imputed, self.config.log_dir + '/video_smoothing_%05d.mp4' % n)
            save_frames(x_reconstr, self.config.log_dir + '/video_reconstruction_%05d.mp4' % n)
            save_true_generated_frames(x_true, x_imputed, self.config.log_dir + '/video_true_smooth_%05d.mp4' % n)
            save_true_generated_frames(x_true, x_filtered, self.config.log_dir + '/video_true_filter_%05d.mp4' % n)
            save_true_generated_frames(x_true, x_reconstr, self.config.log_dir + '/video_true_recon_%05d.mp4' % n)

            plot_alpha_grid(alpha_reconstr, self.config.log_dir + '/alpha_reconstr_%05d.png' % n)

            # Plot z_mu
            plot_auxiliary([smooth_z[0]], self.config.log_dir + '/plot_z_mu_smooth_%05d.png' % n)

            if self.config.dim_a == 2:
                # Plot alpha and corresponding trajectory
                plot_ball_and_alpha(
                    alpha_reconstr[14],
                    a_reconstr[14],
                    cmap='Reds',
                    filename=self.config.log_dir + '/alpha_a_recon_%05d.png' % n
                )

        ###### Sample deterministic generation having access to the first t_init_mask frames for comparison
        # Get initial state z_1
        smooth_z_gen = self.session.run(
            self.model_vars['smooth'],
            {
                self.x: self.test_data.images[slc][:, 0: t_init_mask],
                self.kf.u: self.test_data.controls[slc][:, 0: t_init_mask],
                self.ph_steps: t_init_mask,
                self.mask: mask_impute[:, 0: t_init_mask]
            }
        )

        a_gen_det, _, alpha_gen_det = self.session.run(
            self.out_gen_det_impute,
            {
                self.model_vars['smooth']: smooth_z_gen,
                self.kf.u: np.zeros((self.config.batch_size, self.n_steps_gen, self.config.dim_u)),
                self.ph_steps: self.test_data.timesteps
            }
        )

        x_gen_det = self.session.run(
            self.model_vars['x_hat'],
            {
                self.model_vars['a_seq']: a_gen_det,
                self.ph_steps: self.test_data.timesteps
            }
        )

        if plot:
            save_true_generated_frames(
                x_true,
                x_gen_det,
                self.config.log_dir + '/video_true_gen_%05d.mp4' % n
            )
            if self.config.dim_a == 2:
                plot_ball_trajectories_comparison(
                    a_reconstr,
                    a_gen_det,
                    a_imputed,
                    self.config.log_dir + '/plot_imputation_%05d.png' % n,
                    nrows=4,
                    ncols=4,
                    mask=mask_impute
                )
            else:
                plot_auxiliary(
                    [a_reconstr, a_gen_det, a_imputed],
                    self.config.log_dir + '/plot_imputation_%05d.png' % n
                )

        # For a more fair comparison against pure generation only look at time steps with no observed variables
        mask_unobs = mask_impute < 0.5
        x_true_unobs = x_true[mask_unobs]

        # Get hamming distance on unobserved variables
        ham_unobs = dict()
        for key, value in zip(('gen', 'filt', 'smooth'), (x_gen_det, x_filtered, x_imputed)):
            ham_unobs[key] = hamming(x_true_unobs.flatten() > 0.5, value[mask_unobs].flatten() > 0.5)

        # Baseline is considered as the biggest hamming distance between two frames in the data
        hamming_baseline = 0.0
        for i in [0, 3, 6]:
            for j in [9, 12, 15]:
                tmp_dist = hamming((x_true[0, i] > 0.5).flatten(), (x_true[0, j] > 0.5).flatten())
                hamming_baseline = np.max([hamming_baseline, tmp_dist])

        # Return results
        a_reconstr_unobs = a_reconstr[mask_unobs]
        norm_rmse_a_imputed = norm_rmse(a_imputed[mask_unobs], a_reconstr_unobs)
        norm_rmse_a_gen_det = norm_rmse(a_gen_det[mask_unobs], a_reconstr_unobs)

        if plot:
            self.stdout.print_hamming_unobs(ham_unobs, hamming_baseline)
            self.stdout.print_norm_rmse(norm_rmse_a_imputed, norm_rmse_a_gen_det)

        out_res = (
            ham_unobs['smooth'],
            ham_unobs['filt'],
            ham_unobs['gen'],
            hamming_baseline, norm_rmse_a_imputed,
            norm_rmse_a_gen_det
        )
        return out_res

    def img_alpha_nn(
        self,
        range_x=(-30, 30),
        range_y=(-30, 30),
        n_points=50,
        n=99999
    ):
        """ Visualise the output of the dynamics parameter network alpha over _a_ when dim_a == 2 and alpha_rnn=False
        params
        ----------
        range_x: range of first dimension of a
        range_y: range of second dimension of a
        n_points: points to sample
        n: epoch number

        returns
        ----------
        None
        """
        x = np.linspace(range_x[0], range_x[1], n_points)
        y = np.linspace(range_y[0], range_y[1], n_points)
        xv, yv = np.meshgrid(x, y)

        f, ax = plt.subplots(1, self.config.K, figsize=(18, 6))
        for k in range(self.config.K):
            out = np.zeros_like(xv)
            for i in range(n_points):
                for j in range(n_points):
                    a_prev = np.expand_dims(np.array([xv[i, j], yv[i, j]]), 0)
                    alpha_out = self.session.run(self.out_alpha, {self.a_prev: a_prev})
                    out[i, j] = alpha_out[0][k]

            np.save(self.config.log_dir + '/image_alpha_%05d_%d' % (n, k), out)

            ax[k].pcolor(xv, yv, out, cmap='Greys')
            ax[k].set_aspect(1)
            ax[k].set_yticks([])
            ax[k].set_xticks([])

        plt.savefig(
            self.config.log_dir + '/image_alpha_%05d.png' % n,
            format='png',
            bbox_inches='tight',
            dpi=80
        )
        plt.close()

    def mask_impute_planning(self, t_init_mask=4, t_steps_mask=12):
        """ Create mask with missing values in the middle of the sequence
        params
        -------------
        t_init_mask: observed steps in the beginning of the sequence
        t_steps_mask: observed steps in the end

        returns
        -------------
        mask_impute: np.ndarray
        """
        mask_impute = np.ones(
            (self.config.batch_size, self.test_data.timesteps),
            dtype=np.float32
        )
        t_end_mask = t_init_mask + t_steps_mask
        mask_impute[:, t_init_mask: t_end_mask] = 0.0

        return mask_impute

    def mask_impute_random(self, t_init_mask=4, drop_prob=0.5):
        """ Create mask with values missing at random

        params
        -------------
        t_init_mask: observed steps in the beginning of the sequence
        drop_prob: probability of not observing a step

        returns
        -------------
        mask_impute: np.ndarray
        """
        mask_impute = np.ones(
            (self.config.batch_size, self.test_data.timesteps),
            dtype=np.float32
        )

        n_steps = self.test_data.timesteps - t_init_mask

        mask_impute[:, t_init_mask:] = np.random.choice(
            [0, 1],
            size=(self.config.batch_size, n_steps),
            p=[drop_prob, 1.0 - drop_prob]
        )

        return mask_impute

    def impute_all(self, mask_impute, t_init_mask, n=99999, plot=True):
        """ Iterate over batches in the test set
        :param mask_impute: mask to apply
        :param t_init_mask: observed steps in the beginning of the sequence
        :param n: epoch number
        :param plot: Save plots
        :return: average of imputation errors
        """
        results = []
        for i in range(self.test_data.sequences // self.config.batch_size):
            results.append(
                self.impute(
                    mask_impute,
                    t_init_mask=t_init_mask,
                    idx_batch=i,
                    n=n,
                    plot=plot
                )
            )
        return np.array(results).mean(axis=0)

    def imputation_plot(self):
        """ Generate imputation plots for varying levels of observed data
        :return: None
        """
        self._handle_imputation_plot(
            vec=range(1, 17, 2),
            title="missing_planning",
            xlab="Number of unobserved steps"
        )

        self._handle_imputation_plot(
            vec=np.linspace(0.1, 1.0, num=10),
            title="missing_random",
            xlab="Drop probability"
        )

    def _handle_imputation_plot(
        self,
        vec=range(1, 17, 2),
        title="missing_planning",
        xlab="Number of unobserved steps"
    ):
        """ Generate imputation plots for varying levels of observed data
        :param mask_type: str, missing or planning
        :return: None
        """
        time_imput_start = time.time()
        out_res_all = []

        for i, v in enumerate(vec):
            self.stdout.print_imputation_title(title, v)

            mask_impute = self.mask_impute_planning(
                t_init_mask=self.config.t_init_mask,
                t_steps_mask=v
            )
            out_res = self.impute_all(
                mask_impute,
                t_init_mask=self.config.t_init_mask,
                plot=False,
                n=100+i
            )
            out_res_all.append(out_res)

        out_res_all = np.array(out_res_all)
        hamm_x_imputed = out_res_all[:, 0]
        hamm_x_filtered = out_res_all[:, 1]
        baseline = out_res_all[:, 3]

        results = [
            (baseline, 'Baseline'),
            (hamm_x_imputed, 'KVAE smoothing'),
            (hamm_x_filtered, 'KVAE filtering')
        ]

        for dist, label in results:
            if label == 'Baseline':
                linestyle = '--'
            else:
                linestyle = '.-'
            plt.plot(vec, dist, linestyle, linewidth=3, ms=20, label=label)

        plt.xlabel(xlab, fontsize=20)
        plt.ylabel('Hamming distance', fontsize=20)
        plt.legend(fontsize=20, loc=1)
        plt.savefig(self.config.log_dir + '/imputation_%s.png' % title)
        plt.close()

        np.savez(
            self.config.log_dir + '/imputation_results_%s' % title,
            results=results
        )

        self.stdout.print_imputation_time(time_imput_start)

    @staticmethod
    def get_summary(prefix, elbo_tot, elbo_kf, kf_log_probs, elbo_vae, log_px, log_qa):
        """ Add ELBO terms to a TF Summary object for Tensorboard
        """
        mean_kf_log_probs = np.mean(kf_log_probs, axis=0)

        summary = tf.Summary()
        summary.value.add(tag=prefix + '_elbo_tot', simple_value=np.mean(elbo_tot))
        summary.value.add(tag=prefix + '_elbo_kf', simple_value=np.mean(elbo_kf))
        summary.value.add(tag=prefix + '_elbo_vae', simple_value=np.mean(elbo_vae))
        summary.value.add(tag=prefix + '_vae_px', simple_value=np.mean(log_px))
        summary.value.add(tag=prefix + '_vae_qa', simple_value=np.mean(log_qa))
        summary.value.add(tag=prefix + '_kf_transitions', simple_value=mean_kf_log_probs[0])
        summary.value.add(tag=prefix + '_kf_emissions', simple_value=mean_kf_log_probs[1])
        summary.value.add(tag=prefix + '_kf_init', simple_value=mean_kf_log_probs[2])
        summary.value.add(tag=prefix + '_kf_entropy', simple_value=mean_kf_log_probs[3])

        return summary
