import time

# VAE
dataset = 'box'
out_distr = 'bernoulli'
conv = True
use_vae = True
activation = 'relu'
filter_size = 3
num_filters = [32, 32, 32]
vae_num_units = 25
num_layers = 2
noise_pixel_var = 0.1
ll_keep_prob = 1.0


# LGSSM
dim_a = 2
dim_z = 4
dim_u = 1
K = 3
noise_emission = 0.03
noise_transition = 0.08
init_cov = 20.0

# Parameter network
alpha_rnn = True
alpha_units = 50
alpha_layers = 2
alpha_activation = 'relu'
fifo_size = 1
learn_u = False

# Training
batch_size = 32
init_lr = 0.007
init_kf_matrices = 0.05
max_grad_norm = 150.0
scale_reconstruction = 0.3
only_vae_epochs = 0
kf_update_steps = 10
decay_rate = 0.85
decay_steps = 20
sample_z = False
num_epochs = 80
train_miss_prob = 0.0
t_init_train_miss = 3

# Utils
gpu = ''
reload_model = ''

# Logs/Plotting
run_name = time.strftime('%Y%m%d%H%M%S', time.localtime())
log_dir = 'logs'
display_step = 1
generate_step = 20
n_steps_gen = 80
t_init_mask = 4
t_steps_mask = 12
