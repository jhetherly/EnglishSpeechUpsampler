import numpy as np
import json
import librosa
from inputs import get_bit_rates_and_waveforms
from inputs import randomly_batch, next_batch
from inputs import read_file_pair, gather_all_files_by_tags
import tensorflow as tf
from models import deep_residual_network
from losses import mse
from optimizers import make_variable_learning_rate, setup_optimizer

settings_file = 'preprocessing/data_settings.json'

settings = json.load(open(settings_file))

# Constants describing the training process.
# BATCH_SIZE = 64                     # Samples per batch
BATCH_SIZE = 8                     # Samples per batch
# NUMBER_OF_EPOCHS = 1800             # Number of epochs to train
NUMBER_OF_EPOCHS = 300              # Number of epochs to train
NUM_EPOCHS_PER_DECAY = 50           # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1    # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

example_number = 0
write_tb = False
file_name_lists_dir = settings['output_dir_name_base']
# selected_files = ['BillGates_2010']
selected_files = ['RobertGupta_2010U']


# ###########
# DATA IMPORT
# ###########

train_truth_ds_pairs = gather_all_files_by_tags(file_name_lists_dir,
                                                selected_files)

br_pairs, wf_pairs = get_bit_rates_and_waveforms(train_truth_ds_pairs[0])
true_br = br_pairs[0]
true_wf = wf_pairs[0]
# reshape for mono waveforms
true_wf = true_wf.reshape((-1, 1))

SAMPLES_PER_EPOCH = len(train_truth_ds_pairs)
print('Number of epochs: {}'.format(NUMBER_OF_EPOCHS))
print('Samples per epoch: {}'.format(SAMPLES_PER_EPOCH))
print('Batch size: {}'.format(BATCH_SIZE))

# ###########
# ###########


# ################
# MODEL DEFINITION
# ################

train_flag, x, model = deep_residual_network(true_wf.dtype, true_wf.shape)

# placeholder for the true waveform
y_true = tf.placeholder(true_wf.dtype,
                        shape=x.get_shape())

# ################
# ################


# #############
# LOSS FUNCTION
# #############

loss = mse('waveform_loss', y_true, model)

# #############
# #############


# ####################
# OPTIMIZATION ROUTINE
# ####################

# Variable that affect learning rate.
num_batches_per_epoch = float(SAMPLES_PER_EPOCH)/BATCH_SIZE
decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

# Decay the learning rate based on the number of steps.
lr, global_step = make_variable_learning_rate(INITIAL_LEARNING_RATE,
                                              decay_steps,
                                              LEARNING_RATE_DECAY_FACTOR,
                                              False)

# lr = 1e-4
# min_args = {}
min_args = {'global_step': global_step}
# tf.train.RMSPropOptimizer, tf.train.GradientDescentOptimizer,
# tf.train.AdamOptimizer, tf.train.AdagradOptimizer
train_step = setup_optimizer(lr, loss, tf.train.AdamOptimizer,
                             using_batch_norm=True,
                             min_args=min_args)

# ####################
# ####################


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# create session
sess = tf.Session()

# initialize tensorboard file writers
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('aux/tensorboard/overtrain',
                                     sess.graph)

# initialize the variables for the session
sess.run(tf.global_variables_initializer())


# #############
# TRAINING LOOP
# #############

# Calulate loss on training sample
example_loss = 0.0
example_loss_count = 0
for truth, example in next_batch(1, train_truth_ds_pairs):
    example_loss += np.mean((truth[0].flatten()-example[0].flatten())**2)
    example_loss_count += 1
example_loss = example_loss/float(example_loss_count)

print('loss score of example {}'.format(example_loss))

# training loop
train_loss_file = open('overtrain_loss.txt', 'w')
for i in range(NUMBER_OF_EPOCHS):
    for pair in next_batch(BATCH_SIZE, train_truth_ds_pairs):
        train_step.run(feed_dict={train_flag: True,
                                  x: pair[1],
                                  y_true: pair[0]},
                       session=sess)
    batch = randomly_batch(BATCH_SIZE, train_truth_ds_pairs)
    loss_val = loss.eval(
        feed_dict={train_flag: False,
                   x: batch[1],
                   y_true: batch[0]},
        session=sess)
    print("Epoch {}, Loss {}".format((i + 1), loss_val))
    train_loss_file.write('{}\n'.format(loss_val))
    if write_tb:
        summary = sess.run([merged],
                           feed_dict={train_flag: True,
                                      x: batch[1],
                                      y_true: batch[0]})
        train_writer.add_summary(summary, i)

save_path = saver.save(sess, "aux/model_checkpoints/overtrain_final.ckpt")
print("Model checkpoints will be saved in file: {}".format(save_path))
train_loss_file.close()

truth, example = read_file_pair(train_truth_ds_pairs[example_number])

y_reco = model.eval(feed_dict={train_flag: True,
                               x: example.reshape(1, -1, 1)},
                    session=sess).flatten()

print('difference between truth and example (first 20 elements)')
print(truth.flatten()[:20] - example.flatten()[:20])
print('difference between truth and reconstruction (first 20 elements)')
print(truth.flatten()[:20] - y_reco[:20])

print('writting output audio files')
librosa.output.write_wav('overtrain_true.wav',
                         y=truth.flatten(), sr=true_br)
librosa.output.write_wav('overtrain_ds.wav',
                         y=example.flatten(), sr=true_br)
librosa.output.write_wav('overtrain_reco.wav',
                         y=y_reco, sr=true_br)
# #############
# #############
