import numpy as np
import json
import librosa
from inputs import get_bit_rates_and_waveforms
from inputs import randomly_batch, next_batch
from inputs import read_file_pair, get_truth_ds_filename_pairs
import tensorflow as tf
from models import deep_residual_network
from losses import mse
from optimizers import make_variable_learning_rate, setup_optimizer

settings_file = 'settings/data_settings.json'
training_settings_file = 'settings/training_settings.json'
model_settings_file = 'settings/model_settings.json'

settings = json.load(open(settings_file))
training_settings = json.load(open(training_settings_file))
model_settings = json.load(open(model_settings_file))

# Constants describing the training process.
# Samples per batch.
BATCH_SIZE = training_settings['batch_size']
# Number of epochs to train.
NUMBER_OF_EPOCHS = training_settings['number_of_epochs']
# Epochs after which learning rate decays.
NUM_EPOCHS_PER_DECAY = training_settings['num_epochs_per_decay']
# Learning rate decay factor.
LEARNING_RATE_DECAY_FACTOR = training_settings['learning_rate_decay_factor']
# Initial learning rate.
INITIAL_LEARNING_RATE = training_settings['initial_learning_rate']

example_number = 0
write_tb = False
file_name_lists_dir = settings['output_dir_name_base']


# ###########
# DATA IMPORT
# ###########

train_truth_ds_pairs = get_truth_ds_filename_pairs(file_name_lists_dir,
                                                   'train')
val_truth_ds_pairs = get_truth_ds_filename_pairs(file_name_lists_dir,
                                                 'validation')

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

train_flag, x, model = deep_residual_network(true_wf.dtype,
                                             true_wf.shape,
                                             **model_settings)

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

model_name = model.name.replace('/', '_').replace(':', '_')
val_loss_file = open('val_loss.txt', 'w')
train_loss_file = open('train_loss.txt', 'w')
epoch_scale = int(SAMPLES_PER_EPOCH/BATCH_SIZE)
for i in range(NUMBER_OF_EPOCHS*epoch_scale):
    is_new_epoch = ((i + 1) % epoch_scale == 0)
    if is_new_epoch:
        epoch_num = int((i + 1) / epoch_scale)
    if is_new_epoch:
        print('Calculating validation loss ({} iterations)'.format(
            len(val_truth_ds_pairs)/BATCH_SIZE))
        total_val_loss = 0
        val_count = 0
        for pair in next_batch(BATCH_SIZE, val_truth_ds_pairs):
            loss_value = sess.run([loss],
                                feed_dict={train_flag: False,
                                           x: pair[1],
                                           y_true: pair[0]}
                                )
            total_val_loss += np.mean(loss_value)
            val_count += 1
        loss_value = total_val_loss/val_count
        val_loss_file.write('{},{}\n'.format(epoch_num,
                                             loss_value))
        print("Epoch {}, Val Loss {}".format(epoch_num, loss_value))
    batch = randomly_batch(BATCH_SIZE, train_truth_ds_pairs)
    if write_tb:
        if is_new_epoch:
            summary, _, loss = sess.run([merged, train_step, loss],
                                        feed_dict={train_flag: True,
                                                   x: batch[1],
                                                   y_true: batch[0]})
            print("Epoch {}, Loss {}".format(epoch_num, loss))
            # train_writer.add_summary(summary, i)
            train_loss_file.write('{}, {}\n'.format(epoch_num, loss))
            if epoch_num % 3 == 0:
                save_path =\
                    saver.save(sess, "aux/model_checkpoints/{}_{}.ckpt".format(
                                        model_name, epoch_num))

    train_step.run(feed_dict={train_flag: True,
                              x: batch[1],
                              y_true: batch[0]},
                   session=sess)
    if (i + 1) % 500 == 0 and not is_new_epoch:
        loss_val = np.mean(sess.run([loss],
                                feed_dict={train_flag: True,
                                           x: batch[1],
                                           y_true: batch[0]}))
        print("Iteration {}, Loss {}".format(i + 1, loss_val))

val_loss_file.close()
train_loss_file.close()
# Save the variables to disk.
save_path = saver.save(sess, "aux/model_checkpoints/{}_final.ckpt".format(
    model_name))
print("Model checkpoints will be saved in file: {}".format(save_path))

truth, example = read_file_pair(val_truth_ds_pairs[1])
y_reco = model.eval(feed_dict={train_flag: False,
                               x: example.reshape(1, -1, 1)},
                    session=sess).flatten()

print('difference between truth and example (first 20 elements)')
print(truth.flatten()[:20] - example.flatten()[:20])
print('difference between truth and reconstruction (first 20 elements)')
print(truth.flatten()[:20] - y_reco[:20])

print('writting output audio files')
librosa.output.write_wav('full_train_validation_true.wav',
                         y=truth.flatten(), sr=true_br)
librosa.output.write_wav('full_train_validation_ds.wav',
                         y=example.flatten(), sr=true_br)
librosa.output.write_wav('full_train_validation_reco.wav',
                         y=y_reco, sr=true_br)

# #############
# #############
