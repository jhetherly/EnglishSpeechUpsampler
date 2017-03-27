import numpy as np
import librosa
from inputs import get_bit_rates_and_waveforms, get_truth_ds_filename_pairs
from inputs import read_file_pair, randomly_batch
import tensorflow as tf
from models import deep_residual_network

# Constants describing the training process.
BATCH_SIZE = 10
NUM_EPOCHS_PER_DECAY = 500.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

waveform_reduction_factor = 1
write_tb = True
file_name_lists_dir = '/home/paperspace/Documents/EnglishSpeechUpsampler/aux'


train_truth_ds_pairs = get_truth_ds_filename_pairs(file_name_lists_dir,
                                                   'train')
val_truth_ds_pairs = get_truth_ds_filename_pairs(file_name_lists_dir,
                                                 'validation')

br_pairs, wf_pairs = get_bit_rates_and_waveforms(train_truth_ds_pairs[0])
true_br = br_pairs[0]
true_wf = wf_pairs[0]
waveform_max = int(true_wf.size/waveform_reduction_factor)
# true_wf = true_wf[::waveform_reduction_factor]
true_wf = true_wf[:waveform_max]
# reshape for mono waveforms
true_wf = true_wf.reshape((-1, 1))


# ################
# MODEL DEFINITION
# ################

bits_per_second = true_wf.size/10

train_flag, x, model = deep_residual_network(true_wf.dtype, true_wf.shape,
                                             tensorboard_output=write_tb)

# placeholder for the truth label
y_true = tf.placeholder(true_wf.dtype,
                        shape=x.get_shape())

# ################
# ################


# #############
# LOSS FUNCTION
# #############

with tf.name_scope('waveform_mse'):
    waveform_mse = tf.reduce_mean(tf.square(tf.subtract(y_true, model)))
tf.summary.scalar('waveform_mse', waveform_mse)

# #############
# #############


# ####################
# OPTIMIZATION ROUTINE
# ####################

# Variables that affect learning rate.
num_batches_per_epoch = 1
decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

# Decay the learning rate exponentially based on the number of steps.
global_step = tf.Variable(0, trainable=False)
with tf.name_scope('learning_rate'):
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
tf.summary.scalar('learning_rate', lr)


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step
    # (for batch normalization)
    with tf.name_scope('train'):
        # train_step = tf.train.RMSPropOptimizer(lr).minimize(waveform_mse,
        #                                                     global_step=global_step
        #                                                     )
        # train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(waveform_mse)
        # train_step = tf.train.AdamOptimizer(1e-4,
        #                                     epsilon=1e-01).minimize(waveform_mse)
        train_step = tf.train.AdamOptimizer(1e-5,
                                        epsilon=1e-08).minimize(waveform_mse)
        # train_step = tf.train.AdagradOptimizer(1e-3).minimize(waveform_mse)

# ####################
# ####################


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# create session
sess = tf.Session()

# initialize tensorboard file writers
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('aux/tensorboard/train',
                                     sess.graph)
# validation_writer = tf.summary.FileWriter('aux/tensorboard/validation')

# initialize the variables for the session
sess.run(tf.global_variables_initializer())

# #############
# TRAINING LOOP
# #############

model_name = model.name.replace('/', '_').replace(':', '_')
val_loss_file = open('val_loss.txt', 'w')
train_loss_file = open('train_loss.txt', 'w')
for i in range(24000):
    batch = randomly_batch(BATCH_SIZE, train_truth_ds_pairs)
    if (i + 1) % 500 == 0 or i == 0:
        vbatch = randomly_batch(BATCH_SIZE, val_truth_ds_pairs)
        loss_val = sess.run([waveform_mse],
                            feed_dict={train_flag: False,
                                       x: vbatch[1],
                                       y_true: vbatch[0]}
                            )
        val_loss_file.write('{}\n'.format(np.mean(loss_val)))
        print("Iteration {}, Val Loss {}".format((i + 1), np.mean(loss_val)))
    if write_tb:
        if (i + 1) % 500 == 0 or i == 0:
            summary, _, loss = sess.run([merged, train_step, waveform_mse],
                                        feed_dict={train_flag: True,
                                                   x: batch[1],
                                                   y_true: batch[0]})
            train_writer.add_summary(summary, i)
            train_loss_file.write('{}\n'.format(loss))
            save_path = saver.save(sess,
                                   "aux/model_checkpoints/{}_{}.ckpt".format(
                                        model_name, i))
            print("Iteration {}, Loss {}".format((i + 1), loss))

    train_step.run(feed_dict={train_flag: True,
                              x: batch[1],
                              y_true: batch[0]},
                   session=sess)

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

# if waveform_reduction_factor == 1:
print('writting output audio files')
librosa.output.write_wav('full_train_validation_true.wav',
                         y=truth.flatten(), sr=true_br)
librosa.output.write_wav('full_train_validation_ds.wav',
                         y=example.flatten(), sr=true_br)
librosa.output.write_wav('full_train_validation_reco.wav',
                         y=y_reco, sr=true_br)
# #############
# #############
