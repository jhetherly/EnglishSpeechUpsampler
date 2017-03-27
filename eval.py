import os
import numpy as np
import librosa
from inputs import get_bit_rates_and_waveforms, get_truth_ds_filename_pairs
from inputs import read_file_pair, next_batch
import tensorflow as tf
from models import deep_residual_network

# Constants describing the training process.
BATCH_SIZE = 10

waveform_reduction_factor = 1
file_name_lists_dir = '/home/paperspace/Documents/EnglishSpeechUpsampler/aux'
model_checkpoint_file_name = os.path.join(file_name_lists_dir,
                                    'model_checkpoints/' +\
                                    'deep_residual_deep_residual_0_final.ckpt')


test_truth_ds_pairs = get_truth_ds_filename_pairs(file_name_lists_dir,
                                                  'test')

br_pairs, wf_pairs = get_bit_rates_and_waveforms(test_truth_ds_pairs[0])
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
                                             tensorboard_output=False)

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


# Add ops to restore all the variables.
saver = tf.train.Saver()

# create session
sess = tf.Session()

# restore model from checkpoint file
saver.restore(sess, model_checkpoint_file_name)


# #############
# TRAINING LOOP
# #############

test_loss_file = open('test_loss.txt', 'w')
count = 0
for pair in next_batch(BATCH_SIZE, test_truth_ds_pairs):
    loss_test = sess.run([waveform_mse],
                         feed_dict={train_flag: False,
                                    x: pair[1],
                                    y_true: pair[0]}
                         )
    test_loss_file.write('{}\n'.format(np.mean(loss_test)))
    print("Iteration {}, Test Loss {}".format((count + 1), loss_test))
    count += 1
    if count >= 3:
        break

test_loss_file.close()

truth, example = read_file_pair(test_truth_ds_pairs[0])
y_reco = model.eval(feed_dict={train_flag: False,
                               x: example.reshape(1, -1, 1)},
                    session=sess).flatten()

print('difference between truth and example (first 20 elements)')
print(truth.flatten()[:20] - example.flatten()[:20])
print('difference between truth and reconstruction (first 20 elements)')
print(truth.flatten()[:20] - y_reco[:20])

# if waveform_reduction_factor == 1:
print('writting output audio files')
librosa.output.write_wav('full_train_test_true.wav',
                         y=truth.flatten(), sr=true_br)
librosa.output.write_wav('full_train_test_ds.wav',
                         y=example.flatten(), sr=true_br)
librosa.output.write_wav('full_train_test_reco.wav',
                         y=y_reco, sr=true_br)
# #############
# #############
