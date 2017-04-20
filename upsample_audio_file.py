import os
import json
import numpy as np
import librosa
from inputs import get_bit_rates_and_waveforms
from inputs import get_truth_ds_filename_pairs
import tensorflow as tf
from models import deep_residual_network

data_settings_file = 'settings/data_settings.json'
model_settings_file = 'settings/model_settings.json'
upsampling_settings_file = 'settings/upsampling_settings.json'

data_settings = json.load(open(data_settings_file))
model_settings = json.load(open(model_settings_file))
upsampling_settings = json.load(open(upsampling_settings_file))

file_name_lists_dir = data_settings['output_dir_name_base']

train_truth_ds_pairs = get_truth_ds_filename_pairs(file_name_lists_dir,
                                                   'train')

br_pairs, wf_pairs = get_bit_rates_and_waveforms(train_truth_ds_pairs[0])
true_br = br_pairs[0]
true_wf = wf_pairs[0]
# reshape for mono waveforms
true_wf = true_wf.reshape((-1, 1))

KBPS = true_br
SECONDS_PER_INPUT = data_settings_file['splice_duration']
INPUT_SIZE = KBPS*SECONDS_PER_INPUT
DOWNSAMPLE_FACTOR = KBPS//data_settings_file['downsample_rate']
BEGIN_OFFSET = data_settings_file['start_time']
END_OFFSET = data_settings_file['end_time']

file_name = upsampling_settings['input_file']
source_dir = os.path.split(file_name)[0]
file_name_base = os.path.split(file_name)[1]


model_checkpoint_file_name = upsampling_settings['model_checkpoint_file']


true_wf, true_br = librosa.load(file_name, sr=None, mono=True)
ds_wf, ds_br = librosa.load(file_name, sr=int(true_br/DOWNSAMPLE_FACTOR),
                            mono=True)
ds_wf = librosa.core.resample(ds_wf, ds_br, true_br)

# trim waveforms
if END_OFFSET == 0:
    true_wf = true_wf[BEGIN_OFFSET*true_br:]
    ds_wf = ds_wf[BEGIN_OFFSET*true_br:]
else:
    true_wf = true_wf[BEGIN_OFFSET*true_br:END_OFFSET*true_br]
    ds_wf = ds_wf[BEGIN_OFFSET*true_br:END_OFFSET*true_br]
true_wf = true_wf[:int(true_wf.size/INPUT_SIZE)*INPUT_SIZE]
ds_wf = ds_wf[:int(ds_wf.size/INPUT_SIZE)*INPUT_SIZE]
number_of_reco_iterations = int(ds_wf.size/INPUT_SIZE)


# ################
# MODEL DEFINITION
# ################

train_flag, x, model = deep_residual_network(true_wf.dtype,
                                             true_wf.shape,
                                             **model_settings)

# ################
# ################


# Add ops to restore all the variables.
saver = tf.train.Saver()

# create session
sess = tf.Session()

# restore model from checkpoint file
saver.restore(sess, model_checkpoint_file_name)


# ###################
# RECONSTRUCTION LOOP
# ###################

reco_wf = np.empty(ds_wf.size)
for i in range(number_of_reco_iterations):
    print('Segement {} of {}'.format(i + 1, number_of_reco_iterations))
    example = ds_wf[i*INPUT_SIZE:(i + 1)*INPUT_SIZE]
    reco_wf[i*INPUT_SIZE:(i + 1)*INPUT_SIZE] = \
        model.eval(feed_dict={train_flag: False,
                              x: example.reshape(1, -1, 1)},
                   session=sess).flatten()

librosa.output.write_wav(os.path.join(source_dir, 'true_' + file_name_base),
                         y=true_wf.flatten(), sr=true_br)
librosa.output.write_wav(os.path.join(source_dir, 'ds_' + file_name_base),
                         y=ds_wf.flatten(), sr=true_br)
librosa.output.write_wav(os.path.join(source_dir, 'reco_' + file_name_base),
                         y=reco_wf.flatten(), sr=true_br)

# ###################
# ###################
