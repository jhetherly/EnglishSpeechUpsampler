import os
import numpy as np
import librosa
# import sox
import tensorflow as tf
from models import deep_residual_network

KBPS = 16000
SECONDS_PER_INPUT = 10  # must match restored model
# SECONDS_PER_INPUT = 2
INPUT_SIZE = KBPS*SECONDS_PER_INPUT
OUTPUT_LENGTH = 20*KBPS  # in bits
DOWNSAMPLE_FACTOR = 4  # in kbps
BEGIN_OFFSET = 30  # in seconds
END_OFFSET = -30  # in seconds

# TODO find a way to convert to wav format internally
source_dir = '/home/paperspace/Documents/'
file_name_base = 'GeorgeAyittey_2007G.wav'
file_name = os.path.join(source_dir, file_name_base)
# file_name = '/home/paperspace/Documents/TEDLIUM/TEDLIUM_release2/' +\
#             'train/sph/GeorgeAyittey_2007G.sph'

file_name_lists_dir = '/home/paperspace/Documents/EnglishSpeechUpsampler/aux'
model_checkpoint_file_name = os.path.join(file_name_lists_dir,
                                    'model_checkpoints/' +\
                                    'deep_residual_deep_residual_0_final.ckpt')


true_wf, true_br = librosa.load(file_name, sr=None, mono=True)
ds_wf, ds_br = librosa.load(file_name, sr=int(true_br/DOWNSAMPLE_FACTOR),
                            mono=True)
ds_wf = librosa.core.resample(ds_wf, ds_br, true_br)

# trim waveforms
true_wf = true_wf[BEGIN_OFFSET*true_br:END_OFFSET*true_br]
ds_wf = ds_wf[BEGIN_OFFSET*true_br:END_OFFSET*true_br]
true_wf = true_wf[:int(true_wf.size/INPUT_SIZE)*INPUT_SIZE]
ds_wf = ds_wf[:int(ds_wf.size/INPUT_SIZE)*INPUT_SIZE]
number_of_reco_iterations = int(ds_wf.size/INPUT_SIZE)


# ################
# MODEL DEFINITION
# ################

train_flag, x, model = deep_residual_network(ds_wf.dtype, (INPUT_SIZE, 1),
                                             tensorboard_output=False)

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
                         y=true_wf.flatten()[:OUTPUT_LENGTH], sr=true_br)
librosa.output.write_wav(os.path.join(source_dir, 'ds_' + file_name_base),
                         y=ds_wf.flatten()[:OUTPUT_LENGTH], sr=true_br)
librosa.output.write_wav(os.path.join(source_dir, 'reco_' + file_name_base),
                         y=reco_wf.flatten()[:OUTPUT_LENGTH], sr=true_br)

# ###################
# ###################
