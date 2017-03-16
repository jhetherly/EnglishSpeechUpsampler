import os
import shutil
import numpy as np
import sox
import librosa
import tensorflow as tf

input_dir_name_base = '/home/paperspace/Documents' +\
                      '/TEDLIUM/TEDLIUM_release2/{}/sph'
# input_dir_name_dirs = ['dev', 'test', 'train']
input_dir_name_dirs = ['dev']
downsample_rate = 4000  # in bps
duration_chunks = 10  # in seconds
start_time = 30  # in seconds
end_time = -30  # in seconds
tfrecord_output_dir_name = '/home/paperspace/Documents' +\
                           '/TEDLIUM/TEDLIUM_release2/tfrecords'
max_putput_file_size = 10  # in MB


def create_new_output_file(file_count, output_dir_name):
    return tf.python_io.TFRecordWriter(
        os.path.join(output_dir_name,
                     "waveforms_{}.tfrecords".format(file_count + 1)))


def create_float_list_feature(l):
    return tf.train.Feature(float_list=tf.train.FloatList(value=l))


def write_to_records_file(writer, true_waveform, ds_waveform):
    # construct the Example proto boject
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'truth_waveform': create_float_list_feature(true_waveform),
                'ds_waveform': create_float_list_feature(ds_waveform)
            }
        )
    )
    serialized = example.SerializeToString()
    writer.write(serialized)


truth_ds_pairs = []
true_wf_size = None
true_sr = None

spliced_output_dir_name_base = input_dir_name_base.format('tmp')
output_dir_name = os.path.join(spliced_output_dir_name_base, 'splices')
ds_output_dir_name = os.path.join(spliced_output_dir_name_base,
                                  'downsampled_splices')

if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)
if not os.path.exists(ds_output_dir_name):
    os.makedirs(ds_output_dir_name)

print('Will send spliced audio to {}'.format(output_dir_name))
print('Will send spliced and downsampled audio to' +
      ' {}'.format(ds_output_dir_name))

for input_dir_name_dir in input_dir_name_dirs:
    input_dir_name = input_dir_name_base.format(input_dir_name_dir)

    # Loop over all files within the input directory
    for filename in os.listdir(input_dir_name):
        input_filename = os.path.join(input_dir_name, filename)
        if not os.path.isfile(input_filename) or '.sph' not in filename:
            continue
        filename_base = os.path.splitext(filename)[0]

        # This is the total audio track duration less the
        # start and end times
        duration = sox.file_info.duration(input_filename) - \
            (start_time - end_time)

        n_iterations = int(duration/duration_chunks)

        for i in range(n_iterations):
            # create trasnformer
            splice = sox.Transformer()
            splice_and_downsample = sox.Transformer()

            begin = start_time + i*duration_chunks
            end = begin + duration_chunks
            output_filename = '{}_{}-{}.wav'.format(filename_base, begin, end)
            output_filename = os.path.join(output_dir_name, output_filename)
            ds_output_filename = '{}_{}-{}.wav'.format(filename_base,
                                                       begin, end)
            ds_output_filename = os.path.join(ds_output_dir_name,
                                              ds_output_filename)

            splice.trim(begin, end)
            splice_and_downsample.trim(begin, end)
            splice_and_downsample.convert(samplerate=downsample_rate)

            splice.build(input_filename, output_filename)
            splice_and_downsample.build(input_filename, ds_output_filename)

            truth_ds_pairs.append([output_filename, ds_output_filename])
            if true_wf_size is None and true_sr is None:
                true_sr = int(sox.file_info.sample_rate(input_filename))
                true_wf_size = sox.file_info.num_samples(input_filename)
            print('Finished split {} of {} for {}'.format(i + 1, n_iterations,
                                                          filename_base))

np.random.shuffle(truth_ds_pairs)

if not os.path.exists(tfrecord_output_dir_name):
    os.makedirs(tfrecord_output_dir_name)

max_putput_file_size *= 1000000  # convert to actual byte size
current_file_size_estimate = 0
tfrecord_file_count = 0
writer = create_new_output_file(tfrecord_file_count, tfrecord_output_dir_name)
for truth_filename, ds_filename in truth_ds_pairs:
    additional_file_size = os.path.getsize(truth_filename) +\
        os.path.getsize(ds_filename)
    if current_file_size_estimate + additional_file_size >\
            max_putput_file_size:
        writer.close()
        writer = create_new_output_file(tfrecord_file_count,
                                        tfrecord_output_dir_name)
        current_file_size_estimate = 0
        tfrecord_file_count += 1
    current_file_size_estimate += additional_file_size

    true_waveform, true_br = librosa.load(truth_filename, sr=None)
    ds_waveform, ds_br = librosa.load(ds_filename, sr=None)

    write_to_records_file(writer, true_waveform, ds_waveform)

# close last file
writer.close()

# remove temporary spliced files
shutil.rmtree(output_dir_name)
shutil.rmtree(ds_output_dir_name)
