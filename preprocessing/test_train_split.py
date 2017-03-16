import os
import csv
import numpy as np

validation_fraction = 0.2
test_fraction = 0.2
file_dir_base = '/home/paperspace/Documents/TEDLIUM/TEDLIUM_release2' +\
                '/preprocessed/'
output_dir = '/home/paperspace/Documents/EnglishSpeechUpsampler/aux'

np.random.seed(0)


def write_csv(filename, pairs):
    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        for pair in pairs:
            spamwriter.writerow(pair)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

truth_ds_pairs = []
file_dir_truth = os.path.join(file_dir_base, 'splices')
file_dir_ds = os.path.join(file_dir_base, 'downsampled_splices')

for filename in os.listdir(file_dir_truth):
    truth_input_filename = os.path.join(file_dir_truth, filename)
    ds_input_filename = os.path.join(file_dir_ds, filename)
    if not os.path.isfile(truth_input_filename) or not \
            os.path.isfile(ds_input_filename):
        continue
    truth_ds_pairs.append([truth_input_filename, ds_input_filename])

np.random.shuffle(truth_ds_pairs)

validation_start_index = 0
validation_end_index = validation_start_index +\
    int(len(truth_ds_pairs)*validation_fraction)
test_start_index = validation_end_index
test_end_index = test_start_index +\
    int(len(truth_ds_pairs)*validation_fraction)
train_start_index = test_end_index

validation_truth_ds_pairs =\
    truth_ds_pairs[validation_start_index:validation_end_index]
write_csv(os.path.join(output_dir, 'validation_files.csv'),
          validation_truth_ds_pairs)

test_truth_ds_pairs = truth_ds_pairs[test_start_index:test_end_index]
write_csv(os.path.join(output_dir, 'test_files.csv'), test_truth_ds_pairs)

train_truth_ds_pairs = truth_ds_pairs[train_start_index:]
write_csv(os.path.join(output_dir, 'train_files.csv'), train_truth_ds_pairs)
