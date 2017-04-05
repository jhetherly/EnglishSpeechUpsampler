import os
import csv
import numpy as np
import librosa

"""
The inputs are assumed to already be preprocessed and spliced into 10 second
clips and split into a training, testing, and validation set.
These sets are stored as csv files which are later read in in batches with
functions in this module.

ex:
train_truth_ds_pairs = get_truth_ds_filename_pairs(file_name_lists_dir,
                                                   'train')
br_pairs, wf_pairs = get_bit_rates_and_waveforms(train_truth_ds_pairs[0])
true_br = br_pairs[0]
true_wf_points = wf_pairs[0].size

# ...

for i in range(250):
    batch = randomly_batch(20, train_truth_ds_pairs)
    # ...
"""


def get_bit_rates_and_waveforms(filename_pair):
    """
    given a file name pair this function returns the truth and downsampled bit
    rates as well as the truth and downsampled waveforms
    note that the downsampled bit rate is largely irrelavant as the downsampled
    waveform is upsampled during learning through librosa
    """
    true_waveform, true_br = librosa.load(filename_pair[0], sr=None)
    ds_waveform, ds_br = librosa.load(filename_pair[1], sr=None)
    return [[true_br, ds_br], [true_waveform, ds_waveform]]


def get_truth_ds_filename_pairs(directory, dataset='train'):
    """
    returns a list of file name pairs that represent:
    ["true waveform", "downsampled waveform"]

    directory is a string representing directory of the csv files containing
    the actual file name pairs
    dataset is one of "train," "test," or "validation"
    """
    result = []
    with open(os.path.join(directory,
              '{}_files.csv'.format(dataset)), 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            result.append(row)
    return result


def get_selected_truth_ds_filename_pairs(directory, selected_files_list,
                                         dataset='train'):
    """
    returns a list of selected file name pairs that represent:
    ["true waveform", "downsampled waveform"]

    directory is a string representing directory of the csv files containing
    the actual file name pairs
    selected_files_list is a list of file name tags to match
    dataset is one of "train," "test," or "validation"
    """
    result = []
    with open(os.path.join(directory,
              '{}_files.csv'.format(dataset)), 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            for file_tag in selected_files_list:
                if file_tag in row[0]:
                    result.append(row)
                    break

    return result


def read_file_pair(filename_pair, mono=True):
    """
    given a pair of file names, read in both waveforms and upsample (through
    librosa's default interpolation) the downsampled waveform
    assumes the file name pair is of the form ("original", "downsampled")
    mono selects whether to read in mono or stereo formatted waveforms

    returns a pair of numpy arrays representing the original and upsampled
    waveform
    """
    channel = 1 if mono else 2
    true_waveform, true_br = librosa.load(filename_pair[0], sr=None,
                                          mono=mono)
    ds_waveform, _ = librosa.load(filename_pair[1], sr=true_br, mono=mono)
    # truth, example
    return true_waveform.reshape((-1, channel)), \
        ds_waveform.reshape((-1, channel))


def randomly_batch(batch_size, filename_pairs, mono=True):
    """
    randomly selects batch_size number of samples from a list of file name
    pairs

    returns a tuple of lists containing the true waveforms and downsampled
    waveforms
    """
    batch_truth = []
    batch_ds = []
    indices = range(len(filename_pairs))
    chosen_pairs = np.random.choice(indices,
                                    size=batch_size,
                                    replace=False)
    for pair_index in chosen_pairs:
        truth, ds = read_file_pair(filename_pairs[pair_index], mono)
        batch_truth.append(truth)
        batch_ds.append(ds)

    return batch_truth, batch_ds


def next_batch(batch_size, filename_pairs, mono=True):
    """
    sequentially selects batch_size number of samples from a list of file name
    pairs

    returns a tuple of lists containing the true waveforms and downsampled
    waveforms
    """
    num_pairs = len(filename_pairs)

    for i in range(0, num_pairs, batch_size):
        batch_truth = []
        batch_ds = []
        end_index = i + batch_size
        if end_index >= num_pairs:
            chosen_pairs = filename_pairs[i:]
        else:
            chosen_pairs = filename_pairs[i:end_index]
        for pair in chosen_pairs:
            truth, ds = read_file_pair(pair, mono)
            batch_truth.append(truth)
            batch_ds.append(ds)

        yield batch_truth, batch_ds
