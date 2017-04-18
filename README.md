# EnglishSpeechUpsampler

Upsample speech audio in wav format using a deep neural network based on the
U-Net architecture found in this
[paper](https://openreview.net/pdf?id=S1gNakBFx).
This model is trianed to upsample 4 kbps audio up to 16 kbps.
The training set is a collection of TED talks found
[here](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus).

## Installation Instructions

Make sure the [required software](##requirements-and-dependencies) is
installed.
To compile the custom C++ library that enables fast
[subpixel shuffling](https://arxiv.org/pdf/1609.05158.pdf),
`cd` into the `src` directory and run the
[COMPILE_FROM_BINARY](src/COMPILE_FROM_BINARY.sh) Bash script.
This should be all that is required to run the upsampling script.

## Usage

Since GitHub doesn't allow for files larger than 100 MB, the model must be
retrained in order to perform the upsampling.
The steps to retraining the model are:

1. Download and unzip the [TEDLIUM](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus) datase.
2. Configure the [data_settings](preprocessing/data_settings.json) JSON file to
point to the location of the dataset (`input_dir_name_base`) as well as
specifying the location to store the output (`output_dir_name_base`).
3. This JSON file also contains the duration of the spliced samples which are
also used as the input size to the model (`splice_duration`). Smaller durations
lead to faster model evaluations at the cost of more files being stored on disk.
4. Run the [splice_raw_data](preprocessing/splice_raw_data.py) script from the
preprocessing directory (or run it from any directory as long as the
`splice_settings_file` variable points to the correct JSON file).
5. Next, run the [test_train_split](preprocessing/test_train_split.py) script to
create the CSV files that store which samples are used for training, validation,
and testing.
6. Now that the data is properly preprocessed, the training script
([train.py](train.py)) can be run. The settings for the training script are
found in the JSON file.

## Requirements and Dependencies

The following packages are required and the version numbers that have been
tested are given for reference.

* Python 2.7 or 3.6
* Tensorflow 1.0.1
* Numpy 1.12.1
* Librosa 0.5.0
* tqdm 4.11.2 (only for preprocessing training data)
* Sox 1.2.7 (only for preprocessing training data)
