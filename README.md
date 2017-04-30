# EnglishSpeechUpsampler

This repository contains the nessecary Python scripts to train and run a deep
neural network to perform upsampling on an audio waveform.
This upsampling (also known as super-resolution) learns to infer missing high
frequencies in a downsampled audio waveform and is based on the work presented
in [this paper](https://openreview.net/pdf?id=S1gNakBFx).
<!-- Upsample speech audio in wav format using a deep neural network based on the
U-Net architecture found in this
This model is trianed to upsample 4 kbps audio up to 16 kbps.
The training set is a collection of TED talks found
[here](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus). -->

## Overview

Audio super-resolution aims to reconstruct a high-resolution audio waveform
given a lower-resolution waveform as input.
There are several potential applications for this type of upsampling in such
areas as streaming audio and audio restoration.
A non-deep learning solution is to use a database of audio clips to fill in
the missing frequencies in the downsampled waveform using a similarity metric
(see [this](http://ieeexplore.ieee.org/abstract/document/7251945) and
[this](http://ieeexplore.ieee.org/document/7336890) paper).
However, there is recent interest in using deep neural networks to accomplish
this upsampling.

## Dataset \& Preprocessing

There are a variety of domains where audio upsampling is useful.
Since I focused on a potential voice-over-IP application, the dataset I chose
for this repository is a collection of TED talks about 35 GB in size found
[here](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus).
Each talk is located in separate files with bit rates of 16 kbps which is
considered high quality for speech audio.
This dataset contains primarily well-articulated English speech in front an
audience from a variety of speakers.
These qualities about the TED talks are an approximation to what one may expect
during a voice-over-IP conversation.

![Preprocessing Workflow](images/Preprocessing_flow.png)
The steps I use during preprocessing are outlined in the above figure.
I start by trimming the first and last 30 seconds from each file to remove the
TED logo.
I then split the files into 2 second clips and create a separate, 4x
downsampled set of clips at 4 kbps along with a set at the original 16 kbps.

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

1. Download and unzip the
[TEDLIUM](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus) datase.
2. Configure the [data_settings](settings/data_settings.json) JSON file to
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
found in the [training_settings](settings/training_settings.json) JSON file.
Several
aspects of training including the learning rate schedule and batch size. Model
parameters are found in the [model_settings](settings/model_settings.json) JSON
file.
7. After running the training (which likely take several days), the
[upsample_audio_file](upsample_audio_file.py) script can be used to upsample
a WAV formatted audio file from 4 kbps to 16 kbps. The settings for this script
are found in the [upsampling_settings](settings/upsampling_settings.json) JSON file.

## Requirements and Dependencies

The following packages are required (the version numbers that have been tested
are given for reference):

* Python 2.7 or 3.6
* Tensorflow 1.0.1
* Numpy 1.12.1
* Librosa 0.5.0
* tqdm 4.11.2 (only for preprocessing training data)
* Sox 1.2.7 (only for preprocessing training data)
