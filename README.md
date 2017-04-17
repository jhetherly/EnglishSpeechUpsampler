# EnglishSpeechUpsampler

Upsample speech audio in wav format using a deep neural network based on the
U-Net architecture found in this
[paper](https://openreview.net/pdf?id=S1gNakBFx).
This model is trianed to upsample 4 kbps audio up to 16 kbps.
The training set is a collection of TED talks found
[here](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus).

## Installation Instructions

First, make sure the [required software](##Requirements-and-Dependencies) is
installed.

## Requirements and Dependencies

The following packages are required and the version numbers that have been
tested are given for reference.

* Python 2.7 or 3.6
* Tensorflow 1.0.1
* Numpy 1.12.1
* Librosa 0.5.0
* tqdm 4.11.2 (only for preprocessing training data)
* Sox 1.2.7 (only for preprocessing training data)
