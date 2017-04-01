import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
import librosa
import librosa.display


# original_sample = 'full_train_test_true.wav'
# ds_sample = 'full_train_test_ds.wav'
# reco_sample = 'full_train_test_reco.wav'
# output_file_name = 'full_train_test_spec_comp.pdf'
# original_sample = 'full_train_validation_true.wav'
# ds_sample = 'full_train_validation_ds.wav'
# reco_sample = 'full_train_validation_reco.wav'
# output_file_name = 'full_train_validation_spec_comp.pdf'
original_sample = 'overtrain_true.wav'
ds_sample = 'overtrain_ds.wav'
reco_sample = 'overtrain_reco.wav'
output_file_name = 'overtrain_spec_comp.pdf'
# original_sample = 'loss_function_comparison/gm_overtrain_true.wav'
# ds_sample = 'loss_function_comparison/gm_overtrain_ds.wav'
# reco_sample = 'loss_function_comparison/gm_overtrain_reco.wav'
# output_file_name = 'loss_function_comparison/gm_overtrain_spec_comp.pdf'
# n_fft = 512
n_fft = 256


y_true, sr_true = librosa.load(original_sample, sr=None)
y_ds, sr_ds = librosa.load(ds_sample, sr=None)
y_reco, sr_reco = librosa.load(reco_sample, sr=None)


def compute_signal_to_noise(truth, reco):
    return 10.*np.log10(np.sqrt(np.sum(truth**2))/np.sqrt(
        np.sum((truth - reco)**2)))


def plot_all(true_spectrogram, ds_spectrogram, reco_spectrogram,
             true_waveform, ds_waveform, reco_waveform,
             true_sr, ds_sr, reco_sr, ofile, n_fft):
    max_frame = 200
    plt.figure(figsize=(8, 6))

    if not (true_sr == ds_sr == reco_sr):
        print('Warning: time axis on waveform plots will be meaningless')

    # compute dB-scale magnitudes
    true_dB = librosa.amplitude_to_db(true_spectrogram, ref=np.max)
    ds_dB = librosa.amplitude_to_db(ds_spectrogram, ref=np.max)
    reco_dB = librosa.amplitude_to_db(reco_spectrogram, ref=np.max)

    # compute LSD
    true_X = np.log10(np.abs(true_spectrogram)**2)
    ds_X = np.log10(np.abs(ds_spectrogram)**2)
    reco_X = np.log10(np.abs(reco_spectrogram)**2)
    ds_X_diff_squared = (true_X - ds_X)**2
    reco_X_diff_squared = (true_X - reco_X)**2
    ds_lsd = np.mean(np.sqrt(np.mean(ds_X_diff_squared, axis=0)))
    reco_lsd = np.mean(np.sqrt(np.mean(reco_X_diff_squared, axis=0)))

    # spectrogram plots
    cmap = 'inferno'
    # cmap = 'magma'
    # cmap = 'plasma'
    plt.subplot(3, 2, 1)
    plt.title('True Spectrum (dB)')
    fig = librosa.display.specshow(librosa.amplitude_to_db(true_dB,
                                                           ref=np.max),
                                   sr=true_sr, y_axis='hz', x_axis='time',
                                   hop_length=n_fft/4, cmap=cmap)
    fig.axes.set_xticklabels([])
    plt.xlabel('')
    plt.ylabel('frequency (Hz)')

    ax = plt.subplot(3, 2, 3)
    plt.title('Downsampled Spectrum (dB)')
    fig = librosa.display.specshow(librosa.amplitude_to_db(ds_dB,
                                                           ref=np.max),
                                   sr=ds_sr, y_axis='hz', x_axis='time',
                                   hop_length=n_fft/4, cmap=cmap)
    fig.axes.set_xticklabels([])
    plt.xlabel('')
    plt.ylabel('frequency (Hz)')
    ax.text(0.65, 0.25, r'LSD={:.2}'.format(ds_lsd),
            color='black', fontsize=13, transform=ax.transAxes)

    ax = plt.subplot(3, 2, 5)
    plt.title('Reconstructed Spectrum (dB)')
    fig = librosa.display.specshow(librosa.amplitude_to_db(reco_dB,
                                                           ref=np.max),
                                   sr=reco_sr, y_axis='hz', x_axis='time',
                                   hop_length=n_fft/4, cmap=cmap)
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    ax.text(0.65, 0.25, r'LSD={:.2}'.format(reco_lsd),
            color='black', fontsize=13, transform=ax.transAxes)

    # compute SNR for waveform plots
    ds_snr = compute_signal_to_noise(true_waveform, ds_waveform)
    reco_snr = compute_signal_to_noise(true_waveform, reco_waveform)

    # waveform plots
    ax = plt.subplot(3, 2, 2)
    ax.set_xticklabels([])
    true_time = np.arange(max_frame, dtype=np.float)/float(true_sr)
    plt.title('True Waveform (16 kbps)')
    fig = plt.plot(true_time, true_waveform[:max_frame])
    plt.ylabel('amplitude')

    ax = plt.subplot(3, 2, 4)
    ax.set_xticklabels([])
    ds_time = np.arange(max_frame, dtype=np.float)/float(ds_sr)
    plt.title('Downsampled Waveform (4 kbps)')
    fig = plt.plot(ds_time, ds_waveform[:max_frame])
    plt.ylabel('amplitude')
    ax.text(0.05, 0.1, r'SNR={:.1f}'.format(ds_snr),
            color='blue', fontsize=13, transform=ax.transAxes)

    ax = plt.subplot(3, 2, 6)
    reco_time = np.arange(max_frame, dtype=np.float)/float(reco_sr)
    plt.title('Reconstructed Waveform (16 kbps)')
    fig = plt.plot(reco_time, reco_waveform[:max_frame])
    plt.ylabel('amplitude')
    plt.xlabel('time (s)')
    ax.text(0.05, 0.1, r'SNR={:.1f}'.format(reco_snr),
            color='blue', fontsize=13, transform=ax.transAxes)

    plt.tight_layout()

    plt.savefig(ofile)


# Reads wav file and produces spectrum
def read_audio_spectrum(x, **kwd_args):
    return librosa.core.stft(x, **kwd_args)


true_spectrogram = read_audio_spectrum(y_true, n_fft=n_fft)
ds_spectrogram = read_audio_spectrum(y_ds, n_fft=n_fft)
reco_spectrogram = read_audio_spectrum(y_reco, n_fft=n_fft)

plot_all(true_spectrogram, ds_spectrogram, reco_spectrogram,
         y_true, y_ds, y_reco,
         sr_true, sr_ds, sr_reco,
         output_file_name, n_fft)
