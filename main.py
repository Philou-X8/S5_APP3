# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sp


def make_envelop(sample_arr, sample_rate, should_plot):
    sample_arr = abs(sample_arr)

    N = 64 # ordre
    f = np.pi / 1000 # freq de coupure
    Fe = sample_rate # échantillonnage
    m = N / 8 # position (index) de la raie
    K = 2 * m + 1

    if should_plot:
        plt.plot(sample_arr)
        plt.title('Envelop')
        plt.show()

def get_peaks(freq_signal, should_plot):

    peaks1, _ = sp.find_peaks(freq_signal, distance=1000, prominence=1)
    if should_plot:
        plt.title('peak finding')
        plt.plot(peaks1, freq_signal[peaks1], 'xr')
        plt.plot(freq_signal)
        plt.show()

    return peaks1[1:33], freq_signal[peaks1[1:33]] # returns position X of peaks, and value Y of peaks

def get_sounds(harmonic_base, harmonic_gains, N, Fe):
    print('get_sounds: ')

    k = np.arange(1, 33)
    harmonic_arr = harmonic_base * k
    sound_freq = (harmonic_arr/Fe)*N # convert sound from Hz to discreet index

    print(k)
    print(harmonic_arr)
    print(sound_freq)

    return 0

def note_guitare(file_name, should_plot):
    sample_arr, sample_rate = sf.read(file_name)
    sample_count = len(sample_arr)
    print('sample rate: ' + str(sample_rate))

    x = range(0, len(sample_arr))

    freq_gain = np.abs( np.fft.fft(sample_arr) )
    freq_db = 20*np.log10(freq_gain)

    harmonics_freq, harmonics_gains = get_peaks(freq_gain, False) # get gains of 32 first harmonics
    print(harmonics_gains)
    get_sounds(440.0, harmonics_gains, sample_count, sample_rate)

    make_envelop(sample_arr, sample_rate, False)

    if should_plot:
        plt.subplot(2, 1, 1)
        plt.title('Guitare sound')
        plt.plot(x, sample_arr)

        plt.subplot(2, 1, 2)
        plt.title('freq')
        plt.plot(freq_gain)
        #plt.plot(np.fft.fftshift(freq_db))
        plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    note_guitare('note_guitare_lad.wav', True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
