# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sp

def try_n(N, w):
    ret = np.sum(np.exp(-1j * w * np.arange(N)) / N)
    return np.abs(ret)

def make_envelop(sample_arr, sample_rate, should_plot):
    sample_abs = np.abs(sample_arr)

    w = np.pi / 1000 # freq normalizer (rad/sample)
    N_l = 1
    N_h = 2000
    mag = 20*np.log10( try_n(N_l, w) )
    ret_found = False
    failsafe = 20
    while not ret_found and failsafe > 1: # binary search for N
        test_N = np.round((N_l + N_h) / 2)
        mag = 20*np.log10( try_n(test_N, w) )
        if mag > -3.0:
            N_l = test_N
        if mag < -3.0:
            N_h = test_N

        print('test N = ' + str(test_N) + ' , mag = ' + str(mag))
        print('N low = ' + str(N_l) + ' , N high = ' + str(N_h) + '\n')
        if N_h - N_l <= 1:
            ret_found = True

        failsafe = failsafe - 1
        if failsafe < 1:
            ret_found = True

    if should_plot:
        plt.plot(sample_arr)
        plt.title('Envelop')
        plt.show()


    size = int(len(sample_arr))
    copium = np.zeros(size)
    damping = 0
    for itt in range(size):
        damping = 0.99 * damping + 0.01 * (2*sample_abs[itt])
        copium[itt] = damping
    copium2 = np.zeros(size)
    damping2 = 0
    for itt in range(size):
        damping2 = 0.999 * damping2 + 0.001 * copium[itt]
        copium2[itt] = damping2

    plt.plot(sample_abs, 'b')
    plt.plot(copium2, 'g')
    plt.show()
    return copium2

def get_peaks(freq_signal, should_plot):

    peaks1, _ = sp.find_peaks(freq_signal, distance=1000, prominence=1)
    if should_plot:
        plt.title('peak finding')
        plt.plot(peaks1, freq_signal[peaks1], 'xr')
        plt.plot(freq_signal)
        plt.show()

    return peaks1[1:33], freq_signal[peaks1[1:33]] # returns position X of peaks, and value Y of peaks

def get_sounds(harmonic_base, harmonic_gains, N, Fe, should_plot):
    print('get_sounds: ')

    k = np.arange(1, 33)
    harmonic_arr = harmonic_base * k
    sound_freq = np.round( (harmonic_arr/Fe)*N ) # convert sound from Hz to discreet index
    sound_freq = sound_freq.astype(int)

    print(k)
    print(harmonic_arr)
    print(sound_freq)

    X = np.zeros(N)
    X[sound_freq] = harmonic_gains

    clean_sound = np.abs(np.fft.ifft(X))

    if should_plot:
        plt.subplot(2, 1, 1)
        plt.title('pure frequencies')
        plt.plot(X)

        plt.subplot(2, 1, 2)
        plt.title('synth soundwave')
        plt.plot(clean_sound)
        plt.show()

    return clean_sound

def note_guitare(file_name, should_plot):
    sample_arr, sample_rate = sf.read(file_name)
    sample_count = len(sample_arr)
    print('sample rate: ' + str(sample_rate))

    x = range(0, len(sample_arr))

    freq_gain = np.abs( np.fft.fft(sample_arr) )
    freq_db = 20*np.log10(freq_gain)

    harmonics_freq, harmonics_gains = get_peaks(freq_gain, False) # get gains of 32 first harmonics
    print(harmonics_gains)
    get_sounds(440.0, harmonics_gains, sample_count, sample_rate, False)

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

    note_guitare('note_guitare_lad.wav', False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
