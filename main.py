# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sp
from scipy.io import wavfile as wf


def reponse_echelon(Order, fc):

    n = np.arange(Order/2, Order/2 +1)
    k = (fc * Order /np.pi) +1
    h = np.sin(np.pi*n*k/Order)/(Order*np.sin(np.pi*n/k))

    return h

def try_n(N, w):
    ret = np.sum(np.exp(-1j * w * np.arange(N)) / N)
    return np.abs(ret)

def make_envelop(sample_arr, sample_rate, should_plot):
    sample_arr = abs(sample_arr)

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

        #print('test N = ' + str(test_N) + ' , mag = ' + str(mag))
        #print('N low = ' + str(N_l) + ' , N high = ' + str(N_h) + '\n')
        if N_h - N_l <= 1:
            ret_found = True

        failsafe = failsafe - 1
        if failsafe < 1:
            ret_found = True


    impulse_response = reponse_echelon(test_N,np.pi/1000)

    envelop = np.convolve(sample_arr,impulse_response)

    if should_plot:
        plt.figure()

        plt.subplot(2,1,1)
        plt.plot(envelop)
        plt.title('Envelop')

        plt.subplot(2,1,2)
        plt.plot(sample_arr)
        plt.title('sample_arr')
        plt.show()

    return envelop

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

        plt.figure()

        plt.subplot(2, 1, 1)
        plt.title('pure frequencies')
        plt.plot(X)

        plt.subplot(2, 1, 2)
        plt.title('synth soundwave')
        plt.plot(clean_sound)
        plt.show()

    return clean_sound

def note_guitare(file_name, should_plot,note_freq):
    sample_arr, sample_rate = sf.read(file_name)
    sample_count = len(sample_arr)
    print('sample rate: ' + str(sample_rate))

    x = range(0, len(sample_arr))

    freq_gain = np.abs( np.fft.fft(sample_arr) )
    freq_db = 20*np.log10(freq_gain)

    harmonics_freq, harmonics_gains = get_peaks(freq_gain, should_plot) # get gains of 32 first harmonics
    print(harmonics_gains)
    sounds =  get_sounds(note_freq, harmonics_gains, sample_count, sample_rate, should_plot)

    envelope = make_envelop(sounds, sample_rate, should_plot)

    synthesis_signal = np.sum(freq_gain) * envelope

    #wf.write('output.wav', sample_rate,synthesis_signal.astype('int16')*10000) # fois 10000 pour augmenter l'amplitude et donc pouvoir l'entendre car très petite.

    

    if should_plot:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title('Guitare sound')
        plt.plot(x, sample_arr)

        plt.subplot(3, 1, 2)
        plt.title('freq')
        plt.plot(freq_gain)
        plt.plot(np.fft.fftshift(freq_db))

        plt.subplot(3,1,3)
        plt.title('Synth Note')
        plt.plot(synthesis_signal)
        plt.show()


    return synthesis_signal


def beethoven_guitare() :


    _, sample_rate = sf.read('note_guitare_lad.wav')

    sol = note_guitare('note_guitare_lad.wav',False,392)
    mi_b = note_guitare('note_guitare_lad.wav',False,311.1) # mi bemol = re dieze
    fa = note_guitare('note_guitare_lad.wav',False,349.2)
    re = note_guitare('note_guitare_lad.wav',False,293.7)

    length_notes = len(sol)
    sol = sol[0:int(length_notes/3)]
    mi_b = mi_b[0:int(length_notes/3)]
    fa = fa[0:int(length_notes/3)]
    re = re[0:int(length_notes/3)]

    silence = np.zeros(int(length_notes/6))
    small_silence = np.zeros(int(length_notes/12))

    final = np.concatenate((sol,small_silence))
    final = np.concatenate((final,sol))
    final = np.concatenate((final,small_silence))
    final = np.concatenate((final,sol))
    final = np.concatenate((final,small_silence))
    final = np.concatenate((final,mi_b))
    final = np.concat((final,silence))
    final = np.concatenate((final,fa))
    final = np.concatenate((final,small_silence))
    final = np.concatenate((final,fa))
    final = np.concatenate((final,small_silence))
    final = np.concatenate((final,fa))
    final = np.concatenate((final,small_silence))
    final = np.concatenate((final,re))

    wf.write('output_guitare.wav',sample_rate,final.astype('int16') * 100) # * 100 car sinon l'amplitude est très basse pour l'entendre avec des haut parleurs




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    beethoven_guitare()

    

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
