# This is a sample Python script.
from cmath import log10

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sp

def to_freq_to_dB(signal_h):
    signal_H_dB = 20 * np.log10( np.fft.fftshift(np.abs(np.fft.fft(signal_h))) )
    return signal_H_dB

def band_cut(sample_arr, sample_count, sample_rate, should_plot):
    sample_db = 20*np.log10( np.abs( np.fft.fft(sample_arr) ) )

    N = 6000
    w = 2*np.pi*1000/sample_rate
    K = 2 * (N * 40 / sample_rate) + 1

    win_mask = np.hanning(N) # window function to smooth the low band

    n = np.arange(-N/2, N/2)
    low_band_pulse_h = np.sin(np.pi*n*K/N) / (N * np.sin(np.pi*n/N) + 1e-20)
    low_band_pulse_h[int(N/2)] = K / N

    low_band_pulse_H = np.fft.fft(low_band_pulse_h * win_mask, sample_count) # apply window function

    center_range = np.arange(-sample_count/2, sample_count/2)
    if should_plot:
        plt.subplot(2, 1, 1)
        plt.title('Low band impulse response (h)'); plt.xlabel("Sample index (n)"); plt.ylabel("gain")
        plt.plot(n, np.abs(low_band_pulse_h) )
        plt.subplot(2, 1, 2)
        plt.title('Low band impulse response (H)'); plt.xlabel("Sample index (m)"); plt.ylabel("gain")
        plt.plot(center_range, np.fft.fftshift(np.abs(low_band_pulse_H)) )
        plt.xlim(-1000, 1000)
        plt.show()

    n_range = np.arange(sample_count)
    diracte = np.zeros(sample_count)
    diracte[0] = 1

    mid_band_h = diracte - 2.0 * np.fft.ifft(low_band_pulse_H) * np.cos(w*n_range) # mid band filter creation

    filtered = np.fft.fft(sample_arr) * np.fft.fft(mid_band_h)
    result_sound = np.fft.ifft(filtered)

    plt.show()
    if should_plot:
        plt.subplot(3, 1, 1)
        plt.title('Original audio frequency distribution (x)'); plt.xlabel("Sample index (n)"); plt.ylabel("mag (dB)")
        plt.plot(center_range, to_freq_to_dB(sample_arr))
        plt.xlim(-10000, 10000)
        plt.subplot(3, 1, 2)
        plt.title('Mid band filter (H)'); plt.xlabel("Sample index (m)"); plt.ylabel("mag (dB)")
        plt.plot(center_range, to_freq_to_dB(mid_band_h))
        plt.xlim(-10000, 10000)
        plt.subplot(3, 1, 3)
        plt.title('Filtered audio (Y)'); plt.xlabel("Sample index (m)"); plt.ylabel("mag (dB)")
        plt.plot(np.abs(result_sound))
        plt.show()

    return mid_band_h

def response_echelon(Order, fc):
    n = np.arange(-Order/2, Order/2 +1)
    k = (fc * Order /np.pi) +1
    h = np.sin(np.pi*n*k/Order) / (Order * np.sin(np.pi*n/Order) )
    return h


def try_n(N, w):
    ret = np.sum(np.exp(-1j * w * np.arange(N)) / N)
    return np.abs(ret)


def make_envelop(sample_arr, sample_count, should_plot):
    sample_abs = np.abs(sample_arr)

    w = np.pi / 1000 # freq normalized (rad/sample)
    N_l = 1
    N_h = 2000
    test_N = 0
    ret_found = False
    failsafe = 20
    while not ret_found and failsafe > 1: # binary search for N
        test_N = np.round((N_l + N_h) / 2)
        mag = 20*np.log10( try_n(test_N, w) ) # test new N
        if mag > -3.0:
            N_l = test_N
        if mag < -3.0:
            N_h = test_N

        if N_h - N_l <= 1: # check if we're done
            ret_found = True
        failsafe = failsafe - 1 # check failsafe to avoid [while(true)]

        if should_plot: print('test N = ' + str(test_N) + ' , mag = ' + str(mag))
        if should_plot: print('N low = ' + str(N_l) + ' , N high = ' + str(N_h) + '\n')

    print('Chosen N order: ' + str(test_N) + ' , mag (dB): ' + str(mag))
    impulse_response = response_echelon(test_N, w)
    envelop = np.convolve(sample_abs,impulse_response)

    if should_plot:
        plt.subplot(2, 1, 1)
        plt.title('Envelop\'s filter impulse response'); plt.xlabel("Sample index (m)"); plt.ylabel("Mag (dB)")
        #plt.plot(np.arange(-test_N/2, test_N/2+1), np.abs(impulse_response))
        plt.plot(np.arange(-test_N/2+1, test_N/2), to_freq_to_dB(impulse_response)[1:int(test_N)])
        plt.subplot(2, 1, 2)
        plt.title('Envelop'); plt.xlabel("Sample index (n)"); plt.ylabel("gain")
        plt.plot(np.abs(envelop))
        plt.show()

    return envelop[0:sample_count]

def get_peaks(freq_signal, should_plot):

    peaks1, _ = sp.find_peaks(freq_signal, distance=1000, prominence=1)
    if should_plot:
        plt.title('Harmonics identification'); plt.xlabel("Sample index (m)"); plt.ylabel("Mag (dB)")
        plt.xlim(0, 70000)
        plt.plot(peaks1[1:33], 20*np.log10( freq_signal[peaks1[1:33]] ), 'xr')
        plt.plot(20*np.log10( freq_signal ))
        plt.show()

    return peaks1[1:33], freq_signal[peaks1[1:33]] # returns position X of peaks, and value Y of peaks


def get_sounds(harmonic_base, harmonic_gains, N, Fe, should_plot):
    if should_plot: print('get_sounds: ')

    k = np.arange(1, 33)
    harmonic_arr = harmonic_base * k
    sound_freq = np.round( (harmonic_arr/Fe)*N ).astype(int) # convert sound from Hz to discreet index

    if should_plot:
        print('k: ' + str(k))
        print('harmonic_arr: ' + str(harmonic_arr))
        print('sound_freq: ' + str(harmonic_arr))

    X = np.zeros(N)
    X[sound_freq] = harmonic_gains

    #clean_sound = np.abs(np.fft.ifft(X))
    clean_sound = np.real(np.fft.ifft(X))

    if should_plot:
        plt.subplot(2, 1, 1).set_yscale('log')
        plt.title('Generated frequencies (X)'); plt.xlabel("Sample index (m)")
        plt.xlim(0, 70000)
        plt.plot(X)

        plt.subplot(2, 1, 2)
        plt.title('Generated soundwave (x)'); plt.xlabel("Sample index (n)")
        plt.plot(clean_sound)
        plt.show()

    return clean_sound

def synth_note(base_freq, harmonics_gains, sample_count, sample_rate, envelop, crop_len=0):
    sound = get_sounds(base_freq, harmonics_gains, sample_count, sample_rate, False)
    synth = sound * envelop
    if crop_len != 0 :
        synth = synth[0:int(sample_count/crop_len)]
    return synth

def play_music(file_name, harmonics_gains, sample_count, sample_rate, envelop):

    sol = synth_note(392.0, harmonics_gains, sample_count, sample_rate, envelop, crop_len=3)
    mi_b = synth_note(311.1, harmonics_gains, sample_count, sample_rate, envelop, crop_len=3)
    fa = synth_note(349.2, harmonics_gains, sample_count, sample_rate, envelop, crop_len=3)
    re = synth_note(293.7, harmonics_gains, sample_count, sample_rate, envelop, crop_len=3)

    silence = np.zeros(int(sample_count / 6))

    final = sol
    final = np.concatenate((final, sol))
    final = np.concatenate((final, sol))
    final = np.concatenate((final, mi_b))
    final = np.concatenate((final, silence))
    final = np.concatenate((final, fa))
    final = np.concatenate((final, fa))
    final = np.concatenate((final, fa))
    final = np.concatenate((final, re))
    sf.write(file_name, 20 * final, sample_rate)


def note_guitare(file_name, should_plot):
    sample_arr, sample_rate = sf.read(file_name)
    sample_count = len(sample_arr)
    if should_plot: print('sample rate: ' + str(sample_rate))

    freq_gain = np.abs( np.fft.fft(sample_arr) )
    freq_db = 20*np.log10(freq_gain)

    harmonics_freq, harmonics_gains = get_peaks(freq_gain, False) # get gains of 32 first harmonics
    #retarded_print = "{:,2f}".format(harmonics_gains)
    np.set_printoptions(suppress=True)
    if should_plot: print('harmonics_gains: ' + str(harmonics_gains))
    sound = get_sounds(466.2, harmonics_gains, sample_count, sample_rate, False) # generate new sound

    envelop = make_envelop(sample_arr, sample_count, True) # generate envelop

    synth = sound * envelop # synthesize the note

    sf.write('synth_sound_guitar.wav', 20*synth, sample_rate)

    play_music('synth_music.wav', harmonics_gains, sample_count, sample_rate, envelop) # synthesize "music"

    x_scale = np.arange(-sample_count / 2, sample_count / 2)
    if should_plot:
        plt.subplot(4, 1, 1)
        plt.title('Original sound'); plt.xlabel("Sample index (n)"); plt.ylabel("Gain")
        plt.plot(sample_arr)

        plt.subplot(4, 1, 2)
        plt.title('Fadeout envelop'); plt.xlabel("Sample index (n)"); plt.ylabel("Gain")
        plt.plot(envelop)

        plt.subplot(4, 1, 3)
        plt.title('Synthesized wave'); plt.xlabel("Sample index (n)"); plt.ylabel("Gain")
        plt.plot(x_scale, sound)

        plt.subplot(4, 1, 4)
        plt.title('Frequency analysis of synthesized signal (Y)'); plt.xlabel("Sample index (m)"); plt.ylabel("Mag (dB)")
        plt.plot(x_scale, np.fft.fftshift(freq_db))

        plt.show()


        plt.subplot(4, 1, 1)
        plt.title('Original sound (x)'); plt.xlabel("Sample index (n)"); plt.ylabel("Gain")
        plt.plot(sample_arr)
        plt.subplot(4, 1, 2)
        plt.title('Synthesized sound (y)'); plt.xlabel("Sample index (n)"); plt.ylabel("Gain")
        plt.plot(synth)

        plt.subplot(4, 1, 3)
        plt.title('Original sound (X)'); plt.xlabel("Sample index (m)"); plt.ylabel("Mag (dB)")
        plt.plot(x_scale, to_freq_to_dB(sample_arr))

        plt.subplot(4, 1, 4)
        plt.title('Filtered sound (Y)'); plt.xlabel("Sample index (m)"); plt.ylabel("Mag (dB)")
        plt.plot(x_scale, to_freq_to_dB(synth))

        plt.show()


def note_basson(file_name, should_plot):
    sample_arr, sample_rate = sf.read(file_name)
    sample_count = len(sample_arr)
    if should_plot: print('sample rate: ' + str(sample_rate))

    band_filter = band_cut(sample_arr, sample_count, sample_rate, True)
    filtered = np.fft.fft(sample_arr) * np.fft.fft(band_filter)
    cleaned_sound = np.fft.ifft(filtered)
    sf.write('synth_basson.wav', 1 * np.real(cleaned_sound), sample_rate)

    #get_peaks(np.fft.fft(cleaned_sound), False)


    x_scale = np.arange(-sample_count/2, sample_count/2)

    freq1000 = np.round((1000 / sample_rate) * sample_count).astype(int)
    temp_sin_H = np.zeros(sample_count)
    temp_sin_H[int(freq1000)] = 1
    temp_sin_H[int(sample_count- freq1000)] = 1

    damped_sin = np.fft.ifft( np.fft.fft(band_filter) * temp_sin_H)
    plt.subplot(2, 1, 1)
    plt.title('pure sin (H)'); plt.xlabel("Sample index (n)"); plt.ylabel("Gain")
    plt.plot(x_scale, np.fft.fftshift( temp_sin_H))

    plt.subplot(2, 1, 2)
    plt.title('pure sin (h)'); plt.xlabel("Sample index (m)"); plt.ylabel("Gain")
    plt.plot( np.fft.ifft(temp_sin_H))

    plt.subplot(2, 1, 2)
    #plt.title('filtered sin (h)');
    plt.plot( damped_sin, 'g')

    plt.show()



    if should_plot:
        plt.subplot(4, 1, 1)
        plt.title('Original sound (x)'); plt.xlabel("Sample index (n)"); plt.ylabel("Gain")
        plt.plot(sample_arr)

        plt.subplot(4, 1, 2)
        plt.title('Filtered sound (y)'); plt.xlabel("Sample index (n)"); plt.ylabel("Gain")
        plt.plot(cleaned_sound)

        plt.subplot(4, 1, 3)
        plt.title('Original sound (X)'); plt.xlabel("Sample index (m)"); plt.ylabel("Mag (dB)")
        #plt.plot(x_scale, np.fft.fftshift(np.fft.fft(sample_arr)))
        plt.plot(x_scale, to_freq_to_dB(sample_arr))

        plt.subplot(4, 1, 4)
        plt.title('Filtered sound (Y)'); plt.xlabel("Sample index (m)"); plt.ylabel("Mag (dB)")
        #plt.plot(x_scale, np.fft.fftshift(np.fft.fft(cleaned_sound)))
        plt.plot(x_scale, to_freq_to_dB(cleaned_sound))

        plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #temp = 466.2 * np.arange(1, 33)
    #print(repr(temp))

    note_basson('note_basson_plus_sinus_1000_hz.wav', True)

    note_guitare('note_guitare_lad.wav', False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
