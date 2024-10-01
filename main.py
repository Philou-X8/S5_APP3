# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sp


def band_cut(sample_arr, sample_count, sample_rate, should_plot):
    N = 6000
    m = (N * 1000 / sample_rate)
    w = 2*np.pi*1000/sample_rate
    K = 2 * (N * 40 / sample_rate) + 1
    m_low = int( sample_count * (960 / sample_rate) )
    m_high = int( sample_count * (1040 / sample_rate) )
    print('m_low: ' + str(m_low) + 'm_high: ' + str(m_high))
    m_range = np.arange(-sample_count/2, sample_count/2)

    win_mask = np.hanning(N)

    n = np.arange(-N/2, N/2)
    filter2_h = np.sin(np.pi*n*K/N) / (N * np.sin(np.pi*n/N) + 1e-20)
    filter2_h[int(N/2)] = K / N
    #filter2_H[int(N/2)] = 1
    buffer = np.zeros(sample_count)
    buffer[0:len(filter2_h)] = filter2_h
    filter2_H = np.fft.fft(filter2_h * win_mask, sample_count)

    print('len:')
    print(len(filter2_H))

    center_range = np.arange(-sample_count/2, sample_count/2)
    n_range = np.arange(sample_count)
    diracte = np.zeros(sample_count)
    diracte[0] = 1

    #print(len(center_range))
    #band_pass = filter2_H * ((-1) ** center_range)
    band_pass = diracte - 2.0 * np.fft.ifft(filter2_H) * np.cos(w*n_range) # <-----------------------

    #plt.plot(center_range, np.fft.fftshift(np.abs(np.fft.fft(band_pass))))
    #plt.show()
    #band_pass = band_pass[]
    if should_plot:
        plt.subplot(2, 1, 1)
        plt.title('pulse (h)')
        plt.plot(n, np.abs(filter2_h) )
        plt.subplot(2, 1, 2)
        plt.title('pulse (H)')
        plt.plot(center_range, np.fft.fftshift(np.abs(filter2_H)) )
        plt.xlim(-1000, 1000)
        plt.show()

    sample_db = 20*np.log10( np.abs( np.fft.fft(sample_arr) ) )

    #20 * np.log10(np.abs(h))
    #plt.stem(np.fft.fftshift( np.fft.fft(filter2_H) ) )

    filtered = np.fft.fft(sample_arr) * np.fft.fft(band_pass)
    #filtered = filtered * band_pass # apply filter again

    result_sound = np.fft.ifft(filtered)

    plt.show()
    if should_plot:
        plt.subplot(3, 1, 1)
        plt.title('sample freq')
        plt.plot(center_range, np.fft.fftshift(np.abs(np.fft.fft(sample_arr))))
        plt.xlim(-10000, 10000)
        plt.subplot(3, 1, 2)
        plt.title('filter (H)')
        plt.plot(center_range, np.fft.fftshift(np.abs(np.fft.fft(band_pass))))
        plt.xlim(-10000, 10000)
        #plt.plot(m_range, np.fft.fftshift(filter_h))
        plt.subplot(3, 1, 3)
        plt.title('filtered')
        plt.plot(np.abs(result_sound))
        plt.show()

    return band_pass

def reponse_echelon(Order, fc):

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
        mag = 20*np.log10( try_n(test_N, w) )
        if mag > -3.0:
            N_l = test_N
        if mag < -3.0:
            N_h = test_N

        if N_h - N_l <= 1: # check if we're done
            ret_found = True
        failsafe = failsafe - 1 # check failsafe

        if should_plot: print('test N = ' + str(test_N) + ' , mag = ' + str(mag))
        if should_plot: print('N low = ' + str(N_l) + ' , N high = ' + str(N_h) + '\n')

    print('Chosen N order: ' + str(test_N) + ' , mag (dB): ' + str(mag))
    impulse_response = reponse_echelon(test_N, w)
    envelop = np.convolve(sample_abs,impulse_response)

    if should_plot:
        plt.subplot(2, 1, 1)
        plt.title('Envelop impulse response')
        plt.plot(np.abs(impulse_response))
        plt.subplot(2, 1, 2)
        plt.title('Envelop')
        plt.plot(np.abs(envelop))
        plt.show()

    return envelop[0:sample_count]

def get_peaks(freq_signal, should_plot):

    peaks1, _ = sp.find_peaks(freq_signal, distance=1000, prominence=1)
    if should_plot:
        plt.title('peak finding')
        plt.plot(peaks1[1:33], freq_signal[peaks1[1:33]], 'xr')
        plt.plot(freq_signal)
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
        plt.subplot(2, 1, 1)
        plt.title('pure frequencies')
        plt.plot(X)

        plt.subplot(2, 1, 2)
        plt.title('synth soundwave')
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
    small_silence = np.zeros(int(sample_count / 12))

    final = sol
    #final = np.concatenate((final, small_silence))
    final = np.concatenate((final, sol))
    #final = np.concatenate((final, small_silence))
    final = np.concatenate((final, sol))
    #final = np.concatenate((final, small_silence))
    final = np.concatenate((final, mi_b))
    final = np.concatenate((final, silence))
    final = np.concatenate((final, fa))
    #final = np.concatenate((final, small_silence))
    final = np.concatenate((final, fa))
    #final = np.concatenate((final, small_silence))
    final = np.concatenate((final, fa))
    #final = np.concatenate((final, small_silence))
    final = np.concatenate((final, re))
    sf.write(file_name, 20 * final, sample_rate)


def note_guitare(file_name, should_plot):
    sample_arr, sample_rate = sf.read(file_name)
    sample_count = len(sample_arr)
    if should_plot: print('sample rate: ' + str(sample_rate))

    freq_gain = np.abs( np.fft.fft(sample_arr) )
    freq_db = 20*np.log10(freq_gain)

    harmonics_freq, harmonics_gains = get_peaks(freq_gain, True) # get gains of 32 first harmonics
    if should_plot: print('harmonics_gains: ' + str(harmonics_gains))
    sound = get_sounds(466.2, harmonics_gains, sample_count, sample_rate, False)

    envelop = make_envelop(sample_arr, sample_count, False)
    # if should_plot: print('len sample: ' + str(sample_count) + ' , len envelop: ' + str(len(envelop)))
    synth = sound * envelop
    sf.write('synth_sound_guitar.wav', 20*synth, sample_rate)

    play_music('synth_music.wav', harmonics_gains, sample_count, sample_rate, envelop)

    if should_plot:
        plt.subplot(4, 1, 1)
        plt.title('Original sound')
        plt.plot(sample_arr)

        plt.subplot(4, 1, 2)
        plt.title('Fadeout envelop')
        plt.plot(envelop)

        plt.subplot(4, 1, 3)
        plt.title('Synthesised wave')
        plt.plot(sound)

        plt.subplot(4, 1, 4)
        plt.title('Frequency analysis')
        #plt.plot(freq_gain)
        plt.plot(np.fft.fftshift(freq_db))
        plt.show()


        plt.subplot(2, 1, 1)
        plt.title('Original sound')
        plt.plot(sample_arr)
        plt.subplot(2, 1, 2)
        plt.title('Generated sound')
        plt.plot(synth)
        plt.show()


def note_basson(file_name, should_plot):
    sample_arr, sample_rate = sf.read(file_name)
    sample_count = len(sample_arr)
    if should_plot: print('sample rate: ' + str(sample_rate))

    band_filter = band_cut(sample_arr, sample_count, sample_rate, True)
    filtered = np.fft.fft(sample_arr) * np.fft.fft(band_filter)
    cleaned_sound = np.fft.ifft(filtered)
    sf.write('synth_basson.wav', 1 * np.real(cleaned_sound), sample_rate)

    get_peaks(np.fft.fft(cleaned_sound), True)

    x_scale = np.arange(-sample_count/2, sample_count/2)

    if should_plot:
        plt.subplot(4, 1, 1)
        plt.title('Original sound')
        plt.plot(sample_arr)

        plt.subplot(4, 1, 2)
        plt.title('Filtered sound')
        plt.plot(cleaned_sound)

        plt.subplot(4, 1, 3)
        plt.title('Original sound (freq)')
        plt.plot(x_scale, np.fft.fftshift(np.fft.fft(sample_arr)))
        plt.xlim(-10000, 10000)

        plt.subplot(4, 1, 4)
        plt.title('Filtered sound (freq)')
        plt.plot(x_scale, np.fft.fftshift(np.fft.fft(cleaned_sound)))
        plt.xlim(-10000, 10000)

        plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    note_basson('note_basson_plus_sinus_1000_hz.wav', True)

    note_guitare('note_guitare_lad.wav', True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
