import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

from scipy.io import wavfile
from scipy import signal
from scipy.fft import fftfreq,fft
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
from acoustics.signal import bandpass
from scipy import stats
import sys
from scipy.signal import decimate


def measure_rt60(h, fs=16000, decay_db=15):
    """
    Analyze the RT60 of an impulse response. Optionaly plots some useful information.

    Parameters
    ----------
    h: array_like
        The impulse response.
    fs: float or int, optional
        The sampling frequency of h (default to 1, i.e., samples).
    decay_db: float or int, optional
        The decay in decibels for which we actually estimate the time. Although
        we would like to estimate the RT60, it might not be practical. Instead,
        we measure the RT20 or RT30 and extrapolate to RT60.
    plot: bool, optional
        If set to ``True``, the power decay and different estimated values will
        be plotted (default False).
    rt60_tgt: float
        This parameter can be used to indicate a target RT60 to which we want
        to compare the estimated value.
    """

    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    # -5 dB headroom
    i_5db = np.min(np.where(-5 - energy_db > 0)[0])
    e_5db = energy_db[i_5db]
    t_5db = i_5db / fs

    # after decay
    i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time

    return est_rt60


req_bands=np.array([125 * pow(2,a) for a in range(6)])

path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/dEchorate/zenodo.org/record/4626590/files/"

rooms=['000000', '000001', '000010', '000100', '001000', '010000', '011000', '011100', '011110', '011111', '020002']

abc=h5py.File(path+"dechorate.hdf5",'r')
for r in rooms:
    rt60_r=[]

    for s in abc[r]['rir'].keys():

        for m in abc[r]['rir'][s].keys():
            if 'loopback' not in m:

                sp=decimate(abc[r]['rir'][s][m][()][:,0],int(48000/16000))
                rt60_r.append(measure_rt60(sp,16000))

    sps=np.array(rt60_r)
    print(r)
    print(np.median(sps,axis=0))
