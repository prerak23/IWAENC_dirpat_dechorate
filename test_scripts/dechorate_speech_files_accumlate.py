import h5py
import numpy as np
from scipy.signal import decimate


path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/dEchorate/zenodo.org/record/4626590/files/dechorate.hdf5"

dechorate_file=h5py.File(path)
arr=['000000', '000001', '000010', '000100', '010000', '011000', '011100', '011110', '011111', '020002']

mics=[str(i+1) for i in np.arange(30)]
source=[str(j+1) for j in np.arange(7)]


for rooms in arr:
    av=np.empty((416427,30,7))
    print(rooms)
    for s in source:
        for m in mics:

            sp=decimate(dechorate_file[rooms]["speech"][s][m][()],int(48000/16000))
            av[:,int(m)-1,int(s)-1]=sp

    np.save("/home/psrivastava/axis-2/IWAENC/z_test/dechorate_real_speech/dechorate_"+rooms+".npy",av)
