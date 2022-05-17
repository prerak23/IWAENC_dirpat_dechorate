import numpy as np
import h5py
import matplotlib.pyplot as plt
import os


path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/"

datasets=["D1_0000/"]


def measure_rt60(h, fs=16000, decay_db=30):
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




for data_sets in datasets:
    print(data_sets)
    path_final=path+data_sets
    for files in os.listdir(path_final):
        if "generated_rirs_" in files:

            room_start=files.split(".")[0].split("_")[-2]
            room_end=files.split(".")[0].split("_")[-1]

            rirs_files=h5py.File(path_final+"/"+files,"r")
            with h5py.File(path_final+"/"+"D1_0000_anno_rooms_"+room_start+"_"+room_end+".hdf5","a") as anno_files:
                print(path_final+"/"+"D1_0000_anno_rooms_"+room_start+"_"+room_end+".hdf5")

                for roomss in rirs_files["rirs"].keys():

                    impulse_response_lens=rirs_files["rirs"][roomss]["rirs_length"][()]
                    impulse_response_1=rirs_files["rirs"][roomss]["rir"][0,:impulse_response_lens[0][0]]
                    impulse_response_2=rirs_files["rirs"][roomss]["rir"][2,:impulse_response_lens[0][2]]
                    impulse_response_3=rirs_files["rirs"][roomss]["rir"][4,:impulse_response_lens[0][4]]
                    rt60_1=measure_rt60(impulse_response_1,fs=16000)
                    rt60_2=measure_rt60(impulse_response_2,fs=16000)
                    rt60_3=measure_rt60(impulse_response_1,fs=16000)
                    mean_rt60=np.mean([rt60_1,rt60_2,rt60_3])[()]
                    k=anno_files["rirs_save_anno"][roomss]
                    k.create_dataset("rt60_global_t30",1,data=mean_rt60)

                anno_files.close()
                #rirs_files.close()
