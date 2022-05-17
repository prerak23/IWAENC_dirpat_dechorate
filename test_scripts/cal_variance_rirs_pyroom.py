import os
import h5py
import numpy as np
from tqdm import tqdm

path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D3_1010/"

ar_var=[]
for file_name in tqdm(os.listdir(path)):
    if "generated_rirs" in file_name:
        print(file_name)
        rir_data=h5py.File(path+file_name,'r')
        for j in rir_data["rirs"].keys():
            for k in range(6):
                ar_var.append(np.var(rir_data["rirs"][j]["rir"][k,:rir_data["rirs"][j]["rirs_length"][0,k]]))


print(np.mean(ar_var))
np.save("varince_rirs_D3.npy",np.array(ar_var))
