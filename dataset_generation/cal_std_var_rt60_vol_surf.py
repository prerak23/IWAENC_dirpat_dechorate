import h5py
import numpy as np
import os


root_path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D5_1111/noisy_mixtures/"

anno_file=h5py.File(root_path+"D5_1111_aggregated_mixture_annotations.hdf5","r")
surface=[]
vol=[]
rt60_=np.empty((6,9000))

count=0

for i in anno_file["room_nos"].keys():
    if int(i.split("_")[1])< 9000: #Calcualate std and var only in the training rooms
        print(i)
        sa=anno_file["room_nos"][i]["surface"][0]
        vol_=anno_file["room_nos"][i]["volume"][0]
        surface.append(sa)
        vol.append(vol_)

        for j in range(6):
            rt60_[j,count]=anno_file["room_nos"][i]["rt60"][0,j]
        count+=1

print(count)
print("Std and Variance of surface", np.std(np.array(surface)),np.var(np.array(surface)))
print("Std and Variance of volume", np.std(np.array(vol)),np.var(np.array(vol)))
print("Std and Variance of rt60 125", np.std(rt60_[0,:]),np.var(rt60_[0,:]))
print("Std and Variance of rt60 250", np.std(rt60_[1,:]),np.var(rt60_[1,:]))
print("Std and Variance of rt60 500", np.std(rt60_[2,:]),np.var(rt60_[2,:]))
print("Std and Variance of rt60 1000", np.std(rt60_[3,:]),np.var(rt60_[3,:]))
print("Std and Variance of rt60 2000", np.std(rt60_[4,:]),np.var(rt60_[4,:]))
print("Std and Variance of rt60 4000", np.std(rt60_[5,:]),np.var(rt60_[5,:]))
