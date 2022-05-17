import h5py
import numpy as np
import os


#-----------Change this path to aggregate different batch datasets -------------------------------

path_noisy_mix="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D7_1101/noisy_mixtures/"
new_file=h5py.File(path_noisy_mix+"D7_1101_aggregated_mixture.hdf5","w")
room_save=new_file.create_group("room_nos")

parallel_jobs=100
nj=0
for file_name in os.listdir(path_noisy_mix):
    if "noisy" in file_name:
        abc=h5py.File(path_noisy_mix+file_name,"r")
        room_start=int(file_name.split("_")[3])
        print(file_name,nj)
        for i in range(room_start,room_start+parallel_jobs): #File name used to iterate through rooms

            room_mixture=abc["room_nos"]["room_"+str(i)]["nsmix_f"][()]
            room_id=room_save.create_group("room_"+str(i))
            room_id.create_dataset("nsmix_f",(6,48000),data=room_mixture)
        nj+=1

#Aggregate annotations"
#-----------Change this path to aggregate different batch datasets -------------------------------
path_noisy_mix="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D7_1101/"
new_file_anno=h5py.File(path_noisy_mix+"/noisy_mixtures/"+"D7_1101_aggregated_mixture_annotations.hdf5","w")
room_save=new_file_anno.create_group("room_nos")
parallel_jobs=100
vol_arr=np.empty((1,40000))
surf_area=np.empty((1,40000))
rt60=np.empty((6,40000))

count_room_no=0
for file_name in os.listdir(path_noisy_mix):
    if "_anno_rooms_" in file_name:
        abc=h5py.File(path_noisy_mix+file_name,"r")

        for room_no in abc["rirs_save_anno"].keys(): #Keys used to iterate through rooms.
            sa=abc["rirs_save_anno"][room_no]["surf_area"][0]
            vol=abc["rirs_save_anno"][room_no]["volume"][0]
            rt60_=abc["rirs_save_anno"][room_no]["rt60_median"][()]


            surf_area[0,count_room_no]=sa
            vol_arr[0,count_room_no]=vol
            rt60[0,count_room_no]=rt60_[0,0]
            rt60[1,count_room_no]=rt60_[0,1]
            rt60[2,count_room_no]=rt60_[0,2]
            rt60[3,count_room_no]=rt60_[0,3]
            rt60[4,count_room_no]=rt60_[0,4]
            rt60[5,count_room_no]=rt60_[0,5]

            room_id=room_save.create_group(room_no) #Important step
            room_id.create_dataset("surface",1,data=sa)
            room_id.create_dataset("volume",1,data=vol)
            room_id.create_dataset("rt60",(1,6),data=rt60_)

            count_room_no+=1


print("Volume",np.std(vol_arr),np.var(vol_arr))
print("Surface",np.std(surf_area),np.var(surf_area))
print("RT 60 125",np.std(rt60[0,:]),np.var(rt60[0,:]))
print("RT 60 250",np.std(rt60[1,:]),np.var(rt60[1,:]))
print("RT 60 500",np.std(rt60[2,:]),np.var(rt60[2,:]))
print("RT 60 1000",np.std(rt60[3,:]),np.var(rt60[3,:]))
print("RT 60 2000",np.std(rt60[4,:]),np.var(rt60[4,:]))
print("RT 60 4000",np.std(rt60[5,:]),np.var(rt60[5,:]))
