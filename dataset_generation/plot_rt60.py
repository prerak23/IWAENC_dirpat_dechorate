import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
sns.set_theme()


path_noisy_mix="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/"



vol_arr=np.empty((6,40000))
surf_area=np.empty((6,40000))
#rt60_global=np.empty((5,40000))
#rt60_global_t10=np.empty((5,40000))


rt60_125=np.empty((6,40000))
rt60_250=np.empty((6,40000))
rt60_500=np.empty((6,40000))
rt60_1000=np.empty((6,40000))
rt60_2000=np.empty((6,40000))
rt60_4000=np.empty((6,40000))


count=1
name_d=[]
#k=os.listdir(path_noisy_mix)
#k.sort()
k=["D2_1000","D3_1010","D4_1100","D5_1111","D6_0111"]
#k=k[0]

room_nos=0
for dataset in k:
    print(dataset)
    path_dataset=path_noisy_mix+dataset
    name_d.append(dataset)
    room_nos=0

    for room_annotation in os.listdir(path_dataset):
        if "_anno_rooms_" in room_annotation:
            open_file=h5py.File(path_dataset+'/'+room_annotation,"r")
            print(path_dataset+'/'+room_annotation)
            for room_no in open_file["rirs_save_anno"].keys():

                sa=open_file["rirs_save_anno"][room_no]["surf_area"][0]
                vol=open_file["rirs_save_anno"][room_no]["volume"][0]
                rt60=open_file["rirs_save_anno"][room_no]["rt60_median"][()]

                '''
                if "D3_1010" in dataset:
                    rt60_t10=open_file["rirs_save_anno"][room_no]["rt60_global_t10_1"][()]
                else:
                    rt60_t10=open_file["rirs_save_anno"][room_no]["rt60_global_t10"][()]
                '''

                surf_area[count,room_nos]=sa
                vol_arr[count,room_nos]=vol
                #rt60_global[count,room_nos]=rt60[0]
                #rt60_global_t10[count,room_nos]=rt60_t10[0]


                rt60_125[count,room_nos]=rt60[0,0]
                rt60_250[count,room_nos]=rt60[0,1]
                rt60_500[count,room_nos]=rt60[0,2]
                rt60_1000[count,room_nos]=rt60[0,3]
                rt60_2000[count,room_nos]=rt60[0,4]
                rt60_4000[count,room_nos]=rt60[0,5]


                room_nos+=1
    print(room_nos)
    count+=1



D1_=h5py.File(path_noisy_mix+"D1_0000/noisy_mixtures"+"/D1_0000_aggregated_mixture_annotations.hdf5")
room_nos=0
for rs in D1_["room_nos"].keys():
    rt6=D1_["room_nos"][rs]["rt60"][()]
    rt60_125[0,room_nos]=rt6[0,0]
    rt60_250[0,room_nos]=rt6[0,1]
    rt60_500[0,room_nos]=rt6[0,2]
    rt60_1000[0,room_nos]=rt6[0,3]
    rt60_2000[0,room_nos]=rt6[0,4]
    rt60_4000[0,room_nos]=rt6[0,5]
    room_nos+=1


#rt60_global[0,:]=np.load("D1_rt60_t20.npy")
#rt60_global[1,:39900]=np.load("D2_rt60_t20.npy")
#rt60_global[2,:]=np.load("D3_rt60_t20.npy")
#rt60_global[3,:]=np.load("D4_rt60_t20.npy")
#rt60_global[4,:]=np.load("D5_rt60_t20.npy")

fig = plt.figure(figsize=(6,12), constrained_layout=True)

spec = fig.add_gridspec(6,1)

name_d=["D1_0000","D2_1000","D3_1010","D4_1100","D5_1111","D6_0111"]

#name_d=["D5_1111"]
n_bins=np.arange(2,step=0.05)

for i in range(6):


    '''
    ax0=fig.add_subplot(spec[i, 0])
    ax0.hist(surf_area[i,:])
    ax0.set_xlabel("Surface m2")
    ax0.set_ylabel("Number of Rooms")
    ax0.set_title("Surface "+name_d[i])
    '''

    '''
    ax1=fig.add_subplot(spec[i, 0])
    ax1.hist(vol_arr[i,:])
    ax1.set_xlabel("Volume m3")
    ax1.set_ylabel("Number of Rooms")
    ax1.set_title("Volume "+name_d[i])
    '''

    ax2=fig.add_subplot(spec[i,0])
    ax2.hist([rt60_125[i,:],rt60_250[i,:],rt60_500[i,:],rt60_1000[i,:],rt60_2000[i,:],rt60_4000[i,:]],bins=n_bins,stacked=True,alpha=0.5,label=["rt_125","rt_250","rt_500","rt_1000","rt_2000","rt_4000"])
    #ax2.hist(rt60_global[i,:],bins=n_bins)
    #ax2.hist(rt60_global_t10[i,:],bins=n_bins)
    ax2.set_xlabel("RT60")
    ax2.set_ylabel("Number of Rooms")
    ax2.set_title("RT60 s "+name_d[i])




    #ax7.set_title("RT60 s "+name_d[i])

plt.legend()
plt.savefig("RT_60_ALL_DATASETS_incl_D6.jpeg")
