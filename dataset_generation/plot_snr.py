import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
sns.set_theme()

path_noisy_mix="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D7_1101/noisy_mixtures/"

snr_f=[]
snr_diff=[]
snr_static=[]
parallel_jobs=100
count=1
for file_name in os.listdir(path_noisy_mix):
    if "noisy" in file_name:
        abc=h5py.File(path_noisy_mix+file_name,"r")
        room_start=int(file_name.split("_")[3])
        print(file_name)
        for i in range(room_start,room_start+parallel_jobs):
            l=abc["room_nos"]["room_"+str(i)]["nsmix_snr_f"][()]
            k=abc["room_nos"]["room_"+str(i)]["nsmix_snr_diff"][()]
            g=abc["room_nos"]["room_"+str(i)]["nsmix_snr_static"][()]


            for i in range(3):
                    snr_w_s=g[i]
                    snr_w_d=k[i]
                    snr_w_f=l[i]
                    a_=10**(snr_w_s/10)
                    b_=10**(snr_w_d/10)
                    c_=10**(snr_w_f/10)

                    snr_static_=10*np.log10(a_*(1-(1/c_)))
                    snr_diff_=10*np.log10(b_*(1-(1/c_)))
                    snr_all=10*np.log10(c_-1)

                    snr_f.append(snr_all)

                    snr_diff.append(snr_diff_)

                    snr_static.append(snr_static_)

        count+=1



fig,ax2=plt.subplots()
ax2.hist(snr_f,bins=np.arange(100,step=5),color="orange")
ax2.set_ylabel("No of Samples")
ax2.set_xlabel("Negative log of variance in dB")
ax2.set_title("SNR Mixture")
fig.tight_layout()
plt.savefig("/home/psrivastava/axis-2/IWAENC/dataset_generation/D7_1101/histf_.jpeg")

fig,ax2=plt.subplots()
ax2.hist(snr_diff,bins=np.arange(100,step=5),color="green")
ax2.set_ylabel("No of Samples")
ax2.set_xlabel("Negative log of variance in dB")
ax2.set_title("SNR DIFFUSE")
fig.tight_layout()

plt.savefig("/home/psrivastava/axis-2/IWAENC/dataset_generation/D7_1101/histd_.jpeg")


fig,ax2=plt.subplots()
ax2.hist(snr_static,bins=np.arange(100,step=5),color="blue")
ax2.set_ylabel("No of Samples")
ax2.set_xlabel("Negative log of variance in dB")
ax2.set_title("SNR STATIC")
fig.tight_layout()
plt.savefig("/home/psrivastava/axis-2/IWAENC/dataset_generation/D7_1101/hists_.jpeg")
