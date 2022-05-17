import numpy as np
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
sns.set_theme()



path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D1_0000/noisy_mixtures/"

anno_file=h5py.File(path+"D1_0000_aggregated_mixture_annotations.hdf5","r")
surface_=[]
volume_=[]
rt60_500=[]
rt60_1000=[]
rt60_2000=[]
rt60_4000=[]

for rooms in anno_file["room_nos"].keys():
    sa=anno_file["room_nos"][rooms]["surface"][()][0]
    vo=anno_file["room_nos"][rooms]["volume"][()][0]
    rt_500=anno_file["room_nos"][rooms]["rt60"][()][0,2]
    rt_1000=anno_file["room_nos"][rooms]["rt60"][()][0,3]
    rt_2000=anno_file["room_nos"][rooms]["rt60"][()][0,4]
    rt_4000=anno_file["room_nos"][rooms]["rt60"][()][0,5]

    surface_.append(sa)
    volume_.append(vo)
    rt60_500.append(rt_500)
    rt60_1000.append(rt_1000)
    rt60_2000.append(rt_2000)
    rt60_4000.append(rt_4000)


rt60={'000000':[0,0,0.18,0.14,0.16,0.22],
'011000':[0,0,0.4,0.33,0.25,0.25],
'011100':[0,0,0.46,0.34,0.30,0.37],
'011110':[0,0,0.60,0.56,0.48,0.55],
'011111':[0,0,0.75,0.73,0.68,0.81],
'010000':[0,0,0.22,0.19,0.18,0.22],
'001000':[0,0,0.22,0.19,0.18,0.22],
'000100':[0,0,0.21,0.19,0.20,0.23],
'000010':[0,0,0.21,0.18,0.18,0.26],
'000001':[0,0,0.22,0.19,0.18,0.24],
'020002':[0,0,0.37,0.26,0.24,0.28]}

volr=[80.141]*10
sa=[123.026]*10
de1=[rt60['000000'][2],rt60['000001'][2],rt60['000010'][2],rt60['000100'][2],rt60['010000'][2],rt60['011000'][2],rt60['011100'][2],rt60['011110'][2],rt60['011111'][2],rt60['020002'][2]]
de2=[rt60['000000'][3],rt60['000001'][3],rt60['000010'][3],rt60['000100'][3],rt60['010000'][3],rt60['011000'][3],rt60['011100'][3],rt60['011110'][3],rt60['011111'][3],rt60['020002'][3]]
de3=[rt60['000000'][4],rt60['000001'][4],rt60['000010'][4],rt60['000100'][4],rt60['010000'][4],rt60['011000'][4],rt60['011100'][4],rt60['011110'][4],rt60['011111'][4],rt60['020002'][4]]
de4=[rt60['000000'][5],rt60['000001'][5],rt60['000010'][5],rt60['000100'][5],rt60['010000'][5],rt60['011000'][5],rt60['011100'][5],rt60['011110'][5],rt60['011111'][5],rt60['020002'][5]]
print(len(de1))

out_rt=False

fig,axs=plt.subplots(4,2,figsize=(10,25))

axs[0,0].scatter(rt60_500,surface_,color="lightblue")
axs[0,0].scatter(de1,sa,marker="x",color="orange")
axs[0,0].set_xlabel("RT60 500 seconds")
axs[0,0].set_ylabel("Surface m2")
axs[0,0].set_title("RT 60 500hz")


axs[0,1].scatter(rt60_500,volume_,color="lightblue")
axs[0,1].scatter(de1,volr,marker="x",color="orange")
axs[0,1].set_xlabel("RT60 500 seconds")
axs[0,1].set_ylabel("Volume m3")
axs[0,1].set_title("RT 60 500hz")

axs[1,0].scatter(rt60_1000,surface_,color="lightblue")
axs[1,0].scatter(de2,sa,marker="x",color="orange")
axs[1,0].set_xlabel("RT60 1000 seconds")
axs[1,0].set_ylabel("Surface m2")
axs[1,0].set_title("RT 60 1000hz")


axs[1,1].scatter(rt60_1000,volume_,color="lightblue")
axs[1,1].scatter(de2,volr,marker="x",color="orange")
axs[1,1].set_xlabel("RT60 1000 seconds")
axs[1,1].set_ylabel("Volume m3")
axs[1,1].set_title("RT 60 1000hz")


axs[2,0].scatter(rt60_2000,surface_,color="lightblue")
axs[2,0].scatter(de3,sa,marker="x",color="orange")
axs[2,0].set_xlabel("RT60 2000 seconds")
axs[2,0].set_ylabel("Surface m2")
axs[2,0].set_title("RT 60 2000hz")


axs[2,1].scatter(rt60_2000,volume_,color="lightblue")
axs[2,1].scatter(de3,volr,marker="x",color="orange")
axs[2,1].set_xlabel("RT60 2000 seconds")
axs[2,1].set_ylabel("Volume m3")
axs[2,1].set_title("RT 60 2000hz")


axs[3,0].scatter(rt60_4000,surface_,color="lightblue")
axs[3,0].scatter(de4,sa,marker="x",color="orange")
axs[3,0].set_xlabel("RT60 4000 seconds")
axs[3,0].set_ylabel("Surface m2")
axs[3,0].set_title("RT 60 4000hz")


axs[3,1].scatter(rt60_4000,volume_,color="lightblue")
axs[3,1].scatter(de4,volr,marker="x",color="orange")
axs[3,1].set_xlabel("RT60 4000 seconds")
axs[3,1].set_ylabel("Volume m3")
axs[3,1].set_title("RT 60 4000hz")







plt.savefig("correlation_D1.png")
