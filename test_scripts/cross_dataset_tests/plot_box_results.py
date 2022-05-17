import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import *

root_path="/home/psrivastava/axis-2/IWAENC/z_test/"

#Checkpoint file load

D7_chkp=np.load(root_path+"cross_dataset_tests"+"/D5_D7.npy")
D6_chkp=np.load(root_path+"cross_dataset_tests"+"/D5_D6.npy")
D5_chkp=np.load(root_path+"cross_dataset_tests"+"/D5_D5.npy")
D4_chkp=np.load(root_path+"cross_dataset_tests"+"/D5_D4.npy")
D3_chkp=np.load(root_path+"cross_dataset_tests"+"/D5_D3.npy")
D2_chkp=np.load(root_path+"cross_dataset_tests"+"/D5_D2.npy")
D1_chkp=np.load(root_path+"cross_dataset_tests"+"/D5_D1.npy")




D7_std_volume = 72.261470
D7_std_surface = 63.79672

D7_std_rt60=[ 0.40564, 0.38226, 0.30367, 0.26679, 0.27204, 0.23819]


D6_std_volume = 72.518733
D6_std_surface = 63.944219

D6_std_rt60=[ 0.38510, 0.341487, 0.250104, 0.207987, 0.214506, 0.194992]



D5_std_volume = 72.6143118
D5_std_surface = 64.029048

D5_std_rt60=[ 0.40875, 0.38669, 0.303711, 0.268418, 0.273678, 0.239206]

D4_std_volume = 72.45310
D4_std_surface = 63.83711

D4_std_rt60=[ 0.397998, 0.370756, 0.289561, 0.259480, 0.238140, 0.209692]

D3_std_volume = 72.334923
D3_std_surface = 63.80030

D3_std_rt60=[ 0.403562, 0.388824, 0.295057, 0.263443, 0.242138, 0.213495]


D2_std_volume = 72.760007
D2_std_surface = 64.187581

D2_std_rt60=[ 0.395032, 0.376963, 0.297600, 0.263775, 0.242831, 0.214957]


D1_std_volume = 72.14526
D1_std_surface = 63.60605

D1_std_rt60=[ 0.364063, 0.302710, 0.222462, 0.193005, 0.183209, 0.170326]

'''
D5_std_volume = 72.87116
D5_std_surface = 64.2039

D5_std_rt60=[ 0.284315, 0.306951, 0.278283, 0.238538, 0.219537, 0.193914]

D4_std_volume = 73.1247
D4_std_surface = 64.4198

D4_std_rt60=[ 0.28154,0.30246, 0.27551, 0.23776, 0.20887, 0.18190]

D3_std_volume = 72.68577
D3_std_surface = 63.88418

D3_std_rt60=[ 0.281505, 0.305646, 0.274250, 0.237036, 0.207895, 0.181039]


D2_std_volume = 72.03044
D2_std_surface = 63.72679

D2_std_rt60=[ 0.276834, 0.302112, 0.276772, 0.237419, 0.208285, 0.180662]


D1_std_volume = 72.085330
D1_std_surface = 63.73524

D1_std_rt60=[ 0.267740, 0.264852, 0.218090, 0.176837, 0.162080, 0.150589]
'''

D7_rt60_125=np.abs((D7_chkp[:,2]*D7_std_rt60[0])-(D7_chkp[:,10]))
where_are_NaNs = isnan(D7_rt60_125)
D7_rt60_125[where_are_NaNs] = 0
D7_rt60_250=np.abs((D7_chkp[:,3]*D7_std_rt60[1])-(D7_chkp[:,11]))
where_are_NaNs = isnan(D7_rt60_250)
D7_rt60_250[where_are_NaNs] = 0

D7_rt60_500=np.abs((D7_chkp[:,4]*D7_std_rt60[2])-(D7_chkp[:,12]))
where_are_NaNs = isnan(D7_rt60_500)
D7_rt60_500[where_are_NaNs] = 0


D7_rt60_1000=np.abs((D7_chkp[:,5]*D7_std_rt60[3])-(D7_chkp[:,13]))
where_are_NaNs = isnan(D7_rt60_1000)
D7_rt60_1000[where_are_NaNs] = 0

D7_rt60_2000=np.abs((D7_chkp[:,6]*D7_std_rt60[4])-(D7_chkp[:,14]))
where_are_NaNs = isnan(D7_rt60_2000)
D7_rt60_2000[where_are_NaNs] = 0

D7_rt60_4000=np.abs((D7_chkp[:,7]*D7_std_rt60[5])-(D7_chkp[:,15]))
where_are_NaNs = isnan(D7_rt60_4000)
D7_rt60_4000[where_are_NaNs] = 0


D7_surf=np.abs((D7_chkp[:,0]*D7_std_surface)-(D7_chkp[:,8]))
where_are_NaNs = isnan(D7_surf)
D7_surf[where_are_NaNs] = 0

D7_vol=np.abs((D7_chkp[:,1]*D7_std_volume)-(D7_chkp[:,9]))

where_are_NaNs = isnan(D7_vol)
D7_vol[where_are_NaNs] = 0


D6_rt60_125=np.abs((D6_chkp[:,2]*D6_std_rt60[0])-(D6_chkp[:,10]))
where_are_NaNs = isnan(D6_rt60_125)
D6_rt60_125[where_are_NaNs] = 0
D6_rt60_250=np.abs((D6_chkp[:,3]*D6_std_rt60[1])-(D6_chkp[:,11]))
where_are_NaNs = isnan(D6_rt60_250)
D6_rt60_250[where_are_NaNs] = 0

D6_rt60_500=np.abs((D6_chkp[:,4]*D6_std_rt60[2])-(D6_chkp[:,12]))
where_are_NaNs = isnan(D6_rt60_500)
D6_rt60_500[where_are_NaNs] = 0


D6_rt60_1000=np.abs((D6_chkp[:,5]*D6_std_rt60[3])-(D6_chkp[:,13]))
where_are_NaNs = isnan(D6_rt60_1000)
D6_rt60_1000[where_are_NaNs] = 0

D6_rt60_2000=np.abs((D6_chkp[:,6]*D6_std_rt60[4])-(D6_chkp[:,14]))
where_are_NaNs = isnan(D6_rt60_2000)
D6_rt60_2000[where_are_NaNs] = 0

D6_rt60_4000=np.abs((D6_chkp[:,7]*D6_std_rt60[5])-(D6_chkp[:,15]))
where_are_NaNs = isnan(D6_rt60_4000)
D6_rt60_4000[where_are_NaNs] = 0


D6_surf=np.abs((D6_chkp[:,0]*D6_std_surface)-(D6_chkp[:,8]))
where_are_NaNs = isnan(D6_surf)
D6_surf[where_are_NaNs] = 0

D6_vol=np.abs((D6_chkp[:,1]*D6_std_volume)-(D6_chkp[:,9]))

where_are_NaNs = isnan(D6_vol)
D6_vol[where_are_NaNs] = 0



















D5_rt60_125=np.abs((D5_chkp[:,2]*D5_std_rt60[0])-(D5_chkp[:,10]))
where_are_NaNs = isnan(D5_rt60_125)
D5_rt60_125[where_are_NaNs] = 0
D5_rt60_250=np.abs((D5_chkp[:,3]*D5_std_rt60[1])-(D5_chkp[:,11]))
where_are_NaNs = isnan(D5_rt60_250)
D5_rt60_250[where_are_NaNs] = 0

D5_rt60_500=np.abs((D5_chkp[:,4]*D5_std_rt60[2])-(D5_chkp[:,12]))
where_are_NaNs = isnan(D5_rt60_500)
D5_rt60_500[where_are_NaNs] = 0


D5_rt60_1000=np.abs((D5_chkp[:,5]*D5_std_rt60[3])-(D5_chkp[:,13]))
where_are_NaNs = isnan(D5_rt60_1000)
D5_rt60_1000[where_are_NaNs] = 0

D5_rt60_2000=np.abs((D5_chkp[:,6]*D5_std_rt60[4])-(D5_chkp[:,14]))
where_are_NaNs = isnan(D5_rt60_2000)
D5_rt60_2000[where_are_NaNs] = 0

D5_rt60_4000=np.abs((D5_chkp[:,7]*D5_std_rt60[5])-(D5_chkp[:,15]))
where_are_NaNs = isnan(D5_rt60_4000)
D5_rt60_4000[where_are_NaNs] = 0


D5_surf=np.abs((D5_chkp[:,0]*D5_std_surface)-(D5_chkp[:,8]))
where_are_NaNs = isnan(D5_surf)
D5_surf[where_are_NaNs] = 0

D5_vol=np.abs((D5_chkp[:,1]*D5_std_volume)-(D5_chkp[:,9]))

where_are_NaNs = isnan(D5_vol)
D5_vol[where_are_NaNs] = 0


D4_rt60_125=np.abs((D4_chkp[:,2]*D4_std_rt60[0])-(D4_chkp[:,10]))
where_are_NaNs = isnan(D4_rt60_125)
D4_rt60_125[where_are_NaNs] = 0

D4_rt60_250=np.abs((D4_chkp[:,3]*D4_std_rt60[1])-(D4_chkp[:,11]))
where_are_NaNs = isnan(D4_rt60_250)
D4_rt60_250[where_are_NaNs] = 0

D4_rt60_500=np.abs((D4_chkp[:,4]*D4_std_rt60[2])-(D4_chkp[:,12]))
where_are_NaNs = isnan(D4_rt60_500)
D4_rt60_500[where_are_NaNs] = 0

D4_rt60_1000=np.abs((D4_chkp[:,5]*D4_std_rt60[3])-(D4_chkp[:,13]))
where_are_NaNs = isnan(D4_rt60_1000)
D4_rt60_1000[where_are_NaNs] = 0

D4_rt60_2000=np.abs((D4_chkp[:,6]*D4_std_rt60[4])-(D4_chkp[:,14]))
where_are_NaNs = isnan(D4_rt60_2000)
D4_rt60_2000[where_are_NaNs] = 0

D4_rt60_4000=np.abs((D4_chkp[:,7]*D4_std_rt60[5])-(D4_chkp[:,15]))
where_are_NaNs = isnan(D4_rt60_4000)
D4_rt60_4000[where_are_NaNs] = 0


D4_surf=np.abs((D4_chkp[:,0]*D4_std_surface)-(D4_chkp[:,8]))
where_are_NaNs = isnan(D4_surf)
D4_surf[where_are_NaNs] = 0



D4_vol=np.abs((D4_chkp[:,1]*D4_std_volume)-(D4_chkp[:,9]))
where_are_NaNs = isnan(D4_vol)
D4_vol[where_are_NaNs] = 0



D3_rt60_125=np.abs((D3_chkp[:,2]*D3_std_rt60[0])-(D3_chkp[:,10]))
where_are_NaNs = isnan(D3_rt60_125)
D3_rt60_125[where_are_NaNs] = 0

D3_rt60_250=np.abs((D3_chkp[:,3]*D3_std_rt60[1])-(D3_chkp[:,11]))
where_are_NaNs = isnan(D3_rt60_250)
D3_rt60_250[where_are_NaNs] = 0

D3_rt60_500=np.abs((D3_chkp[:,4]*D3_std_rt60[2])-(D3_chkp[:,12]))
where_are_NaNs = isnan(D3_rt60_500)
D3_rt60_500[where_are_NaNs] = 0

D3_rt60_1000=np.abs((D3_chkp[:,5]*D3_std_rt60[3])-(D3_chkp[:,13]))
where_are_NaNs = isnan(D3_rt60_1000)
D3_rt60_1000[where_are_NaNs] = 0

D3_rt60_2000=np.abs((D3_chkp[:,6]*D3_std_rt60[4])-(D3_chkp[:,14]))
where_are_NaNs = isnan(D3_rt60_2000)
D3_rt60_2000[where_are_NaNs] = 0

D3_rt60_4000=np.abs((D3_chkp[:,7]*D3_std_rt60[5])-(D3_chkp[:,15]))
where_are_NaNs = isnan(D3_rt60_4000)
D3_rt60_4000[where_are_NaNs] = 0

D3_surf=np.abs((D3_chkp[:,0]*D3_std_surface)-(D3_chkp[:,8]))
where_are_NaNs = isnan(D3_surf)
D3_surf[where_are_NaNs] = 0

D3_vol=np.abs((D3_chkp[:,1]*D3_std_volume)-(D3_chkp[:,9]))
where_are_NaNs = isnan(D3_vol)
D3_vol[where_are_NaNs] = 0




D2_rt60_125=np.abs((D2_chkp[:,2]*D2_std_rt60[0])-(D2_chkp[:,10]))
where_are_NaNs = isnan(D2_rt60_125)
D2_rt60_125[where_are_NaNs] = 0


D2_rt60_250=np.abs((D2_chkp[:,3]*D2_std_rt60[1])-(D2_chkp[:,11]))
where_are_NaNs = isnan(D2_rt60_250)
D2_rt60_250[where_are_NaNs] = 0

D2_rt60_500=np.abs((D2_chkp[:,4]*D2_std_rt60[2])-(D2_chkp[:,12]))
where_are_NaNs = isnan(D2_rt60_500)
D2_rt60_500[where_are_NaNs] = 0


D2_rt60_1000=np.abs((D2_chkp[:,5]*D2_std_rt60[3])-(D2_chkp[:,13]))
where_are_NaNs = isnan(D2_rt60_1000)
D2_rt60_1000[where_are_NaNs] = 0

D2_rt60_2000=np.abs((D2_chkp[:,6]*D2_std_rt60[4])-(D2_chkp[:,14]))
where_are_NaNs = isnan(D2_rt60_2000)
D2_rt60_2000[where_are_NaNs] = 0

D2_rt60_4000=np.abs((D2_chkp[:,7]*D2_std_rt60[5])-(D2_chkp[:,15]))
where_are_NaNs = isnan(D2_rt60_4000)
D2_rt60_4000[where_are_NaNs] = 0

D2_surf=np.abs((D2_chkp[:,0]*D2_std_surface)-(D2_chkp[:,8]))
where_are_NaNs = isnan(D2_surf)
D2_surf[where_are_NaNs] = 0

D2_vol=np.abs((D2_chkp[:,1]*D2_std_volume)-(D2_chkp[:,9]))
where_are_NaNs = isnan(D2_vol)
D2_vol[where_are_NaNs] = 0


D1_rt60_125=np.abs((D1_chkp[:,2]*D1_std_rt60[0])-(D1_chkp[:,10]))
where_are_NaNs = isnan(D1_rt60_125)
D1_rt60_125[where_are_NaNs] = 0

D1_rt60_250=np.abs((D1_chkp[:,3]*D1_std_rt60[1])-(D1_chkp[:,11]))
where_are_NaNs = isnan(D1_rt60_250)
D1_rt60_250[where_are_NaNs] = 0

D1_rt60_500=np.abs((D1_chkp[:,4]*D1_std_rt60[2])-(D1_chkp[:,12]))
where_are_NaNs = isnan(D1_rt60_500)
D1_rt60_500[where_are_NaNs] = 0


D1_rt60_1000=np.abs((D1_chkp[:,5]*D1_std_rt60[3])-(D1_chkp[:,13]))
where_are_NaNs = isnan(D1_rt60_1000)
D1_rt60_1000[where_are_NaNs] = 0

D1_rt60_2000=np.abs((D1_chkp[:,6]*D1_std_rt60[4])-(D1_chkp[:,14]))
where_are_NaNs = isnan(D1_rt60_2000)
D1_rt60_2000[where_are_NaNs] = 0

D1_rt60_4000=np.abs((D1_chkp[:,7]*D1_std_rt60[5])-(D1_chkp[:,15]))
where_are_NaNs = isnan(D1_rt60_4000)
D1_rt60_4000[where_are_NaNs] = 0


D1_surf=np.abs((D1_chkp[:,0]*D1_std_surface)-(D1_chkp[:,8]))
where_are_NaNs = isnan(D1_surf)
D1_surf[where_are_NaNs] = 0

D1_vol=np.abs((D1_chkp[:,1]*D1_std_volume)-(D1_chkp[:,9]))

where_are_NaNs = isnan(D1_vol)
D1_vol[where_are_NaNs] = 0

print("D7","D6","D5","D4","D3","D2","D1")

print(np.mean(D7_rt60_125),np.mean(D6_rt60_125),np.mean(D5_rt60_125),np.mean(D4_rt60_125),np.mean(D3_rt60_125),np.mean(D2_rt60_125),np.mean(D1_rt60_125))
print(np.mean(D7_rt60_250),np.mean(D6_rt60_250),np.mean(D5_rt60_250),np.mean(D4_rt60_250),np.mean(D3_rt60_250),np.mean(D2_rt60_250),np.mean(D1_rt60_250))
print(np.mean(D7_rt60_500),np.mean(D6_rt60_500),np.mean(D5_rt60_500),np.mean(D4_rt60_500),np.mean(D3_rt60_500),np.mean(D2_rt60_500),np.mean(D1_rt60_500))
print(np.mean(D7_rt60_1000),np.mean(D6_rt60_1000),np.mean(D5_rt60_1000),np.mean(D4_rt60_1000),np.mean(D3_rt60_1000),np.mean(D2_rt60_1000),np.mean(D1_rt60_1000))
print(np.mean(D7_rt60_2000),np.mean(D6_rt60_2000),np.mean(D5_rt60_2000),np.mean(D4_rt60_2000),np.mean(D3_rt60_2000),np.mean(D2_rt60_2000),np.mean(D1_rt60_2000))
print(np.mean(D7_rt60_4000),np.mean(D6_rt60_4000),np.mean(D5_rt60_4000),np.mean(D4_rt60_4000),np.mean(D3_rt60_4000),np.mean(D2_rt60_4000),np.mean(D1_rt60_4000))
print(np.mean(D7_surf),np.mean(D6_surf),np.mean(D5_surf),np.mean(D4_surf),np.mean(D3_surf),np.mean(D2_surf),np.mean(D1_surf))
print(np.mean(D7_vol),np.mean(D6_vol),np.mean(D5_vol),np.mean(D4_vol),np.mean(D3_vol),np.mean(D2_vol),np.mean(D1_vol))

out_rt=False

fig,axs=plt.subplots(4,2,figsize=(10,25))

bplot1=axs[0,0].boxplot([D7_rt60_125,D6_rt60_125,D5_rt60_125,D4_rt60_125,D3_rt60_125,D2_rt60_125,D1_rt60_125],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,0].set_xticks([1,2,3,4,5,6,7])
axs[0,0].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[0,0].set_ylabel("Abs Err Sec")
axs[0,0].set_title("RT 60 125hz")

bplot2=axs[0,1].boxplot([D7_rt60_250,D6_rt60_250,D5_rt60_250,D4_rt60_250,D3_rt60_250,D2_rt60_250,D1_rt60_250],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,1].set_xticks([1,2,3,4,5,6,7])
axs[0,1].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[0,1].set_ylabel("Abs Err Sec")
axs[0,1].set_title("RT 60 250hz")

bplot3=axs[1,0].boxplot([D7_rt60_500,D6_rt60_500,D5_rt60_500,D4_rt60_500,D3_rt60_500,D2_rt60_500,D1_rt60_500],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[1,0].set_xticks([1,2,3,4,5,6,7])
axs[1,0].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[1,0].set_ylabel("Abs Err Sec")
axs[1,0].set_title("RT 60 500hz")

bplot4=axs[1,1].boxplot([D7_rt60_1000,D6_rt60_1000,D5_rt60_1000,D4_rt60_1000,D3_rt60_1000,D2_rt60_1000,D1_rt60_1000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[1,1].set_xticks([1,2,3,4,5,6,7])
axs[1,1].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[1,1].set_ylabel("Abs Err Sec")
axs[1,1].set_title("RT 60 1000hz")

bplot5=axs[2,0].boxplot([D7_rt60_2000,D6_rt60_2000,D5_rt60_2000,D4_rt60_2000,D3_rt60_2000,D2_rt60_2000,D1_rt60_2000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[2,0].set_xticks([1,2,3,4,5,6,7])
axs[2,0].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[2,0].set_ylabel("Abs Err Sec")
axs[2,0].set_title("RT 60 2000hz")

bplot6=axs[2,1].boxplot([D7_rt60_4000,D6_rt60_4000,D5_rt60_4000,D4_rt60_4000,D3_rt60_4000,D2_rt60_4000,D1_rt60_4000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[2,1].set_xticks([1,2,3,4,5,6,7])
axs[2,1].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[2,1].set_ylabel("Abs Err Sec")
axs[2,1].set_title("RT 60 4000hz")

bplot7=axs[3,0].boxplot([D7_surf,D6_surf,D5_surf,D4_surf,D3_surf,D2_surf,D1_surf],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,0].set_xticks([1,2,3,4,5,6,7])
axs[3,0].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[3,0].set_ylabel("Abs Err M2")
axs[3,0].set_title("Surface Err")

bplot8=axs[3,1].boxplot([D7_vol,D6_vol,D5_vol,D4_vol,D3_vol,D2_vol,D1_vol],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,1].set_xticks([1,2,3,4,5,6,7])
axs[3,1].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[3,1].set_ylabel("Abs Err M3")
axs[3,1].set_title("Volume Err")


colors=['pink','lightblue','lightgreen','orange','cyan','gold']


for bplot in (bplot1,bplot2,bplot3,bplot4,bplot5,bplot6,bplot7,bplot8):
    for patch,color in zip(bplot['boxes'],colors):
        patch.set_facecolor(color)


fig.tight_layout(pad=3.0)
#plt.xticks([1,2,3],('Dummy Bnf','bnf','Dummy M','M'))
#plt.title("Absolute Diff Estimated Mean And Target RT60")
plt.savefig("test_on_D7.png")
