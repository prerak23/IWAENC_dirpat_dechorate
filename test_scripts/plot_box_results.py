import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import *

root_path="/home/psrivastava/axis-2/IWAENC/z_test/"

#Checkpoint file load

D5_chkp=np.load(root_path+"D5_test_set"+"/vp_3_test_same_src.npy")
D4_chkp=np.load(root_path+"D4_test_set"+"/vp_3_test.npy")
D3_chkp=np.load(root_path+"D3_test_set"+"/vp_3_test.npy")
D2_chkp=np.load(root_path+"D2_test_set"+"/vp_3_test.npy")
D1_chkp=np.load(root_path+"D1_test_set"+"/vp_3_test.npy")


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



D5_rt60_125=np.abs((D5_chkp[:,2]*D5_std_rt60[0])-(D5_chkp[:,10]))
D5_rt60_250=np.abs((D5_chkp[:,3]*D5_std_rt60[1])-(D5_chkp[:,11]))
D5_rt60_500=np.abs((D5_chkp[:,4]*D5_std_rt60[2])-(D5_chkp[:,12]))
D5_rt60_1000=np.abs((D5_chkp[:,5]*D5_std_rt60[3])-(D5_chkp[:,13]))
D5_rt60_2000=np.abs((D5_chkp[:,6]*D5_std_rt60[4])-(D5_chkp[:,14]))
D5_rt60_4000=np.abs((D5_chkp[:,7]*D5_std_rt60[5])-(D5_chkp[:,15]))

D5_surf=np.abs((D5_chkp[:,0]*D5_std_surface)-(D5_chkp[:,8]))
D5_vol=np.abs((D5_chkp[:,1]*D5_std_volume)-(D5_chkp[:,9]))




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


out_rt=False

fig,axs=plt.subplots(4,2,figsize=(10,25))

bplot1=axs[0,0].boxplot([D5_rt60_125,D4_rt60_125,D3_rt60_125,D2_rt60_125,D1_rt60_125],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,0].set_xticks([1,2,3,4,5])
axs[0,0].set_xticklabels(['D5','D4','D3','D2','D1'],rotation=45)
axs[0,0].set_ylabel("Abs Err Sec")
axs[0,0].set_title("RT 60 125hz")

bplot2=axs[0,1].boxplot([D5_rt60_250,D4_rt60_250,D3_rt60_250,D2_rt60_250,D1_rt60_250],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,1].set_xticks([1,2,3,4,5])
axs[0,1].set_xticklabels(['D5','D4','D3','D2','D1'],rotation=45)
axs[0,1].set_ylabel("Abs Err Sec")
axs[0,1].set_title("RT 60 250hz")

bplot3=axs[1,0].boxplot([D5_rt60_500,D4_rt60_500,D3_rt60_500,D2_rt60_500,D1_rt60_500],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[1,0].set_xticks([1,2,3,4,5])
axs[1,0].set_xticklabels(['D5','D4','D3','D2','D1'],rotation=45)
axs[1,0].set_ylabel("Abs Err Sec")
axs[1,0].set_title("RT 60 500hz")

bplot4=axs[1,1].boxplot([D5_rt60_1000,D4_rt60_1000,D3_rt60_1000,D2_rt60_1000,D1_rt60_1000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[1,1].set_xticks([1,2,3,4,5])
axs[1,1].set_xticklabels(['D5','D4','D3','D2','D1'],rotation=45)
axs[1,1].set_ylabel("Abs Err Sec")
axs[1,1].set_title("RT 60 1000hz")

bplot5=axs[2,0].boxplot([D5_rt60_2000,D4_rt60_2000,D3_rt60_2000,D2_rt60_2000,D1_rt60_2000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[2,0].set_xticks([1,2,3,4,5])
axs[2,0].set_xticklabels(['D5','D4','D3','D2','D1'],rotation=45)
axs[2,0].set_ylabel("Abs Err Sec")
axs[2,0].set_title("RT 60 2000hz")

bplot6=axs[2,1].boxplot([D5_rt60_4000,D4_rt60_4000,D3_rt60_4000,D2_rt60_4000,D1_rt60_4000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[2,1].set_xticks([1,2,3,4,5])
axs[2,1].set_xticklabels(['D5','D4','D3','D2','D1'],rotation=45)
axs[2,1].set_ylabel("Abs Err Sec")
axs[2,1].set_title("RT 60 4000hz")

bplot7=axs[3,0].boxplot([D5_surf,D4_surf,D3_surf,D2_surf,D1_surf],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,0].set_xticks([1,2,3,4,5])
axs[3,0].set_xticklabels(['D5','D4','D3','D2','D1'],rotation=45)
axs[3,0].set_ylabel("Abs Err M2")
axs[3,0].set_title("Surface Err")

bplot8=axs[3,1].boxplot([D5_vol,D4_vol,D3_vol,D2_vol,D1_vol],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,1].set_xticks([1,2,3,4,5])
axs[3,1].set_xticklabels(['D5','D4','D3','D2','D1'],rotation=45)
axs[3,1].set_ylabel("Abs Err M3")
axs[3,1].set_title("Volume Err")


colors=['pink','lightblue','lightgreen','orange','cyan']


for bplot in (bplot1,bplot2,bplot3,bplot4,bplot5,bplot6,bplot7,bplot8):
    for patch,color in zip(bplot['boxes'],colors):
        patch.set_facecolor(color)


fig.tight_layout(pad=3.0)
#plt.xticks([1,2,3],('Dummy Bnf','bnf','Dummy M','M'))
#plt.title("Absolute Diff Estimated Mean And Target RT60")
plt.savefig("dataset_comparasion_test_set_vp3_same_src_D5.png")
