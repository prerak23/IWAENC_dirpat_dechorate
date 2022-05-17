import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import *

root_path="/home/psrivastava/axis-2/IWAENC/z_test/"

D6_chkp=np.load(root_path+"D6_test_set"+"/D6_real_data_src_6.npy")
D5_chkp=np.load(root_path+"D5_test_set"+"/D5_real_data_src_6.npy")
D4_chkp=np.load(root_path+"D4_test_set"+"/D4_real_data_src_6.npy")
D3_chkp=np.load(root_path+"D3_test_set"+"/D3_real_data_src_6.npy")
D2_chkp=np.load(root_path+"D2_test_set"+"/D2_real_data_src_6.npy")
D1_chkp=np.load(root_path+"D1_test_set"+"/D1_real_data_src_6.npy")

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


volr=[80.141]
sa=[123.026]

sa=np.array([[123.026]]*D5_chkp.shape[0])
volr=np.array([[80.141]]*D5_chkp.shape[0])
rt60_=np.array([rt60['000000']]*D5_chkp.shape[0])

'''


#D5_chkp=np.concatenate((D5_chkp,sa,volr,rt60_),axis=1)
#print(D5_chkp.shape)
#D4_chkp=np.concatenate((D4_chkp,sa,volr,rt60_),axis=1)
#D3_chkp=np.concatenate((D3_chkp,sa,volr,rt60_),axis=1)
#D2_chkp=np.concatenate((D2_chkp,sa,volr,rt60_),axis=1)
#D1_chkp=np.concatenate((D1_chkp,sa,volr,rt60_),axis=1)


#D5_rt60_125=np.abs((D5_chkp[:,2]*D5_std_rt60[0])-(D5_chkp[:,10]))
#D5_rt60_250=np.abs((D5_chkp[:,3]*D5_std_rt60[1])-(D5_chkp[:,11]))


D6_rt60_500=np.abs(D6_chkp[:,4])
D6_rt60_1000=np.abs(D6_chkp[:,5])
D6_rt60_2000=np.abs(D6_chkp[:,6])
D6_rt60_4000=np.abs(D6_chkp[:,7])

D6_surf=np.abs(D6_chkp[:,0])
D6_vol=np.abs(D6_chkp[:,1])


D5_rt60_500=np.abs(D5_chkp[:,4])
D5_rt60_1000=np.abs(D5_chkp[:,5])
D5_rt60_2000=np.abs(D5_chkp[:,6])
D5_rt60_4000=np.abs(D5_chkp[:,7])

D5_surf=np.abs(D5_chkp[:,0])
D5_vol=np.abs(D5_chkp[:,1])




#D4_rt60_125=np.abs((D4_chkp[:,2]*D4_std_rt60[0])-(D4_chkp[:,10]))
#where_are_NaNs = isnan(D4_rt60_125)
#D4_rt60_125[where_are_NaNs] = 0

#D4_rt60_250=np.abs((D4_chkp[:,3]*D4_std_rt60[1])-(D4_chkp[:,11]))
#where_are_NaNs = isnan(D4_rt60_250)
#D4_rt60_250[where_are_NaNs] = 0

D4_rt60_500=np.abs(D4_chkp[:,4])
where_are_NaNs = isnan(D4_rt60_500)
D4_rt60_500[where_are_NaNs] = 0

D4_rt60_1000=np.abs(D4_chkp[:,5])
where_are_NaNs = isnan(D4_rt60_1000)
D4_rt60_1000[where_are_NaNs] = 0

D4_rt60_2000=np.abs(D4_chkp[:,6])
where_are_NaNs = isnan(D4_rt60_2000)
D4_rt60_2000[where_are_NaNs] = 0

D4_rt60_4000=np.abs(D4_chkp[:,7])
where_are_NaNs = isnan(D4_rt60_4000)
D4_rt60_4000[where_are_NaNs] = 0


D4_surf=np.abs(D4_chkp[:,0])
where_are_NaNs = isnan(D4_surf)
D4_surf[where_are_NaNs] = 0



D4_vol=np.abs(D4_chkp[:,1])
where_are_NaNs = isnan(D4_vol)
D4_vol[where_are_NaNs] = 0



#D3_rt60_125=np.abs((D3_chkp[:,2]*D3_std_rt60[0])-(D3_chkp[:,10]))
#where_are_NaNs = isnan(D3_rt60_125)
#D3_rt60_125[where_are_NaNs] = 0

#D3_rt60_250=np.abs((D3_chkp[:,3]*D3_std_rt60[1])-(D3_chkp[:,11]))
#where_are_NaNs = isnan(D3_rt60_250)
#D3_rt60_250[where_are_NaNs] = 0

D3_rt60_500=np.abs(D3_chkp[:,4])
where_are_NaNs = isnan(D3_rt60_500)
D3_rt60_500[where_are_NaNs] = 0

D3_rt60_1000=np.abs(D3_chkp[:,5])
where_are_NaNs = isnan(D3_rt60_1000)
D3_rt60_1000[where_are_NaNs] = 0

D3_rt60_2000=np.abs(D3_chkp[:,6])
where_are_NaNs = isnan(D3_rt60_2000)
D3_rt60_2000[where_are_NaNs] = 0

D3_rt60_4000=np.abs(D3_chkp[:,7])
where_are_NaNs = isnan(D3_rt60_4000)
D3_rt60_4000[where_are_NaNs] = 0

D3_surf=np.abs(D3_chkp[:,0])
where_are_NaNs = isnan(D3_surf)
D3_surf[where_are_NaNs] = 0

D3_vol=np.abs(D3_chkp[:,1])
where_are_NaNs = isnan(D3_vol)
D3_vol[where_are_NaNs] = 0




#D2_rt60_125=np.abs((D2_chkp[:,2]*D2_std_rt60[0])-(D2_chkp[:,10]))
#where_are_NaNs = isnan(D2_rt60_125)
#D2_rt60_125[where_are_NaNs] = 0


#D2_rt60_250=np.abs((D2_chkp[:,3]*D2_std_rt60[1])-(D2_chkp[:,11]))
#where_are_NaNs = isnan(D2_rt60_250)
#D2_rt60_250[where_are_NaNs] = 0

D2_rt60_500=np.abs(D2_chkp[:,4])
where_are_NaNs = isnan(D2_rt60_500)
D2_rt60_500[where_are_NaNs] = 0


D2_rt60_1000=np.abs(D2_chkp[:,5])
where_are_NaNs = isnan(D2_rt60_1000)
D2_rt60_1000[where_are_NaNs] = 0

D2_rt60_2000=np.abs(D2_chkp[:,6])
where_are_NaNs = isnan(D2_rt60_2000)
D2_rt60_2000[where_are_NaNs] = 0

D2_rt60_4000=np.abs(D2_chkp[:,7])
where_are_NaNs = isnan(D2_rt60_4000)
D2_rt60_4000[where_are_NaNs] = 0

D2_surf=np.abs(D2_chkp[:,0])
where_are_NaNs = isnan(D2_surf)
D2_surf[where_are_NaNs] = 0

D2_vol=np.abs(D2_chkp[:,1])
where_are_NaNs = isnan(D2_vol)
D2_vol[where_are_NaNs] = 0


#D1_rt60_125=np.abs((D1_chkp[:,2]*D1_std_rt60[0])-(D1_chkp[:,10]))
#where_are_NaNs = isnan(D1_rt60_125)
#D1_rt60_125[where_are_NaNs] = 0

#D1_rt60_250=np.abs((D1_chkp[:,3]*D1_std_rt60[1])-(D1_chkp[:,11]))
#where_are_NaNs = isnan(D1_rt60_250)
#D1_rt60_250[where_are_NaNs] = 0

D1_rt60_500=np.abs(D1_chkp[:,4])
where_are_NaNs = isnan(D1_rt60_500)
D1_rt60_500[where_are_NaNs] = 0


D1_rt60_1000=np.abs(D1_chkp[:,5])
where_are_NaNs = isnan(D1_rt60_1000)
D1_rt60_1000[where_are_NaNs] = 0

D1_rt60_2000=np.abs(D1_chkp[:,6])
where_are_NaNs = isnan(D1_rt60_2000)
D1_rt60_2000[where_are_NaNs] = 0

D1_rt60_4000=np.abs(D1_chkp[:,7])
where_are_NaNs = isnan(D1_rt60_4000)
D1_rt60_4000[where_are_NaNs] = 0


D1_surf=np.abs(D1_chkp[:,0])
where_are_NaNs = isnan(D1_surf)
D1_surf[where_are_NaNs] = 0

D1_vol=np.abs(D1_chkp[:,1])

where_are_NaNs = isnan(D1_vol)
D1_vol[where_are_NaNs] = 0




out_rt=False

fig,axs=plt.subplots(3,2,figsize=(10,25))
'''
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
'''


bplot3=axs[0,0].boxplot([D6_rt60_500,D5_rt60_500,D4_rt60_500,D3_rt60_500,D2_rt60_500,D1_rt60_500],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,0].set_xticks([1,2,3,4,5,6])
axs[0,0].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
axs[0,0].set_ylabel("Abs Err Sec")
axs[0,0].set_title("RT 60 500hz")

bplot4=axs[0,1].boxplot([D6_rt60_1000,D5_rt60_1000,D4_rt60_1000,D3_rt60_1000,D2_rt60_1000,D1_rt60_1000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,1].set_xticks([1,2,3,4,5,6])
axs[0,1].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
axs[0,1].set_ylabel("Abs Err Sec")
axs[0,1].set_title("RT 60 1000hz")

bplot5=axs[1,0].boxplot([D6_rt60_2000,D5_rt60_2000,D4_rt60_2000,D3_rt60_2000,D2_rt60_2000,D1_rt60_2000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[1,0].set_xticks([1,2,3,4,5,6])
axs[1,0].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
axs[1,0].set_ylabel("Abs Err Sec")
axs[1,0].set_title("RT 60 2000hz")

bplot6=axs[1,1].boxplot([D6_rt60_4000,D5_rt60_4000,D4_rt60_4000,D3_rt60_4000,D2_rt60_4000,D1_rt60_4000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[1,1].set_xticks([1,2,3,4,5,6])
axs[1,1].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
axs[1,1].set_ylabel("Abs Err Sec")
axs[1,1].set_title("RT 60 4000hz")

bplot7=axs[2,0].boxplot([D6_surf,D5_surf,D4_surf,D3_surf,D2_surf,D1_surf],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[2,0].set_xticks([1,2,3,4,5,6])
axs[2,0].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
axs[2,0].set_ylabel("Abs Err M2")
axs[2,0].set_title("Surface Err")

bplot8=axs[2,1].boxplot([D6_vol,D5_vol,D4_vol,D3_vol,D2_vol,D1_vol],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[2,1].set_xticks([1,2,3,4,5,6])
axs[2,1].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
axs[2,1].set_ylabel("Abs Err M3")
axs[2,1].set_title("Volume Err")


colors=['pink','lightblue','lightgreen','orange','cyan']


for bplot in (bplot3,bplot4,bplot5,bplot6,bplot7,bplot8):
    for patch,color in zip(bplot['boxes'],colors):
        patch.set_facecolor(color)


fig.tight_layout(pad=3.0)
#plt.xticks([1,2,3],('Dummy Bnf','bnf','Dummy M','M'))
#plt.title("Absolute Diff Estimated Mean And Target RT60")
plt.savefig("real_decorate_src_6.png")
