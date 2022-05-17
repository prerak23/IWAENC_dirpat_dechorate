import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import *

root_path="/home/psrivastava/axis-2/IWAENC/z_test/"

D7_chkp=np.load(root_path+"D7_test_set"+"/D7_real_data_020002.npy")
D6_chkp=np.load(root_path+"D6_test_set"+"/D6_real_data_020002.npy")
D5_chkp=np.load(root_path+"D5_test_set"+"/D5_real_data_020002.npy")
D4_chkp=np.load(root_path+"D4_test_set"+"/D4_real_data_020002.npy")
D3_chkp=np.load(root_path+"D3_test_set"+"/D3_real_data_020002.npy")
D2_chkp=np.load(root_path+"D2_test_set"+"/D2_real_data_020002.npy")
D1_chkp=np.load(root_path+"D1_test_set"+"/D1_real_data_020002.npy")




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



#[0.31773088 0.26774256 0.16348391 0.09895241 0.09950609 0.10233923]

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


'''
rt60={'000000':[0.31773088 ,0.26774256, 0.16348391, 0.09895241, 0.09950609, 0.10233923],
'011000':[0.30211865 ,0.3081504 , 0.30591411, 0.24827514, 0.2037211 , 0.17500304],
'011100':[0.30828875 ,0.31672684, 0.34249594, 0.27772315, 0.23367071, 0.20984031],
'011110':[0.29298232 ,0.36625056, 0.41388529, 0.46165814, 0.38912755, 0.33056917],
'011111':[0.29994518 ,0.42850027, 0.56735991, 0.62755461, 0.5809405 , 0.55043453],
'010000':[0.33279878 ,0.32403993, 0.29313552, 0.20991954, 0.16010863, 0.14187958],
'001000':[0.2776678  ,0.2694092 , 0.18622227, 0.13326911, 0.12637406, 0.13210042],
'000100':[0.24561399 ,0.26454351, 0.19228435, 0.1219367 , 0.12532883, 0.12059272],
'000010':[0.28172177 ,0.27036614, 0.17492325, 0.12604166, 0.12242046, 0.1297927 ],
'000001':[0.31993617 ,0.2815543 , 0.1788875 , 0.11093442, 0.11391684, 0.12472031],
'020002':[0.32970609 ,0.28980944, 0.25241602, 0.19910788, 0.17197203, 0.15972939]}
'''


volr=[80.141]
sa=[123.026]
sa=np.array([[123.026]]*D5_chkp.shape[0])
volr=np.array([[80.141]]*D5_chkp.shape[0])
rt60_=np.array([rt60['020002']]*D5_chkp.shape[0])

D6_chkp=np.concatenate((D6_chkp,sa,volr,rt60_),axis=1)
D5_chkp=np.concatenate((D5_chkp,sa,volr,rt60_),axis=1)
print(D5_chkp.shape)
D4_chkp=np.concatenate((D4_chkp,sa,volr,rt60_),axis=1)
D3_chkp=np.concatenate((D3_chkp,sa,volr,rt60_),axis=1)
D2_chkp=np.concatenate((D2_chkp,sa,volr,rt60_),axis=1)
D1_chkp=np.concatenate((D1_chkp,sa,volr,rt60_),axis=1)




D6_rt60_125=np.abs((D6_chkp[:,2]*D6_std_rt60[0])-(D6_chkp[:,10]))
D6_rt60_250=np.abs((D6_chkp[:,3]*D6_std_rt60[1])-(D6_chkp[:,11]))

D6_rt60_500=np.abs((D6_chkp[:,4]*D6_std_rt60[2])-(D6_chkp[:,12]))
D6_rt60_1000=np.abs((D6_chkp[:,5]*D6_std_rt60[3])-(D6_chkp[:,13]))
D6_rt60_2000=np.abs((D6_chkp[:,6]*D6_std_rt60[4])-(D6_chkp[:,14]))
D6_rt60_4000=np.abs((D6_chkp[:,7]*D6_std_rt60[5])-(D6_chkp[:,15]))

D6_surf=np.abs((D6_chkp[:,0]*D6_std_surface)-(D6_chkp[:,8]))
D6_vol=np.abs((D6_chkp[:,1]*D6_std_volume)-(D6_chkp[:,9]))


D5_rt60_125=np.abs((D5_chkp[:,2]*D5_std_rt60[0])-(D5_chkp[:,10]))
D5_rt60_250=np.abs((D5_chkp[:,3]*D5_std_rt60[1])-(D5_chkp[:,11]))

D5_rt60_500=np.abs((D5_chkp[:,4]*D5_std_rt60[2])-(D5_chkp[:,12]))
D5_rt60_1000=np.abs((D5_chkp[:,5]*D5_std_rt60[3])-(D5_chkp[:,13]))
D5_rt60_2000=np.abs((D5_chkp[:,6]*D5_std_rt60[4])-(D5_chkp[:,14]))
D5_rt60_4000=np.abs((D5_chkp[:,7]*D5_std_rt60[5])-(D5_chkp[:,15]))

D5_surf=np.abs((D5_chkp[:,0]*D5_std_surface)-(D5_chkp[:,8]))
D5_vol=np.abs((D5_chkp[:,1]*D5_std_volume)-(D5_chkp[:,9]))




D4_rt60_125=np.abs((D4_chkp[:,2]*D4_std_rt60[0])-(D4_chkp[:,10]))
#where_are_NaNs = isnan(D4_rt60_125)
#D4_rt60_125[where_are_NaNs] = 0

D4_rt60_250=np.abs((D4_chkp[:,3]*D4_std_rt60[1])-(D4_chkp[:,11]))
#where_are_NaNs = isnan(D4_rt60_250)
#D4_rt60_250[where_are_NaNs] = 0

D4_rt60_500=np.abs((D4_chkp[:,4]*D4_std_rt60[2])-(D4_chkp[:,12]))
#where_are_NaNs = isnan(D4_rt60_500)
#D4_rt60_500[where_are_NaNs] = 0

D4_rt60_1000=np.abs((D4_chkp[:,5]*D4_std_rt60[3])-(D4_chkp[:,13]))
#where_are_NaNs = isnan(D4_rt60_1000)
#D4_rt60_1000[where_are_NaNs] = 0

D4_rt60_2000=np.abs((D4_chkp[:,6]*D4_std_rt60[4])-(D4_chkp[:,14]))
#where_are_NaNs = isnan(D4_rt60_2000)
#D4_rt60_2000[where_are_NaNs] = 0

D4_rt60_4000=np.abs((D4_chkp[:,7]*D4_std_rt60[5])-(D4_chkp[:,15]))
#where_are_NaNs = isnan(D4_rt60_4000)
#D4_rt60_4000[where_are_NaNs] = 0


D4_surf=np.abs((D4_chkp[:,0]*D4_std_surface)-(D4_chkp[:,8]))

where_are_NaNs = isnan(D4_surf)
print("D4 NAN",where_are_NaNs)
D4_surf[where_are_NaNs] = 0



D4_vol=np.abs((D4_chkp[:,1]*D4_std_volume)-(D4_chkp[:,9]))
where_are_NaNs = isnan(D4_vol)
print("D4 NAN vol",where_are_NaNs)
D4_vol[where_are_NaNs] = 0



D3_rt60_125=np.abs((D3_chkp[:,2]*D3_std_rt60[0])-(D3_chkp[:,10]))
#where_are_NaNs = isnan(D3_rt60_125)
#D3_rt60_125[where_are_NaNs] = 0

D3_rt60_250=np.abs((D3_chkp[:,3]*D3_std_rt60[1])-(D3_chkp[:,11]))
#where_are_NaNs = isnan(D3_rt60_250)
#D3_rt60_250[where_are_NaNs] = 0

D3_rt60_500=np.abs((D3_chkp[:,4]*D3_std_rt60[2])-(D3_chkp[:,12]))
#where_are_NaNs = isnan(D3_rt60_500)
#D3_rt60_500[where_are_NaNs] = 0

D3_rt60_1000=np.abs((D3_chkp[:,5]*D3_std_rt60[3])-(D3_chkp[:,13]))
#where_are_NaNs = isnan(D3_rt60_1000)
#D3_rt60_1000[where_are_NaNs] = 0

D3_rt60_2000=np.abs((D3_chkp[:,6]*D3_std_rt60[4])-(D3_chkp[:,14]))
#where_are_NaNs = isnan(D3_rt60_2000)
#D3_rt60_2000[where_are_NaNs] = 0

D3_rt60_4000=np.abs((D3_chkp[:,7]*D3_std_rt60[5])-(D3_chkp[:,15]))
#where_are_NaNs = isnan(D3_rt60_4000)
#D3_rt60_4000[where_are_NaNs] = 0

D3_surf=np.abs((D3_chkp[:,0]*D3_std_surface)-(D3_chkp[:,8]))
#where_are_NaNs = isnan(D3_surf)
#D3_surf[where_are_NaNs] = 0

D3_vol=np.abs((D3_chkp[:,1]*D3_std_volume)-(D3_chkp[:,9]))
#where_are_NaNs = isnan(D3_vol)
#D3_vol[where_are_NaNs] = 0




D2_rt60_125=np.abs((D2_chkp[:,2]*D2_std_rt60[0])-(D2_chkp[:,10]))
#where_are_NaNs = isnan(D2_rt60_125)
#D2_rt60_125[where_are_NaNs] = 0


D2_rt60_250=np.abs((D2_chkp[:,3]*D2_std_rt60[1])-(D2_chkp[:,11]))
#where_are_NaNs = isnan(D2_rt60_250)
#D2_rt60_250[where_are_NaNs] = 0

D2_rt60_500=np.abs((D2_chkp[:,4]*D2_std_rt60[2])-(D2_chkp[:,12]))
#where_are_NaNs = isnan(D2_rt60_500)
#D2_rt60_500[where_are_NaNs] = 0


D2_rt60_1000=np.abs((D2_chkp[:,5]*D2_std_rt60[3])-(D2_chkp[:,13]))
#where_are_NaNs = isnan(D2_rt60_1000)
#D2_rt60_1000[where_are_NaNs] = 0

D2_rt60_2000=np.abs((D2_chkp[:,6]*D2_std_rt60[4])-(D2_chkp[:,14]))
#where_are_NaNs = isnan(D2_rt60_2000)
#D2_rt60_2000[where_are_NaNs] = 0

D2_rt60_4000=np.abs((D2_chkp[:,7]*D2_std_rt60[5])-(D2_chkp[:,15]))
#where_are_NaNs = isnan(D2_rt60_4000)
#D2_rt60_4000[where_are_NaNs] = 0

D2_surf=np.abs((D2_chkp[:,0]*D2_std_surface)-(D2_chkp[:,8]))
where_are_NaNs = isnan(D2_surf)
print("D2 NAN",where_are_NaNs)
D2_surf[where_are_NaNs] = 0

D2_vol=np.abs((D2_chkp[:,1]*D2_std_volume)-(D2_chkp[:,9]))
where_are_NaNs = isnan(D2_vol)
print("D2 NAN vol",where_are_NaNs)

D2_vol[where_are_NaNs] = 0


D1_rt60_125=np.abs((D1_chkp[:,2]*D1_std_rt60[0])-(D1_chkp[:,10]))
#where_are_NaNs = isnan(D1_rt60_125)
#D1_rt60_125[where_are_NaNs] = 0

D1_rt60_250=np.abs((D1_chkp[:,3]*D1_std_rt60[1])-(D1_chkp[:,11]))
#where_are_NaNs = isnan(D1_rt60_250)
#D1_rt60_250[where_are_NaNs] = 0

D1_rt60_500=np.abs((D1_chkp[:,4]*D1_std_rt60[2])-(D1_chkp[:,12]))
#where_are_NaNs = isnan(D1_rt60_500)
#D1_rt60_500[where_are_NaNs] = 0


D1_rt60_1000=np.abs((D1_chkp[:,5]*D1_std_rt60[3])-(D1_chkp[:,13]))
#where_are_NaNs = isnan(D1_rt60_1000)
#D1_rt60_1000[where_are_NaNs] = 0

D1_rt60_2000=np.abs((D1_chkp[:,6]*D1_std_rt60[4])-(D1_chkp[:,14]))
#where_are_NaNs = isnan(D1_rt60_2000)
#D1_rt60_2000[where_are_NaNs] = 0

D1_rt60_4000=np.abs((D1_chkp[:,7]*D1_std_rt60[5])-(D1_chkp[:,15]))
#where_are_NaNs = isnan(D1_rt60_4000)
#D1_rt60_4000[where_are_NaNs] = 0


D1_surf=np.abs((D1_chkp[:,0]*D1_std_surface)-(D1_chkp[:,8]))
where_are_NaNs = isnan(D1_surf)
print("D1 NAN",where_are_NaNs)
D1_surf[where_are_NaNs] = 0

D1_vol=np.abs((D1_chkp[:,1]*D1_std_volume)-(D1_chkp[:,9]))

where_are_NaNs = isnan(D1_vol)
print("D1 NAN vol",where_are_NaNs)
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
plt.savefig("real_decorate_020002_vp3.png")
