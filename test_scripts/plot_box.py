import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


root_path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/training/"

naive_D6=np.load(root_path+"D6_0111/"+"/mlh_dummy_input_mean_sh.npy")
naive_D5=np.load(root_path+"D5_1111/"+"/mlh_dummy_input_mean_sh.npy")
naive_D4=np.load(root_path+"D4_1100"+"/mlh_dummy_input_mean_sh.npy")
naive_D3=np.load(root_path+"D3_1010"+"/mlh_dummy_input_mean_sh.npy")
naive_D2=np.load(root_path+"D2_1000"+"/mlh_dummy_input_mean_sh.npy")
naive_D1=np.load(root_path+"D1_0000"+"/mlh_dummy_input_mean_sh.npy")


#Checkpoint file load

D6_chkp=np.load(root_path+"D6_0111"+"/mlh_bnf_mag_96ms_98.npy")
D5_chkp=np.load(root_path+"D5_1111"+"/mlh_bnf_mag_96ms_163.npy")
D4_chkp=np.load(root_path+"D4_1100"+"/mlh_bnf_mag_96ms_111.npy")
D3_chkp=np.load(root_path+"D3_1010"+"/mlh_bnf_mag_96ms_117.npy")
D2_chkp=np.load(root_path+"D2_1000"+"/mlh_bnf_mag_96ms_62.npy")
D1_chkp=np.load(root_path+"D1_0000"+"/mlh_bnf_mag_96ms_110.npy")












dum_bnf_1=np.abs((np.array([np.mean(naive_D5[1:,2])]*(D5_chkp.shape[0]-1)))-(D5_chkp[1:,18]))
#dum_m_1=np.abs((np.array([np.mean(dummy_m[1:,2])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,30]))

dum_bnf_2=np.abs((np.array([np.mean(naive_D5[1:,3])]*(D5_chkp.shape[0]-1)))-(D5_chkp[1:,19]))
#dum_m_2=np.abs((np.array([np.mean(dummy_m[1:,3])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,31]))


dum_bnf_3=np.abs((np.array([np.mean(naive_D5[1:,4])]*(D5_chkp.shape[0]-1)))-(D5_chkp[1:,20]))
#dum_m_3=np.abs((np.array([np.mean(dummy_m[1:,10])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,38]))


dum_bnf_4=np.abs((np.array([np.mean(naive_D5[1:,5])]*(D5_chkp.shape[0]-1)))-(D5_chkp[1:,21]))
#dum_m_4=np.abs((np.array([np.mean(dummy_m[1:,11])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,39]))

dum_bnf_5=np.abs((np.array([np.mean(naive_D5[1:,6])]*(D5_chkp.shape[0]-1)))-(D5_chkp[1:,22]))
#dum_m_5=np.abs((np.array([np.mean(dummy_m[1:,12])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,40]))

dum_bnf_6=np.abs((np.array([np.mean(naive_D5[1:,7])]*(D5_chkp.shape[0]-1)))-(D5_chkp[1:,23]))

#Surface
dum_bnf_7=np.abs((np.array([np.mean(naive_D5[1:,0])]*(D5_chkp.shape[0]-1)))-(D5_chkp[1:,16]))
#Volume
dum_bnf_8=np.abs((np.array([np.mean(naive_D5[1:,1])]*(D5_chkp.shape[0]-1)))-(D5_chkp[1:,17]))

##########################################################################################################

d4um_bnf_1=np.abs((np.array([np.mean(naive_D4[1:,2])]*(D4_chkp.shape[0]-1)))-(D4_chkp[1:,18]))
#dum_m_1=np.abs((np.array([np.mean(dummy_m[1:,2])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,30]))

d4um_bnf_2=np.abs((np.array([np.mean(naive_D4[1:,3])]*(D4_chkp.shape[0]-1)))-(D4_chkp[1:,19]))
#dum_m_2=np.abs((np.array([np.mean(dummy_m[1:,3])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,31]))


d4um_bnf_3=np.abs((np.array([np.mean(naive_D4[1:,4])]*(D4_chkp.shape[0]-1)))-(D4_chkp[1:,20]))
#dum_m_3=np.abs((np.array([np.mean(dummy_m[1:,10])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,38]))


d4um_bnf_4=np.abs((np.array([np.mean(naive_D4[1:,5])]*(D4_chkp.shape[0]-1)))-(D4_chkp[1:,21]))
#dum_m_4=np.abs((np.array([np.mean(dummy_m[1:,11])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,39]))

d4um_bnf_5=np.abs((np.array([np.mean(naive_D4[1:,6])]*(D4_chkp.shape[0]-1)))-(D4_chkp[1:,22]))
#dum_m_5=np.abs((np.array([np.mean(dummy_m[1:,12])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,40]))

d4um_bnf_6=np.abs((np.array([np.mean(naive_D4[1:,7])]*(D4_chkp.shape[0]-1)))-(D4_chkp[1:,23]))

#Surface
d4um_bnf_7=np.abs((np.array([np.mean(naive_D4[1:,0])]*(D4_chkp.shape[0]-1)))-(D4_chkp[1:,16]))
#Volume
d4um_bnf_8=np.abs((np.array([np.mean(naive_D4[1:,1])]*(D4_chkp.shape[0]-1)))-(D4_chkp[1:,17]))

########################################################################################################

d3um_bnf_1=np.abs((np.array([np.mean(naive_D3[1:,2])]*(D3_chkp.shape[0]-1)))-(D3_chkp[1:,18]))
#dum_m_1=np.abs((np.array([np.mean(dummy_m[1:,2])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,30]))

d3um_bnf_2=np.abs((np.array([np.mean(naive_D3[1:,3])]*(D3_chkp.shape[0]-1)))-(D3_chkp[1:,19]))
#dum_m_2=np.abs((np.array([np.mean(dummy_m[1:,3])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,31]))


d3um_bnf_3=np.abs((np.array([np.mean(naive_D3[1:,4])]*(D3_chkp.shape[0]-1)))-(D3_chkp[1:,20]))
#dum_m_3=np.abs((np.array([np.mean(dummy_m[1:,10])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,38]))


d3um_bnf_4=np.abs((np.array([np.mean(naive_D3[1:,5])]*(D3_chkp.shape[0]-1)))-(D3_chkp[1:,21]))
#dum_m_4=np.abs((np.array([np.mean(dummy_m[1:,11])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,39]))

d3um_bnf_5=np.abs((np.array([np.mean(naive_D3[1:,6])]*(D3_chkp.shape[0]-1)))-(D3_chkp[1:,22]))
#dum_m_5=np.abs((np.array([np.mean(dummy_m[1:,12])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,40]))

d3um_bnf_6=np.abs((np.array([np.mean(naive_D3[1:,7])]*(D3_chkp.shape[0]-1)))-(D3_chkp[1:,23]))

#Surface
d3um_bnf_7=np.abs((np.array([np.mean(naive_D3[1:,0])]*(D3_chkp.shape[0]-1)))-(D3_chkp[1:,16]))
#Volume
d3um_bnf_8=np.abs((np.array([np.mean(naive_D3[1:,1])]*(D3_chkp.shape[0]-1)))-(D3_chkp[1:,17]))

#######################################################################################################

d2um_bnf_1=np.abs((np.array([np.mean(naive_D2[1:,2])]*(D2_chkp.shape[0]-1)))-(D2_chkp[1:,18]))
#dum_m_1=np.abs((np.array([np.mean(dummy_m[1:,2])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,30]))

d2um_bnf_2=np.abs((np.array([np.mean(naive_D2[1:,3])]*(D2_chkp.shape[0]-1)))-(D2_chkp[1:,19]))
#dum_m_2=np.abs((np.array([np.mean(dummy_m[1:,3])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,31]))


d2um_bnf_3=np.abs((np.array([np.mean(naive_D2[1:,4])]*(D2_chkp.shape[0]-1)))-(D2_chkp[1:,20]))
#dum_m_3=np.abs((np.array([np.mean(dummy_m[1:,10])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,38]))


d2um_bnf_4=np.abs((np.array([np.mean(naive_D2[1:,5])]*(D2_chkp.shape[0]-1)))-(D2_chkp[1:,21]))
#dum_m_4=np.abs((np.array([np.mean(dummy_m[1:,11])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,39]))

d2um_bnf_5=np.abs((np.array([np.mean(naive_D2[1:,6])]*(D2_chkp.shape[0]-1)))-(D2_chkp[1:,22]))
#dum_m_5=np.abs((np.array([np.mean(dummy_m[1:,12])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,40]))

d2um_bnf_6=np.abs((np.array([np.mean(naive_D2[1:,7])]*(D2_chkp.shape[0]-1)))-(D2_chkp[1:,23]))

#Surface
d2um_bnf_7=np.abs((np.array([np.mean(naive_D2[1:,0])]*(D2_chkp.shape[0]-1)))-(D2_chkp[1:,16]))
#Volume
d2um_bnf_8=np.abs((np.array([np.mean(naive_D2[1:,1])]*(D2_chkp.shape[0]-1)))-(D2_chkp[1:,17]))

##################################################################################################333

d1um_bnf_1=np.abs((np.array([np.mean(naive_D1[1:,2])]*(D1_chkp.shape[0]-1)))-(D1_chkp[1:,18]))
#dum_m_1=np.abs((np.array([np.mean(dummy_m[1:,2])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,30]))

d1um_bnf_2=np.abs((np.array([np.mean(naive_D1[1:,3])]*(D1_chkp.shape[0]-1)))-(D1_chkp[1:,19]))
#dum_m_2=np.abs((np.array([np.mean(dummy_m[1:,3])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,31]))


d1um_bnf_3=np.abs((np.array([np.mean(naive_D1[1:,4])]*(D1_chkp.shape[0]-1)))-(D1_chkp[1:,20]))
#dum_m_3=np.abs((np.array([np.mean(dummy_m[1:,10])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,38]))


d1um_bnf_4=np.abs((np.array([np.mean(naive_D1[1:,5])]*(D1_chkp.shape[0]-1)))-(D1_chkp[1:,21]))
#dum_m_4=np.abs((np.array([np.mean(dummy_m[1:,11])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,39]))

d1um_bnf_5=np.abs((np.array([np.mean(naive_D1[1:,6])]*(D1_chkp.shape[0]-1)))-(D1_chkp[1:,22]))
#dum_m_5=np.abs((np.array([np.mean(dummy_m[1:,12])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,40]))

d1um_bnf_6=np.abs((np.array([np.mean(naive_D1[1:,7])]*(D1_chkp.shape[0]-1)))-(D1_chkp[1:,23]))

#Surface
d1um_bnf_7=np.abs((np.array([np.mean(naive_D1[1:,0])]*(D1_chkp.shape[0]-1)))-(D1_chkp[1:,16]))
#Volume
d1um_bnf_8=np.abs((np.array([np.mean(naive_D1[1:,1])]*(D1_chkp.shape[0]-1)))-(D1_chkp[1:,17]))

#####################################################################################################3

d6um_bnf_1=np.abs((np.array([np.mean(naive_D6[1:,2])]*(D6_chkp.shape[0]-1)))-(D6_chkp[1:,18]))
#dum_m_1=np.abs((np.array([np.mean(dummy_m[1:,2])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,30]))

d6um_bnf_2=np.abs((np.array([np.mean(naive_D6[1:,3])]*(D6_chkp.shape[0]-1)))-(D6_chkp[1:,19]))
#dum_m_2=np.abs((np.array([np.mean(dummy_m[1:,3])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,31]))


d6um_bnf_3=np.abs((np.array([np.mean(naive_D6[1:,4])]*(D6_chkp.shape[0]-1)))-(D6_chkp[1:,20]))
#dum_m_3=np.abs((np.array([np.mean(dummy_m[1:,10])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,38]))


d6um_bnf_4=np.abs((np.array([np.mean(naive_D6[1:,5])]*(D6_chkp.shape[0]-1)))-(D6_chkp[1:,21]))
#dum_m_4=np.abs((np.array([np.mean(dummy_m[1:,11])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,39]))

d6um_bnf_5=np.abs((np.array([np.mean(naive_D6[1:,6])]*(D6_chkp.shape[0]-1)))-(D6_chkp[1:,22]))
#dum_m_5=np.abs((np.array([np.mean(dummy_m[1:,12])]*(m_mag_96.shape[0]-1)))-(m_mag_96[1:,40]))

d6um_bnf_6=np.abs((np.array([np.mean(naive_D6[1:,7])]*(D6_chkp.shape[0]-1)))-(D6_chkp[1:,23]))

#Surface
d6um_bnf_7=np.abs((np.array([np.mean(naive_D6[1:,0])]*(D6_chkp.shape[0]-1)))-(D6_chkp[1:,16]))
#Volume
d6um_bnf_8=np.abs((np.array([np.mean(naive_D6[1:,1])]*(D6_chkp.shape[0]-1)))-(D6_chkp[1:,17]))



















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
D2_std_surface =  64.187581

D2_std_rt60=[ 0.395032, 0.376963, 0.297600, 0.263775, 0.242831, 0.214957]


D1_std_volume = 72.14526
D1_std_surface = 63.60605

D1_std_rt60=[ 0.364063, 0.302710, 0.222462, 0.193005, 0.183209, 0.170326]



D6_rt60_125=np.abs((D6_chkp[:,2]*D6_std_rt60[0])-(D6_chkp[:,18]))
D6_rt60_250=np.abs((D6_chkp[:,3]*D6_std_rt60[1])-(D6_chkp[:,19]))
D6_rt60_500=np.abs((D6_chkp[:,4]*D6_std_rt60[2])-(D6_chkp[:,20]))
D6_rt60_1000=np.abs((D6_chkp[:,5]*D6_std_rt60[3])-(D6_chkp[:,21]))
D6_rt60_2000=np.abs((D6_chkp[:,6]*D6_std_rt60[4])-(D6_chkp[:,22]))
D6_rt60_4000=np.abs((D6_chkp[:,7]*D6_std_rt60[5])-(D6_chkp[:,23]))

D6_surf=np.abs((D6_chkp[:,0]*D6_std_surface)-(D6_chkp[:,16]))
D6_vol=np.abs((D6_chkp[:,1]*D6_std_volume)-(D6_chkp[:,17]))


D5_rt60_125=np.abs((D5_chkp[:,2]*D5_std_rt60[0])-(D5_chkp[:,18]))
D5_rt60_250=np.abs((D5_chkp[:,3]*D5_std_rt60[1])-(D5_chkp[:,19]))
D5_rt60_500=np.abs((D5_chkp[:,4]*D5_std_rt60[2])-(D5_chkp[:,20]))
D5_rt60_1000=np.abs((D5_chkp[:,5]*D5_std_rt60[3])-(D5_chkp[:,21]))
D5_rt60_2000=np.abs((D5_chkp[:,6]*D5_std_rt60[4])-(D5_chkp[:,22]))
D5_rt60_4000=np.abs((D5_chkp[:,7]*D5_std_rt60[5])-(D5_chkp[:,23]))

D5_surf=np.abs((D5_chkp[:,0]*D5_std_surface)-(D5_chkp[:,16]))
D5_vol=np.abs((D5_chkp[:,1]*D5_std_volume)-(D5_chkp[:,17]))




D4_rt60_125=np.abs((D4_chkp[:,2]*D4_std_rt60[0])-(D4_chkp[:,18]))
D4_rt60_250=np.abs((D4_chkp[:,3]*D4_std_rt60[1])-(D4_chkp[:,19]))
D4_rt60_500=np.abs((D4_chkp[:,4]*D4_std_rt60[2])-(D4_chkp[:,20]))
D4_rt60_1000=np.abs((D4_chkp[:,5]*D4_std_rt60[3])-(D4_chkp[:,21]))
D4_rt60_2000=np.abs((D4_chkp[:,6]*D4_std_rt60[4])-(D4_chkp[:,22]))
D4_rt60_4000=np.abs((D4_chkp[:,7]*D4_std_rt60[5])-(D4_chkp[:,23]))

D4_surf=np.abs((D4_chkp[:,0]*D4_std_surface)-(D4_chkp[:,16]))
D4_vol=np.abs((D4_chkp[:,1]*D4_std_volume)-(D4_chkp[:,17]))



D3_rt60_125=np.abs((D3_chkp[:,2]*D3_std_rt60[0])-(D3_chkp[:,18]))
D3_rt60_250=np.abs((D3_chkp[:,3]*D3_std_rt60[1])-(D3_chkp[:,19]))
D3_rt60_500=np.abs((D3_chkp[:,4]*D3_std_rt60[2])-(D3_chkp[:,20]))
D3_rt60_1000=np.abs((D3_chkp[:,5]*D3_std_rt60[3])-(D3_chkp[:,21]))
D3_rt60_2000=np.abs((D3_chkp[:,6]*D3_std_rt60[4])-(D3_chkp[:,22]))
D3_rt60_4000=np.abs((D3_chkp[:,7]*D3_std_rt60[5])-(D3_chkp[:,23]))

D3_surf=np.abs((D3_chkp[:,0]*D3_std_surface)-(D3_chkp[:,16]))
D3_vol=np.abs((D3_chkp[:,1]*D3_std_volume)-(D3_chkp[:,17]))




D2_rt60_125=np.abs((D2_chkp[:,2]*D2_std_rt60[0])-(D2_chkp[:,18]))
D2_rt60_250=np.abs((D2_chkp[:,3]*D2_std_rt60[1])-(D2_chkp[:,19]))
D2_rt60_500=np.abs((D2_chkp[:,4]*D2_std_rt60[2])-(D2_chkp[:,20]))
D2_rt60_1000=np.abs((D2_chkp[:,5]*D2_std_rt60[3])-(D2_chkp[:,21]))
D2_rt60_2000=np.abs((D2_chkp[:,6]*D2_std_rt60[4])-(D2_chkp[:,22]))
D2_rt60_4000=np.abs((D2_chkp[:,7]*D2_std_rt60[5])-(D2_chkp[:,23]))

D2_surf=np.abs((D2_chkp[:,0]*D2_std_surface)-(D2_chkp[:,16]))
D2_vol=np.abs((D2_chkp[:,1]*D2_std_volume)-(D2_chkp[:,17]))



D1_rt60_125=np.abs((D1_chkp[:,2]*D1_std_rt60[0])-(D1_chkp[:,18]))
D1_rt60_250=np.abs((D1_chkp[:,3]*D1_std_rt60[1])-(D1_chkp[:,19]))
D1_rt60_500=np.abs((D1_chkp[:,4]*D1_std_rt60[2])-(D1_chkp[:,20]))
D1_rt60_1000=np.abs((D1_chkp[:,5]*D1_std_rt60[3])-(D1_chkp[:,21]))
D1_rt60_2000=np.abs((D1_chkp[:,6]*D1_std_rt60[4])-(D1_chkp[:,22]))
D1_rt60_4000=np.abs((D1_chkp[:,7]*D1_std_rt60[5])-(D1_chkp[:,23]))

D1_surf=np.abs((D1_chkp[:,0]*D1_std_surface)-(D1_chkp[:,16]))
D1_vol=np.abs((D1_chkp[:,1]*D1_std_volume)-(D1_chkp[:,17]))









out_rt=False

fig,axs=plt.subplots(4,2,figsize=(10,25))

bplot1=axs[0,0].boxplot([d6um_bnf_1,D6_rt60_125,dum_bnf_1,D5_rt60_125,d4um_bnf_1,D4_rt60_125,d3um_bnf_1,D3_rt60_125,d2um_bnf_1,D2_rt60_125,d1um_bnf_1,D1_rt60_125],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
axs[0,0].set_xticklabels(['NM D6','D6','NM D5','D5','NM D4','D4','NM D3','D3','NM D2','D2','NM D1','D1'],rotation=45)
axs[0,0].set_ylabel("Abs Err Sec")
axs[0,0].set_title("RT 60 125hz")

bplot2=axs[0,1].boxplot([d6um_bnf_2,D6_rt60_250,dum_bnf_2,D5_rt60_250,d4um_bnf_2,D4_rt60_250,d3um_bnf_2,D3_rt60_250,d2um_bnf_2,D2_rt60_250,d1um_bnf_2,D1_rt60_250],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
axs[0,1].set_xticklabels(['NM D6','D6','NM D5','D5','NM D4','D4','NM D3','D3','NM D2','D2','NM D1','D1'],rotation=45)
axs[0,1].set_ylabel("Abs Err Sec")
axs[0,1].set_title("RT 60 250hz")

bplot3=axs[1,0].boxplot([d6um_bnf_3,D6_rt60_500,dum_bnf_3,D5_rt60_500,d4um_bnf_3,D4_rt60_500,d3um_bnf_3,D3_rt60_500,d2um_bnf_3,D2_rt60_500,d1um_bnf_3,D1_rt60_500],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[1,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
axs[1,0].set_xticklabels(['NM D6','D6','NM D5','D5','NM D4','D4','NM D3','D3','NM D2','D2','NM D1','D1'],rotation=45)
axs[1,0].set_ylabel("Abs Err Sec")
axs[1,0].set_title("RT 60 500hz")

bplot4=axs[1,1].boxplot([d6um_bnf_4,D6_rt60_1000,dum_bnf_4,D5_rt60_1000,d4um_bnf_4,D4_rt60_1000,d3um_bnf_4,D3_rt60_1000,d2um_bnf_4,D2_rt60_1000,d1um_bnf_4,D1_rt60_1000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[1,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
axs[1,1].set_xticklabels(['NM D6','D6','NM D5','D5','NM D4','D4','NM D3','D3','NM D2','D2','NM D1','D1'],rotation=45)
axs[1,1].set_ylabel("Abs Err Sec")
axs[1,1].set_title("RT 60 1000hz")

bplot5=axs[2,0].boxplot([d6um_bnf_5,D6_rt60_2000,dum_bnf_5,D5_rt60_2000,d4um_bnf_5,D4_rt60_2000,d3um_bnf_5,D3_rt60_2000,d2um_bnf_5,D2_rt60_2000,d1um_bnf_5,D1_rt60_2000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[2,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
axs[2,0].set_xticklabels(['NM D6','D6','NM D5','D5','NM D4','D4','NM D3','D3','NM D2','D2','NM D1','D1'],rotation=45)
axs[2,0].set_ylabel("Abs Err Sec")
axs[2,0].set_title("RT 60 2000hz")

bplot6=axs[2,1].boxplot([d6um_bnf_6,D6_rt60_4000,dum_bnf_6,D5_rt60_4000,d4um_bnf_6,D4_rt60_4000,d3um_bnf_6,D3_rt60_4000,d2um_bnf_6,D2_rt60_4000,d1um_bnf_6,D1_rt60_4000],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[2,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
axs[2,1].set_xticklabels(['NM D6','D6','NM D5','D5','NM D4','D4','NM D3','D3','NM D2','D2','NM D1','D1'],rotation=45)
axs[2,1].set_ylabel("Abs Err Sec")
axs[2,1].set_title("RT 60 4000hz")

bplot7=axs[3,0].boxplot([d6um_bnf_7,D6_surf,dum_bnf_7,D5_surf,d4um_bnf_7,D4_surf,d3um_bnf_7,D3_surf,d2um_bnf_7,D2_surf,d1um_bnf_7,D1_surf],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
axs[3,0].set_xticklabels(['NM D6','D6','NM D5','D5','NM D4','D4','NM D3','D3','NM D2','D2','NM D1','D1'],rotation=45)
axs[3,0].set_ylabel("Abs Err M2")
axs[3,0].set_title("Surface Err")

bplot8=axs[3,1].boxplot([d6um_bnf_8,D6_vol,dum_bnf_8,D5_vol,d4um_bnf_8,D4_vol,d3um_bnf_8,D3_vol,d2um_bnf_8,D2_vol,d1um_bnf_8,D1_vol],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
axs[3,1].set_xticklabels(['NM D6','D6','NM D5','D5','NM D4','D4','NM D3','D3','NM D2','D2','NM D1','D1'],rotation=45)
axs[3,1].set_ylabel("Abs Err M3")
axs[3,1].set_title("Volume Err")


colors=['pink','lightblue','lightgreen','orange','cyan','red','gold','skyblue','yellow','magenta','violet','blue']


for bplot in (bplot1,bplot2,bplot3,bplot4,bplot5,bplot6,bplot7,bplot8):
    for patch,color in zip(bplot['boxes'],colors):
        patch.set_facecolor(color)


fig.tight_layout(pad=3.0)
#plt.xticks([1,2,3],('Dummy Bnf','bnf','Dummy M','M'))
#plt.title("Absolute Diff Estimated Mean And Target RT60")
plt.savefig("volume_surface_comparasion_ff_6.png")
