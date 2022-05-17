import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import *


root_path="/home/psrivastava/axis-2/IWAENC/z_test/"


D6_chkp_1=np.load(root_path+"D6_test_set"+"/D6_real_data_000000.npy")
D6_chkp_2=np.load(root_path+"D6_test_set"+"/D6_real_data_000001.npy")
D6_chkp_3=np.load(root_path+"D6_test_set"+"/D6_real_data_000010.npy")
D6_chkp_4=np.load(root_path+"D6_test_set"+"/D6_real_data_000100.npy")
D6_chkp_5=np.load(root_path+"D6_test_set"+"/D6_real_data_010000.npy")
D6_chkp_6=np.load(root_path+"D6_test_set"+"/D6_real_data_011000.npy")
D6_chkp_7=np.load(root_path+"D6_test_set"+"/D6_real_data_011100.npy")
D6_chkp_8=np.load(root_path+"D6_test_set"+"/D6_real_data_011110.npy")
D6_chkp_9=np.load(root_path+"D6_test_set"+"/D6_real_data_011111.npy")
D6_chkp_10=np.load(root_path+"D6_test_set"+"/D6_real_data_020002.npy")

D5_chkp_1=np.load(root_path+"D5_test_set"+"/D5_real_data_000000.npy")
D5_chkp_2=np.load(root_path+"D5_test_set"+"/D5_real_data_000001.npy")
D5_chkp_3=np.load(root_path+"D5_test_set"+"/D5_real_data_000010.npy")
D5_chkp_4=np.load(root_path+"D5_test_set"+"/D5_real_data_000100.npy")
D5_chkp_5=np.load(root_path+"D5_test_set"+"/D5_real_data_010000.npy")
D5_chkp_6=np.load(root_path+"D5_test_set"+"/D5_real_data_011000.npy")
D5_chkp_7=np.load(root_path+"D5_test_set"+"/D5_real_data_011100.npy")
D5_chkp_8=np.load(root_path+"D5_test_set"+"/D5_real_data_011110.npy")
D5_chkp_9=np.load(root_path+"D5_test_set"+"/D5_real_data_011111.npy")
D5_chkp_10=np.load(root_path+"D5_test_set"+"/D5_real_data_020002.npy")


D4_chkp_1=np.load(root_path+"D4_test_set"+"/D4_real_data_000000.npy")
D4_chkp_2=np.load(root_path+"D4_test_set"+"/D4_real_data_000001.npy")
D4_chkp_3=np.load(root_path+"D4_test_set"+"/D4_real_data_000010.npy")
D4_chkp_4=np.load(root_path+"D4_test_set"+"/D4_real_data_000100.npy")
D4_chkp_5=np.load(root_path+"D4_test_set"+"/D4_real_data_010000.npy")
D4_chkp_6=np.load(root_path+"D4_test_set"+"/D4_real_data_011000.npy")
D4_chkp_7=np.load(root_path+"D4_test_set"+"/D4_real_data_011100.npy")
D4_chkp_8=np.load(root_path+"D4_test_set"+"/D4_real_data_011110.npy")
D4_chkp_9=np.load(root_path+"D4_test_set"+"/D4_real_data_011111.npy")
D4_chkp_10=np.load(root_path+"D4_test_set"+"/D4_real_data_020002.npy")


D3_chkp_1=np.load(root_path+"D3_test_set"+"/D3_real_data_000000.npy")
D3_chkp_2=np.load(root_path+"D3_test_set"+"/D3_real_data_000001.npy")
D3_chkp_3=np.load(root_path+"D3_test_set"+"/D3_real_data_000010.npy")
D3_chkp_4=np.load(root_path+"D3_test_set"+"/D3_real_data_000100.npy")
D3_chkp_5=np.load(root_path+"D3_test_set"+"/D3_real_data_010000.npy")
D3_chkp_6=np.load(root_path+"D3_test_set"+"/D3_real_data_011000.npy")
D3_chkp_7=np.load(root_path+"D3_test_set"+"/D3_real_data_011100.npy")
D3_chkp_8=np.load(root_path+"D3_test_set"+"/D3_real_data_011110.npy")
D3_chkp_9=np.load(root_path+"D3_test_set"+"/D3_real_data_011111.npy")
D3_chkp_10=np.load(root_path+"D3_test_set"+"/D3_real_data_020002.npy")


D2_chkp_1=np.load(root_path+"D2_test_set"+"/D2_real_data_000000.npy")
D2_chkp_2=np.load(root_path+"D2_test_set"+"/D2_real_data_000001.npy")
D2_chkp_3=np.load(root_path+"D2_test_set"+"/D2_real_data_000010.npy")
D2_chkp_4=np.load(root_path+"D2_test_set"+"/D2_real_data_000100.npy")
D2_chkp_5=np.load(root_path+"D2_test_set"+"/D2_real_data_010000.npy")
D2_chkp_6=np.load(root_path+"D2_test_set"+"/D2_real_data_011000.npy")
D2_chkp_7=np.load(root_path+"D2_test_set"+"/D2_real_data_011100.npy")
D2_chkp_8=np.load(root_path+"D2_test_set"+"/D2_real_data_011110.npy")
D2_chkp_9=np.load(root_path+"D2_test_set"+"/D2_real_data_011111.npy")
D2_chkp_10=np.load(root_path+"D2_test_set"+"/D2_real_data_020002.npy")


D1_chkp_1=np.load(root_path+"D1_test_set"+"/D1_real_data_000000.npy")
D1_chkp_2=np.load(root_path+"D1_test_set"+"/D1_real_data_000001.npy")
D1_chkp_3=np.load(root_path+"D1_test_set"+"/D1_real_data_000010.npy")
D1_chkp_4=np.load(root_path+"D1_test_set"+"/D1_real_data_000100.npy")
D1_chkp_5=np.load(root_path+"D1_test_set"+"/D1_real_data_010000.npy")
D1_chkp_6=np.load(root_path+"D1_test_set"+"/D1_real_data_011000.npy")
D1_chkp_7=np.load(root_path+"D1_test_set"+"/D1_real_data_011100.npy")
D1_chkp_8=np.load(root_path+"D1_test_set"+"/D1_real_data_011110.npy")
D1_chkp_9=np.load(root_path+"D1_test_set"+"/D1_real_data_011111.npy")
D1_chkp_10=np.load(root_path+"D1_test_set"+"/D1_real_data_020002.npy")




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
src_0=np.empty((10,20))
src_1=np.empty((10,20))
src_2=np.empty((10,20))
src_3=np.empty((10,20))
src_4=np.empty((10,20))
src_5=np.empty((10,20))
src_6=np.empty((10,20))
'''

#srcs=[src_0,src_1,src_2,src_3,src_4,src_5,src_6]
rooms_=['000000','000001','000010','000100','010000','011000','011100','011110','011111','020002']


D6_rooms=[D6_chkp_1,D6_chkp_2,D6_chkp_3,D6_chkp_4,D6_chkp_5,D6_chkp_6,D6_chkp_7,D6_chkp_8,D6_chkp_9,D6_chkp_10]

D5_rooms=[D5_chkp_1,D5_chkp_2,D5_chkp_3,D5_chkp_4,D5_chkp_5,D5_chkp_6,D5_chkp_7,D5_chkp_8,D5_chkp_9,D5_chkp_10]
D4_rooms=[D4_chkp_1,D4_chkp_2,D4_chkp_3,D4_chkp_4,D4_chkp_5,D4_chkp_6,D4_chkp_7,D4_chkp_8,D4_chkp_9,D5_chkp_10]
D3_rooms=[D3_chkp_1,D3_chkp_2,D3_chkp_3,D3_chkp_4,D3_chkp_5,D3_chkp_6,D3_chkp_7,D3_chkp_8,D3_chkp_9,D3_chkp_10]
D2_rooms=[D2_chkp_1,D2_chkp_2,D2_chkp_3,D2_chkp_4,D2_chkp_5,D2_chkp_6,D2_chkp_7,D2_chkp_8,D2_chkp_9,D2_chkp_10]
D1_rooms=[D1_chkp_1,D1_chkp_2,D1_chkp_3,D1_chkp_4,D1_chkp_5,D1_chkp_6,D1_chkp_7,D1_chkp_8,D1_chkp_9,D1_chkp_10]

sr_=0
sr_2=20
out_rt=False



for j in range(7):
    fig,axs=plt.subplots(6,10,figsize=(30,65))

    for i in range(10):

        sa=np.array([[123.026]]*D5_rooms[i].shape[0])
        volr=np.array([[80.141]]*D5_rooms[i].shape[0])
        rt60_=np.array([rt60[rooms_[i]]]*D5_rooms[i].shape[0])

        D6_chkp=np.concatenate((D6_rooms[i],sa,volr,rt60_),axis=1)
        D5_chkp=np.concatenate((D5_rooms[i],sa,volr,rt60_),axis=1)
        D4_chkp=np.concatenate((D4_rooms[i],sa,volr,rt60_),axis=1)
        D3_chkp=np.concatenate((D3_rooms[i],sa,volr,rt60_),axis=1)
        D2_chkp=np.concatenate((D2_rooms[i],sa,volr,rt60_),axis=1)
        D1_chkp=np.concatenate((D1_rooms[i],sa,volr,rt60_),axis=1)



        D6_rt60_500=np.abs((D6_chkp[:,4]*D6_std_rt60[2])-(D6_chkp[:,12])).reshape(-1,1)


        D6_rt60_1000=np.abs((D6_chkp[:,5]*D6_std_rt60[3])-(D6_chkp[:,13])).reshape(-1,1)

        D6_rt60_2000=np.abs((D6_chkp[:,6]*D6_std_rt60[4])-(D6_chkp[:,14])).reshape(-1,1)


        D6_rt60_4000=np.abs((D6_chkp[:,7]*D6_std_rt60[5])-(D6_chkp[:,15])).reshape(-1,1)



        D6_surf=np.abs((D6_chkp[:,0]*D6_std_surface)-(D6_chkp[:,8])).reshape(-1,1)
        D6_vol=np.abs((D6_chkp[:,1]*D6_std_volume)-(D6_chkp[:,9])).reshape(-1,1)

        D6_err=np.concatenate((D6_surf,D6_vol,D6_rt60_500,D6_rt60_1000,D6_rt60_2000,D6_rt60_4000),axis=1)

        D5_rt60_500=np.abs((D5_chkp[:,4]*D5_std_rt60[2])-(D5_chkp[:,12])).reshape(-1,1)


        D5_rt60_1000=np.abs((D5_chkp[:,5]*D5_std_rt60[3])-(D5_chkp[:,13])).reshape(-1,1)

        D5_rt60_2000=np.abs((D5_chkp[:,6]*D5_std_rt60[4])-(D5_chkp[:,14])).reshape(-1,1)


        D5_rt60_4000=np.abs((D5_chkp[:,7]*D5_std_rt60[5])-(D5_chkp[:,15])).reshape(-1,1)



        D5_surf=np.abs((D5_chkp[:,0]*D5_std_surface)-(D5_chkp[:,8])).reshape(-1,1)
        D5_vol=np.abs((D5_chkp[:,1]*D5_std_volume)-(D5_chkp[:,9])).reshape(-1,1)

        D5_err=np.concatenate((D5_surf,D5_vol,D5_rt60_500,D5_rt60_1000,D5_rt60_2000,D5_rt60_4000),axis=1)

        D4_rt60_500=np.abs((D4_chkp[:,4]*D4_std_rt60[2])-(D4_chkp[:,12])).reshape(-1,1)
        where_are_NaNs = isnan(D4_rt60_500)
        D4_rt60_500[where_are_NaNs] = 0

        D4_rt60_1000=np.abs((D4_chkp[:,5]*D4_std_rt60[3])-(D4_chkp[:,13])).reshape(-1,1)
        where_are_NaNs = isnan(D4_rt60_1000)
        D4_rt60_1000[where_are_NaNs] = 0

        D4_rt60_2000=np.abs((D4_chkp[:,6]*D4_std_rt60[4])-(D4_chkp[:,14])).reshape(-1,1)
        where_are_NaNs = isnan(D4_rt60_2000)
        D4_rt60_2000[where_are_NaNs] = 0

        D4_rt60_4000=np.abs((D4_chkp[:,7]*D4_std_rt60[5])-(D4_chkp[:,15])).reshape(-1,1)
        where_are_NaNs = isnan(D4_rt60_4000)
        D4_rt60_4000[where_are_NaNs] = 0


        D4_surf=np.abs((D4_chkp[:,0]*D4_std_surface)-(D4_chkp[:,8])).reshape(-1,1)
        D4_vol=np.abs((D4_chkp[:,1]*D4_std_volume)-(D4_chkp[:,9])).reshape(-1,1)

        D4_err=np.concatenate((D4_surf,D4_vol,D4_rt60_500,D4_rt60_1000,D4_rt60_2000,D4_rt60_4000),axis=1)

        D3_rt60_500=np.abs((D3_chkp[:,4]*D3_std_rt60[2])-(D3_chkp[:,12])).reshape(-1,1)
        where_are_NaNs = isnan(D3_rt60_500)
        D3_rt60_500[where_are_NaNs] = 0

        D3_rt60_1000=np.abs((D3_chkp[:,5]*D3_std_rt60[3])-(D3_chkp[:,13])).reshape(-1,1)
        where_are_NaNs = isnan(D3_rt60_1000)
        D3_rt60_1000[where_are_NaNs] = 0

        D3_rt60_2000=np.abs((D3_chkp[:,6]*D3_std_rt60[4])-(D3_chkp[:,14])).reshape(-1,1)
        where_are_NaNs = isnan(D3_rt60_2000)
        D3_rt60_2000[where_are_NaNs] = 0

        D3_rt60_4000=np.abs((D3_chkp[:,7]*D3_std_rt60[5])-(D3_chkp[:,15])).reshape(-1,1)
        where_are_NaNs = isnan(D3_rt60_4000)
        D3_rt60_4000[where_are_NaNs] = 0


        D3_surf=np.abs((D3_chkp[:,0]*D3_std_surface)-(D3_chkp[:,8])).reshape(-1,1)
        D3_vol=np.abs((D3_chkp[:,1]*D3_std_volume)-(D3_chkp[:,9])).reshape(-1,1)

        D3_err=np.concatenate((D3_surf,D3_vol,D3_rt60_500,D3_rt60_1000,D3_rt60_2000,D3_rt60_4000),axis=1)

        D2_rt60_500=np.abs((D2_chkp[:,4]*D2_std_rt60[2])-(D2_chkp[:,12])).reshape(-1,1)
        where_are_NaNs = isnan(D2_rt60_500)
        D2_rt60_500[where_are_NaNs] = 0

        D2_rt60_1000=np.abs((D2_chkp[:,5]*D2_std_rt60[3])-(D2_chkp[:,13])).reshape(-1,1)
        where_are_NaNs = isnan(D2_rt60_1000)
        D2_rt60_1000[where_are_NaNs] = 0

        D2_rt60_2000=np.abs((D2_chkp[:,6]*D2_std_rt60[4])-(D2_chkp[:,14])).reshape(-1,1)
        where_are_NaNs = isnan(D2_rt60_2000)
        D2_rt60_2000[where_are_NaNs] = 0

        D2_rt60_4000=np.abs((D2_chkp[:,7]*D2_std_rt60[5])-(D2_chkp[:,15])).reshape(-1,1)
        where_are_NaNs = isnan(D2_rt60_4000)
        D2_rt60_4000[where_are_NaNs] = 0


        D2_surf=np.abs((D2_chkp[:,0]*D2_std_surface)-(D2_chkp[:,8])).reshape(-1,1)
        D2_vol=np.abs((D2_chkp[:,1]*D2_std_volume)-(D2_chkp[:,9])).reshape(-1,1)
        D2_err=np.concatenate((D2_surf,D2_vol,D2_rt60_500,D2_rt60_1000,D2_rt60_2000,D2_rt60_4000),axis=1)


        D1_rt60_500=np.abs((D1_chkp[:,4]*D1_std_rt60[2])-(D1_chkp[:,12])).reshape(-1,1)
        where_are_NaNs = isnan(D1_rt60_500)
        D1_rt60_500[where_are_NaNs] = 0

        D1_rt60_1000=np.abs((D1_chkp[:,5]*D1_std_rt60[3])-(D1_chkp[:,13])).reshape(-1,1)
        where_are_NaNs = isnan(D1_rt60_1000)
        D1_rt60_1000[where_are_NaNs] = 0

        D1_rt60_2000=np.abs((D1_chkp[:,6]*D1_std_rt60[4])-(D1_chkp[:,14])).reshape(-1,1)
        where_are_NaNs = isnan(D1_rt60_2000)
        D1_rt60_2000[where_are_NaNs] = 0

        D1_rt60_4000=np.abs((D1_chkp[:,7]*D1_std_rt60[5])-(D1_chkp[:,15])).reshape(-1,1)
        where_are_NaNs = isnan(D1_rt60_4000)
        D1_rt60_4000[where_are_NaNs] = 0


        D1_surf=np.abs((D1_chkp[:,0]*D1_std_surface)-(D1_chkp[:,8])).reshape(-1,1)
        D1_vol=np.abs((D1_chkp[:,1]*D1_std_volume)-(D1_chkp[:,9])).reshape(-1,1)

        D1_err=np.concatenate((D1_surf,D1_vol,D1_rt60_500,D1_rt60_1000,D1_rt60_2000,D1_rt60_4000),axis=1)

        bplot3=axs[0,i].boxplot([D6_err[sr_:sr_2,2],D5_err[sr_:sr_2,2],D4_err[sr_:sr_2,2],D3_err[sr_:sr_2,2],D2_err[sr_:sr_2,2],D1_err[sr_:sr_2,2]],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
        axs[0,i].set_xticks([1,2,3,4,5,6])
        axs[0,i].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
        axs[0,i].set_ylabel("Abs Err Sec")
        axs[0,i].set_title("RT 60 500hz"+rooms_[i])

        bplot4=axs[1,i].boxplot([D6_err[sr_:sr_2,3],D5_err[sr_:sr_2,3],D4_err[sr_:sr_2,3],D3_err[sr_:sr_2,3],D2_err[sr_:sr_2,3],D1_err[sr_:sr_2,3]],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
        axs[1,i].set_xticks([1,2,3,4,5,6])
        axs[1,i].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
        axs[1,i].set_ylabel("Abs Err Sec")
        axs[1,i].set_title("RT 60 1000hz"+rooms_[i])

        bplot5=axs[2,i].boxplot([D6_err[sr_:sr_2,4],D5_err[sr_:sr_2,4],D4_err[sr_:sr_2,4],D3_err[sr_:sr_2,4],D2_err[sr_:sr_2,4],D1_err[sr_:sr_2,4]],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
        axs[2,i].set_xticks([1,2,3,4,5,6])
        axs[2,i].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
        axs[2,i].set_ylabel("Abs Err Sec")
        axs[2,i].set_title("RT 60 2000hz"+rooms_[i])

        bplot6=axs[3,i].boxplot([D6_err[sr_:sr_2,5],D5_err[sr_:sr_2,5],D4_err[sr_:sr_2,5],D3_err[sr_:sr_2,5],D2_err[sr_:sr_2,5],D1_err[sr_:sr_2,5]],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
        axs[3,i].set_xticks([1,2,3,4,5,6])
        axs[3,i].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
        axs[3,i].set_ylabel("Abs Err Sec")
        axs[3,i].set_title("RT 60 4000hz"+rooms_[i])

        bplot7=axs[4,i].boxplot([D6_err[sr_:sr_2,0],D5_err[sr_:sr_2,0],D4_err[sr_:sr_2,0],D3_err[sr_:sr_2,0],D2_err[sr_:sr_2,0],D1_err[sr_:sr_2,0]],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
        axs[4,i].set_xticks([1,2,3,4,5,6])
        axs[4,i].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
        axs[4,i].set_ylabel("Abs Err M2")
        axs[4,i].set_title("Surface Err"+rooms_[i])

        bplot8=axs[5,i].boxplot([D6_err[sr_:sr_2,1],D5_err[sr_:sr_2,1],D4_err[sr_:sr_2,1],D3_err[sr_:sr_2,1],D2_err[sr_:sr_2,1],D1_err[sr_:sr_2,1]],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
        axs[5,i].set_xticks([1,2,3,4,5,6])
        axs[5,i].set_xticklabels(['D6','D5','D4','D3','D2','D1'],rotation=45)
        axs[5,i].set_ylabel("Abs Err M3")
        axs[5,i].set_title("Volume Err"+rooms_[i])

        colors=['pink','lightblue','lightgreen','orange','cyan']


        for bplot in (bplot3,bplot4,bplot5,bplot6,bplot7,bplot8):
            for patch,color in zip(bplot['boxes'],colors):
                patch.set_facecolor(color)

    fig.tight_layout(pad=5.0)
    #plt.xticks([1,2,3],('Dummy Bnf','bnf','Dummy M','M'))
    #plt.title("Absolute Diff Estimated Mean And Target RT60")
    plt.savefig("real_decorate_src_rec_"+str(j)+".png")




    sr_+=20
    sr_2+=20
