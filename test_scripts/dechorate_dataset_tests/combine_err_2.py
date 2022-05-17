import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import *


root_path="/home/psrivastava/axis-2/IWAENC/z_test/"



D7_chkp_6=np.load(root_path+"D7_test_set"+"/D7_real_data_011000.npy")
D7_chkp_7=np.load(root_path+"D7_test_set"+"/D7_real_data_011100.npy")
D7_chkp_8=np.load(root_path+"D7_test_set"+"/D7_real_data_011110.npy")
D7_chkp_9=np.load(root_path+"D7_test_set"+"/D7_real_data_011111.npy")


'''
D6_chkp_1=np.load(root_path+"D6_test_set"+"/D6_real_data_000000.npy")
D6_chkp_2=np.load(root_path+"D6_test_set"+"/D6_real_data_000001.npy")
D6_chkp_3=np.load(root_path+"D6_test_set"+"/D6_real_data_000010.npy")
D6_chkp_4=np.load(root_path+"D6_test_set"+"/D6_real_data_000100.npy")
D6_chkp_5=np.load(root_path+"D6_test_set"+"/D6_real_data_010000.npy")
'''

D6_chkp_6=np.load(root_path+"D6_test_set"+"/D6_real_data_011000.npy")
D6_chkp_7=np.load(root_path+"D6_test_set"+"/D6_real_data_011100.npy")
D6_chkp_8=np.load(root_path+"D6_test_set"+"/D6_real_data_011110.npy")
D6_chkp_9=np.load(root_path+"D6_test_set"+"/D6_real_data_011111.npy")



'''
D5_chkp_1=np.load(root_path+"D5_test_set"+"/D5_real_data_000000.npy")
D5_chkp_2=np.load(root_path+"D5_test_set"+"/D5_real_data_000001.npy")
D5_chkp_3=np.load(root_path+"D5_test_set"+"/D5_real_data_000010.npy")
D5_chkp_4=np.load(root_path+"D5_test_set"+"/D5_real_data_000100.npy")
D5_chkp_5=np.load(root_path+"D5_test_set"+"/D5_real_data_010000.npy")
'''

D5_chkp_6=np.load(root_path+"D5_test_set"+"/D5_real_data_011000.npy")
D5_chkp_7=np.load(root_path+"D5_test_set"+"/D5_real_data_011100.npy")
D5_chkp_8=np.load(root_path+"D5_test_set"+"/D5_real_data_011110.npy")
D5_chkp_9=np.load(root_path+"D5_test_set"+"/D5_real_data_011111.npy")
#D5_chkp_10=np.load(root_path+"D5_test_set"+"/D5_real_data_020002.npy")

'''
D4_chkp_1=np.load(root_path+"D4_test_set"+"/D4_real_data_000000.npy")
D4_chkp_2=np.load(root_path+"D4_test_set"+"/D4_real_data_000001.npy")
D4_chkp_3=np.load(root_path+"D4_test_set"+"/D4_real_data_000010.npy")
D4_chkp_4=np.load(root_path+"D4_test_set"+"/D4_real_data_000100.npy")
D4_chkp_5=np.load(root_path+"D4_test_set"+"/D4_real_data_010000.npy")
'''
D4_chkp_6=np.load(root_path+"D4_test_set"+"/D4_real_data_011000.npy")
D4_chkp_7=np.load(root_path+"D4_test_set"+"/D4_real_data_011100.npy")
D4_chkp_8=np.load(root_path+"D4_test_set"+"/D4_real_data_011110.npy")
D4_chkp_9=np.load(root_path+"D4_test_set"+"/D4_real_data_011111.npy")
#D4_chkp_10=np.load(root_path+"D4_test_set"+"/D4_real_data_020002.npy")

'''
D3_chkp_1=np.load(root_path+"D3_test_set"+"/D3_real_data_000000.npy")
D3_chkp_2=np.load(root_path+"D3_test_set"+"/D3_real_data_000001.npy")
D3_chkp_3=np.load(root_path+"D3_test_set"+"/D3_real_data_000010.npy")
D3_chkp_4=np.load(root_path+"D3_test_set"+"/D3_real_data_000100.npy")
D3_chkp_5=np.load(root_path+"D3_test_set"+"/D3_real_data_010000.npy")
'''
D3_chkp_6=np.load(root_path+"D3_test_set"+"/D3_real_data_011000.npy")
D3_chkp_7=np.load(root_path+"D3_test_set"+"/D3_real_data_011100.npy")
D3_chkp_8=np.load(root_path+"D3_test_set"+"/D3_real_data_011110.npy")
D3_chkp_9=np.load(root_path+"D3_test_set"+"/D3_real_data_011111.npy")
#D3_chkp_10=np.load(root_path+"D3_test_set"+"/D3_real_data_020002.npy")

'''
D2_chkp_1=np.load(root_path+"D2_test_set"+"/D2_real_data_000000.npy")
D2_chkp_2=np.load(root_path+"D2_test_set"+"/D2_real_data_000001.npy")
D2_chkp_3=np.load(root_path+"D2_test_set"+"/D2_real_data_000010.npy")
D2_chkp_4=np.load(root_path+"D2_test_set"+"/D2_real_data_000100.npy")
D2_chkp_5=np.load(root_path+"D2_test_set"+"/D2_real_data_010000.npy")
'''
D2_chkp_6=np.load(root_path+"D2_test_set"+"/D2_real_data_011000.npy")
D2_chkp_7=np.load(root_path+"D2_test_set"+"/D2_real_data_011100.npy")
D2_chkp_8=np.load(root_path+"D2_test_set"+"/D2_real_data_011110.npy")
D2_chkp_9=np.load(root_path+"D2_test_set"+"/D2_real_data_011111.npy")
#D2_chkp_10=np.load(root_path+"D2_test_set"+"/D2_real_data_020002.npy")

'''
D1_chkp_1=np.load(root_path+"D1_test_set"+"/D1_real_data_000000.npy")
D1_chkp_2=np.load(root_path+"D1_test_set"+"/D1_real_data_000010.npy")
D1_chkp_3=np.load(root_path+"D1_test_set"+"/D1_real_data_000110.npy")
D1_chkp_4=np.load(root_path+"D1_test_set"+"/D1_real_data_010100.npy")
D1_chkp_5=np.load(root_path+"D1_test_set"+"/D1_real_data_010000.npy")
'''
D1_chkp_6=np.load(root_path+"D1_test_set"+"/D1_real_data_011000.npy")
D1_chkp_7=np.load(root_path+"D1_test_set"+"/D1_real_data_011100.npy")
D1_chkp_8=np.load(root_path+"D1_test_set"+"/D1_real_data_011110.npy")
D1_chkp_9=np.load(root_path+"D1_test_set"+"/D1_real_data_011111.npy")
#D1_chkp_10=np.load(root_path+"D1_test_set"+"/D1_real_data_020002.npy")


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
#sa=np.array([[123.026]]*D5_chkp.shape[0])
#volr=np.array([[80.141]]*D5_chkp.shape[0])


#rooms_=['000000','000001','000010','000100','010000','011000','011100','011110','011111','020002']
#rooms_=['000001','000010','000100','010000']
rooms_=['011000','011100','011110','011111']
'''
D5_rooms=[D5_chkp_1,D5_chkp_2,D5_chkp_3,D5_chkp_4,D5_chkp_5,D5_chkp_6,D5_chkp_7,D5_chkp_8,D5_chkp_9,D5_chkp_10]
D4_rooms=[D4_chkp_1,D4_chkp_2,D4_chkp_3,D4_chkp_4,D4_chkp_5,D4_chkp_6,D4_chkp_7,D4_chkp_8,D4_chkp_9,D5_chkp_10]
D3_rooms=[D3_chkp_1,D3_chkp_2,D3_chkp_3,D3_chkp_4,D3_chkp_5,D3_chkp_6,D3_chkp_7,D3_chkp_8,D3_chkp_9,D3_chkp_10]
D2_rooms=[D2_chkp_1,D2_chkp_2,D2_chkp_3,D2_chkp_4,D2_chkp_5,D2_chkp_6,D2_chkp_7,D2_chkp_8,D2_chkp_9,D2_chkp_10]
D1_rooms=[D1_chkp_1,D1_chkp_2,D1_chkp_3,D1_chkp_4,D1_chkp_5,D1_chkp_6,D1_chkp_7,D1_chkp_8,D1_chkp_9,D1_chkp_10]
'''
D7_rooms=[D7_chkp_6,D7_chkp_7,D7_chkp_8,D7_chkp_9]
D6_rooms=[D6_chkp_6,D6_chkp_7,D6_chkp_8,D6_chkp_9]
D5_rooms=[D5_chkp_6,D5_chkp_7,D5_chkp_8,D5_chkp_9]#,D5_chkp_10]
D4_rooms=[D4_chkp_6,D4_chkp_7,D4_chkp_8,D4_chkp_9]#,D5_chkp_10]
D3_rooms=[D3_chkp_6,D3_chkp_7,D3_chkp_8,D3_chkp_9]#,D3_chkp_10]
D2_rooms=[D2_chkp_6,D2_chkp_7,D2_chkp_8,D2_chkp_9]#,D2_chkp_10]
D1_rooms=[D1_chkp_6,D1_chkp_7,D1_chkp_8,D1_chkp_9]#,D1_chkp_10]

D7_err=np.empty((1,6))
D6_err=np.empty((1,6))
D5_err=np.empty((1,6))
D4_err=np.empty((1,6))
D3_err=np.empty((1,6))
D2_err=np.empty((1,6))
D1_err=np.empty((1,6))

inclu_src=[i for i in range(81)]+[i for i in range(101,141)]
for i in np.arange(4):

        sa=np.array([[123.026]]*D5_rooms[i].shape[0])
        volr=np.array([[80.141]]*D5_rooms[i].shape[0])
        rt60_=np.array([rt60[rooms_[i]]]*D5_rooms[i].shape[0])

        D7_chkp=np.concatenate((D7_rooms[i],sa,volr,rt60_),axis=1)
        D6_chkp=np.concatenate((D6_rooms[i],sa,volr,rt60_),axis=1)
        D5_chkp=np.concatenate((D5_rooms[i],sa,volr,rt60_),axis=1)
        D4_chkp=np.concatenate((D4_rooms[i],sa,volr,rt60_),axis=1)
        D3_chkp=np.concatenate((D3_rooms[i],sa,volr,rt60_),axis=1)
        D2_chkp=np.concatenate((D2_rooms[i],sa,volr,rt60_),axis=1)
        D1_chkp=np.concatenate((D1_rooms[i],sa,volr,rt60_),axis=1)



        D7_rt60_500=np.abs((D7_chkp[:,4]*D7_std_rt60[2])-(D7_chkp[:,12])).reshape(-1,1)


        D7_rt60_1000=np.abs((D7_chkp[:,5]*D7_std_rt60[3])-(D7_chkp[:,13])).reshape(-1,1)


        D7_rt60_2000=np.abs((D7_chkp[:,6]*D7_std_rt60[4])-(D7_chkp[:,14])).reshape(-1,1)


        D7_rt60_4000=np.abs((D7_chkp[:,7]*D7_std_rt60[5])-(D7_chkp[:,15])).reshape(-1,1)


        D7_surf=np.abs((D7_chkp[:,0]*D7_std_surface)-(D7_chkp[:,8])).reshape(-1,1)
        D7_vol=np.abs((D7_chkp[:,1]*D7_std_volume)-(D7_chkp[:,9])).reshape(-1,1)
        D7_err_=np.concatenate((D7_surf[inclu_src],D7_vol[inclu_src],D7_rt60_500[inclu_src],D7_rt60_1000[inclu_src],D7_rt60_2000[inclu_src],D7_rt60_4000[inclu_src]),axis=1)
        D7_err=np.concatenate((D7_err,D7_err_),axis=0)

        '''
        D5_rt60_125=np.abs((D5_chkp[:,2]*D5_std_rt60[0])-(D5_chkp[:,10])).reshape(-1,1)
        where_are_NaNs = isnan(D5_rt60_125)
        D5_rt60_125[where_are_NaNs] = 0

        D5_rt60_250=np.abs((D5_chkp[:,3]*D5_std_rt60[1])-(D5_chkp[:,11])).reshape(-1,1)
        where_are_NaNs = isnan(D5_rt60_250)
        D5_rt60_250[where_are_NaNs] = 0
        '''

        D6_rt60_500=np.abs((D6_chkp[:,4]*D6_std_rt60[2])-(D6_chkp[:,12])).reshape(-1,1)


        D6_rt60_1000=np.abs((D6_chkp[:,5]*D6_std_rt60[3])-(D6_chkp[:,13])).reshape(-1,1)


        D6_rt60_2000=np.abs((D6_chkp[:,6]*D6_std_rt60[4])-(D6_chkp[:,14])).reshape(-1,1)


        D6_rt60_4000=np.abs((D6_chkp[:,7]*D6_std_rt60[5])-(D6_chkp[:,15])).reshape(-1,1)


        D6_surf=np.abs((D6_chkp[:,0]*D6_std_surface)-(D6_chkp[:,8])).reshape(-1,1)
        D6_vol=np.abs((D6_chkp[:,1]*D6_std_volume)-(D6_chkp[:,9])).reshape(-1,1)
        D6_err_=np.concatenate((D6_surf[inclu_src],D6_vol[inclu_src],D6_rt60_500[inclu_src],D6_rt60_1000[inclu_src],D6_rt60_2000[inclu_src],D6_rt60_4000[inclu_src]),axis=1)
        D6_err=np.concatenate((D6_err,D6_err_),axis=0)

        '''
        D5_rt60_125=np.abs((D5_chkp[:,2]*D5_std_rt60[0])-(D5_chkp[:,10])).reshape(-1,1)
        where_are_NaNs = isnan(D5_rt60_125)
        D5_rt60_125[where_are_NaNs] = 0

        D5_rt60_250=np.abs((D5_chkp[:,3]*D5_std_rt60[1])-(D5_chkp[:,11])).reshape(-1,1)
        where_are_NaNs = isnan(D5_rt60_250)
        D5_rt60_250[where_are_NaNs] = 0
        '''

        D5_rt60_500=np.abs((D5_chkp[:,4]*D5_std_rt60[2])-(D5_chkp[:,12])).reshape(-1,1)
        where_are_NaNs = isnan(D5_rt60_500)
        D5_rt60_500[where_are_NaNs] = 0

        D5_rt60_1000=np.abs((D5_chkp[:,5]*D5_std_rt60[3])-(D5_chkp[:,13])).reshape(-1,1)
        where_are_NaNs = isnan(D5_rt60_1000)
        D5_rt60_1000[where_are_NaNs] = 0

        D5_rt60_2000=np.abs((D5_chkp[:,6]*D5_std_rt60[4])-(D5_chkp[:,14])).reshape(-1,1)
        where_are_NaNs = isnan(D5_rt60_2000)
        D5_rt60_2000[where_are_NaNs] = 0

        D5_rt60_4000=np.abs((D5_chkp[:,7]*D5_std_rt60[5])-(D5_chkp[:,15])).reshape(-1,1)
        where_are_NaNs = isnan(D5_rt60_4000)
        D5_rt60_4000[where_are_NaNs] = 0


        D5_surf=np.abs((D5_chkp[:,0]*D5_std_surface)-(D5_chkp[:,8])).reshape(-1,1)
        D5_vol=np.abs((D5_chkp[:,1]*D5_std_volume)-(D5_chkp[:,9])).reshape(-1,1)
        D5_err_=np.concatenate((D5_surf[inclu_src],D5_vol[inclu_src],D5_rt60_500[inclu_src],D5_rt60_1000[inclu_src],D5_rt60_2000[inclu_src],D5_rt60_4000[inclu_src]),axis=1)
        D5_err=np.concatenate((D5_err,D5_err_),axis=0)



        '''
        D4_rt60_125=np.abs((D4_chkp[:,2]*D4_std_rt60[0])-(D4_chkp[:,10])).reshape(-1,1)
        where_are_NaNs = isnan(D4_rt60_125)
        D4_rt60_125[where_are_NaNs] = 0

        D4_rt60_250=np.abs((D4_chkp[:,3]*D4_std_rt60[1])-(D4_chkp[:,11])).reshape(-1,1)
        where_are_NaNs = isnan(D4_rt60_250)
        D4_rt60_250[where_are_NaNs] = 0
        '''

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
        #where_are_NaNs = isnan(D4_surf)
        #D4_surf[where_are_NaNs] = 0



        D4_vol=np.abs((D4_chkp[:,1]*D4_std_volume)-(D4_chkp[:,9])).reshape(-1,1)
        #where_are_NaNs = isnan(D4_vol)
        #D4_vol[where_are_NaNs] = 0

        D4_err_=np.concatenate((D4_surf[inclu_src],D4_vol[inclu_src],D4_rt60_500[inclu_src],D4_rt60_1000[inclu_src],D4_rt60_2000[inclu_src],D4_rt60_4000[inclu_src]),axis=1)
        D4_err=np.concatenate((D4_err,D4_err_),axis=0)


        '''
        D3_rt60_125=np.abs((D3_chkp[:,2]*D3_std_rt60[0])-(D3_chkp[:,10])).reshape(-1,1)
        where_are_NaNs = isnan(D3_rt60_125)
        D3_rt60_125[where_are_NaNs] = 0

        D3_rt60_250=np.abs((D3_chkp[:,3]*D3_std_rt60[1])-(D3_chkp[:,11])).reshape(-1,1)
        where_are_NaNs = isnan(D3_rt60_250)
        D3_rt60_250[where_are_NaNs] = 0
        '''

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
        #where_are_NaNs = isnan(D3_surf)
        #D3_surf[where_are_NaNs] = 0

        D3_vol=np.abs((D3_chkp[:,1]*D3_std_volume)-(D3_chkp[:,9])).reshape(-1,1)
        #where_are_NaNs = isnan(D3_vol)
        #D3_vol[where_are_NaNs] = 0
        D3_err_=np.concatenate((D3_surf[inclu_src],D3_vol[inclu_src],D3_rt60_500[inclu_src],D3_rt60_1000[inclu_src],D3_rt60_2000[inclu_src],D3_rt60_4000[inclu_src]),axis=1)
        D3_err=np.concatenate((D3_err,D3_err_),axis=0)



        '''
        D2_rt60_125=np.abs((D2_chkp[:,2]*D2_std_rt60[0])-(D2_chkp[:,10])).reshape(-1,1)
        where_are_NaNs = isnan(D2_rt60_125)
        D2_rt60_125[where_are_NaNs] = 0


        D2_rt60_250=np.abs((D2_chkp[:,3]*D2_std_rt60[1])-(D2_chkp[:,11])).reshape(-1,1)
        where_are_NaNs = isnan(D2_rt60_250)
        D2_rt60_250[where_are_NaNs] = 0
        '''

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
        where_are_NaNs = isnan(D2_surf)
        D2_surf[where_are_NaNs] = 0

        D2_vol=np.abs((D2_chkp[:,1]*D2_std_volume)-(D2_chkp[:,9])).reshape(-1,1)
        where_are_NaNs = isnan(D2_vol)
        D2_vol[where_are_NaNs] = 0

        D2_err_=np.concatenate((D2_surf[inclu_src],D2_vol[inclu_src],D2_rt60_500[inclu_src],D2_rt60_1000[inclu_src],D2_rt60_2000[inclu_src],D2_rt60_4000[inclu_src]),axis=1)
        D2_err=np.concatenate((D2_err,D2_err_),axis=0)

        '''
        D1_rt60_125=np.abs((D1_chkp[:,2]*D1_std_rt60[0])-(D1_chkp[:,10])).reshape(-1,1)
        where_are_NaNs = isnan(D1_rt60_125)
        D1_rt60_125[where_are_NaNs] = 0

        D1_rt60_250=np.abs((D1_chkp[:,3]*D1_std_rt60[1])-(D1_chkp[:,11])).reshape(-1,1)
        where_are_NaNs = isnan(D1_rt60_250)
        D1_rt60_250[where_are_NaNs] = 0
        '''

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
        #where_are_NaNs = isnan(D1_surf)
        #D1_surf[where_are_NaNs] = 0

        D1_vol=np.abs((D1_chkp[:,1]*D1_std_volume)-(D1_chkp[:,9])).reshape(-1,1)

        #where_are_NaNs = isnan(D1_vol)
        #D1_vol[where_are_NaNs] = 0
        D1_err_=np.concatenate((D1_surf[inclu_src],D1_vol[inclu_src],D1_rt60_500[inclu_src],D1_rt60_1000[inclu_src],D1_rt60_2000[inclu_src],D1_rt60_4000[inclu_src]),axis=1)
        D1_err=np.concatenate((D1_err,D1_err_),axis=0)

print(D7_err.shape)
print(D6_err.shape)
print(D5_err.shape)
print(D4_err.shape)
print(D3_err.shape)
print(D2_err.shape)
print(D1_err.shape)


out_rt=False

fig,axs=plt.subplots(3,2,figsize=(10,25))

'''
where_are_NaNs = isnan(D6_err)
D6_err[where_are_NaNs] = 0

where_are_NaNs = isnan(D5_err)
D5_err[where_are_NaNs] = 0


D4_err[where_are_NaNs] = 0
where_are_NaNs = isnan(D4_err)

D3_err[where_are_NaNs] = 0
where_are_NaNs = isnan(D3_err)

D2_err[where_are_NaNs] = 0
where_are_NaNs = isnan(D2_err)

D1_err[where_are_NaNs] = 0
where_are_NaNs = isnan(D1_err)
'''


'''
bplot1=axs[0,0].boxplot([D5_err[:,2],D4_err[:,2],D3_err[:,2],D2_err[:,2],D1_err[:,2]],showmeans=False,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,0].set_xticks([1,2,3,4,5])
axs[0,0].set_xticklabels(['D5','D4','D3','D2','D1'],rotation=45)
axs[0,0].set_ylabel("Abs Err Sec")
axs[0,0].set_title("RT 60 125hz")

bplot2=axs[0,1].boxplot([D5_err[:,3],D4_err[:,3],D3_err[:,3],D2_err[:,3],D1_err[:,3]],showmeans=False,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,1].set_xticks([1,2,3,4,5])
axs[0,1].set_xticklabels(['D5','D4','D3','D2','D1'],rotation=45)
axs[0,1].set_ylabel("Abs Err Sec")
axs[0,1].set_title("RT 60 250hz")
'''


print(np.mean(D7_err[:,2]),np.mean(D6_err[:,2]),np.mean(D5_err[:,2]),np.mean(D4_err[:,2]),np.mean(D3_err[:,2]),np.mean(D2_err[:,2]),np.mean(D1_err[:,2]))


print(np.mean(D7_err[:,3]),np.mean(D6_err[:,3]),np.mean(D5_err[:,3]),np.mean(D4_err[:,3]),np.mean(D3_err[:,3]),np.mean(D2_err[:,3]),np.mean(D1_err[:,3]))

print(np.mean(D7_err[:,4]),np.mean(D6_err[:,4]),np.mean(D5_err[:,4]),np.mean(D4_err[:,4]),np.mean(D3_err[:,4]),np.mean(D2_err[:,4]),np.mean(D1_err[:,4]))

print(np.mean(D7_err[:,5]),np.mean(D6_err[:,5]),np.mean(D5_err[:,5]),np.mean(D4_err[:,5]),np.mean(D3_err[:,5]),np.mean(D2_err[:,5]),np.mean(D1_err[:,5]))

print(np.mean(D7_err[:,0]),np.mean(D6_err[:,0]),np.mean(D5_err[:,0]),np.mean(D4_err[:,0]),np.mean(D3_err[:,0]),np.mean(D2_err[:,0]),np.mean(D1_err[:,0]))

print(np.mean(D7_err[:,1]),np.mean(D6_err[:,1]),np.mean(D5_err[:,1]),np.mean(D4_err[:,1]),np.mean(D3_err[:,1]),np.mean(D2_err[:,1]),np.mean(D1_err[:,1]))

#print(np.mean(D2_err[0,4]))



bplot3=axs[0,0].boxplot([D7_err[:,2],D6_err[:,2],D5_err[:,2],D4_err[:,2],D3_err[:,2],D2_err[:,2],D1_err[:,2]],showmeans=False,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,0].set_xticks([1,2,3,4,5,6,7])
axs[0,0].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[0,0].set_ylabel("Abs Err Sec")
axs[0,0].set_title("RT 60 500hz")

bplot4=axs[0,1].boxplot([D7_err[:,3],D6_err[:,3],D5_err[:,3],D4_err[:,3],D3_err[:,3],D2_err[:,3],D1_err[:,3]],showmeans=False,vert=True,showfliers=out_rt,patch_artist=True)
axs[0,1].set_xticks([1,2,3,4,5,6,7])
axs[0,1].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[0,1].set_ylabel("Abs Err Sec")
axs[0,1].set_title("RT 60 1000hz")

bplot5=axs[1,0].boxplot([D7_err[:,4],D6_err[:,4],D5_err[:,4],D4_err[:,4],D3_err[:,4],D2_err[:,4],D1_err[:,4]],showmeans=False,vert=True,showfliers=out_rt,patch_artist=True)
axs[1,0].set_xticks([1,2,3,4,5,6,7])
axs[1,0].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[1,0].set_ylabel("Abs Err Sec")
axs[1,0].set_title("RT 60 2000hz")

bplot6=axs[1,1].boxplot([D7_err[:,5],D6_err[:,5],D5_err[:,5],D4_err[:,5],D3_err[:,5],D2_err[:,5],D1_err[:,5]],showmeans=False,vert=True,showfliers=out_rt,patch_artist=True)
axs[1,1].set_xticks([1,2,3,4,5,6,7])
axs[1,1].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[1,1].set_ylabel("Abs Err Sec")
axs[1,1].set_title("RT 60 4000hz")

bplot7=axs[2,0].boxplot([D7_err[:,0],D6_err[:,0],D5_err[:,0],D4_err[:,0],D3_err[:,0],D2_err[:,0],D1_err[:,0]],showmeans=False,vert=True,showfliers=out_rt,patch_artist=True)
axs[2,0].set_xticks([1,2,3,4,5,6,7])
axs[2,0].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[2,0].set_ylabel("Abs Err M2")
axs[2,0].set_title("Surface Err")

bplot8=axs[2,1].boxplot([D7_err[:,1],D6_err[:,1],D5_err[:,1],D4_err[:,1],D3_err[:,1],D2_err[:,1],D1_err[:,1]],showmeans=False,vert=True,showfliers=out_rt,patch_artist=True)
axs[2,1].set_xticks([1,2,3,4,5,6,7])
axs[2,1].set_xticklabels(['D7','D6','D5','D4','D3','D2','D1'],rotation=45)
axs[2,1].set_ylabel("Abs Err M3")
axs[2,1].set_title("Volume Err")


colors=['pink','lightblue','lightgreen','orange','cyan','green','magenta']


for bplot in (bplot3,bplot4,bplot5,bplot6,bplot7,bplot8):
    for patch,color in zip(bplot['boxes'],colors):
        patch.set_facecolor(color)


#fig.tight_layout(pad=3.0)
#plt.xticks([1,2,3],('Dummy Bnf','bnf','Dummy M','M'))
#plt.title("Absolute Diff Estimated Mean And Target RT60")
plt.savefig("real_decorate_nonreflective_rooms_src_0-6_excluding_4.png")
