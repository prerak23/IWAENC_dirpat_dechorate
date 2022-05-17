import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import h5py
import yaml
import random

#From big file
#abcd=h5py.File('/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D5_1111/noisy_mixtures/D5_1111_aggregated_mixture.hdf5','r')

#Specialized test set
abcd=h5py.File('/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/test_400_rooms/D5/D5_1111_aggregated_mixture.hdf5')

#From big file
#anno_file=h5py.File('/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D5_1111/noisy_mixtures/D5_1111_aggregated_mixture_annotations.hdf5','r')

anno_file=h5py.File('/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/test_400_rooms/D5/D5_1111_aggregated_mixture_annotations.hdf5','r')
#test_set=np.load("/home/psrivastava/baseline/scripts/pre_processing/test_random_ar.npy")

class load_data_test():
    def __init__(self,vps):
        self.vps=vps


    def return_data(self,room_id):
        random_vps=np.array(random.sample(range(3),self.vps))+1
        sample_col_ch1=np.zeros((1,48000))
        sample_col_ch2=np.zeros((1,48000))
        #rt60_col=np.zeros((1,6))
        print(random_vps)
        for vp in random_vps:
            bn_sample_vp_ch1=abcd['room_nos']["room_"+str(room_id)]['nsmix_f'][(vp-1)*2,:]
            bn_sample_vp_ch2=abcd['room_nos']["room_"+str(room_id)]['nsmix_f'][((vp*2)-1),:]

            #rt60=rt60_file['room_nos']["room_"+str(room_id)]['rt60'][()][(vp-1),:]

            sample_col_ch1=np.concatenate((sample_col_ch1,bn_sample_vp_ch1.reshape(1,48000)),axis=0)

            sample_col_ch2=np.concatenate((sample_col_ch2,bn_sample_vp_ch2.reshape(1,48000)),axis=0)


            #rt60_col=np.concatenate((rt60_col,rt60.reshape(1,6)),axis=0)




        surface=anno_file['room_nos']["room_"+str(room_id)]['surface'][0]
        volume=anno_file['room_nos']["room_"+str(room_id)]['volume'][0]
        rt60=anno_file['room_nos']['room_'+str(room_id)]['rt60'][()].reshape(6)
        return torch.tensor(sample_col_ch1[1:,:]).float(),torch.tensor(sample_col_ch2[1:,:]).float(),torch.tensor(rt60).float().float(),torch.tensor(surface).float(),torch.tensor(volume).float()


'''
kd=load_data_test(3)

samp1,samp2,rt,ab,surf,vol=kd.return_data(18001)
print(samp1.shape)
print(rt)
'''
