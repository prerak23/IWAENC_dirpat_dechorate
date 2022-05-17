import matplotlib
#matplotlib.use('Agg')
import numpy as np
import torch
import h5py
import data_loader_test
import mlh_baseline as net

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools



sns.set_theme()



device=torch.device("cpu")
model_1=net.Model_1().to(device="cpu")
model_3=net.Model_3().to(device="cpu")
model_4=net.Ensemble(model_1,model_3,1).to(device="cpu")
model_4.batch_size=1

chkp=torch.load("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/training/D6_0111/mlh_tas_save_best_sh.pt",map_location=device)

model_1.load_state_dict(chkp['model_dict_1'])
model_3.load_state_dict(chkp['model_dict_3'])
model_4.load_state_dict(chkp['model_dict_ens'])
optimizer = optim.Adam(model_4.parameters(), lr=0.0001,weight_decay=1e-5)
optimizer.load_state_dict(chkp['optimizer_dic'])

model_1.eval()
model_3.eval()
model_4.eval()

#abc=np.load("/home/psrivastava/baseline/scripts/pre_processing/test_random_ar.npy")

no_of_vps=1

#test_data=data_loader_test.load_data_test(no_of_vps)



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


#no_of_rooms=np.arange(18001,20000)


#no_of_rooms=np.load("/home/psrivastava/baseline/scripts/pre_processing/robustness/high_reverb_rooms.npy")


def cal_features(ch_1,ch_2):

    bz=no_of_vps

    enc_ch1 = torch.stft(ch_1.view(bz, -1), n_fft=768, hop_length=384, return_complex=True)

    enc_ch2 = torch.stft(ch_2.view(bz, -1), n_fft=768, hop_length=384, return_complex=True)

    f = torch.view_as_real(enc_ch1)

    f = torch.sqrt(f[:, :, :, 0] ** 2 + f[:, :, :, 1] ** 2)  # Magnitude


    #f2 = torch.view_as_real(enc_ch2)

    #f2 = torch.sqrt(f2[:, :, :, 0] ** 2 + f2[:, :, :, 1] ** 2)  # Magnitude


    # Ipd ild calculation

    cc = enc_ch1 * torch.conj(enc_ch2)
    ipd = cc / (torch.abs(cc)+10e-8)
    ipd_ri = torch.view_as_real(ipd)
    ild = torch.log(torch.abs(enc_ch1) + 10e-8) - torch.log(torch.abs(enc_ch2) + 10e-8)

    x2 = torch.cat((ipd_ri[:, :, :, 0], ipd_ri[:, :, :, 1], ild), axis=1)

    #print(f.shape, x2.shape)


    return f ,x2


estimate=np.zeros((1,2)) #28

#m=nn.AvgPool1d(32,stride=1)
#o=nn.AvgPool1d(54,stride=1)
#n=nn.AvgPool1d(2,stride=2)

#rt60_hist=np.zeros((1,6))
#ab_hist=np.zeros((1,6))
#suf_vol=np.zeros((1,2))

path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/dechorate_real_speech/"
files_=os.listdir(path)

print(os.listdir(path))
#itr=[[(1,5),(6,10),(11,15),(16,20),(21,25)],[(6,10),(11,15),(16,20),(21,25),(26,30)],[(1,5),(6,10),(16,20),(21,25),(26,30)],[(1,5),(6,10),(11,15),(21,25),(26,30)],[(1,5),(6,10),(11,15),(16,20),(26,30)]]
#mic=[(1,5),(21,25),(11,15)] #,(16,20),(21,25)]

itr=[(1,5),(6,10),(11,15),(16,20),(21,25),(26,30)]
#src=[3]
src=[0,1,2,3,4,5,6]
#total_pred=np.zeros((1,8))


for s in src:

    #f_=np.concatenate((np.array([sa]),np.array([volr]),np.array([rt60[room_no]]).reshape(1,6)),axis=1)
    estimated=np.zeros((1,2))
    total_pred=np.zeros((1,8))
    print(s)
    for r in files_:
        for mic_it in list(itertools.combinations([0,1,2,3,4,5],3)):
            room_wet_speech=np.load(path+r)
            room_no=r.split("_")[1].split(".")[0]

            total_num=np.zeros((1,8))
            total_precision=np.zeros((1,8))

            #for mic in itr:
            combined_ch_1=room_wet_speech[48000:96000,itr[mic_it[0]][0]:itr[mic_it[0]][1]:3,s]  #mic[0],mic[1]  #32000,80000
            combined_ch_2=room_wet_speech[48000:96000,itr[mic_it[1]][0]:itr[mic_it[1]][1]:3,s]  #mic[0],mic[1]
            combined_ch_3=room_wet_speech[48000:96000,itr[mic_it[2]][0]:itr[mic_it[2]][1]:3,s]  #mic[0],mic[1]
            #combined_ch_4=room_wet_speech[48000:96000,mic[3][0]:mic[3][1]:3,s]  #mic[0],mic[1]
            #combined_ch_5=room_wet_speech[48000:96000,mic[4][0]:mic[4][1]:3,s]  #mic[0],mic[1]

            combined_ch_1=torch.tensor(combined_ch_1).float()
            combined_ch_2=torch.tensor(combined_ch_2).float()
            combined_ch_3=torch.tensor(combined_ch_3).float()
            #combined_ch_4=torch.tensor(combined_ch_4).float()
            #combined_ch_5=torch.tensor(combined_ch_5).float()


            ch1_feat_1,ch2_feat_1=cal_features(combined_ch_1[:,0],combined_ch_1[:,1])

            ch1_feat_2,ch2_feat_2=cal_features(combined_ch_2[:,0],combined_ch_2[:,1])
            ch1_feat_3,ch2_feat_3=cal_features(combined_ch_3[:,0],combined_ch_3[:,1])

            #ch1_feat_4,ch2_feat_4=cal_features(combined_ch_4[:,0],combined_ch_4[:,1])
            #ch1_feat_5,ch2_feat_5=cal_features(combined_ch_5[:,0],combined_ch_5[:,1])

            #ch1_feat=torch.tensor(ch1_feat).float().to(device="cpu")
            #ch2_feat=torch.tensor(ch2_feat).float().to(device="cpu")

            #x=model_1(mic_x.reshape(1,769,63))#ch2_feat_1.reshape(1,2307,63))

            #x=model_1(torch.tensor(mic_x).unsqueeze(0).float())
            #x=model_1(ch1_feat_1.reshape(1,769,63))
            #print(x.shape)
            #x_2=model_1(ch1_feat_2.reshape(1,769,63))#ch2_feat_2.reshape(1,2307,63))
            #x_3=model_1(ch1_feat_3.reshape(1,769,63))#ch2_feat_3.reshape(1,2307,63))
            #x_4=model_1(ch1_feat_4.reshape(1,769,63))#ch2_feat_4.reshape(1,2307,63))
            #x_5=model_1(ch1_feat_5.reshape(1,769,63))#ch2_feat_5.reshape(1,2307,63))





            #x=torch.cat([x,x2],dim=1)
            #x_2=torch.cat([x_2,x_2_],dim=1)
            #x_3=torch.cat([x_3,x_3_],dim=1)
            #x_4=torch.cat([x_4,x_4_],dim=1)
            #x_5=torch.cat([x_5,x_5_],dim=1)





            #x=m(x)
            #x_2=m(x_2)
            #x_3=m(x_3)
            #x_4=m(x_4)
            #x_5=m(x_5)

            mean,variance=model_4(ch1_feat_1,ch2_feat_1,0)
            mean_2,variance_2=model_4(ch1_feat_2,ch2_feat_2,0)
            mean_3,variance_3=model_4(ch1_feat_3,ch2_feat_3,0)



            precision_1=1/variance
            precision_2=1/variance_2
            precision_3=1/variance_3
            #precision_4=1/variance_4
            #precision_5=1/variance_5

            total_num=total_num+(mean.detach().numpy()*precision_1.detach().numpy())
            total_num=total_num+(mean_2.detach().numpy()*precision_2.detach().numpy())
            total_num=total_num+(mean_3.detach().numpy()*precision_3.detach().numpy())
            #total_num=total_num+(mean_4.detach().numpy()*precision_4.detach().numpy())
            #total_num=total_num+(mean_5.detach().numpy()*precision_5.detach().numpy())





            total_precision=total_precision+precision_1.detach().numpy()
            total_precision=total_precision+precision_2.detach().numpy()
            total_precision=total_precision+precision_3.detach().numpy()
            #total_precision=total_precision+precision_4.detach().numpy()
            #total_precision=total_precision+precision_5.detach().numpy()






            final_mean=total_num/total_precision

            final_precision=1/total_precision

            actual_value=np.concatenate((np.array([sa]),np.array([volr]),np.array([rt60[room_no]])),axis=1)
            final_mean[:,0]=final_mean[:,0]*D6_std_surface
            final_mean[:,1]=final_mean[:,1]*D6_std_volume
            final_mean[:,2]=final_mean[:,2]*D6_std_rt60[0]
            final_mean[:,3]=final_mean[:,3]*D6_std_rt60[1]
            final_mean[:,4]=final_mean[:,4]*D6_std_rt60[2]
            final_mean[:,5]=final_mean[:,5]*D6_std_rt60[3]
            final_mean[:,6]=final_mean[:,6]*D6_std_rt60[4]
            final_mean[:,7]=final_mean[:,7]*D6_std_rt60[5]

            final_mean=final_mean-actual_value


            total_pred=np.concatenate((total_pred,final_mean.reshape(1,8)),axis=0)

            #print(total_pred.shape)


        #tmp=np.concatenate((x.detach().numpy(),np.array([volr]).reshape(1,1)),axis=1) #final_mean
        #estimated=np.concatenate((estimated,tmp),axis=0)

    np.save("/home/psrivastava/axis-2/IWAENC/z_test/D6_test_set/D6_real_data_src_"+str(s)+".npy",total_pred)




















'''

for r in no_of_rooms:

    print('room no',r)
    bnf_mixture_ch1,bnf_mixture_ch2,rt60,ab,surf,vol=test_data.return_data(r)

    #rt60_hist=np.concatenate((rt60_hist,rt60.reshape(1,6)),axis=0)
    #ab_hist=np.concatenate((ab_hist,ab.reshape(1,6)),axis=0)

    #s_v=np.array([surf,vol]).reshape(1,2)
    #suf_vol=np.concatenate((suf_vol,s_v),axis=0)


    #For 15 sec signals
    #first_ex=bnf_mixture_ch1[0,:48000]
    #second_ex=bnf_mixture_ch1[0,48000:96000]
    #third_ex=bnf_mixture_ch1[0,96000:144000]
    #fourth_ex=bnf_mixture_ch1[0,144000:192000]
    #fifth_ex=bnf_mixture_ch1[0,192000:240000]

    #first_ex_ch2=bnf_mixture_ch1[0,:48000]
    #second_ex_ch2=bnf_mixture_ch1[0,48000:96000]
    #third_ex_ch2=bnf_mixture_ch1[0,96000:144000]
    #fourth_ex_ch2=bnf_mixture_ch1[0,144000:192000]
    #fifth_ex_ch2=bnf_mixture_ch1[0,192000:240000]

    #ch1_feat_first_ex,ch2_feat_first_ex=cal_features(first_ex,first_ex_ch2)
    #ch1_feat_second_ex,ch2_feat_second_ex=cal_features(second_ex,second_ex_ch2)



    #ch1_feat_third_ex,ch2_feat_third_ex=cal_features(third_ex,third_ex_ch2)
    #ch1_feat_fourth_ex,ch2_feat_fourth_ex=cal_features(fourth_ex,fourth_ex_ch2)
    #ch1_feat_fifth_ex,ch2_feat_fifth_ex=cal_features(fifth_ex,fifth_ex_ch2)






    ch1_feat,ch2_feat=cal_features(bnf_mixture_ch1,bnf_mixture_ch2)

    #print("vp feat ch1",ch1_feat.shape)

    #print("vp feat ch2",ch2_feat.shape)


    total_mean_local=np.zeros((1,14))

    total_variance_local=np.zeros((1,14))

    total_precision=np.zeros((1,14))

    total_num=np.zeros((1,14))

    f=np.concatenate((surf.reshape(1,1),vol.reshape(1,1),rt60.reshape(1,6),ab.reshape(1,6)),axis=1)


    for vp in range(no_of_vps):

        feat_1=ch1_feat[vp,:,:].clone().requires_grad_(True).to(device="cpu")
        feat_2=ch2_feat[vp,:,:].clone().requires_grad_(True).to(device="cpu")
        #print(feat_1)
        #print(feat_2)
        feat_1=torch.unsqueeze(feat_1,axis=0)
        feat_2=torch.unsqueeze(feat_2,axis=0)

        #print(model_1(feat_1).shape)
        #print(model_2(feat_2).shape)
        #print(model_2(torch.randn((1,2307,63)).float().to(device="cuda")))

        #x1=feat_1.reshape(1,769,63)

        #x2=feat_2.reshape(1,2307,63)




    #print(x1.shape)
    #print(x2.shape)

    x,x2=model_1(ch1_feat[0,:,:].reshape(1,769,63),ch2_feat[0,:,:].reshape(1,2307,63))
    x_2,x2_2=model_1(ch1_feat[1,:,:].reshape(1,769,63),ch2_feat[1,:,:].reshape(1,2307,63))
    x_3,x2_3=model_1(ch1_feat[2,:,:].reshape(1,769,63),ch2_feat[2,:,:].reshape(1,2307,63))
    x_4,x2_4=model_1(ch1_feat[3,:,:].reshape(1,769,63),ch2_feat[3,:,:].reshape(1,2307,63))
    x_5,x2_5=model_1(ch1_feat[4,:,:].reshape(1,769,63),ch2_feat[4,:,:].reshape(1,2307,63))

    #x1,x2=model_1(ch1_feat[0,:,:].reshape(1,769,63),ch2_feat[0,:,:].reshape(1,2307,63))

    #print(x.shape)
    #x2=model_1(ch1_feat_second_ex[0,:,:].reshape(1,769,63)) #1
    #x3=model_1(ch1_feat_third_ex[0,:,:].reshape(1,769,63))  #2
    #x4=model_1(ch1_feat_fourth_ex[0,:,:].reshape(1,769,63)) #3
    #x5=model_1(ch1_feat_fifth_ex[0,:,:].reshape(1,769,63))  #4



    #x=torch.cat((x,x2),axis=1)
    #print(x.shape)
    #x=n(x.reshape(1,32,-1))
    #print(x.shape)


    #x_con=torch.cat([x,x2],dim=-1).view(-1,x.shape[-1])
    #x=n(x_con.reshape(1,32,192))

    #x=torch.cat([x1,x2],dim=1)
    #x=m(x.reshape(1,1248,32))
    #x2=o(x2.reshape(1,1152,54))
    #x2=torch.cat([x,x2],dim=1)

    #print(x.shape)
    #x=x.reshape(1,-1)
    #mean,variance=model_3(x.reshape(1,1248))
    #mean,variance=model_3(x2.reshape(1,1248))


    #x=m(x)
    #x_2=m(x_2)
    #x_3=m(x_3)
    #x_4=m(x_4)
    #x_5=m(x_5)
    #x2=o(x2)
    #x2_2=o(x2_2)
    #x2_3=o(x2_3)
    #x2_4=o(x2_4)
    #x2_5=o(x2_5)

    x=torch.cat([x,x2],dim=1)
    x2=torch.cat([x_2,x2_2],dim=1)
    x3=torch.cat([x_3,x2_3],dim=1)
    x4=torch.cat([x_4,x2_4],dim=1)
    x5=torch.cat([x_5,x2_5],dim=1)

    x=m(x)
    x2=m(x2)
    x3=m(x3)
    x4=m(x4)
    x5=m(x5)





    mean_1,variance_1=model_3(x.reshape(1,1248))
    mean_2,variance_2=model_3(x2.reshape(1,1248))
    mean_3,variance_3=model_3(x3.reshape(1,1248))
    mean_4,variance_4=model_3(x4.reshape(1,1248))
    mean_5,variance_5=model_3(x5.reshape(1,1248))





    x=m(x)
    x2=m(x2)
    x3=m(x3)
    x4=m(x4)
    x5=m(x5)

    mean_1,variance_1=model_3(x.reshape(1,96))
    mean_2,variance_2=model_3(x2.reshape(1,96))
    mean_3,variance_3=model_3(x3.reshape(1,96))
    mean_4,variance_4=model_3(x4.reshape(1,96))
    mean_5,variance_5=model_3(x5.reshape(1,96))




    #mean=[]

    #print(mean_1.shape)

    #print(mean_1[0,0])


    #print(mean_2[0,0])

    #Average the final estimate


    #x=x.reshape(1,-1)

    #mean,variance=model_4(feat_1,feat_2)

    #print(mean)
    #print(variance)
    #total_mean_local=np.concatenate((total_mean_local,mean.detach().numpy()),axis=0)

    #total_variance_local=np.concatenate((total_variance_local,variance.detach().numpy()),axis=0)

    #total_num=total_num+(mean.detach().numpy()*precision.detach().numpy())

    #final_precision=1/total_precision
    #mean=np.array(mean).reshape(1,14)


    total_mean_local=np.concatenate((total_mean_local,mean_1.detach().numpy()),axis=0)
    total_mean_local=np.concatenate((total_mean_local,mean_2.detach().numpy()),axis=0)
    total_mean_local=np.concatenate((total_mean_local,mean_3.detach().numpy()),axis=0)
    total_mean_local=np.concatenate((total_mean_local,mean_4.detach().numpy()),axis=0)
    total_mean_local=np.concatenate((total_mean_local,mean_5.detach().numpy()),axis=0)


    total_variance_local=np.concatenate((total_variance_local,variance_1.detach().numpy()),axis=0)
    total_variance_local=np.concatenate((total_variance_local,variance_2.detach().numpy()),axis=0)
    total_variance_local=np.concatenate((total_variance_local,variance_3.detach().numpy()),axis=0)
    total_variance_local=np.concatenate((total_variance_local,variance_4.detach().numpy()),axis=0)
    total_variance_local=np.concatenate((total_variance_local,variance_5.detach().numpy()),axis=0)





    precision_1=1/variance_1
    precision_2=1/variance_2
    precision_3=1/variance_3
    precision_4=1/variance_4
    precision_5=1/variance_5

    total_num=total_num+(mean_1.detach().numpy()*precision_1.detach().numpy())
    total_num=total_num+(mean_2.detach().numpy()*precision_2.detach().numpy())
    total_num=total_num+(mean_3.detach().numpy()*precision_3.detach().numpy())
    total_num=total_num+(mean_4.detach().numpy()*precision_4.detach().numpy())
    total_num=total_num+(mean_5.detach().numpy()*precision_5.detach().numpy())





    total_precision=total_precision+precision_1.detach().numpy()
    total_precision=total_precision+precision_2.detach().numpy()
    total_precision=total_precision+precision_3.detach().numpy()
    total_precision=total_precision+precision_4.detach().numpy()
    total_precision=total_precision+precision_5.detach().numpy()






    final_mean=total_num/total_precision

    final_precision=1/total_precision

    f_=np.concatenate((final_mean,f),axis=1)

    estimate=np.concatenate((estimate,f_),axis=0)





    #f_=np.concatenate((mean.detach().numpy(),f),axis=1) #mean.detach.numpy()
    #estimate=np.concatenate((estimate,f_),axis=0)


np.save("bn_vp_5_test_weightmean.npy",estimate)
'''
#colors=['lightgreen','lightblue','red','cyan','orange','purple']
'''
colors=['lightblue','deepskyblue','dodgerblue','cornflowerblue','royalblue','navy']

n,bins,patches=plt.hist(rt60_hist[1:,:],25,stacked=True,density=True,color=colors,label=['125','250','500','1000','2000','4000'])
plt.legend()
plt.savefig("hist_testset_rt60.png")

plt.clf()
n,bins,patches=plt.hist(ab_hist[1:,:],25,stacked=True,density=True,color=colors,label=['125 ab','250 ab','500 ab','1000 ab','2000 ab','4000 ab'])
plt.legend()
plt.savefig("hist_testset_ab.png")



plt.clf()
n,bins,patches=plt.hist(suf_vol[1:,:],20,stacked=True,density=True,color=['lightgreen','lightblue'],label=['Surface','Volume'])
plt.legend()
plt.savefig("hist_testset_surface_vol.png")

'''
