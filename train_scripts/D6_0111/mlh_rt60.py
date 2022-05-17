#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader


import numpy as np
import data_loader as dl
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


#Calculate MSE loss function
def pred(mean_values, target_value):
    param_mse_loss = 0

    for i in range(16):
        param_mse_loss += (mean_values[i] - target_value[i]) ** 2

    return param_mse_loss / 16


def NLLloss(y, mean, var):
    """ Negative log-likelihood loss function. """
    return (torch.log(var) + ((y - mean).pow(2)) / var).sum()

dvs="cuda"

#Function used to calculate 2 features STFT and IPD ILD , given 2 channels from the same view point.

def cal_features(ch1, ch2):
    enc_ch1 = torch.stft(ch1.view(16, -1), n_fft=768, hop_length=384, return_complex=True)

    #enc_ch2 = torch.stft(ch2.view(128, -1), n_fft=768, hop_length=384, return_complex=True)

    f = torch.view_as_real(enc_ch1)

    f = torch.sqrt(f[:, :, :, 0] ** 2 + f[:, :, :, 1] ** 2)  # Magnitude


    return f




#Std deviation and varince of all the parameters that we are estimating from the training set , for the sole purpose of scaling all the parameters on the same scale

std_volume = 72.6143118
std_surface = 64.029048

std_rt60=[ 0.250104, 0.207987, 0.214506, 0.194992]  #For 125,250,500,1000,2000,4000 Hz

#std_abs=[0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663016] #For 125,250,500,1000,2000,4000  Hz

#vari_vol= 5347.319611103444
#vari_surface=4157.618919651688
#vari_rt60=[0.06384,0.061628,0.04562,0.02882,0.02164,0.02023]

#vari_abs=[0.00571186,0.00716683,0.00592367,0.00545315,0.00556748,0.00549911]

'''
std_rt60_125 = 0.7793691
std_rt60_250 = 0.7605436
std_rt60_500 = 0.6995225
std_rt60_1000 = 0.7076665
std_rt60_2000 = 0.6520753
std_rt60_4000 = 0.51794204
'''
#0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663016












def train(model, train_loader, optimizer, epoch, ar_loss, batch_loss_ar):
    model.train()
    #print("training....Epoch", epoch)

    loss_batch = 0

    tr_loss = 0

    adcc = np.zeros((1, 4))  # Total 8 Values are saved for the purpose of analysis of the data after the model is trained.

    track_var = np.zeros((1, 4))  # Total 8 values for the tracking of the variance while the model is getting trained

    #Intialization of the varaibles for mse tracking
    rt60_500_ = 0
    rt60_1000_ = 0
    rt60_2000_ = 0
    rt60_4000_ = 0



    for batch_idx, sample_batched in enumerate(train_loader):

        data, surface, volume, rt60 = sample_batched['bnsample'].float(),sample_batched['surface'].float().to(device=dvs), sample_batched['volume'].float().to(device=dvs), sample_batched['rt60'].float().to(device=dvs)

        # rt60=torch.log(rt60)
        # absorption=torch.log(absorption)

        optimizer.zero_grad()

        x_1 = cal_features(data[:, 0, :].to(device=dvs), data[:, 1, :].to(device=dvs)) #Calculate features STFT and IPD_ILD

        mean, variance = model(x_1,epoch) #The model is being trained to predict two values mean and variance

        # target_,idx=torch.sort(target,1)

        #Varaible intialization for loss accumalation for freq dependent parameters.

        rt60_loss = 0

        # Calculate NLL Loss for freq dependent rt60 and absorption coeff.

        for i in range(4):
            rt60_loss += NLLloss(rt60[:, i+2]/std_rt60[i], mean[:,  i], variance[:, i])


        #Add all the loss and normalize by 14 as we are estimating 14 parameters.

        loss = (rt60_loss)/ 4


        # rt60_loss=rt60_loss/6
        # absorp_loss=absorp_loss/6
        # loss=loss_surface+loss_volume+rt60+absorp_loss/4
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss_1 = loss + l1_lambda*l1_norm

        loss_1.backward()
        optimizer.step()

        #Calulate MSE and add MSE for all the param's

        rt60_500_ += float(pred(mean[:, 0]*std_rt60[0], rt60[:, 2]).item())
        rt60_1000_ += float(pred(mean[:, 1]*std_rt60[1], rt60[:, 3]).item())
        rt60_2000_ += float(pred(mean[:, 2]*std_rt60[2], rt60[:, 4]).item())
        rt60_4000_ += float(pred(mean[:, 3]*std_rt60[3], rt60[:, 5]).item())


        loss_batch = float(loss.item()) + loss_batch

        tr_loss = float(loss.item()) + tr_loss

        # print(surface.shape,volume.shape,absorption.shape,rt60.shape)

        #Concat all the real annotations

        total = rt60[:,2:]

        # For tracking the variance while traning.

        track_var = np.concatenate((track_var, variance.detach().cpu().clone().numpy().reshape(16, 4)), axis=0)


        if epoch == 0:

            adcc = np.concatenate((adcc, total.detach().cpu().clone().numpy().reshape(16, 4)), axis=0)



        del loss, data, rt60_loss

        #Save MSE loss after 100 epochs.

        if batch_idx % 50 == 49:
            # print("Running loss after 100 batches",(loss_batch/100),loss_batch)
            batch_loss_ar.append(((rt60_500_) / 50, (rt60_1000_) / 50, (rt60_2000_) / 50, (rt60_4000_) / 50))

            rt60_500_ = 0
            rt60_1000_ = 0
            rt60_2000_ = 0
            rt60_4000_ = 0


            # del surface_mse,volume_mse,rt60_125_,rt60_250_,rt60_500_,rt60_1000_,rt60_2000_,rt60_4000_,ab_125_,ab_250_,ab_500_,ab_1000_,ab_2000_,ab_4000_

            # loss_batch=0

    #print("Epoch Loss", (tr_loss / batch_idx), epoch)

    ar_loss.append(tr_loss / batch_idx)
    return ar_loss, batch_loss_ar, adcc, track_var


def val(model, train_loader, optimizer, epoch, val_data_ar, acc_data_ar):

    model.eval()

    val_loss = 0

    rt60_loss_mse_500 = 0
    rt60_loss_mse_1000 = 0
    rt60_loss_mse_2000 = 0
    rt60_loss_mse_4000 = 0


    local_data_sp = np.zeros([1, 14])  # Save validation data(real annotations params + estimated params 8(mean)+8(varaince)+8(real annotations)+room_id(1)+vp_id(1) ~ 26 )  for analysis purposes 42

    for batch_idx, sample_batched in enumerate(train_loader):

        data, surface, volume,  rt60, rm, vp = sample_batched['bnsample'].float(), sample_batched[
            'surface'].float().to(device=dvs), sample_batched['volume'].float().to(device=dvs), sample_batched[
                                                              'rt60'].float().to(device=dvs), sample_batched[
                                                              'room'].to(device=dvs), sample_batched['vp'].to(device=dvs)


        rt60_loss_mse_500_t = 0
        rt60_loss_mse_1000_t = 0
        rt60_loss_mse_2000_t = 0
        rt60_loss_mse_4000_t = 0


        x_1 = cal_features(data[:, 0, :].to(device=dvs), data[:, 1, :].to(device=dvs))

        mean, variance = model(x_1,epoch)

        # rt60=torch.log(rt60)
        # absorption=torch.log(absorption)


        rt60_loss = 0


        for i in range(4):
            rt60_loss += NLLloss(rt60[:, i+2]/std_rt60[i] , mean[:, i], variance[:,  i])

        # rt60_loss=rt60_loss/6
        # absorp_loss=absorp_loss/6

        # val_loss_t=(rt60_loss+absorp_loss+loss_surface+loss_volume)/4

        val_loss_t = (rt60_loss ) / 4

        val_loss = float(val_loss_t.item()) + val_loss


        #Calculate MSE between actual value and the estimated mean.



        rt60_loss_mse_500_t += pred(mean[:, 0]*std_rt60[0], rt60[:, 2])
        rt60_loss_mse_1000_t += pred(mean[:, 1]*std_rt60[1], rt60[:, 3])
        rt60_loss_mse_2000_t += pred(mean[:, 2]*std_rt60[2], rt60[:, 4])
        rt60_loss_mse_4000_t += pred(mean[:, 3]*std_rt60[3], rt60[:, 5])


        target_ = torch.cat((rt60[:,2:],  rm.reshape(16, 1), vp.reshape(16, 1)), axis=1)  # absorption , rt60 (real annotations)
        output = torch.cat((mean, variance), axis=1) # (Output of the model mean and variance )

        #Concatenate everything to save for analysis purpose
        analysis_data = np.concatenate((output.detach().cpu().clone().numpy().reshape(16, 8),
                              target_.detach().cpu().clone().numpy().reshape(16, 6)), axis=1)

        local_data_sp = np.concatenate((local_data_sp, analysis_data), axis=0)


        rt60_loss_mse_500 = rt60_loss_mse_500 + float(rt60_loss_mse_500_t.item())
        rt60_loss_mse_1000 = rt60_loss_mse_1000 + float(rt60_loss_mse_1000_t.item())
        rt60_loss_mse_2000 = rt60_loss_mse_2000 + float(rt60_loss_mse_2000_t.item())
        rt60_loss_mse_4000 = rt60_loss_mse_4000 + float(rt60_loss_mse_4000_t.item())


        del val_loss_t, rt60_loss_mse_500_t, rt60_loss_mse_1000_t, rt60_loss_mse_2000_t, rt60_loss_mse_4000_t
        # del val_loss_t,volume_loss_mse_t,surface_loss_mse_t

        #Track mse after every 50 batches.

        if batch_idx % 25 == 24:


            rt60_500_acc = (rt60_loss_mse_500 / 25)
            rt60_1000_acc = (rt60_loss_mse_1000 / 25)
            rt60_2000_acc = (rt60_loss_mse_2000 / 25)
            rt60_4000_acc = (rt60_loss_mse_4000 / 25)



            acc_data_ar.append(( rt60_500_acc, rt60_1000_acc,rt60_2000_acc, rt60_4000_acc))

            # acc_data_ar.append((surface_acc,volume_acc))

            rt60_loss_mse_500 = 0
            rt60_loss_mse_1000 = 0
            rt60_loss_mse_2000 = 0
            rt60_loss_mse_4000 = 0


            del  rt60_500_acc, rt60_1000_acc, rt60_2000_acc, rt60_4000_acc
            # del surface_acc,volume_acc

    #Track loss per batch.

    val_data_ar.append((val_loss / batch_idx))

    return val_data_ar, acc_data_ar, local_data_sp
    #print(val_data_ar, acc_data_ar)



#The whole model is divided into different sub-modules of models which are trained at once
#It is done so that we can load any module when ever we require it.

class Model_1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.batch_size = 16


        #Series of seperable convolution layers followed by a layer norm , with increasing dilation and padding.

        #Pipeline 1
        self.conv1d_down_1_depth = nn.Conv1d(385, 385, kernel_size=11, stride=1, groups=385, padding=5)
        self.conv1d_down_1_point = nn.Conv1d(385, 192, kernel_size=1, stride=1, padding=0)
        self.ln_1 = nn.LayerNorm([192])

        self.relu=nn.ReLU()



        #Pipeline 1
        self.conv1d_down_2_depth = nn.Conv1d(192, 192, kernel_size=11, stride=1, groups=192, dilation=2, padding=10)
        self.conv1d_down_2_point = nn.Conv1d(192, 96, kernel_size=1, stride=1)
        self.ln_2 = nn.LayerNorm([96])


        '''
        self.pip_conv_2d = nn.Conv1d(1152, 1152, kernel_size=10, stride=1, groups=1152, dilation=2, padding=0)
        self.pip_conv_2p = nn.Conv1d(1152, 576, kernel_size=1, stride=1, padding=0)
        self.bn_pip_2 = nn.LayerNorm([576, 36])
        '''
        #Pipeline 1
        self.conv1d_down_3_depth = nn.Conv1d(96, 96, kernel_size=11, stride=1, groups=96, dilation=4, padding=20)
        self.conv1d_down_3_point = nn.Conv1d(96, 48, kernel_size=1, stride=1)
        self.ln_3 = nn.LayerNorm([48])

        '''
        self.pip_conv_3d = nn.Conv1d(576, 576, kernel_size=2, stride=1, groups=576, dilation=4, padding=0)
        self.pip_conv_3p = nn.Conv1d(576, 288, kernel_size=1, stride=1, padding=0)
        self.bn_pip_3 = nn.LayerNorm([288, 32])
        '''


        self.drp_1=nn.Dropout(p=0.5)

        self.drp = nn.Dropout(p=0.5)



        #self.fc = nn.Linear(384, 28)




        #self.fc = nn.Linear(384, 28)

    def forward(self, x):
        # x~ ch1, x2~ ch2

        # x~ ch1, x2~ ch2

        x = self.relu(self.conv1d_down_1_depth(x))

        x = self.relu(self.conv1d_down_1_point(x))

        x = self.ln_1(x.reshape(self.batch_size,-1,192))
        x = self.drp_1(x.reshape(self.batch_size,192,-1))



        x = self.relu(self.conv1d_down_2_depth(x))

        x = self.relu(self.conv1d_down_2_point(x))

        x = self.ln_2(x.reshape(self.batch_size,-1,96))

        x = self.drp_1(x.reshape(self.batch_size,96,-1))

        #print(x.shape)
        #x2=self.relu(self.pip_conv_2d(x2))
        #print(x2)
        #x2=self.bn_pip_2(self.relu(self.pip_conv_2p(x2)))

        #x2=self.drp_1(x2)

        #print(x2)

        x = self.relu(self.conv1d_down_3_depth(x))

        x = self.relu(self.conv1d_down_3_point(x))

        x = self.ln_3(x.reshape(self.batch_size,-1,48))
        x= x.reshape(self.batch_size,48,-1)

        #print(x.shape)
        #x2=self.relu(self.pip_conv_3d(x2))
        #print(x2)
        #x2=self.bn_pip_3(self.relu(self.pip_conv_3p(x2)))

        #print(x2)

        #print(x.shape)

        '''
        x = torch.cat((x, x2), axis=1)
        #print(x.shape)
        x= self.avgpool(x)
        #print(x.shape)
        x = self.drp(x)
        x = self.fc(x.reshape(self.batch_size, -1))
        mean,variance = x[:,:14],x[:,14:] #:14,14:
        variance=self.softplus(variance)
        return (mean+10e-7),(variance+10e-7)
        '''

        return x



#The two pipelines described above is followed by a series of linear layers

class Model_3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.drp = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(48, 32)
        self.fc_2 = nn.Linear(32, 16)
        self.fc_3 = nn.Linear(16, 8)
        self.softplus = nn.Softplus()

    def forward(self, x):

        x = self.fc_1(x)

        x = self.drp(x)
        x = self.fc_3(self.fc_2(x))
        #x=self.fc_3(x)


        # First 14 estimated values are mean , the next 14 estimated values are variance.

        mean, variance = x[:, :4], x[:, 4:]

        variance = self.softplus(variance)

        #Varaince should'nt be less than 0 hence we add 10e-7 to it , a very negligible value, because if the varince becames 0 it produces an infinty at NLL Loss and model does no train/learn .

        return mean, (variance + 10e-7)


#Learning of both the models together at once, this class just act as a place-holder.

class Ensemble(torch.nn.Module):
    def __init__(self, model1, model2,bs):
        super().__init__()
        self.batch_size = bs
        self.model_a = model1
        #self.model_b = model2
        self.model_c = model2
        self.avgpool_1 = nn.AvgPool1d(126, stride=1)
        #self.avgpool_2 = nn.AvgPool1d(54, stride=1)


    def forward(self, ch1, ep):
        '''
        if ep > 3:
            print("============================Input Values  Ch2 ==================")
            print("Ch2 input value",ch2)
            print(ch2.shape)
        '''

        x = self.model_a(ch1)

        '''
        if ep > 3:
            print("=================== After Computation (Method Ensemble) ==============")
            print("x1",x)
            print(x.shape)
            print("x2",x2)
            print(x2.shape)
        '''

        print(x.shape)
        x=self.avgpool_1(x)

        #x2=self.avgpool_2(x2)

        #x = torch.cat((x, x2), axis=1)
        #print(x.shape)
        #x = self.avgpool(x)


        x = x.reshape(self.batch_size, -1)

        mean, variance = self.model_c(x)

        return mean, variance


train_data = dl.binuaral_dataset('/home/psrivastava/axis-2/IWAENC/dataset_generation/train_random_arr_2.npy')
val_data = dl.binuaral_dataset('/home/psrivastava/axis-2/IWAENC/dataset_generation/val_random_arr_2.npy')

train_dl = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
val_dl = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=0, drop_last=True)


net_1 = Model_1().to(torch.device("cuda"))
#net_2 = Model_2().to(torch.device("cuda"))
net_3 = Model_3().to(torch.device("cuda"))

net = Ensemble(net_1, net_3,16).to(torch.device("cuda"))

# net=Model()

optimizer = optim.Adam(net.parameters(), lr=0.0001 ,weight_decay=1e-5)
#a1,b1=cal_features(torch.randn(128,1,48000),torch.randn(128,1,48000))
#print(a1.shape)
#net(a1,b1,1)


ar_loss = []
batch_loss_ar = []
total_batch_idx = 0
val_data_ar = []
acc_data_ar = []
save_best_val = 0
adcc = np.zeros((1, 4))
track_var = np.zeros((1, 4))
local_dt_sp = np.zeros((1, 14))

#path="/home/psrivastava/baseline/scripts/pre_processing/results_bn_exp/"
#path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/bn_mlh_k2"
path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/training/D6_0111/mse_training/"


for epoch in tqdm(range(120)):
    ar_loss, batch_loss_ar, adcc, track_var = train(net, train_dl, optimizer, epoch, ar_loss, batch_loss_ar)
    val_data_ar, acc_data_ar, local_dt_sp = val(net, val_dl, optimizer, epoch, val_data_ar, acc_data_ar)

    np.save(path+"mlh_ar_loss.npy", ar_loss)
    np.save(path+"mlh_batch_loss_ar.npy", batch_loss_ar)
    np.save(path+"mlh_val_data_ar.npy", val_data_ar)
    np.save(path+"mlh_acc_data_ar.npy", acc_data_ar)
    np.save(path+"mlh_bnf_track_" + str(epoch) + "_var_.npy", track_var)

    # save best model
    if epoch == 0:
        save_best_val = val_data_ar[-1]

        np.save(path+"mlh_dummy_input_mean_sh.npy", adcc)
    elif save_best_val > val_data_ar[-1]:


        torch.save(
            {'model_dict_1': net_1.state_dict(),  'model_dict_3':net_3.state_dict(),'model_dict_ens': net.state_dict(),
             'optimizer_dic': optimizer.state_dict(), 'epoch': epoch, 'loss': val_data_ar[-1]},
            path+"mlh_tas_save_best_sh.pt")
        save_best_val = val_data_ar[-1]
        np.save(path+"mlh_bnf_mag_96ms_" + str(epoch) + ".npy", local_dt_sp)
