import matplotlib.pyplot as plt
import numpy as np
import os

path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/training/"

fig = plt.figure(figsize=(5,12), constrained_layout=True)

spec = fig.add_gridspec(5,1)
datasets_in_training=["D1_0000","D2_1000","D3_1010","D4_1100","D5_1111"]
i=0
for data_sets in os.listdir(path):
    print(data_sets)
    if data_sets in datasets_in_training:
        train_loss=np.load(path+data_sets+"/"+"mlh_ar_loss.npy")
        val_loss=np.load(path+data_sets+"/"+"mlh_val_data_ar.npy")
        ax0=fig.add_subplot(spec[i, 0])
        ax0.plot(np.arange(train_loss.shape[0]),train_loss,label="Train loss")
        ax0.plot(np.arange(val_loss.shape[0]),val_loss,label="Val loss")
        ax0.set_xlabel("Epochs")
        ax0.set_ylabel("Loss")
        ax0.legend()
        ax0.set_title("Dataset "+data_sets)
        i+=1


plt.legend()
plt.savefig("All_losses.jpeg")
