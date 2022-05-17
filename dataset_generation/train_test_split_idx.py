import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split



tmp_=[]
for i in range(40000): #It starts from room 1 and goes until room 19999, so basically train-test split misses two room's : room_0, room_20000
    for j in range(3):
        tmp_.append(("room_"+str(i),j+1))

#Take into account every room starting from room_0 to room_9999 , view points are presented as 0,1,2


print(len(tmp_))

total_samples=40000*3

train,val=(total_samples*90)/100,(total_samples*15)/100#,(total_samples*10)/100


train_ar=tmp_[:int(train)]
print(len(train_ar),train_ar[0],train_ar[-1])

val_ar=tmp_[int(train):int(train)+int(val)]
print(len(val_ar),val_ar[0],val_ar[-1])

'''
test_ar=tmp_[int(train)+int(val):int(train)+int(val)+int(test)]
print(len(test_ar),test_ar[0],test_ar[-1])3
'''
#print(train_ar)
#print(val_ar)
np.save("train_random_ar.npy",train_ar,allow_pickle=True)
np.save("val_random_ar.npy",val_ar,allow_pickle=True)
#np.save("test_random_ar.npy",test_ar,allow_pickle=True)
