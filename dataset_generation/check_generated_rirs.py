import numpy as np
import h5py
import matplotlib.pyplot as plt

path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/"

start_room=np.random.choice(np.arange(20000,step=100))
end_room=start_room+99


#Open Datasets



D5=h5py.File(path+"D5_1111/"+"generated_rirs_D5_1111_rooms_"+str(start_room)+"_"+str(end_room)+".hdf5")
D4=h5py.File(path+"D4_1100/"+"generated_rirs_D4_1100_rooms_"+str(start_room)+"_"+str(end_room)+".hdf5")
D3=h5py.File(path+"D3_1010/"+"generated_rirs_D3_1010_rooms_"+str(start_room)+"_"+str(end_room)+".hdf5")
D2=h5py.File(path+"D2_1000/"+"generated_rirs_D2_1000_rooms_"+str(start_room)+"_"+str(end_room)+".hdf5")
D1=h5py.File(path+"D1_0000/"+"generated_rirs_D1_0000_rooms_"+str(start_room)+"_"+str(end_room)+".hdf5")
roomsim=h5py.File("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/generated_rirs_lwh_correct.hdf5")


room_no=np.random.randint(100)+start_room
vp=np.random.randint(3)+1

D5_rirlen=D5["rirs"]["room_"+str(room_no)]["rirs_length"][0,:]
D5_rirlen=D5_rirlen[(vp-1)*2]
D5_rir=D5["rirs"]["room_"+str(room_no)]["rir"][vp,:D5_rirlen]

D4_rirlen=D4["rirs"]["room_"+str(room_no)]["rirs_length"][0,:]
D4_rirlen=D4_rirlen[(vp-1)*2]
D4_rir=D4["rirs"]["room_"+str(room_no)]["rir"][vp,:D4_rirlen]

D3_rirlen=D3["rirs"]["room_"+str(room_no)]["rirs_length"][0,:]
D3_rirlen=D3_rirlen[(vp-1)*2]
D3_rir=D3["rirs"]["room_"+str(room_no)]["rir"][vp,:D3_rirlen]

D2_rirlen=D2["rirs"]["room_"+str(room_no)]["rirs_length"][0,:]
D2_rirlen=D2_rirlen[(vp-1)*2]
D2_rir=D2["rirs"]["room_"+str(room_no)]["rir"][vp,:D2_rirlen]

D1_rirlen=D1["rirs"]["room_"+str(room_no)]["rirs_length"][0,:]
D1_rirlen=D1_rirlen[(vp-1)*2]
D1_rir=D1["rirs"]["room_"+str(room_no)]["rir"][vp,:D1_rirlen]

roomsim_rir=roomsim["rirs"]["room_"+str(room_no)]["rir"][vp,:]






fig = plt.figure(figsize=(5,12), constrained_layout=True)

spec = fig.add_gridspec(6,1)


ax0=fig.add_subplot(spec[0, 0])
ax0.plot(np.arange(D5_rirlen//2),10*np.log10(np.abs(np.fft.fft(D5_rir)[:D5_rirlen//2])**2))
ax0.set_xlabel("Frequency Hz")
ax0.set_ylabel("Power dB")

ax0=fig.add_subplot(spec[1, 0])
ax0.plot(np.arange(roomsim_rir.shape[0]//2),10*np.log10(np.abs(np.fft.fft(roomsim_rir)[:roomsim_rir.shape[0]//2])**2))
ax0.set_xlabel("Frequency Hz")
ax0.set_ylabel("Power dB")






ax0=fig.add_subplot(spec[2, 0])
ax0.plot(np.arange(D4_rirlen//2),10*np.log10(np.abs(np.fft.fft(D4_rir)[:D4_rirlen//2])**2))
ax0.set_xlabel("Frequency Hz")
ax0.set_ylabel("Power dB")

ax0=fig.add_subplot(spec[3, 0])
ax0.plot(np.arange(D3_rirlen//2),10*np.log10(np.abs(np.fft.fft(D3_rir)[:D3_rirlen//2])**2))
ax0.set_xlabel("Frequency Hz")
ax0.set_ylabel("Power dB")

ax0=fig.add_subplot(spec[4, 0])
ax0.plot(np.arange(D2_rirlen//2),10*np.log10(np.abs(np.fft.fft(D2_rir)[:D2_rirlen//2])**2))
ax0.set_xlabel("Frequency Hz")
ax0.set_ylabel("Power dB")


ax0=fig.add_subplot(spec[5, 0])
ax0.plot(np.arange(D1_rirlen//2),10*np.log10(np.abs(np.fft.fft(D1_rir)[:D1_rirlen//2])**2))
ax0.set_xlabel("Frequency Hz")
ax0.set_ylabel("Power dB")








plt.savefig("Test_generated_rirs.jpeg")



#Test generated noisy speech

"""
D5=h5py.File(path+"D5_1111/"+"noisy_mixtures/"+"noisy_mixture_rooms_"+str(start_room)+"_"+str(end_room)+".hdf5")
D4=h5py.File(path+"D4_1100/"+"noisy_mixtures/"+"noisy_mixture_rooms_"+str(start_room)+"_"+str(end_room)+".hdf5")
D3=h5py.File(path+"D3_1010/"+"noisy_mixtures/"+"noisy_mixture_rooms_"+str(start_room)+"_"+str(end_room)+".hdf5")
D2=h5py.File(path+"D2_1000/"+"noisy_mixtures/"+"noisy_mixture_rooms_"+str(start_room)+"_"+str(end_room)+".hdf5")
D1=h5py.File(path+"D1_0000/"+"noisy_mixtures/"+"noisy_mixture_rooms_"+str(start_room)+"_"+str(end_room)+".hdf5")

fs=16000

room_no=np.random.randint(100)+start_room
vp=np.random.randint(3)+1


noisy_mix_d5=D5["room_nos"]["room_"+str(room_no)]["nsmix_f"][(vp-1)*2,:]
noisy_mix_d4=D4["room_nos"]["room_"+str(room_no)]["nsmix_f"][(vp-1)*2,:]
noisy_mix_d3=D3["room_nos"]["room_"+str(room_no)]["nsmix_f"][(vp-1)*2,:]
noisy_mix_d2=D2["room_nos"]["room_"+str(room_no)]["nsmix_f"][(vp-1)*2,:]
noisy_mix_d1=D1["room_nos"]["room_"+str(room_no)]["nsmix_f"][(vp-1)*2,:]

sf.write("D5_mix.wav",noisy_mix_d5,fs)
sf.write("D4_mix.wav",noisy_mix_d4,fs)
sf.write("D3_mix.wav",noisy_mix_d3,fs)
sf.write("D2_mix.wav",noisy_mix_d2,fs)
sf.write("D1_mix.wav",noisy_mix_d1,fs)

"""
