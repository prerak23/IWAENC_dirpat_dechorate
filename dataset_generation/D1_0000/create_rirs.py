import numpy as np
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
import h5py
from tqdm import tqdm
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fftfreq,fft
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
    DIRPATRir,
)
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
from acoustics.signal import bandpass
from scipy import stats
import sys







def measure_rt60(h, fs=16000, decay_db=20):
    """
    Analyze the RT60 of an impulse response. Optionaly plots some useful information.

    Parameters
    ----------
    h: array_like
        The impulse response.
    fs: float or int, optional
        The sampling frequency of h (default to 1, i.e., samples).
    decay_db: float or int, optional
        The decay in decibels for which we actually estimate the time. Although
        we would like to estimate the RT60, it might not be practical. Instead,
        we measure the RT20 or RT30 and extrapolate to RT60.
    plot: bool, optional
        If set to ``True``, the power decay and different estimated values will
        be plotted (default False).
    rt60_tgt: float
        This parameter can be used to indicate a target RT60 to which we want
        to compare the estimated value.
    """

    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    # -5 dB headroom
    i_5db = np.min(np.where(-5 - energy_db > 0)[0])
    e_5db = energy_db[i_5db]
    t_5db = i_5db / fs

    # after decay
    i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time

    return est_rt60

parallel_batch_no = int(sys.argv[1])

no_of_rooms=40000
no_of_vps=3
parallel_jobs=100
divided_rooms=[]
divi_arr=[]

for divi in range(no_of_rooms+1):

    divi_arr.append(divi)
    if divi !=0  and divi%parallel_jobs == 0:
        #divi_arr=[j+30000 for j in divi_arr]
        divided_rooms.append(divi_arr[:-1])
        divi_arr=[]
        divi_arr.append(divi)


file_name_rirs="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D1_0000/generated_rirs_D1_0000_rooms_"+str(divided_rooms[parallel_batch_no][0])+"_"+str(divided_rooms[parallel_batch_no][-1])+".hdf5"
file_name_anno_rirs="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D1_0000/D1_0000_anno_rooms_"+str(divided_rooms[parallel_batch_no][0])+"_"+str(divided_rooms[parallel_batch_no][-1])+".hdf5"


Output_file=h5py.File(file_name_rirs,'w')
Output_anno_file=h5py.File(file_name_anno_rirs,'w')

rirs_save=Output_file.create_group('rirs')

rirs_save_anno=Output_anno_file.create_group('rirs_save_anno')


with open('/home/psrivastava/axis-2/IWAENC/dataset_generation/D1_0000/conf_room_setup_D1_0000.yml') as f:
    room_setup = yaml.load(f, Loader=SafeLoader)


with open('/home/psrivastava/axis-2/IWAENC/dataset_generation/D1_0000/conf_receivers_D1_0000.yml') as f_1:
    receiver_rooms = yaml.load(f_1, Loader=SafeLoader)


with open('/home/psrivastava/axis-2/IWAENC/dataset_generation/D1_0000/conf_source_D1_0000.yml') as f_2:
    source_rooms = yaml.load(f_2, Loader=SafeLoader)

with open('/home/psrivastava/axis-2/IWAENC/dataset_generation/D1_0000/conf_noise_source_D1_0000.yml') as f_3:
    fakesource_rooms = yaml.load(f_3, Loader=SafeLoader)

'''
r_d=room_setup["room_6333"]["dimension"]
m_1=receiver_rooms["room_6333"]["mic_pos_1"][0]
m_2=receiver_rooms["room_6333"]["mic_pos_2"][0]
s=source_rooms["room_6333"]["source_pos"][0]
fs=fakesource_rooms["room_6333"]["source_pos"][0]
'''

req_bands=np.array([125 * pow(2,a) for a in range(6)])


for room_no in tqdm(divided_rooms[parallel_batch_no]):

    room_id="room_"+str(room_no)
    r_d=room_setup[room_id]["dimension"]
    diffusion=room_setup[room_id]["surface"]["diffusion"][0]
    surface_area=room_setup[room_id]["surface_area"]
    volume=room_setup[room_id]["volume"]



    all_materials = {
        "east": pra.Material(energy_absorption={'coeffs': room_setup[room_id]["surface"]["absorption"][0],
                                                'center_freqs': [125,250,500,1000,2000,4000]},
                             scattering=diffusion),
        "west": pra.Material(energy_absorption={'coeffs': room_setup[room_id]["surface"]["absorption"][1],
                                                'center_freqs': [125,250,500,1000,2000,4000]},
                             scattering=diffusion),

        "north": pra.Material(energy_absorption={'coeffs': room_setup[room_id]["surface"]["absorption"][2],
                                                 'center_freqs': [125,250,500,1000,2000,4000]},
                              scattering=diffusion),

        "south": pra.Material(energy_absorption={'coeffs': room_setup[room_id]["surface"]["absorption"][3],
                                                 'center_freqs': [125,250,500,1000,2000,4000]},
                              scattering=diffusion),

        "ceiling": pra.Material(energy_absorption={'coeffs': room_setup[room_id]["surface"]["absorption"][4],
                                                   'center_freqs': [125,250,500,1000,2000,4000]},
                                scattering=diffusion),

        "floor": pra.Material(energy_absorption={'coeffs': room_setup[room_id]["surface"]["absorption"][5],
                                                 'center_freqs': [125,250,500,1000,2000,4000]},
                              scattering=diffusion)}

    room=pra.ShoeBox(r_d,fs=16000,max_order=20,materials=all_materials,air_absorption=True,ray_tracing=False,min_phase=False)#,pyroom_IWAENC=True)#,min_phase=False)
    room.add_source(source_rooms[room_id]["source_pos"][0])
    room.add_source(fakesource_rooms[room_id]["source_pos"][0])

    mic_locs = np.c_[receiver_rooms[room_id]["mic_pos_1"][0],receiver_rooms[room_id]["mic_pos_2"][0],
                      receiver_rooms[room_id]["mic_pos_1"][1],receiver_rooms[room_id]["mic_pos_2"][1],
                      receiver_rooms[room_id]["mic_pos_1"][2],receiver_rooms[room_id]["mic_pos_2"][2]]     #, [3.42, 2.48, 0.91],  # mic 1  # mic 2  #[3.47, 2.57, 1.31], [3.42, 2.48, 0.91]]

    room.add_microphone_array(mic_locs)

    room.compute_rir()

    #Calculate RT_60

    rir_mic_arr_1=room.rir[0][0]
    rir_mic_arr_2=room.rir[2][0]
    rir_mic_arr_3=room.rir[4][0]


    len_rirs=np.array([room.rir[c][0].shape[0] for c in range(no_of_vps*2)])
    max_len_rir=np.max(len_rirs)


    len_rirs_noise=np.array([room.rir[c][1].shape[0] for c in range(no_of_vps*2)])
    max_len_rir_noise=np.max(len_rirs_noise)


    packed_rir=np.zeros((6,max_len_rir))
    packed_noise_rir=np.zeros((6,max_len_rir_noise))

    abc=np.empty((3,1))
    abc[0,0]=measure_rt60(rir_mic_arr_1,fs=16000).reshape(1,-1)
    abc[1,0]=measure_rt60(rir_mic_arr_2,fs=16000).reshape(1,-1)
    abc[2,0]=measure_rt60(rir_mic_arr_3,fs=16000).reshape(1,-1)

    rt60_room=np.mean(abc,axis=0)

    rir_id=rirs_save.create_group(room_id)
    rir_id_anno=rirs_save_anno.create_group(room_id)


    for i in range(no_of_vps*2):
        packed_rir[i,:len_rirs[i]]=room.rir[i][0]
        packed_noise_rir[i,:len_rirs_noise[i]]=room.rir[i][1]

    rir_id.create_dataset('rir',(6,max_len_rir), data=packed_rir)
    rir_id.create_dataset('rirs_length',(1,6), data=len_rirs.reshape(1,6))


    rir_id.create_dataset('rir_noise',(6,max_len_rir_noise), data=packed_noise_rir)
    rir_id.create_dataset('rirs_noise_length',(1,6), data=len_rirs_noise.reshape(1,6))


    rir_id_anno.create_dataset("surf_area",1,data=surface_area)
    rir_id_anno.create_dataset("volume",1,data=volume)
    rir_id_anno.create_dataset("rt60_global",1,data=rt60_room)













































###################
# 3-D Plot script #
# Room setting    #
#                 #
###################



'''

def cuboid_data(center, size):



    #suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point

    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return x,y,z




fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(111,projection='3d')


x=r_d[0]
y=r_d[1]
z=r_d[2]

X, Y, Z = cuboid_data([x/2,y/2,z/2], (x, y, z))
ax.plot_surface(np.array(X), np.array(Y), np.array(Z), color='deepskyblue', rstride=1, cstride=1, alpha=0.05,linewidth=1)


#ax.plot3D([0,x,x,0,0,0,x,x,0,0,0,0,x,x,x,x],
#          [0,0,y,y,0,0,0,y,y,y,0,y,y,y,0,0],
#          [0,0,0,0,0,z,z,z,z,0,z,z,0,z,0,z])


#scamap = plt.cm.ScalarMappable(cmap='summer')
#fcolors_ = scamap.to_rgba(np.abs(fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,300]),norm=False)
#fcolors_1 = scamap.to_rgba(np.abs(fft(dir_obj_mic.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,30]),norm=False)

ax.scatter(m_1[0],m_1[1],m_1[2],c="red") #sh_coeffs_expanded_target_grid[:,300],fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)
ax.scatter(m_2[0],m_2[1],m_2[2],c="red")
ax.scatter(s[0],s[1],s[2],c="blue") #sh_coeffs_expanded_target_grid[:,300],fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)
ax.scatter(fs[0],fs[1],fs[2],c="green")



#ax.scatter(src_p_x,src_p_y,src_p_z,c=np.abs(dir_obj_sr.obj_open_sofa_inter.freq_angles_fft[:,30]), cmap='inferno')

#ax.scatter(mic_p_x,mic_p_y,mic_p_z,c=np.abs(dir_obj_mic.obj_open_sofa_inter.freq_angles_fft[:,30]), cmap='inferno')
#ax.scatter(mic_p_x1,mic_p_y1,mic_p_z1,c=np.abs(fft(dir_obj_mic.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,30]), cmap='inferno')



ax.set_xlabel('Length')
ax.set_xlim(0, r_d[0])
ax.set_ylabel('Width')
ax.set_ylim(0, r_d[1])
ax.set_zlabel('Height')
ax.set_zlim(0, r_d[2])
plt.legend()
plt.savefig("3D_setup.jpeg")
'''
