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


dir_obj_mic = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=1,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
dir_obj_mic_1 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=1,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
dir_obj_mic_2 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=1,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
dir_obj_mic_3 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=1,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
dir_obj_mic_4 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=1,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
dir_obj_mic_5 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=1,
    fs=16000,
    #no_points_on_fibo_sphere=0

)


dir_obj_sr_0 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0, degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
    directivity_pattern=0,
    fs=16000,
    #no_points_on_fibo_sphere=0

)

dir_obj_sr_2 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0, degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
    directivity_pattern=4,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
dir_obj_sr_4 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0, degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
    directivity_pattern=5,
    fs=16000,
    #no_points_on_fibo_sphere=0

)



def t60_impulse(raw_signal, fs, bands, rt='t10'):  # pylint: disable=too-many-locals
    """
    Reverberation time from a WAV impulse response.

    :param file_name: name of the WAV file containing the impulse response.
    :param bands: Octave or third bands as NumPy array.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`

    """
    #fs, raw_signal = wavfile.read(file_name)
    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        # Filtering signal
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]

        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    return t60

parallel_batch_no = int(sys.argv[1])

no_of_rooms=40000
no_of_vps=3
parallel_jobs=100
divided_rooms=[]
divi_arr=[]

for divi in range(no_of_rooms+1):

    divi_arr.append(divi)
    if divi !=0  and divi%parallel_jobs == 0:
        divided_rooms.append(divi_arr[:-1])
        divi_arr=[]
        divi_arr.append(divi)

#/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D5_1111/generated_rirs_D5_1111_rooms_
#/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D5_1111/D5_1111_anno_rooms_
file_name_rirs="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D6_0111/generated_rirs_D6_0111_rooms_"+str(divided_rooms[parallel_batch_no][0])+"_"+str(divided_rooms[parallel_batch_no][-1])+".hdf5"
file_name_anno_rirs="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/IWAENC/D6_0111/D6_0111_anno_rooms_"+str(divided_rooms[parallel_batch_no][0])+"_"+str(divided_rooms[parallel_batch_no][-1])+".hdf5"


Output_file=h5py.File(file_name_rirs,'w')
Output_anno_file=h5py.File(file_name_anno_rirs,'w')

rirs_save=Output_file.create_group('rirs')

rirs_save_anno=Output_anno_file.create_group('rirs_save_anno')


with open('/home/psrivastava/axis-2/IWAENC/dataset_generation/D6_0111/conf_room_setup_D6_0111.yml') as f:
    room_setup = yaml.load(f, Loader=SafeLoader)


with open('/home/psrivastava/axis-2/IWAENC/dataset_generation/D6_0111/conf_receivers_D6_0111.yml') as f_1:
    receiver_rooms = yaml.load(f_1, Loader=SafeLoader)


with open('/home/psrivastava/axis-2/IWAENC/dataset_generation/D6_0111/conf_source_D6_0111.yml') as f_2:
    source_rooms = yaml.load(f_2, Loader=SafeLoader)

with open('/home/psrivastava/axis-2/IWAENC/dataset_generation/D6_0111/conf_noise_source_D6_0111.yml') as f_3:
    fakesource_rooms = yaml.load(f_3, Loader=SafeLoader)

'''
r_d=room_setup["room_6333"]["dimension"]
m_1=receiver_rooms["room_6333"]["mic_pos_1"][0]
m_2=receiver_rooms["room_6333"]["mic_pos_2"][0]
s=source_rooms["room_6333"]["source_pos"][0]
fs=fakesource_rooms["room_6333"]["source_pos"][0]
'''

req_bands=np.array([125 * pow(2,a) for a in range(6)])
print(divided_rooms[parallel_batch_no])


for room_no in tqdm(divided_rooms[parallel_batch_no]):

    room_id="room_"+str(room_no)
    r_d=room_setup[room_id]["dimension"]
    diffusion=room_setup[room_id]["surface"]["diffusion"][0]
    surface_area=room_setup[room_id]["surface_area"]
    volume=room_setup[room_id]["volume"]
    src_directivity=source_rooms[room_id]["directivity"]



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

    room=pra.ShoeBox(r_d,fs=16000,max_order=20,materials=all_materials,air_absorption=True,ray_tracing=False,min_phase=True)#,pyroom_IWAENC=True)#,min_phase=False)


    if src_directivity == 0:
        room.add_source(source_rooms[room_id]["source_pos"][0],directivity=dir_obj_sr_0)
        dir_obj_sr_0.change_orientation(np.radians(source_rooms[room_id]["source_ypr"][0][0]),np.radians(source_rooms[room_id]["source_ypr"][0][1]))
    elif src_directivity == 4:
        room.add_source(source_rooms[room_id]["source_pos"][0],directivity=dir_obj_sr_2)
        dir_obj_sr_2.change_orientation(np.radians(source_rooms[room_id]["source_ypr"][0][0]),np.radians(source_rooms[room_id]["source_ypr"][0][1]))
    elif src_directivity == 5:
        room.add_source(source_rooms[room_id]["source_pos"][0],directivity=dir_obj_sr_4)
        dir_obj_sr_4.change_orientation(np.radians(source_rooms[room_id]["source_ypr"][0][0]),np.radians(source_rooms[room_id]["source_ypr"][0][1]))


    room.add_source(fakesource_rooms[room_id]["source_pos"][0])

    mic_locs = np.c_[receiver_rooms[room_id]["mic_pos_1"][0],receiver_rooms[room_id]["mic_pos_2"][0],
                      receiver_rooms[room_id]["mic_pos_1"][1],receiver_rooms[room_id]["mic_pos_2"][1],
                      receiver_rooms[room_id]["mic_pos_1"][2],receiver_rooms[room_id]["mic_pos_2"][2]]     #, [3.42, 2.48, 0.91],  # mic 1  # mic 2  #[3.47, 2.57, 1.31], [3.42, 2.48, 0.91]]

    room.add_microphone_array(mic_locs,directivity=[dir_obj_mic,dir_obj_mic_1,dir_obj_mic_2,dir_obj_mic_3,dir_obj_mic_4,dir_obj_mic_5])

    dir_obj_mic.change_orientation(np.radians(receiver_rooms[room_id]["mic_pos_1_ypr"][0][0]),np.radians(receiver_rooms[room_id]["mic_pos_1_ypr"][0][1]))
    dir_obj_mic_1.change_orientation(np.radians(receiver_rooms[room_id]["mic_pos_2_ypr"][0][0]),np.radians(receiver_rooms[room_id]["mic_pos_2_ypr"][0][1]))
    dir_obj_mic_2.change_orientation(np.radians(receiver_rooms[room_id]["mic_pos_1_ypr"][1][0]),np.radians(receiver_rooms[room_id]["mic_pos_1_ypr"][1][1]))
    dir_obj_mic_3.change_orientation(np.radians(receiver_rooms[room_id]["mic_pos_2_ypr"][1][0]),np.radians(receiver_rooms[room_id]["mic_pos_2_ypr"][1][1]))
    dir_obj_mic_4.change_orientation(np.radians(receiver_rooms[room_id]["mic_pos_1_ypr"][2][0]),np.radians(receiver_rooms[room_id]["mic_pos_1_ypr"][2][1]))
    dir_obj_mic_5.change_orientation(np.radians(receiver_rooms[room_id]["mic_pos_2_ypr"][2][0]),np.radians(receiver_rooms[room_id]["mic_pos_2_ypr"][2][1]))

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

    abc=np.empty((3,6))
    abc[0,:]=t60_impulse(rir_mic_arr_1,16000,req_bands).reshape(1,-1)
    abc[1,:]=t60_impulse(rir_mic_arr_2,16000,req_bands).reshape(1,-1)
    abc[2,:]=t60_impulse(rir_mic_arr_3,16000,req_bands).reshape(1,-1)

    rt60_room=np.median(abc,axis=0)

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
    rir_id_anno.create_dataset("rt60_median",(1,6),data=rt60_room)













































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
