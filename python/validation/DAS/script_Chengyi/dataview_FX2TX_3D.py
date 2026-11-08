#!/usr/bin/env python
# coding: utf-8

"""
-----------------  Author C.Shen
-----------------  Makutu Inria & Totalenergies, Pau

- Read a one-gauge DAS dataset from Haven Modeling
- Convert one-gauge data to one-rcv data (one gauge is discretized to one rcv series)
- Convert one-rcv data in frequency to one-rcv data in time (sismo)
- Project sismo toward the gauge direction

"""
import numpy as np
import struct
import class_FX as cFXPixil
from acqui_DAS import get_acquiDAS, projee_DAS, projeu_DAS
from phy_SOL import C_EHI
from matplotlib import pyplot as plt
from math import pi, sqrt, sin, cos
from sys import argv


def read_sismos_ascii(path, col, dt=None, d3to2=False, coeff=1):
    """
    Read the seismos in ascii format

    Parameters
    -----------
        path : string
            Path of the ascii data
        col : int
            Column to read
        dt : float
            Time interval
        d3to2 : bool
            Convert 3D data to 2D data
        coeff : float
            Scaling coefficient

    Returns
    --------
        signal : list
    """
    signal = []

    with open(path, 'r') as afile:
        t = 0
        if d3to2:
            for line in afile.readlines():
                signal.append(float(line.split()[col])*sqrt(t)*coeff)
                t += dt
        else:
            for line in afile.readlines():
                signal.append(float(line.split()[col])*coeff)
    
    return signal


Nf = 6135 # 6400 # default number of frequencies from Haven data
if len(argv) < 2:
    print('Using default source ID string : srcsxxsyyszz')
    print('This is an explosive source applied on constraints')
    src = 'srcsxxsyyszz'
    print(f'Using default number of frequencies from Haven data: {Nf}')

elif len(argv) < 3:
    src = argv[1]
    print(f'Using user defined source ID string: {src}')
    print(f'Using default number of frequencies from Haven data: {Nf}')

else:
    src = argv[1]
    print(f'Using user-defined source ID string: {src}'.format(src))
    Nf = int(argv[2])
    print(f'Using user-defined number of frequencies from Haven data: {Nf}')

# medium
dim, vp, vs, rho = 3, 4000, 2000, 2000
C_stiffness = C_EHI(vp=vp, vs=vs, rho=rho, dim=dim)
C_stiffness_inv = np.linalg.inv(C_stiffness)

# analytical DAS
gl = 8.                 # gauge length
Nquad = 7               # number of quadrature points to estimate the integral
dip = pi / 6              # DAS channel dip with respect to the horizontal
az = pi / 4               # DAS channel azimuth with respect to the x-axis
                      # counter-clockwise when looking down
gc = [505., 505., 505.] # gauge center

# gauge direction, x,y,z coords of the discretized gauge
gd_haven, gx_haven, gy_haven, gz_haven =\
    get_acquiDAS(*gc, dip=dip, azimuth=az, gauge_length=gl, quadrature_points=Nquad)
Nrcv = len(gx_haven)    # gauge discretized to Nrcv points

# data path and components to read, use and write
bdata_path = '../hawen/' # Haven binary data path
# binary file name prefix and suffix
prefix, suffix = 'miladhmg.' + src + f'_L{Nf}_', '_subsrc.01.bin'
# component identifiers in the binary file name
compnames = ('VXsuni', 'VYsuni', 'VZsuni', 'SXXuni', 'SYYuni', 'SZZuni', 'SXYuni', 'SXZuni', 'SYZuni')
# corresponding components, raw DAS data with discretized rcv points
comps = ('VX', 'VY', 'VZ', 'SXX', 'SYY', 'SZZ', 'SXY', 'SXZ', 'SYZ')
# strain names
epsilons=('EXX', 'EYY', 'EZZ', 'EXY', 'EXZ', 'EYZ')
# new output component identifiers of DAS data (discretized gauge data to point data)
outputs=('VXduni', 'VYduni', 'VZduni',\
         'EXXuni', 'EYYuni', 'EZZuni', 'EXYuni', 'EXZuni', 'EYZuni',\
         'VXsuni', 'VYsuni', 'VZsuni',\
         'VXruni', 'VYruni', 'VZruni',\
         'VXtuni', 'VYtuni', 'VZtuni',) # s: mid rcv ; r: start rcv ; t: end rcv
# keys of different dictionaries data_dictR_DASdiffv,
#                                data_dictR_DASinteg,
#                                data_dictR - mid, - start, - end
output_keys=('VX', 'VY', 'VZ',\
             'EXX', 'EYY', 'EZZ', 'EXY', 'EXZ', 'EYZ',\
             'VX', 'VY', 'VZ',\
             'VX', 'VY', 'VZ',\
             'VX', 'VY', 'VZ',)
# headers for the new binary outputs (discretized gauge data to point data)
output_keys_hdr=(b'VXd', b'VYd', b'VZd',\
                 b'EXX', b'EYY', b'EZZ', b'EXY', b'EXZ', b'EYZ',\
                 b'VXs', b'VYs', b'VZs',\
                 b'VXr', b'VYr', b'VZr',\
                 b'VXt', b'VYt', b'VZt',)

bfiles = []                   # new binary files after one-discretized-gauge to one-point
data_dictR = {}               # DAS data on key rcv points (mid, start, end)
data_dictR_DASdiffv = {}      # DAS data from diff(u)
data_dictR_DASinteg = {}      # DAS data from integarl(sigma)

# Read the raw binary data, haven simulation, frequency data
for i, comp in enumerate(compnames):
    headr, data_dictR[comps[i]] = cFXPixil.readRINF(bdata_path,\
                                                 f'{prefix}rcvDAS.{comp}{suffix}',\
                                                 kwrdIndex=(0,2),)
    f0, df = headr[2], headr[3]
    Nrcv, Nf = int(headr[4]), int(headr[5])
    print(f'Number of receivers in {prefix}rcvDAS.{comp}{suffix} is {Nrcv}')
    Nc = int(headr[6])

# DAS from diff(u)
data_dictR_DASdiffv['VX'] = (data_dictR['VX'][:,-1,:] - data_dictR['VX'][:,0,:]) / gl
data_dictR_DASdiffv['VY'] = (data_dictR['VY'][:,-1,:] - data_dictR['VY'][:,0,:]) / gl
data_dictR_DASdiffv['VZ'] = (data_dictR['VZ'][:,-1,:] - data_dictR['VZ'][:,0,:]) / gl

# DAS from integral(sigma)
data_dictR_DASinteg['EXX'] = np.zeros((Nf, Nc))
data_dictR_DASinteg['EYY'] = np.zeros((Nf, Nc))
data_dictR_DASinteg['EZZ'] = np.zeros((Nf, Nc))
data_dictR_DASinteg['EXY'] = np.zeros((Nf, Nc))
data_dictR_DASinteg['EXZ'] = np.zeros((Nf, Nc))
data_dictR_DASinteg['EYZ'] = np.zeros((Nf, Nc))

for j in range(Nf):
    for k in range(Nc):
        for i, epsilon in enumerate(epsilons):
            data_dictR_DASinteg[epsilon][j,k] +=\
                (sum(C_stiffness_inv[i,0] * data_dictR['SXX'][j,1:-1,k])\
                 + sum(C_stiffness_inv[i,1] * data_dictR['SYY'][j,1:-1,k])\
                 + sum(C_stiffness_inv[i,2] * data_dictR['SZZ'][j,1:-1,k])\
                 + sum(C_stiffness_inv[i,3] * data_dictR['SXY'][j,1:-1,k])\
                 + sum(C_stiffness_inv[i,4] * data_dictR['SXZ'][j,1:-1,k])\
                 + sum(C_stiffness_inv[i,5] * data_dictR['SYZ'][j,1:-1,k]))
            data_dictR_DASinteg[epsilon][j,k] /= Nquad

data_dictR_DASinteg['EXY'] /= 2
data_dictR_DASinteg['EXZ'] /= 2
data_dictR_DASinteg['EYZ'] /= 2

# Write new DAS data to customized twenty-or-so-byte-header data
nrcv = 1 # new rcv number (one gauge, one receiver)
for i, comp in enumerate(outputs):
    with open(f'{bdata_path}{prefix}rcv001.{comp}{suffix}', 'wb') as bwfile:
        bwfile.write(struct.pack('1s', b'N'))                # identifier : native
        bwfile.write(struct.pack('i', 3))                    # length of dataname
        bwfile.write(struct.pack('3s', output_keys_hdr[i]))  # dataname for dict key
        bwfile.write(struct.pack('f', f0))                   # first freq
        bwfile.write(struct.pack('f', df))                   # freq interval
        bwfile.write(struct.pack('f', nrcv))                 # number of rcvs
        bwfile.write(struct.pack('f', Nf))                   # number of freqs
        bwfile.write(struct.pack('f', Nc))                   # 2 numbers for complex

        if i < 3: # diff(u)
            for j in range(Nf):
                bwfile.write\
                    (struct.pack('ff', *data_dictR_DASdiffv[output_keys[i]][j,:]))
        elif i < 9: # integral(sigma)
            for j in range(Nf):
                bwfile.write\
                    (struct.pack('ff', *data_dictR_DASinteg[output_keys[i]][j,:]))
        elif i < 12: # mid point rcv
            for j in range(Nf):
                bwfile.write\
                    (struct.pack('ff', *data_dictR[output_keys[i]][j,int(Nrcv/2),:]))
        elif i < 15: # start point rcv
            for j in range(Nf):
                bwfile.write\
                    (struct.pack('ff', *data_dictR[output_keys[i]][j,0,:]))
        else: # end point rcv
            for j in range(Nf):
                bwfile.write\
                    (struct.pack('ff', *data_dictR[output_keys[i]][j,-1,:]))

# IFFT on the new binary data (singal receiver)
components_in = ('EZZ', 'EYY', 'EXX', 'EXY', 'EXZ', 'EYZ', 'VXd', 'VYd', 'VZd',\
               'VXs', 'VYs', 'VZs', 'VXr', 'VYr', 'VZr', 'VXt', 'VYt', 'VZt',)
components_use = ('dEZZ', 'dEYY', 'dEXX', 'dEXY', 'dEXZ', 'dEYZ', 'VXd', 'VYd', 'VZd',\
                'VXs', 'VYs', 'VZs', 'VXr', 'VYr', 'VZr', 'VXt', 'VYt', 'VZt',)
components_label = ('dEZZ/dt', 'dEYY/dt', 'dEXX/dt', 'dEXY/dt', 'dEXZ/dt', 'dEYZ/dt',\
                  'dYXd/dt', 'dUYd/dt', 'dUZd/dt', 'dUX/dt', 'dUY/dt', 'dUZ/dt',\
                  'dUXr/dt', 'dUYr/dt', 'dUZr/dt', 'dUXt/dt', 'dUYt/dt', 'dUZt/dt',)
components_out = ('Ezz', 'Eyy', 'Exx', 'Exy', 'Exz', 'Eyz', 'Uxd', 'Uyd', 'Uzd',\
                'Ux', 'Uy', 'Uz', 'Uxstart', 'Uystart', 'Uzstart', 'Uxend', 'Uyend', 'Uzend',)

fc = 10             # Ricker central frequency
t_listen = 1        # time signal duration retained after ifft
das_sismos = {}     # das sismos (time signal) before projection toward the gauge direction
nt_sismos = {}      # time sample numbers

for component_in, component_use, component_label, component_out in\
    zip(components_in, components_use, components_label, components_out):
    if component_in.startswith('E'):
        # in case of strain, take time derive (necessary for quantitative comparison)
        FXDatPixil_srcss = cFXPixil.FX_data(bdata_path,\
                                          f'{prefix}rcv001.{component_in}uni{suffix}',\
                                          dataderive=True)
    else:
        # in case of velocity, take original (it is the time derive of displacement)
        FXDatPixil_srcss = cFXPixil.FX_data(bdata_path,\
                                          f'{prefix}rcv001.{component_in}uni{suffix}')

    FXDatPixil_srcss(0)
    FXDatPixil_srcss.readBinAvch20(kwrdIndex=3, FGreen=True, srcType=0,\
                                   smoothrate=5, verbose=False, fpeak=fc, tpeak=1/fc)
    FXDatPixil_srcss.FX2TX()
    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=True, figsize=(12,7))
    # for i in range(1):
    #     ax[0].plot(FXDatPixil_srcss.tlist[0],\
    #                FXDatPixil_srcss.data_FX2TX["{}".format(component_use)][:,i],\
    #                label=component_label)
    #     ax[1].plot(FXDatPixil_srcss.tlist[0],\
    #                FXDatPixil_srcss.data_FX2TX["{}".format(component_use)][:,i],\
    #                label=component_label)
    # ax[0].legend()
    # ax[1].legend()
    # ax[0].set_xlabel('Time (s)')
    # ax[0].set_ylabel('Amplitude')
    # ax[1].set_ylabel('Amplitude')
    # plt.show()

    nt = int(1//FXDatPixil_srcss.dt_fourier[0])
    nt_sismos[f"{component_use}"] = nt
    das_sismos[f"{component_use}"] = np.zeros((nt, nrcv))
    das_sismos[f"{component_use}"][:,:] =\
        FXDatPixil_srcss.data_FX2TX[f"{component_use}"][:nt, :]
    # with open("{}_{}.bin".format(component_out,src), "wb") as bfile:
    #     for i in range(nrcv):
    #         bfile.write(struct.pack("f"*nt,*FXDatPixil_srcss.data_FX2TX\
    #                                 ["{}".format(component_use)][:nt,i]))

# DAS data projection
das_sismo_dir_ee = {}
das_sismo_dir_eu = {}
das_sismo_dir_ee[f'{src}'] = np.zeros((nt, nrcv))
das_sismo_dir_eu[f'{src}'] = np.zeros((nt, nrcv))

for i in range(nt):
    for j in range(nrcv):
        das_sismo_dir_ee[f'{src}'][i,j] =\
            projee_DAS(*gd_haven, exx=das_sismos['dEXX'][i,j], exy=das_sismos['dEXY'][i,j],\
                       exz=das_sismos['dEXZ'][i,j], eyy=das_sismos['dEYY'][i,j],\
                       eyz=das_sismos['dEYZ'][i,j], ezz=das_sismos['dEZZ'][i,j])
        das_sismo_dir_eu[f'{src}'][i,j]=\
            projeu_DAS(*gd_haven, uxd=das_sismos['VXd'][i,j], uyd=das_sismos['VYd'][i,j],\
                       uzd=das_sismos['VZd'][i,j])

# time info of haven data
dt_haven3D = FXDatPixil_srcss.dt_fourier[0]
nt_haven3D = nt
nrcv_haven3D = nrcv
tlist_haven3D = np.arange(0, dt_haven3D*nt_haven3D, dt_haven3D)

# ascii data path, column to read and coeff on amplitude
adata_prop = {}  # key    (     path,  column to read, coeff)
adata_prop['das_milad'] = ('../script_Milad/dassyn', 0, 1)
adata_prop['ux_milad'] = ('../script_Milad/ux.txt', 0, 1)
adata_prop['uy_milad'] = ('../script_Milad/uy.txt', 0, 1)
adata_prop['uz_milad'] = ('../script_Milad/uz.txt', 0, 1)
adata_prop['ux_gar6more'] = ('../gar6more/Ux.dat', 1, -1)
adata_prop['uy_gar6more'] = ('../gar6more/Uy.dat', 1, -1)
adata_prop['uz_gar6more'] = ('../gar6more/Uz.dat', 1, -1)

# time info of ascii data (Milad, Gar6more)
dt = 1e-3
t_show = 0.25                        # length of time to show
Nt_show = int(t_show/dt)             # number of time samples to show
tsamples = np.arange(0, Nt_show) * dt
data_ascii = {}

# read ascii data into dict
for key in adata_prop.keys():
    data_ascii[key] = read_sismos_ascii(adata_prop[key][0],\
                                      adata_prop[key][1],\
                                      coeff=adata_prop[key][2])
amplifi = 2.5                             # an amplifier on Haven numerical data
ux_haven3D =- amplifi * das_sismos['VXs']
uy_haven3D =- amplifi * das_sismos['VYs']
uz_haven3D =- amplifi * das_sismos['VZs']
ee_haven3D =- amplifi * das_sismo_dir_ee[f'{src}']
eu_haven3D =- amplifi * das_sismo_dir_eu[f'{src}']

ftsize, lbsize, lgsize = 16, 15, 12
figname = 'Figure_DAS_verif3D.png'
fig, ax = plt.subplots(nrows=2, ncols=1,\
                       sharex=True, sharey=False,\
                       figsize=(17,8))
keys,labels=['ux_milad', 'uy_milad', 'uz_milad',\
             'ux_gar6more', 'uy_gar6more', 'uz_gar6more'],\
             ['$U_x$ analytic Milad', '$U_y$ analytic Milad', '$U_z$ analytic Milad',\
              '$U_x$ gar6more', '$U_y$ gar6more', '$U_z$ gar6more']

for i in range(len(keys)):
    ax[0].plot(tsamples, data_ascii[keys[i]][:Nt_show], label=labels[i])
ax[0].plot(tlist_haven3D, ux_haven3D, label='$\dot{U_x}$ numerical Haven')
ax[0].plot(tlist_haven3D, uy_haven3D, label='$\dot{U_y}$ numerical Haven')
ax[0].plot(tlist_haven3D, uz_haven3D, label='$\dot{U_z}$ numerical Haven')
ax[1].plot(tsamples, data_ascii['das_milad'][:Nt_show], label='DAS synthetic Milad')
ax[1].plot(tlist_haven3D, ee_haven3D, label='DAS numerical Haven (Integral on $\dot\epsilon$)')
ax[1].plot(tlist_haven3D, eu_haven3D, label='DAS numerical Haven (Diff on $\dot{U}$)')
ax[0].legend(loc=1, fontsize=lgsize, ncol=4)
ax[1].legend(loc=1, fontsize=lgsize)
ax[0].tick_params(labelsize=lbsize)
ax[1].tick_params(labelsize=lbsize)
ax[1].set_xlabel('Time (s)', fontsize=ftsize)
ax[1].set_ylabel('Amplitude', fontsize=ftsize)
ax[0].set_ylabel('Amplitude', fontsize=ftsize)
plt.savefig(figname)
print(f'Figure saved as {figname}')
# plt.show()
