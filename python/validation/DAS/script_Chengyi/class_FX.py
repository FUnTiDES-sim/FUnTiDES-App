#!/usr/bin/env python
# coding: utf-8

"""
-----------------  Author C.Shen
-----------------  Makutu Inria & Totalenergies, Pau
Version dans ~/Bin
A class of FX data (simulated by Hawen, F.Faucher):
1) Two local functions Normalize_TBT (trace-by-trace normalization) and
   Normalize_SBS (slice-by-slice normalizarion)
2) A FX_data class including:
   +) __init__ function initializing paths and filenames;
   +) __call__ fuctiion initializing t0 and fixed-number of frequencies;
   +) readBinAvch20 (read Binary with a twenty-or-so-byte header) function.
      See "Pixil_txt2bin.sh" for more details;
   +) FX2TX (convert FX data to TX data via ifft) function;
   +) TX_normalize (normalize the TX data amplitudes) fucntion;
   +) DictWrite2Bin (Write data in dictionary to pure Binary) function
"""
import os
import numpy as np
from math import pi, sqrt
from scipy.signal import butter, filtfilt, sosfilt
import struct


# Local functions
def Normalize_TBT(dat_vector, L=None):
    """
    Normalization of one trace:
    with respect to the std value of the vector
    If L is given, only the first L samples will be taken to calculate the std

    Parameters
    -----------
        dat_vector : a list or a 1D-numpy-array
            Vector to normalize
        L          : int
            Length on which the std is evaluated
    
    Returns
    --------
        array-like
            normalized vector
    """
    if isinstance(dat_vector, list):
        # if list, convert to np.ndarray
        dat_vector = np.array(dat_vector)

    elif isinstance(dat_vector, np.ndarray):
        # if np.ndarray
        if len(dat_vector.shape) > 1:
            # if not vector
            print('warning: the input data is not a vector but a matrix')
            print('the results might be incorrect if you continue')

    else:
        raise Exception("Error of input data type:\nIt should be either an 1D list or an 1D ndarray.")

    if L == None:
        return dat_vector / np.std(dat_vector)
    else:
        return dat_vector / np.std(dat_vector[:L])


def Normalize_SBS(dat_array2D, axis=None, L=None):
    """
    Normalization by slice
    with respect to the std value of the 2D slice
    If L is given, only the first L samples will be taken to calculate the std
    and L will be applied along axis given

    Parameters
    -----------
        dat_array2D : a nesteled list (2D) or a 2D-numpy-array
            Data
        L           : int
            Length on which the std is evaluated
        axis        : int
            axis along which L is considered

    Returns
    -------
        2D np.ndarray
            Normalized 2D numpy-array
    """
    if isinstance(dat_array2D, list):
        # if list, convert to np.ndarray
        dat_array2D = np.array(dat_array2D)

    elif isinstance(dat_array2D, np.ndarray):
        # if np.ndarray
        if len(dat_array2D.shape) != 2:
            # if not dim2
            print("Warning: the input data is not a 2D matrix,\n The results might be incorrect if you continue")

    else:
        raise Exception("Error of input data type:\n It should be either a 2D list or a 2D ndarray.")

    if axis == None:
        # Signal in the frequency domain divided by the standard deviation:
        #                                               np.std(dat_array2D)
        return dat_array2D / np.std(dat_array2D)

    else:
        if L == None:
            print('warning: axis is given but not length to keep')
            print('warning: no truncation will be applied')
            return dat_array2D / np.std(dat_array2D)

        else:
            # axis can only be 0, -2, 1, or -1 for a 2D array
            if axis == 0 or axis == -2:
                # Signal in the frequency domain divided by the standard deviation:
                # calculated with truncation on the fisrt axis
                return dat_array2D / np.std(dat_array2D[:L,:])

            elif axis == 1 or axis == -1:
                # Signal in the frequency domain divided by the standard deviation:
                # calculated with truncation on the second axis
                return dat_array2D / np.std(dat_array2D[:,:L])
            
            else:
                print('warning: more than 2 axis ?')
                print('warning: the given length is ignored')
                print('warning: results might be incorrect')
                return dat_array2D / np.std(dat_array2D)

def nextPow2(x, inc_times=1):
    """
    Return the next power of 2 of an integer
    if x is not an integer it will be converted with int(x)
        
    Parameters
    -----------
        x         : int
        inc_times : int or float
            An increase rate to get a bigger nextpow2

    Returns
    --------
         The next power of 2 of x*inc_times
    """
    return 1 if int(x*inc_times) == 0 else 2**(int(x*inc_times)-1).bit_length()


def checkButter(**filt):
    """
    Check butter filter parameters
    report errors if any
    This is only a check of the user-chosen filter

    Parameters
    -----------
        filt : dict
            Dictionary with filter params as keys and filter param values

    Returns
    --------
        int
            0 if error
            1 if lowpass or highpass
            2 if bandpass or bandstop
    """
    kwords_all = ('filterName', 'N', 'Wn', 'btype', 'analog', 'output', 'fs')

    if len(filt) < 3:
        # not enough paramerters
        error_message="Not enough parameters. At least filterName, N and Wn are required"
        return 0, error_message

    else:
        if 'btype' in filt.keys():
            if filt['btype'] in ('lowpass', 'highpass', 'lp', 'hp', 'low', 'high'):
                normal_message = f"using {filt['btype']}"
                return 1, normal_message

            elif filt['btype'] in ('bandpass', 'bandstop', 'bp', 'bs'):
                if type(filt['Wn']) is float or type(filt['Wn']) is int:
                    error_message = f"btype is {filt['btype']}, Wn need at least 2 elements"
                    # in case of bandpass or bandstop, Wn should be a
                    # 2-element list and not float nor int
                    return 0, error_message

                elif len(filt['Wn']) >= 2:
                    normal_message = f"using {filt['btype']}"
                    return 2, normal_message

                else:
                    error_message = "type error in Wn"
                    return 0, error_message

            else: # btype is unknown
                error_message = "btype is unknown"
                return 0, error_message

        else: # btype not in filt params, assume highpass or lowpass
            normal_message = "btype is not in filt params, assume highpass or lowpass"
            return 1, normal_message


def Ricker(tlist, **RickerParameters):
    """
    Calculate fft of a Ricker source

    Parameters
    -----------
        tlist            : 1D array-like
            Time samples
        RickerParameters : dict
            Dictionary with Ricker params as keys and Ricker param values

    Returns
    -------
        1D numpy-array
            Fft values of a time Ricker seismic source
    """
    if isinstance(tlist, list):
        # if list, convert to np.ndarray
        tlist = np.array(tlist)

    ricker_inTime = (1 - 2*(pi*RickerParameters['fpeak']*\
                        (tlist-RickerParameters['tpeak']))**2)*\
                        np.exp(-(pi*RickerParameters['fpeak']*\
                                 (tlist-RickerParameters['tpeak']))**2)
    ricker_inFreq = np.fft.fft(ricker_inTime)

    return ricker_inFreq


def Ricker_FF(freq, dt=1e-4, tstart=-5, tend=5, **RickerParameters):
    """
    Calculate fft for one frequency using discretized integral :
        R(omega)=sum(r(t_n)*exp(-1i*omega*t_n)*delta_t)
    The accuracy of this method depends on the choice on dt, and time range
    and they are adapted to the frequency
    at least 100 points per 2*mainperiod, at most 1000 points per 2*mainperiod
    the time range is centered on t=0 and the duration is at least 4*mainperiod

    Parameters
    -----------
        freq : float
            Frequency on which fft is calculated
        dt : float
            Time interval
        tstart : float
            Time samples start at
        tend : float 
            Time samples end at
        RickerParameters : dict
            Dictionary with Ricker params as keys and Ricker param values

    Returns
    -------
        complex:
            Complex fft value for freq
    """
    period = 1 / freq
    signal_duration = 2 * period
    nbp_signal = signal_duration / dt

    if nbp_signal < 100:
        nbp_signal = 100
        dt = signal_duration / nbp_signal

    elif nbp_signal > 1000:
        nbp_signal = 1000
        dt = signal_duration / nbp_signal

    if (tstart + signal_duration) < 0:
        tstart = -1 * signal_duration

    if (tend - signal_duration) < 0:
        tend = signal_duration

    omega = 2 * pi * freq
    tlist = np.arange(tstart, tend, dt)
    ricker_inTime = (1 - 2*(pi*RickerParameters['fpeak']*\
                        (tlist-RickerParameters['tpeak']))**2)*\
                        np.exp(-(pi*RickerParameters['fpeak']*\
                                 (tlist-RickerParameters['tpeak']))**2)
    iRicker = sum(ricker_inTime*np.exp(-1j*omega*tlist))*dt

    return iRicker


def Ricker_FF_custom(freq, tlist, amplist, verbose=False):
    """
    Calculate fft at one frequency for an external source using discretized integral :
    R_external(omega)=sum(r_external(t_n)*exp(-1i*omega*t_n)*delta_t)
    The accuracy of this method depends on the sampling of the external source
    at least 100 points per 2*mainperiod, at most 1000 points per 2*mainperiod
    the time range is centered on t=0 and the duration is at least 4*mainperiod

    Parameters
    ----------
        freq : float
            Frequency on which fft is calculated
        dt : float
            Time interval
        tstart : float
            Time samples start at
        tend : float
            Time samples end at
        RickerParameters : dict
            Dictionary with Ricker params as keys and Ricker param values

    Returns
    -------
        complex
            Complex fft value for freq
    """
    if verbose:
        print("Using formula of ifft from Florian F. on an external source")

    omega = 2 * pi * freq
    dt = tlist[1] - tlist[0]
    iRicker = sum(amplist * np.exp( -1j * omega * tlist)) * dt

    return iRicker


def Ricker_SF(freq, **RickerParameters):
    """
    calculate fft at one frequency for a Ricker source using an analytical solution
    REF:
    This method is recommended in case of Ricker source.
    Parameters
    -----------
        freq : float
            Frequency on which fft is calculated
        RickerParameters : dict
            Fictionary with Ricker params as keys and Ricker param values
    
    Returns
    --------
        complex
            Complex fft value for freq
    """
    if 'verbose' in RickerParameters.keys():
        if RickerParameters['verbose']:
            print("Using analytical formula of ifft from Stefano F")

    omega = 2 * pi * freq
    iRicker = -omega**2 * np.exp(-omega**2/4/pi**2/RickerParameters['fpeak']**2\
                             +1j*RickerParameters['tpeak']*omega)/\
                             (sqrt(2)*pi*RickerParameters['fpeak'])

    # attention : reverse time use conjugate !
    return np.conjugate(iRicker)


def readRINF_header(filepath, filename, kwrdIndex=(0,2)):
    """
    Read header of a customized binary file containing frequency data
    This will decide on a key for the "data name", e.g. Ux, VZ, EXY, SYY, P
    The header size varies :
    1) header starts with 'R', with a total size of 21 bytes : Raw data,
       the data name is contained in the file name "filename".
       kwrdIndex should be a 2-element tuple. The data name is taken as
       filename[kwrdIndex[0]:kwrdIndex[1]]
    2) header starts with 'I', with a total size of 21 bytes : Input-string data,
       the data name is an input string, namely kwrdIndex (not a tuple anymore).
    3) header starts with 'N', with a total size of 25+X bytes : Native data,
       the data name starts from the 6th byte. The 1st byte is 'N', followed by
       a 4-byte integer indicating the data name of X byte. The data name is thus
       read directly from the header.
    4) header starts with 'F', with a total size of 21+X bytes : Foreign data,
       the data name follows the 1st byte ('F'), and its length in byte is given
       by kwrdIndex (an integer for this case). One has to know the length of
       the data name in the header beforehand.

    Parameters
    -----------
        filepath  : string
            Path to the customized binary file
        filename  : string
            Name of the customized binary file
        kwrdIndex : tuple, list, int or string
            Used to determine the keywrd

    Returns
    --------
        headerbytes
        keywrd
        f0
        df
        Nrcv
        Nf
        Nc
    """
    filedatbin = open('{}/{}'.format(filepath, filename), 'rb')
    headr = struct.unpack('1s', filedatbin.read(1))
    headerbytes = 21
    dataformat = headr[0].decode()
    
    if dataformat == 'R': # raw 20-byte header data
        # kwrdIndex should be a 2-element tuple or list
        # fix dictionnary keywords from filenames
        keywrd = filename[kwrdIndex[0]:\
                        kwrdIndex[1]]
    
    if dataformat == 'I': # a 21-byte header data
        # kwrdIndex should be a string
        keywrd = kwrdIndex
    
    if dataformat == 'N': # a 25+string-byte header data
        # kwrdIndex is read from header
        headr = struct.unpack('i', filedatbin.read(4))
        kwrdIndex = headr[0]
        headerbytes += 4
        headr = struct.unpack(f'{kwrdIndex}s',\
                            filedatbin.read(kwrdIndex))
        headerbytes += kwrdIndex
        keywrd = headr[0].decode()

    if dataformat == 'F': # a 21+string-byte header data
        # kwrdIndex should be an integer
        headr = struct.unpack(f'{kwrdIndex}s',\
                            filedatbin.read(kwrdIndex))
        headerbytes += kwrdIndex
        keywrd = headr[0].decode()
 
    headr = struct.unpack('f'*5, filedatbin.read(20))
    
    # freq start and freq increment
    f0, df = headr[0], headr[1]
    
    # number of receivers, number of freqs
    Nrcv, Nf = int(headr[2]), int(headr[3])
    
    # 2 numbers, real part and imaginary part
    Nc = int(headr[4])
    
    filedatbin.close()

    return headerbytes, keywrd, f0, df, Nrcv, Nf, Nc


def readRINF(filepath, filename, kwrdIndex=(0,2)):
    """
    Read header and data from a customized binary file containing frequency data
    The header will be read with readRINF_header()
    Each dataset will be read in a dictionary data_dictR
    The key will be the "data name", e.g. Ux, VZ, EXY, SYY, P. It is returned by
    readRINF_header(): keywrd

    Parameters
    ------------
        filepath : string 
            Path to the customized binary file
        filename : string 
            Name of the customized binary file
        kwrdIndex : tuple, list, int or string
            Used to determine the keywrd

    Returns
    --------
     (headerbytes, keywrd, f0, df, Nrcv, Nf, Nc) : tuple
     data_dictR : dict
    """
    headerbytes, keywrd, f0, df, Nrcv, Nf, Nc =\
        readRINF_header(filepath,filename,kwrdIndex=(0,2),)

    filedatbin = open(f'{filepath}/{filename}', 'rb')
    filedatbin.seek(headerbytes)

    # Read data (seperated real and imag parts) and reshape
    # data_dictR has 3 dimensions : Nf, Nrcv and Nc
    data_dictR =\
        np.reshape(struct.unpack\
                   ('f'*int(Nrcv*Nf*Nc), filedatbin.read()),\
                   [Nf,Nrcv,Nc])
    filedatbin.close()

    return (headerbytes, keywrd, f0, df, Nrcv, Nf, Nc), data_dictR


def BasicPrint(*args):
    """
    Display function
    
    Parameters
    -----------
        args : label0,value0,label1,value1,label2,value2
            With str labels
    """
    lenmax = 0
    for i in range(int(len(args)//2)):
        if len(args[i*2]) > lenmax:
            lenmax = len(args[i*2])

    print('-'*lenmax)

    for i in range(int(len(args)//2)):
        label = args[i*2] + (lenmax-len(args[i*2])) * ' '
        value = args[2*i+1]

        if type(value) is float:
            print(f"{label}: {value:.5E}")
        else:
            print(f"{label}: {value}")


class FX_data:
    def __init__(self, datapath, FileNames, dataderive=False, datainteg=False):
        """
        Set paths and data file names and check file existance

        Parameters
        -----------
            datapath : tuple/list of strings, or string
                Path to filenames
            FileNames  : tuple/list of strings, or string
                Filenames (length should be equal to length of datapath)
            dataderive : 
                Derived dataset will be calculated and added
            datainteg : 
                Integrated dataset will be calculated and added
        """
        self.derive = dataderive
        self.integral = datainteg
        
        # datapath and FileNames should be contained in a list or a tuple
        if type(datapath) == str:
            self.datapath = (datapath,)
        
        elif type(datapath) == tuple or type(datapath) == list:
            self.datapath = datapath
        
        else:
            raise Exception("Bad path input:\nIt should be a string, a list or a tuple")

        if type(FileNames) == str:
            self.FileNames = (FileNames,)

        elif type(FileNames) == tuple or type(FileNames) == list:
            self.FileNames = FileNames

        else:
            raise Exception("Bad filename input\nIt should be a string, a list or a tuple")

        # Check if datapath and Filenames are consistent
        if len(self.datapath) != len(self.FileNames):
            raise Exception("{The number of paths should be equal to the number of files.}\n Number of paths: {len(self.datapath)}\n Number of files: {len(self.FileNames)}")

        self.nbdatapath = len(self.datapath)
        self.nbfiles = self.nbdatapath

        # check if files exists
        for pathi, path in enumerate(self.datapath):
            if not os.path.isfile(path+self.FileNames[pathi]):
                raise Exception(f"Error: {path} {self.FileNames[pathi]} is missing.")

    def __call__(self, t0=0, dx=1):
        """
        Initialize list          : f0, Nf, Nf_fix, df, Nrcv, dx, Nc, t0, Nt, Nk_fix, dk_fourier
                   list          : df_time_fx (for TX2FX after FX2TX), df_time_fk (for TX2FK after FX2TX),
                   nesteled list : tlist, flist, klist, dt_fourier
                   dict          : data_dictR, data_dictC, data_dictCW (if derive),
                                   data_dictCU (if integral)
        Add checkCalled=True      

        Parameters
        -----------
            t0 : float
                The starting time
            dx : float
                The default receiver increment=1
        """
        ########################################################################
        ##      The following variables are directly related                  ##
        ##              to the FX simulation                                  ##
        ##--------------------------------------------------------------------##
        self.f0 = [None for i in range(self.nbfiles)]     # freq start
        self.Nf = [None for i in range(self.nbfiles)]     # number of freq samples

        # next var: # number of freq samples (2^)
        self.Nf_fix = [None for i in range(self.nbfiles)]
        self.df = [None for i in range(self.nbfiles)]     # freq increment
        self.Nrcv = [None for i in range(self.nbfiles)]   # number of receivers
        self.dx = [dx for i in range(self.nbfiles)]       # receiver increment

        # next var: always 2 (complex number, a+bi)
        self.Nc = [None for i in range(self.nbfiles)]

        ########################################################################
        ##      The following variables are indirectly related                ##
        ##                  to the FX simulation                              ##
        ##      Actually they will be used in case of FX2TX or FX2TX2FX       ##
        ##--------------------------------------------------------------------##
        # time start (could be estimated from source delay)
        self.t0 = [t0 for i in range(self.nbfiles)]

        # freq increment, generally equals to 1/t_max (en cas de FX2TX2FX)
        self.df_time_fx, self.df_time_fk = [None for i in range(self.nbfiles)],\
            [None for i in range(self.nbfiles)]

        # number of time samples
        self.Nt = [None for i in range(self.nbfiles)]
        # number of time samples (1024*2, 2048*2 ...)
        # the number doubles with respect to Nf due to the
        # consideration of the negative frequencies
        #  !!!! self.Nt=[self.Nf_fix*2 for i in range(self.nbfiles)]

        ########################################################################
        ##      The following variables are indirectly related                ##
        ##                to the FX simulation                                ##
        ##      Actually they will be used in case of FX2FK                   ##
        ##--------------------------------------------------------------------##
        # next var: number of wavenumbers (2^)
        self.Nk_fix = [None for i in range(self.nbfiles)]

        # next var: wavenumber increment
        self.dk_fourier = [None for i in range(self.nbfiles)]

        ########################################################################
        ##                               les donnees :                        ##
        ## 1)   chiffres reels seulement, partie reelle imaginaire separee    ##
        ## 2)   chiffres complexes pour donnée originale V                    ##
        ## 3,4) chiffres complexes pour donnée dérivée ou intégrale dV, iV    ##
        ##--------------------------------------------------------------------##
        self.data_dictR = {}            # 1) raw data, separated real and imag parts

        self.data_dictC = {}            # 2) complexe data

        if self.derive:
            self.data_dictCW = {}       # 3) derived data

        if self.integral:
            self.data_dictCU = {}       # 4) integrated data

        self.tlist = [None for i in range(self.nbfiles)] # list of time lags
        self.flist = [None for i in range(self.nbfiles)] # list of freq lags
        self.klist = [None for i in range(self.nbfiles)] # list of wavenumber lags

        # in Haven, the following parameter is a carefully-defined parameter
        # the time sampling interval deduced from fmax
        self.dt_fourier = [None for i in range(self.nbfiles)]
        self.checkCalled = True


    def readBinAvch20(self, kwrdIndex=(-20,-14), optimal=True, FGreen=False,\
                      srcType=0, smoothrate=1, verbose=True, **kwargs):
        """
        Read a customized-style Binary file featuring a Twenty-or-so-byte header:
        This method will read data and put them into the variables initialized
             after call.

        Parameters
        -----------
            kwrdIndex    : used to decide the keyword of each data
                           see readRINF_header() for more info.
                           make sure that they are unique
            optimal      : if True, pad 0 to reach 2^X samples
            FGreen       : if True, data are results of Green Function (impulsive response)
            srcType      : determine the source to convolve in case of Green Function data
            smoothrate   : a smooth factor to pad more 0 and smoothen further the time signal
            verbose      : if True, print infos
            kwargs       : tpeak, fpeak (for Ricker source), filename (for external source)
        Fulfill dict   : data_dictR, data_dictC, data_dictCU (if integral), data_dictCW (if derive)
        Fulfull list   : keywords, keywords_d, keywords_i
                       : f0, df, Nrcv, Nf, Nc, Nf_fix, dx, t0, Nt, Nk_fix, dk_fourier
        """
        if self.checkCalled:
            self.keywords = []              # keywords stocked in a list
            self.keywords_d = []            # keywords for possible derived data
            self.keywords_i = []            # keywords for possible integrated data
            keywrd = 'Unknown'              # keywrd initialized as 'Unknown'
            
            for foldi, fold in enumerate(self.datapath):
                # Open file containing data
                headr = readRINF_header(fold, self.FileNames[foldi],\
                                      kwrdIndex=kwrdIndex,)
                keywrd = headr[1]
                self.keywords.append(headr[1])
                self.keywords_d.append(f'd{headr[1]}')
                self.keywords_i.append(f'i{headr[1]}')

                # read raw data (seperated real and imag parts)
                # data_dictR is 3-dimensional
                headr,self.data_dictR[keywrd] =\
                    readRINF(fold, self.FileNames[foldi], kwrdIndex=kwrdIndex,)
                # freq start and freq increment
                self.f0[foldi], self.df[foldi] = headr[2], headr[3]
                # number of receivers, number of freqs
                self.Nrcv[foldi], self.Nf[foldi] = int(headr[4]), int(headr[5])
                # 2 numbers, real part and imaginary part
                self.Nc[foldi] = int(headr[6])
                # optimize the number of freqs and wavenumbers
                # next var: number of freqs
                self.Nf_fix[foldi] = nextPow2(self.Nf[foldi]+1, inc_times=smoothrate)
                # next var: number of wavenumbers
                self.Nk_fix[foldi] = nextPow2(self.Nrcv[foldi])

                #data_dictC[keywrd] = np.empty((Nf[foldi]+1, Nrcv[foldi]),dtype=complex)
                #data_dictCU[keywrd] = np.empty((Nf[foldi]+1, Nrcv[foldi]),dtype=complex)
                # Initialize complexe data
                # data_dictC and data_dictCU have 2 dimensions
                # Nf_fix + 1 (freq=0)
                if optimal:
                    self.data_dictC[keywrd] = np.zeros((self.Nf_fix[foldi],\
                                                        self.Nk_fix[foldi]),\
                                                       dtype=complex)
                    if self.integral:
                        self.data_dictCU[keywrd] = np.zeros((self.Nf_fix[foldi],\
                                                             self.Nk_fix[foldi]),\
                                                            dtype=complex)
                    if self.derive:
                        self.data_dictCW[keywrd] = np.zeros((self.Nf_fix[foldi],\
                                                             self.Nk_fix[foldi]),\
                                                            dtype=complex)
                    # calculate the freq lists according to the values read from the 20-byte header
                    self.flist[foldi] = self.df[foldi] * np.arange(0, self.Nf_fix[foldi])
                    # calculate the wavenumber list
                    self.klist[foldi] = np.fft.fftfreq(self.Nk_fix[foldi],\
                                                     d=self.dx[foldi])
                    self.dt_fourier[foldi] = 1 / (2*self.flist[foldi][-1])
                    self.tlist[foldi] = self.dt_fourier[foldi]*\
                        np.arange(0, 2*(self.Nf_fix[foldi]-1)) + self.t0[foldi]

                    if verbose:
                        BasicPrint('Registered',keywrd,\
                                   'Optimal',optimal,\
                                   'Number of frequencies (including 0Hz)',self.Nf_fix[foldi],\
                                   'First frequency',self.flist[foldi][0],\
                                   'Frequency increment',self.df[foldi],\
                                   'Last frequency',self.flist[foldi][-1],\
                                   'Time interval from fmax:',self.dt_fourier[foldi],\
                                   'Time starts at:',self.tlist[foldi][0],\
                                   'Time ends at:',self.tlist[foldi][-1],\
                                   'Space interval',self.dx[foldi],\
                                   'Number of wavenumbers',self.Nk_fix[foldi],\
                                   'First wavenumber',self.klist[foldi][0],\
                                   'Max wavenumber:',max(self.klist[foldi]))

                else:
                    self.data_dictC[keywrd] = np.zeros((self.Nf[foldi]+1,\
                                                        self.Nrcv[foldi]),\
                                                       dtype=complex)
                    if self.integral:
                        self.data_dictCU[keywrd] = np.zeros((self.Nf[foldi]+1,\
                                                             self.Nrcv[foldi]),\
                                                            dtype=complex)
                    if self.derive:
                        self.data_dictCW[keywrd] = np.zeros((self.Nf[foldi]+1,\
                                                             self.Nrcv[foldi]),\
                                                            dtype=complex)
                    # calculate the freq lists according to the values read from the 20-byte header
                    self.flist[foldi] = self.df[foldi] * np.arange(0, self.Nf[foldi]+1)
                    # calculate the wavenumber list
                    self.klist[foldi] = np.fft.fftfreq(self.Nrcv[foldi],\
                                                     d=self.dx[foldi])
                    self.dt_fourier[foldi] = 1 / (2*self.flist[foldi][-1])
                    self.tlist[foldi] = self.dt_fourier[foldi] * np.arange(0, 2*self.Nf[foldi]) + self.t0[foldi]

                    if verbose:
                        BasicPrint('Registered',keywrd,\
                                   'Optimal',optimal,\
                                   'Number of frequencies (including 0Hz)',self.Nf[foldi]+1,\
                                   'First frequency',self.flist[foldi][0],\
                                   'Frequency increment',self.df[foldi],\
                                   'Last frequency',self.flist[foldi][-1],\
                                   'Time interval from fmax:',self.dt_fourier[foldi],\
                                   'Time starts at:',self.tlist[foldi][0],\
                                   'Time ends at:',self.tlist[foldi][-1],\
                                   'Space interval',self.dx[foldi],\
                                   'Number of wavenumbers',self.Nrcv[foldi],\
                                   'First wavenumber',self.klist[foldi][0],\
                                   'Max wavenumber:',max(self.klist[foldi]))

                if FGreen:
                    if srcType == 0:
                        if 'tpeak' not in kwargs:
                            kwargs['tpeak'] =\
                                1.2/kwargs['fpeak']
                        coef_src = np.ones(len(self.flist[foldi]), dtype=complex)
                        for fi in range(len(self.flist[foldi])):
                            if fi == 0 and verbose:
                                coef_src[fi] = Ricker_SF(self.flist[foldi][fi],\
                                                        fpeak=kwargs['fpeak'],\
                                                        tpeak=kwargs['tpeak'],\
                                                        verbose=verbose)
                            else:
                                coef_src[fi] = Ricker_SF(self.flist[foldi][fi],\
                                                        fpeak=kwargs['fpeak'],\
                                                        tpeak=kwargs['tpeak'])
                        # coef_src=Ricker(self.tlist[foldi],\
                        #                 fpeak=kwargs['fpeak'],\
                        #                 tpeak=kwargs['tpeak'])
                    elif srcType == 1:
                        timelist = []
                        amplist = []
                        with open(kwargs['filename'], 'r') as afile:
                            for line in afile.readlines():
                                timelist.append(float(line.split()[0]))
                                amplist.append(float(line.split()[1]))

                        coef_src = np.ones(len(self.flist[foldi]), dtype=complex)
                        for fi in range(len(self.flist[foldi])):
                            if fi == 0 and verbose:
                                coef_src[fi] = Ricker_FF_custom(self.flist[foldi][fi],\
                                                                np.array(timelist),\
                                                                np.array(amplist),\
                                                                verbose=verbose)
                            else:
                                coef_src[fi] = Ricker_FF_custom(self.flist[foldi][fi],\
                                                                np.array(timelist),\
                                                                np.array(amplist))
                        if verbose:
                            BasicPrint('Time vector length', len(timelist),\
                                       'Amplitude vector length', len(amplist))
                    else:
                        raise Exception(f"srcType={srcType}, unknown source type")
                else:
                    coef_src = 1
                # Fill in the complexe dicts freq by freq
                for fi in range(self.Nf[foldi]+1):
                    # special case at frequency=0
                    # MEAN of the signal
                    if fi == 0:
                        for ri in range(self.Nrcv[foldi]):
                            #FORCER MOYENNE NULLE
                            self.data_dictC[keywrd][fi,ri] = 0
                            #FORCER MOYENNE NULLE
                            if self.integral:
                                self.data_dictCU[keywrd][fi,ri] = 0
                            if self.derive:
                                self.data_dictCW[keywrd][fi,ri] = 0
                            #?? Interpolation au lieu de les forcer au 0 ??#
                            #?? data_dictC[keywrd][fi,ri]=data_dictR[keywrd][fi,ri,0] ??#
                    else:
                        for ri in range(self.Nrcv[foldi]):
                            # pour V, prendre le conjugue du resultat brut
                            # le conjugue permet de recaler l'axe temps suite à
                            #    la convention RT
                            # RT= reverse time
                            self.data_dictC[keywrd][fi,ri] =\
                                (self.data_dictR[keywrd][fi-1,ri,0]\
                                -1j*self.data_dictR[keywrd][fi-1,ri,1])*\
                                coef_src[fi]
                            # pour U, integral sur V d abord, puis prendre le conjugue
                            # add - sign ?
                            # either a derived or a integrated dataset is added
                            if self.integral:
                                self.data_dictCU[keywrd][fi,ri] =\
                                    (-self.data_dictR[keywrd][fi-1,ri,1]\
                                     -1j*self.data_dictR[keywrd][fi-1,ri,0])*\
                                     coef_src[fi]/(2*pi*self.flist[foldi][fi])
                            if self.derive:
                                self.data_dictCW[keywrd][fi,ri] =\
                                    (self.data_dictR[keywrd][fi-1,ri,1]\
                                     +1j*self.data_dictR[keywrd][fi-1,ri,0])*\
                                     coef_src[fi]*(2*pi*self.flist[foldi][fi])
                # Normally, self.flist[foldi][-1] is the highest frequency
                # dt_fourier=1/(2*fmc), fmc<=fnyquist
                #    (for example: padded to 0 until fnyquist)
                # fmax=2*fnyquist

                #self.dx_fourier[foldi]=1/(2*self.klist[foldi][-1])
        else:
            raise Exception("You need to call the instance first\n In order to set some fundamental parameters")


    def FX2FK(self, axis=1):
        """
        Transformee FX -> FK:
        By default, the fft will act along the second axis (Offset)
           of the 2-dimensional data_dictC and data_dictCU
           whose dimensions are (Nf, Nrcv)
        
        Add dict data_FX2FK
            checkFX2FK=True

        Parameters
        ------------
            axis : int
                Axis along which the FFT (space to wavenumber) is performed
        """
        self.data_FX2FK = {}

        for kyi, ky in enumerate(self.keywords):
            # convert V_fx to V_fk via fft
            self.data_FX2FK[ky] = self.dx[kyi] * np.fft.fft(self.data_dictC[ky],\
                                                        axis=axis)
            # convert U_fx to U_fk via fft
            # The first letter of the keyword will change from the default 'V' to 'U'
            if self.derive:
                self.data_FX2FK[self.keywords_d[kyi]] =\
                    self.dx[kyi] * np.fft.fft(self.data_dictCW[ky], axis=axis)
            if self.integral:
                self.data_FX2FK[self.keywords_i[kyi]] =\
                    self.dx[kyi] * np.fft.fft(self.data_dictCU[ky], axis=axis)

        self.checkFX2FK = True


    def FX2TX(self, axis=0, timeshift=0):
        """
        Convert to temporal
        By default, the irfft will act along the first axis (Freq)
           of the 2-dimensional data_dictC, data_dictCU and data_dictCW if they are required
           whose dimensions are (Nf, Nrcv)
        
        Add dict     : data_FX2TX
        
        Parameters
        -----------
            axis : int
                Axis along which the irfft will be performed
            timeshift : float
                Shift in time of the result of irfft
        
        """
        self.data_FX2TX = {}
        self.ifftAxis = axis

        for kyi, ky in enumerate(self.keywords):
            Nshift = int(timeshift/self.dt_fourier[kyi])
            # convert V(omega) to v(t) via irfft
            self.data_FX2TX[ky] = 1 / self.dt_fourier[kyi] *\
                np.roll(np.fft.irfft(self.data_dictC[ky], axis=axis),\
                        Nshift, axis=self.ifftAxis)
            # convert, if necessary, dV(omega) and iV(omega)
            #                   to dV(t) and iV(t) via irfft
            if self.integral:
                self.data_FX2TX[self.keywords_i[kyi]] = 1 / self.dt_fourier[kyi] *\
                    np.roll(np.fft.irfft(self.data_dictCU[ky], axis=axis),\
                            Nshift, axis=self.ifftAxis)
            if self.derive:
                self.data_FX2TX[self.keywords_d[kyi]] = 1 / self.dt_fourier[kyi] *\
                    np.roll(np.fft.irfft(self.data_dictCW[ky], axis=axis),\
                            Nshift, axis=self.ifftAxis)
        self.checkFX2TX = True


    def TXfilt(self, **filt):
        """
        filtrer les signaux converti en temps after calling FX2TX
        Add dict : data_TXfilt
        
        Parameters
        -----------
            filt : keyword arguments for filter. If not provided, filt is a list
                   keywords are 'filterName','N','Wn' for butterworth
        """
        if self.checkFX2TX:
            axis = self.ifftAxis # the same axis as the precedent ifft
            self.data_TXfilt = {}
            if not filt:
                print('filter not given, butterworth will be used by default')
                filt = ('butter', 1, 10, 'lowpass')

                for kyi, ky in enumerate(self.keywords):
                    bbutter, abutter = butter(filt[1],\
                                              filt[2]*self.dt_fourier[kyi]*2,\
                                              btype='highpass') # Ordre 1, cut at 0.06 X Nyqst (Nqyst = 50 MHz), lowpass

                    self.data_TXfilt[ky] = self.data_FX2TX[ky]-\
                        filtfilt(bbutter, abutter, self.data_FX2TX[ky], axis=axis)

                    if self.integral:
                        self.data_TXfilt[self.keywords_i[kyi]] =\
                            self.data_FX2TX[self.keywords_i[kyi]] -\
                            filtfilt(bbutter, abutter, self.data_FX2TX[self.keywords_i[kyi]], axis=axis)
                    
                    if self.derive:
                        self.data_TXfilt[self.keywords_d[kyi]] =\
                            self.data_FX2TX[self.keywords_d[kyi]] -\
                            filtfilt(bbutter, abutter, self.data_FX2TX[self.keywords_d[kyi]], axis=axis)

                self.checkTXfilt = True
            
            else:
                if filt['filterName'] == 'butter':
                    for kyi, ky in enumerate(self.keywords):
                        ckButter, message = checkButter(**filt)
                        if ckButter == 0:
                            raise Exception(f"checkButter() returned an error:\n{message}")

                        elif ckButter == 1:
                            bsos =\
                                butter(filt['N'],\
                                       filt['Wn']*self.dt_fourier[kyi]*2,\
                                       btype=filt['btype'], output='sos')

                        elif ckButter == 2:
                            # Ordre 1, cut at 0.06 X Nyqst (Nqyst = 50 MHz), lowpass
                            bsos =\
                                butter(filt['N'],\
                                       (filt['Wn'][0]*self.dt_fourier[kyi]*2,\
                                        filt['Wn'][1]*self.dt_fourier[kyi]*2),\
                                       btype=filt['btype'], output='sos')
                        # self.data_FX2TX[ky][0,:]=0
                        # self.data_FX2TX['U'+ky[1:]][0,:]=0
                        # self.data_FX2TX[ky][-1,:]=0
                        # self.data_FX2TX['U'+ky[1:]][-1,:]=0
                        self.data_TXfilt[ky] =\
                            sosfilt(bsos, self.data_FX2TX[ky], axis=axis)

                        if self.integral:
                            self.data_TXfilt[self.keywords_i[kyi]]=\
                                sosfilt(bsos, self.data_FX2TX[self.keywords_i[kyi]], axis=axis)

                        if self.derive:
                            self.data_TXfilt[self.keywords_d[kyi]]=\
                                sosfilt(bsos, self.data_FX2TX[self.keywords_d[kyi]], axis=axis)

                    self.checkTXfilt = True

                else:
                    raise Exception("Unknown filter name")


    def TX2FX(self, timecut=20):
        """
        Convert to frequency data after calling FX2TX

        Add dict : data_TX2FX
        Fulfill list : df_time_fx, flist_newTX2FX (flist can change due to the time truncation)

        Parameters
        -----------
            timecut : 
                TX signal truncated at timecut before doing ifft, unity is second
         """
        if self.checkFX2TX:

            axis = self.ifftAxis # the same axis as the precedent ifft
            self.data_TX2FX = {}
            self.flist_newTX2FX = {}
            for kyi, ky in enumerate(self.keywords):
                if timecut < max(self.tlist[kyi]):
                    Nfirst = list(map(lambda i: i>timecut, self.tlist[kyi])).index(True)
                    self.df_time_fx[kyi] = 1 / self.tlist[kyi][Nfirst-1]
                    self.flist_newTX2FX[kyi] = np.fft.rfftfreq(Nfirst, d=self.dt_fourier[kyi])

                    # convert v to V via fft
                    self.data_TX2FX[ky] = self.dt_fourier[kyi]*\
                        np.fft.rfft(np.moveaxis(np.moveaxis(self.data_FX2TX[ky], axis, 0)\
                                                [:Nfirst], 0, axis), axis=axis)
                    if self.integral:
                        self.data_TX2FX[keywords_i[kyi]] = self.dt_fourier[kyi] *\
                            np.fft.rfft(np.moveaxis(np.moveaxis(self.data_FX2TX[keywords_i[kyi]], axis, 0)\
                                                    [:Nfirst], 0, axis), axis=axis)
                    if self.derive:
                        self.data_TX2FX[keywords_d[kyi]] = self.dt_fourier[kyi] *\
                            np.fft.rfft(np.moveaxis(np.moveaxis(self.data_FX2TX[keywords_d][kyi], axis, 0)\
                                                    [:Nfirst], 0, axis), axis=axis)

                else:
                    self.df_time_fx[kyi] = 1 / self.tlist[kyi][-1]
                    self.data_TX2FX[ky] = self.dt_fourier[kyi] * np.fft.rfft(self.data_FX2TX[ky],  axis=axis)

                    if self.integral:
                        self.data_TX2FX[keywords_i[kyi]] = self.dt_fourier[kyi] *\
                            np.fft.rfft(self.data_FX2TX[keywords_i[kyi]], axis=axis)

                    if self.derive:
                        self.data_TX2FX[keywords_d[kyi]]=self.dt_fourier[kyi] *\
                            np.fft.rfft(self.data_FX2TX[keywords_d[kyi]], axis=axis)

        else:
            raise Exception('Method FX2TX must be applied before TX2FX')


    def TX2FK(self, timecut=False):
        """
        Convert to f-k after calling FX2TX

        Add dict     : data_TX2FK
        Fulfill list : df_time_fk, flist_newTX2FK (flist can change due to the time truncation)

        Parameters
        -----------
            timecut : float
                TX signal truncated at timecut before doing ifft2, unity is second.
        """
        if self.checkFX2TX:
            axis = self.ifftAxis # the same axis as the precedent ifft
            self.data_TX2FK = {}
            self.flist_newTX2FK = {}

            for kyi, ky in enumerate(self.keywords):
                if not timecut:
                    self.df_time_fk[kyi] = 1 / self.tlist[kyi][-1]
                    self.data_TX2FK[ky] = self.dt_fourier[kyi]*self.dx[kyi]*\
                        np.fft.rfft2(self.data_FX2TX[ky], axes=(1-axis,axis))

                    if self.integral:
                        self.data_TX2FK[keywords_i[kyi]] = self.dt_fourier[kyi] * self.dx[kyi] *\
                            np.fft.rfft2(self.data_FX2TX[keywords_i[kyi]], axes=(1-axis,axis))

                    if self.derive:
                        self.data_TX2FK[keywords_d[kyi]] = self.dt_fourier[kyi] * self.dx[kyi] *\
                            np.fft.rfft2(self.data_FX2TX[keywords_d[kyi]], axes=(1-axis,axis))
                else:
                    if timecut<max(self.tlist[kyi]):
                        Nfirst = list(map(lambda i: i>timecut, self.tlist[kyi])).index(True)
                        self.df_time_fk[kyi] = 1 / self.tlist[kyi][Nfirst-1]
                        self.flist_newTX2FK[kyi] = np.fft.rfftfreq(Nfirst, d=self.dt_fourier[kyi])
                        # convert v_tx to V_fk via fft2D
                        self.data_TX2FK[ky] = self.dt_fourier[kyi] * self.dx[kyi] *\
                            np.fft.rfft2(np.moveaxis(np.moveaxis(self.data_FX2TX[ky], axis, 0)\
                                                     [:Nfirst],0,axis), axes=(1-axis,axis))

                        if self.integral:
                            self.data_TX2FK[keywords_i[kyi]] = self.dt_fourier[kyi] * self.dx[kyi] *\
                                np.fft.rfft2(np.moveaxis(np.moveaxis(self.data_FX2TX[keywords_i[kyi]], axis, 0)\
                                                         [:Nfirst], 0, axis), axes=(1-axis, axis))

                        if self.derive:
                            self.data_TX2FK[keywords_d[kyi]] = self.dt_fourier[kyi] * self.dx[kyi] *\
                                np.fft.rfft2(np.moveaxis(np.moveaxis(self.data_FX2TX[keywords_d[kyi]], axis, 0)\
                                                         [:Nfirst], 0, axis), axes=(1-axis, axis))
        else:
            raise Exception('Method FX2TX must be applied before TX2FK')


    def TX_normalize(self, axis=None, lenEff=None, applyON=None):
        """
        Normalize by slice the converted data:
        By default, the normalization acts along the first axis
        
        Parameters
        -----------
            axis : 
                Axis along which the normalization is done, see Normalize_SBS
            lenEff : int
                Effective length for the normalization, see Normalize_SBS
            applyON : char
                'U'-> apply on integral data
                'W'-> apply on derived data
        """
        self.data_FX2TXN = {}

        if self.checkTXfilt:
            self.data_TXfiltN = {}

        if applyON == 'U': # apply on integral
            for kyi, ky in enumerate(self.keywords_i):
                self.data_FX2TXN[ky] = Normalize_SBS(self.data_FX2TX[ky], axis, lenEff)
                if self.checkTXfilt:
                    self.data_TXfiltN[ky] = Normalize_SBS(self.data_TXfilt[ky], axis, lenEff)

        elif applyON == 'W': # apply on derive
            for kyi, ky in enumerate(self.keywords_d):
                self.data_FX2TXN[ky] = Normalize_SBS(self.data_FX2TX[ky], axis, lenEff)
                if self.checkTXfilt:
                    self.data_TXfiltN[ky] = Normalize_SBS(self.data_TXfilt[ky], axis, lenEff)
        else:
            for kyi, ky in enumerate(self.keywords):
                self.data_FX2TXN[ky] = Normalize_SBS(self.data_FX2TX[ky], axis, lenEff)
                if self.checkTXfilt:
                    self.data_TXfiltN[ky] = Normalize_SBS(self.data_TXfilt[ky], axis, lenEff)


    def DictWrite2Bin(self, t_listen=None, filename_suffix='outputFX2TX_',\
                      take=None, pt=None, verbose=False):
        """
        Write binary temporal traces to file outputFX2TX_xxx.bin (name by default):
        By default, only the original data is written
        This method should be improved

        Parameters
        -----------
            t_listen : float
                Duration of time signal to write
            filename_suffix : string
                Suffix of the output file name
            take : char
                U -> take the integral
                W -> take the derive
            pt : char
                post-treatment, normalized, filtered both or none
            verbose : bool, 
                Print info or not
        """
        if take == 'U': # write integrated data
            for kyi, ky in enumerate(self.keywords_i):
                bfilename = f'{filename_suffix}{ky}.bin'

                with open(bfilename, 'wb') as f2writ:
                    # one can choose a duration of signal in second
                    if t_listen==None or t_listen>max(self.tlist[kyi]):
                        # write all samples
                        for i in range(self.Nrcv[kyi]):
                            for j in range(self.Nt[kyi]):
                                if pt in ('N', 'Normalized','Normalised',\
                                          'n', 'normalized','normalised'):
                                    f2writ.write(struct.pack('f',self.data_FX2TXN[ky][j,i]))
                                elif pt in ('f', 'filtered','Filtered','F'):
                                    f2writ.write(struct.pack('f',self.data_TXfilt[ky][j,i]))
                                elif pt in ('nf', 'NF', 'fn', 'FN'):
                                    f2writ.write(struct.pack('f',self.data_TXfiltN[ky][j,i]))
                                else:
                                    f2writ.write(struct.pack('f',self.data_FX2TX[ky][j,i]))

                    else:
                        # write chosen samples (a chosen duration)
                        Nt_listen=int(t_listen/self.dt_fourier[kyi])
                        if verbose:
                            print(f'{Nt_listen} samples in time will be written to binary file {bfilename}')

                        for i in range(self.Nrcv[kyi]):
                            for j in range(Nt_listen):

                                if pt in ('N', 'Normalized', 'Normalised',\
                                          'n', 'normalized', 'normalised'):
                                    f2writ.write(struct.pack('f', self.data_FX2TXN[ky][j,i]))

                                elif pt in ('f', 'filtered', 'Filtered', 'F'):
                                    f2writ.write(struct.pack('f', self.data_TXfilt[ky][j,i]))

                                elif pt in ('nf', 'NF', 'fn', 'FN'):
                                    f2writ.write(struct.pack('f', self.data_TXfiltN[ky][j,i]))

                                else:
                                    f2writ.write(struct.pack('f', self.data_FX2TX[ky][j,i]))

        elif take == 'W': # write derived data
            for kyi, ky in enumerate(self.keywords_d):
                bfilename = f'{filename_suffix}{ky}.bin'

                with open(bfilename, 'wb') as f2writ:
                    # one can choose a duration of signal in second
                    if t_listen == None or t_listen > max(self.tlist[kyi]):
                        # write all samples
                        for i in range(self.Nrcv[kyi]):
                            for j in range(self.Nt[kyi]):
                                if pt in ('N', 'Normalized', 'Normalised',\
                                          'n', 'normalized', 'normalised'):
                                    f2writ.write(struct.pack('f', self.data_FX2TXN[ky][j,i]))

                                elif pt in ('f', 'filtered', 'Filtered', 'F'):
                                    f2writ.write(struct.pack('f', self.data_TXfilt[ky][j,i]))

                                elif pt in ('nf', 'NF', 'fn', 'FN'):
                                    f2writ.write(struct.pack('f', self.data_TXfiltN[ky][j,i]))

                                else:
                                    f2writ.write(struct.pack('f', self.data_FX2TX[ky][j,i]))

                    else:
                        # write chosen samples (a chosen duration)
                        Nt_listen = int(t_listen/self.dt_fourier[kyi])

                        if verbose:
                            print(f'{Nt_listen} samples in time will be written to binary file {bfilename}')

                        for i in range(self.Nrcv[kyi]):
                            for j in range(Nt_listen):
                                if pt in ('N', 'Normalized','Normalised',\
                                          'n', 'normalized','normalised'):
                                    f2writ.write(struct.pack('f', self.data_FX2TXN[ky][j,i]))

                                elif pt in ('f', 'filtered','Filtered', 'F'):
                                    f2writ.write(struct.pack('f', self.data_TXfilt[ky][j,i]))

                                elif pt in ('nf', 'NF', 'fn', 'FN'):
                                    f2writ.write(struct.pack('f', self.data_TXfiltN[ky][j,i]))

                                else:
                                    f2writ.write(struct.pack('f', self.data_FX2TX[ky][j,i]))

        else:
            # write original data
            for kyi, ky in enumerate(self.keywords):
                bfilename = f'{filename_suffix}{ky}.bin'

                with open(bfilename, 'wb') as f2writ:
                    if t_listen == None or t_listen > max(self.tlist[kyi]):

                        for i in range(self.Nrcv[kyi]):
                            for j in range(self.Nt[kyi]):
                                if pt in ('N', 'Normalized', 'Normalised',\
                                          'n', 'normalized', 'normalised'):
                                    f2writ.write(struct.pack('f', self.data_FX2TXN[ky][j,i]))

                                elif pt in ('f', 'filtered', 'Filtered', 'F'):
                                    f2writ.write(struct.pack('f', self.data_TXfilt[ky][j,i]))

                                elif pt in ('nf', 'NF', 'fn', 'FN'):
                                    f2writ.write(struct.pack('f', self.data_TXfiltN[ky][j,i]))

                                else:
                                    f2writ.write(struct.pack('f', self.data_FX2TX[ky][j,i]))

                    else:
                        Nt_listen = int(t_listen/self.dt_fourier[kyi])
                        print(f'{Nt_listen} samples in time will be written to binary file {bfilename}')

                        for i in range(self.Nrcv[kyi]):
                            for j in range(Nt_listen):
                                if pt in ('N', 'Normalized', 'Normalised',\
                                          'n', 'normalized', 'normalised'):
                                    f2writ.write(struct.pack('f', self.data_FX2TXN[ky][j,i]))

                                elif pt in ('f', 'filtered', 'Filtered', 'F'):
                                    f2writ.write(struct.pack('f', self.data_TXfilt[ky][j,i]))

                                elif pt in ('nf', 'NF', 'fn', 'FN'):
                                    f2writ.write(struct.pack('f', self.data_TXfiltN[ky][j,i]))

                                else:
                                    f2writ.write(struct.pack('f', self.data_FX2TX[ky][j,i]))
