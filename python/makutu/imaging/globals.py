def initializeGlobalVariables(solv, fullAcq, listOfAcq, minM, maxM, minD, freqs, runmode="REL"):
    """Initialize global variables for simulations routines (direct, gradient, FWI, optimization...)
    
    Parameters:
    -----------
        solv : AcousticSolver
            AcousticSolver object
        fullAcq : Acquisition
            Acquisition object before splitting
        litsOfAcquisition : list
            List of all acquisitions (after splitting the full one)
        minM : float
            Min Value of model tolerated
        maxM : float 
            Max value of model tolerated
        freqs : list
            List of frequencies for source
        mode : str
            Type of run \
            release (REL) or debug (DBG) \
            Default is `REL`
    """
    global solver, acquisition, listOfAcquisition, totalShots, minModel, maxModel, minDepth, freqList, mode 

    solver = solv
    acquisition = fullAcq
    listOfAcquisition = listOfAcq
    totalShots = len(acquisition.shots)
    minModel = minM
    maxModel = maxM
    minDepth = minD
    freqList = freqs
    mode = runmode


def setModelAsGlobal(m):
    """Set model values array as global for optimization routine
    
    Parameters:
    -----------
        m : np array
            Array containing model values
    """
    global model

    model = m
