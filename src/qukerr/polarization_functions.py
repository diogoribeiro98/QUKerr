#Import libs
import numpy as np
def getlorentzboost(boost, chi):
    """
    Returns the Lorentz transformation associated with a boost in the r-phi plane. 
    
    Parameters
    ----------
    boost : float
        Boost magnitude. Must be between 0 and 1.
    chi : float
        Angle with respect to the radial axis measured torwards positive values of phi. 
    Returns
    -------
    np.array
        4x4 Lorentz boost matrix
    """
	
    #Lorenz factor
    gamma = 1 / np.sqrt(1 - boost**2)
	
    #Direction of boost
    coschi = np.cos(chi)
    sinchi = np.sin(chi)
	
    #Boost matrix
    lorentzboost = np.array([
        [        gamma        ,  -gamma*boost*coschi    ,   -gamma*boost*sinchi    , 0 ],
        [ -gamma*boost*coschi , (gamma-1)*coschi**2+1   , (gamma-1)*sinchi*coschi  , 0 ],
        [ -gamma*boost*sinchi , (gamma-1)*sinchi*coschi ,  (gamma-1)*sinchi**2+1   , 0 ],
        [          0          ,           0             ,            0             , 1 ]])
	
    return lorentzboost
