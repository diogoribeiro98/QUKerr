# Note: All implemented equations can be found in the paper
#
#       [1] 'Lensing by Kerr Black Holes' by Gralla & Lupsasca 2020:
#            https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031
#

import numpy as np
from .quadrature_integrals_numerical import getIphiIntegrate
from .quadrature_integrals import getGphi

def deltaPhi(alpha , beta , spin , theta , r_source , mbar):
    """
    Returns azimuthal coordinate of a photon emmited by an equatorial source detected by an observer at infinity.
    For a discussion of how this is calculated see the paper

    'Lensing by Kerr Black Holes' by S.Gralla & A.Lupsasca 2020:
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031

    
    Note: The returned azimuthal coordinate is offset from the observer's one by pi/2 radians. 
          In practice this is equivalent to putting the observer at varphi = -pi/2 in the kerr geometry.
          The reason for this conventio is to more easily match the usual astropysical definition of 
          argument of periapsis.

    Args:
        alpha    (float): screen position in the x direction
        beta     (float): screen position in the y direction
        spin     (float): dimensionless spin a = J/a . Must be between 0 and 1.
        theta    (float): observer inclination in radians with respect to the BH axis. 0 corresponds to face-on. pi/2 corresponds to edge-on
        r_source (float): radial coordinate of the source
        mbar       (int): Number of winding points before photon reaches the observer

    Returns:
        _float_: azimuthal coordinate of the source
    """
    
    varphi = np.arctan2(beta, alpha)
	
    #If not spinning return Schwarzschild result
    if spin == 0:
        lam = -alpha * np.sin(theta)
        delta_phi = -lam*getGphi(alpha,beta,0.0,theta,mbar)
        return np.mod(delta_phi-np.pi/2,2*np.pi)
    else:
        
        #If on-axis, return limit result
        if theta == 0:
            Iphi = getIphiIntegrate(alpha , beta , spin , theta , r_source , mbar)
            delta_phi =  varphi - np.real(Iphi) - mbar*np.pi
            return np.mod(delta_phi,2*np.pi)

        else:

            lam = -alpha * np.sin(theta)
            
            Gphi = getGphi(alpha , beta , spin , theta , mbar)
            Iphi = getIphiIntegrate(alpha , beta , spin , theta , r_source , mbar)
            
            delta_phi = -np.real(Iphi + lam*Gphi)
            
            return np.mod(delta_phi-np.pi/2,2*np.pi)
