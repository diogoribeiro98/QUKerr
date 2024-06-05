import numpy as np
from .quadrature_integrals_numerical import getIphiIntegrate
from .quadrature_integrals import getGphi

def deltaPhi(alpha , beta , spin , theta , r_source , mbar):
    
    varphi = np.arctan2(beta, alpha)
	
    #If not spinning return simple result (motion on a plane)
    if spin == 0:
        pa = np.arctan2(np.sin(varphi), np.cos(varphi) * np.cos(theta)) + mbar*np.pi
        return np.mod(pa,2*np.pi)
    else:
        
        #If on-axis
        if theta == 0:
            Iphi = getIphiIntegrate(alpha , beta , spin , theta , r_source , mbar)
            pa =  varphi - np.real(Iphi) - mbar*np.pi

            return np.mod(pa,2*np.pi)

        else:

            lam = -alpha * np.sin(theta)

            Gphi = getGphi(alpha , beta , spin , theta , mbar)
            Iphi = getIphiIntegrate(alpha , beta , spin , theta , r_source , mbar)
            
            pa = -np.real(Iphi + lam*Gphi)
            
            return np.mod(pa,2*np.pi)
