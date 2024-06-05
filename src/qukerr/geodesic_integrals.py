import numpy as np
from .quadrature_integrals_numerical import getIphiIntegrate
from .quadrature_integrals import getGphi

def deltaPhi(alpha , beta , spin , theta , r_source , mbar):
    
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
