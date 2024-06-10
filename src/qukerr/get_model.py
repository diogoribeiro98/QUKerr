#
# Helper functions for plotting and modelling
#

import numpy as np
from scipy import interpolate

from .impact_parameter_finder import find_impact_parameter
from .geodesic_integrals import deltaPhi
from .polarization_functions import get_polarization

#
# Output class
#

class _output:

    def __init__(self,time,phi,alpha,beta,falpha,fbeta,Q,U,EVPA):

        self.tmin = time[0]
        self.tmax = time[-1]

        self.phi    = interpolate.interp1d(time,  phi   )
        self.alpha  = interpolate.interp1d(time,  alpha )
        self.beta   = interpolate.interp1d(time,  beta  )
        self.falpha = interpolate.interp1d(time,  falpha)
        self.fbeta  = interpolate.interp1d(time,  fbeta )
        self.Q      = interpolate.interp1d(time,  Q     )
        self.U      = interpolate.interp1d(time,  U     )
        self.EVPA   = interpolate.interp1d(time,  EVPA  )

#
# Generate data from model
#

def get_model_from_orbital_parameters(*,
        #Orbital parameters
        radius,
        inclination,
        Omega,
        #BH parameters
        spin = 0.0,
        #Local magnectic field parameters
        bvec = [ 0 , 0 , 1]     , 
        #Model parameters
        NSEG = 360,
        spectral_index = 1,
        mbar = 0
):

    #Convert angles to radians
    theta_rad = inclination * np.pi/180 
    theta_rot = (90 + Omega) * np.pi/180
    
    #Define angular velocity, boost magnitude and direction for prograde orbit
    Kepler_velocity = 1/(radius**(3/2) + spin)
    boost = (radius**2 - 2*spin*np.sqrt(radius) + spin**2)/ ( np.sqrt(radius**2-2*radius + spin**2)) * Kepler_velocity 
    chi   = np.pi/2    

    #Get storage vectors for points
    phi_source  = []    # Orbital angle

    alpha_list  = []    # Horizontal position on sky
    beta_list   = []    # Vertical position on sky

    fa_list     = []    #horizontal component of electric field
    fb_list     = []    # vertical component of electric field

    Q_list      = []    # Q parameter
    U_list      = []    # U parameter

    evpa_fe     = []    # EVPA of measured field 
    evpa_qu     = []    # EVPA of QU 

    #Define the number of segments and list of sky position angles
    sky_varphi_list = np.linspace(1e-10 , 2*np.pi-1e-10, NSEG, endpoint=True)

    for sphi in sky_varphi_list:

        #Get the exact impact parameter b and correct alpha beta coordinates with no turning points (m=0)
        b = find_impact_parameter( radius , sphi ,  spin , theta_rad , mbar)
        
        alpha = b*np.cos(sphi)
        beta  = b*np.sin(sphi)

        #Get the source phi coordinate
        phi0 =  deltaPhi( alpha , beta , spin , theta_rad , radius , mbar)

        #Get the qu parameters for the Counter Clock wise orbit
        fa, fb, _, Q,U, EVPA = get_polarization(b , sphi , radius , spin , theta_rad ,
                                    boost , chi , bvec , mbar , spectral_index)    

        #On sky after rotation (two options of doing calculation)
        alpha_rotate = alpha*np.cos(theta_rot) - beta*np.sin(theta_rot)
        beta_rotate  = alpha*np.sin(theta_rot) + beta*np.cos(theta_rot)

        #Store position on sky after rotation
        alpha_rotate = b*np.cos(sphi+theta_rot)
        beta_rotate  = b*np.sin(sphi+theta_rot)

        ifa = fa*np.cos(theta_rot) - fb*np.sin(theta_rot)
        ifb = fa*np.sin(theta_rot) + fb*np.cos(theta_rot) 

        iQ  = Q*np.cos(2*theta_rot) - U*np.sin(2*theta_rot)
        iU  = Q*np.sin(2*theta_rot) + U*np.cos(2*theta_rot)
       
        phi_source.append(phi0)

        alpha_list.append(alpha_rotate)
        beta_list.append(beta_rotate)

        fa_list.append(ifa)
        fb_list.append(ifb)

        Q_list.append( iQ )    
        U_list.append( iU )    

        evpa_qu.append(EVPA)
  
    #Sort the data by angular position
    phi_source_sort = np.sort(phi_source)

    data = [phi_source_sort, phi_source_sort, alpha_list, beta_list, fa_list, fb_list, Q_list, U_list, evpa_qu]
    
    data_sorted = []
    #
    for idx,element in enumerate(data):
        if idx ==0 or idx==1:
            data_sorted.append(element)
            continue
        element = [ x for _,x in sorted(zip(phi_source, element))]
        data_sorted.append(element)

    data_sorted[0] = data_sorted[0]/Kepler_velocity

    return _output(data_sorted[0],data_sorted[1],data_sorted[2],data_sorted[3],data_sorted[4],data_sorted[5],data_sorted[6],data_sorted[7],data_sorted[8])


