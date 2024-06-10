#
# Returns the impact parameter for a photon emmited from the equatorial plane
#

# Note: Unlsess stated otherwise, the equations mentioned are taken from
#       https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.044060
#       We shall refer to this article as Gelles et al (2021)

import numpy as np
from lmfit import Parameters, Minimizer

from .quadrature_integrals import getGtheta
from .turning_points import *
from .special_functions import elliptic_F , elliptic_sn, elliptic_K
from .critical_parameters import get_critical_parameters

def rinvert(    alpha   , 
                beta    ,
                spin    ,
                theta   ,
                mbar     ):
    """
    Parameters
    ----------
    alpha : float
        screen position in the x direction
    beta : float
        screen position in the y direction
    spin : float
        dimensionless spin a = J/a . Must be between 0 and 1.
    theta : float
        observer inclination in radians with respect to the BH axis. 0 corresponds to face-on.
        pi/2 corresponds to edge-on
    mbar : int
        Number of winding points before photon reaches the observer
    
    Returns
    -------
    np.array
        Angular integral Gphi
    """

    # Get the angular integral Gtheta that, by definition, is equal to the radial integral Ir
    Ir = getGtheta( alpha , beta , spin , theta , mbar)
    
    #Get radial roots
    r1 , r2 , r3 , r4 = getroots_radial(alpha , beta , spin , theta)
    
    #Calculate helper quantities in Eq.22
    k0 = ((r3-r2)*(r4-r1)) / ((r3-r1)*(r4-r2))

    #Calculate F0
    F0 = elliptic_F( np.arcsin( np.sqrt( (r3-r1)/(r4-r1) ) )  , k0  )

    #Argument for the Jacobi Elliptic Sine function (sn)
    arg_sn = 0.5*np.sqrt((r3-r1)*(r4-r2))*Ir - complex(F0) 

    #Calculate Squared Jacobi function with prefactor (see Eq.20)
    r41sn2 = (r4-r1)*(complex(elliptic_sn( arg_sn , k0)))**2
    
    #Return RHS of Eq(20)
    return (r4 * (r3-r1) - r3*r41sn2) / ( (r3-r1) - r41sn2 )

def find_impact_parameter(  r_source    , 
                            varphi      ,
                            spin        ,
                            theta       ,
                            mbar        ):
    """
    Given the radial coordinate of the equatorial source and on-sky angle measured from
    the alpha direction counterclockwise returns the impact parameter on the observer's
    screen.
    
    Parameters
    ----------
    r_source : float
        Radius of the source
    varphi : float
        Position on sky measured from the alpha direction torwards positive values
        of beta
    spin : float
        dimensionless spin a = J/a . Must be between 0 and 1.
    theta : float
        observer inclination in radians with respect to the BH axis. 0 corresponds to face-on.
        pi/2 corresponds to edge-on
    mbar : int
        Number of winding points before photon reaches the observer
    
    Returns
    -------
    float
        Impact parameter of the geodesic connecting the source to the
        observer's screen.
    """
    
    #Define function whose root will give us the impacct parameter
    def residuals(params, radius, varphi, spin, theta, mbar):
        
        b = params['b']
        bval = np.abs( radius - rinvert( b*np.cos(varphi) , b*np.sin(varphi) , spin , theta , mbar ) )
        return bval
    
    #Get an initial guess for the parameters
    if mbar == 0:
        b_guess =  (r_source + 1)  + (spin**2-1)/(2*r_source) + (50 - 2*spin**2 - 15*np.pi)/(4*r_source**2)
    else:
    
        #Get critical parameters
        _, rho, gamma, cplus, _ = get_critical_parameters(spin , theta , varphi) 

        # For highly spinning black holes the secondary images are very close to the critical curve (helps convergence)
        if spin > 0.9:
            b_guess = rho
        else:
            m = mbar if np.sin(varphi)*np.cos(theta) < 0 else mbar + 1

            #Angular roots
            alpha = rho * np.cos(varphi) 
            beta  = rho * np.sin(varphi)
            if spin == 0.0:
                uroot =  getroots_angular(alpha, beta, 0.0, theta)
                
                #Geometric factor in Eq. 86 of Lensing by Kerr BHs
                F0 = np.arcsin( np.abs(np.cos(theta)) / np.sqrt(uroot)+0j )
                K  = np.pi/2
                f0 = F0/K
                
                #Get distance from critical curve
                sign_beta = np.sign(np.sin(varphi)*np.cos(theta))
                n = (m/2) - 0.25*sign_beta*f0
                d = np.real((1/cplus)*np.exp(-2*n*gamma))
                b_guess = rho + d
            else:

                uplus,uminus = getroots_angular(alpha, beta, spin, theta)
        
                #Geometric factor in Eq. 86 of Lensing by Kerr BHs
                F0 = elliptic_F( np.arcsin( np.abs(np.cos(theta)) / np.sqrt(uplus)+0j )  , uplus/uminus  )
                K = elliptic_K(uplus / uminus)
                f0 = F0/K
            
                #Get distance from critical curve
                sign_beta = np.sign(np.sin(varphi)*np.cos(theta))
                n = (m/2) - 0.25*sign_beta*f0
                d = np.real((1/cplus)*np.exp(-2*n*gamma))
                b_guess = rho + d
        
    #Fitting parameters
    params = Parameters()
    params.add('b' , value=b_guess, min = spin**2 * np.abs(np.cos(theta)))

    #Create minimizer and find root
    fitter = Minimizer(residuals, params, fcn_args=(r_source, varphi, spin, theta, mbar))
    fitter.minimize()
    bres = fitter.result.params['b'].value

    return bres
