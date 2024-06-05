#
# Numerically calculated quadrature integrals associated with 
# Null geodesics in Kerr by numerically integrating them
#

import numpy as np
from scipy import integrate

from .turning_points import getroots_radial, getroots_angular
from .critical_parameters import get_sign_pr, get_critical_parameters
from .potentials import RadialP,AngularP,Delta

######################
# Angular Integrals
######################

def getGphiIntegrate(   alpha   , 
                        beta    ,
                        spin    ,
                        theta   ,
                        mbar    ): 
    """
    Numerically evaluates the angular integral Gphi evaluated between a source and an observer at infinity.
    For a discussion of how this is calculated see the paper

    'Lensing by Kerr Black Holes' by S.Gralla & A.Lupsasca 2020:
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031

    For a analytic alternative see the function getGphi

    Args:
        alpha (float): screen position in the x direction
        beta  (float): screen position in the y direction
        spin  (float): dimensionless spin a = J/a . Must be between 0 and 1.
        theta (float): observer inclination in radians with respect to the BH axis. 0 corresponds to face-on. pi/2 corresponds to edge-on
        mbar    (int): Number of winding points before photon reaches the observer

    Returns:
        _float_:  Radial integral Iphi evaluated between the source radius and infinity
    """

    #Photon conserved quantities
    lam = -alpha * np.sin(theta)
    eta = (alpha**2 - spin**2) * np.cos(theta)**2 + beta**2

    #Get angular roots u+ and u-
    uplus , _ = getroots_angular(alpha , beta , spin , theta)

    #Calculate the limits for the Gtheta integral
    #theta_minus = np.arccos(np.sqrt(uplus))
    theta_plus  = np.arccos(-np.sqrt(uplus))
    
    #Get sign of beta
    sgn_beta = 1.0 if beta*np.cos(theta) >= 0 else -1.0
    
    #Calculate corrected number of turning points mm. See Eq.82 of the above link
    Hbeta = 1 if beta >= 0 else 0
    mm = mbar + Hbeta 

    #Define the integrant (see equation Eq. 13e of Null geodesics of the Kerr Exterior paper)
    def Gphi_Integrand(spin, eta, lam):
        return lambda theta: (1 / np.sin(theta)**2) / np.sqrt(AngularP(theta,spin,eta,lam))
    
    Integrand    = Gphi_Integrand(spin, eta, lam)
    
    # Note: the calculation is subtle because of the turning points.
    #       To see a careful discussion see Eq 20a and 20b of 
    #       the paper 'Particle motion near high spin BHs' by
    #       Lupsasca and Kapec 2020
    #
    #       Here I use expression 20a to calculate the integrals

    I1 = 2*mm * np.abs(integrate.quad( Integrand , np.pi/2 , theta_plus)[0])
    I2 = sgn_beta * np.abs((integrate.quad(Integrand, np.pi/2, theta)[0]))
    
    return (I1-I2)
    

######################
# Radial integrals
######################

def getIphiIntegrate(   alpha   , 
                        beta    ,
                        spin    ,
                        theta   ,
                        radius  ,
                        mbar    ):
    """
    Numerically evaluates the radial integral Iphi evaluated between a source and an observer at infinity.
    For a discussion of how this is calculated see the paper

    'Lensing by Kerr Black Holes' by S.Gralla & A.Lupsasca 2020:
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031

    Args:
        alpha  (float): screen position in the x direction
        beta   (float): screen position in the y direction
        spin   (float): dimensionless spin a = J/a . Must be between 0 and 1.
        theta  (float): observer inclination in radians with respect to the BH axis. 0 corresponds to face-on. pi/2 corresponds to edge-on
        radius (float): source radius
        mbar     (int): Number of winding points before photon reaches the observer

    Returns:
        _float_:  Radial integral Iphi evaluated between the source radius and infinity
    """

    #Photon conserved quantities and on-sky quantities
    lam = -alpha * np.sin(theta)
    eta = (alpha**2 - spin**2) * np.cos(theta)**2 + beta**2
        
    b = np.sqrt(alpha**2 + beta**2)
    varphi = np.arctan2(beta, alpha)
    
    def Iphi_Integrand(spin, eta, lam):
        return lambda r: spin*(2*r-spin*lam)/( Delta(r,spin) * np.sqrt( RadialP(r,spin,eta,lam) ) )

    #Define integrand (see equation 8c of 'Lensing by BHs')          
    Integrand = Iphi_Integrand(spin, eta, lam)
    I1 = integrate.quad(Integrand, radius, np.inf)[0]
    
    #Get critical parameter
    bc = get_critical_parameters(spin , theta , varphi)[1]
    
    if b < bc:
        return I1
    else:
        sign_pr = get_sign_pr(b , spin , theta , varphi , mbar)

        if sign_pr==1:
            return I1
        elif sign_pr==-1:
            r4 = getroots_radial(alpha , beta , spin , theta)[3]
            I2 = integrate.quad(Integrand, r4.real, radius)[1]
            return (I1+2*I2)
      