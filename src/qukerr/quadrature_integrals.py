#
# Returns the quadrature integrals associated with Null geodesics in Kerr and equatorial emitters
#

# Note: All implemented equations can be found in the paper
#       
#       [1] 'Particle motion near high-spin black holes' by D.Kapec and A.Lupsasca 2020:
#            https://iopscience.iop.org/article/10.1088/1361-6382/ab519e
#
#       Other useful papers are
#
#       [2] 'Lensing by Kerr Black Holes' by Gralla & Lupsasca 2020:
#            https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031
#
#       [3] 'Polarized image of qeuatorial emission in the Kerr geometry' by Gelles et. al 2021:    
#            https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031
#      

#Import libs
import numpy as np
from .turning_points    import *
from .special_functions import elliptic_F , elliptic_K 
from .special_functions import elliptic_Pi , elliptic_Pi_complete
from .special_functions import elliptic_Ep,  elliptic_Ep_complete

######################
# Angular Integrals
######################

def getGtheta(  alpha   , 
                beta    ,
                spin    ,
                theta   ,
                mbar    ):
    """
    Returns the theta coordinate quadrature integral Gtheta for sources on the equatorial
    plane. For a discussion of how this integral is calculated see Section II.A of the paper

    'Lensing by Kerr Black Holes' by S.Gralla & A.Lupsasca 2020:
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031

    Args:
        alpha (float): screen position in the x direction
        beta  (float): screen position in the y direction
        spin  (float): dimensionless spin a = J/a . Must be between 0 and 1.
        theta (float): observer inclination in radians with respect to the BH axis. 0 corresponds to face-on. pi/2 corresponds to edge-on
        mbar    (int): Number of winding points before photon reaches the observer

    Returns:
        _float_: Angular integral Gtheta
    """    

    #If not spinning, use simpler Schwarzchild formula (See Eqs. 26 of [1])
    if spin == 0:

        #Calculate photon conserved quantities
        lam = -alpha * np.sin(theta)
        eta = (alpha**2 - spin**2) * np.cos(theta)**2 + beta**2

        #Angular root (see Eq.22 of [1])
        uroot = eta/(eta+lam**2)

        #Get observers cosine
        uobs = np.cos(theta)**2

        #Account for subtle counting of turning points and signn of beta
        mm = mbar if beta*np.cos(theta) < 0 else mbar+1
        sign_beta = 1 if beta >= 0 else -1
        
        #Calculate Gtheta according to Eq. 26a of [1]
        return np.sqrt(uroot/eta) * ( np.pi*mm - sign_beta*np.arcsin( np.sqrt(uobs/uroot) ) )  
    
    #If spinning, use the more general formulas (See Eqs.32 of [1])
    else:

        #Get angular  roots
        uplus , uminus = getroots_angular(alpha , beta , spin , theta)

        #Get observers cosine and associated Psi variable (See Eq.23 of [1])
        uobs = np.cos(theta)**2         
        Psi_obs_plus = np.arcsin(np.sqrt(uobs/uplus)+0j) 

        #Calculate elliptic functions
        K  = elliptic_K(uplus/uminus)
        F0 = elliptic_F( Psi_obs_plus, uplus/uminus)
    
        #Account for subtle counting of turning points and signn of beta
        mm = mbar if beta*np.cos(theta) < 0 else mbar+1
        sign_beta = 1 if beta >= 0 else -1

        #Calculate Gtheta according to Eq.32a of [1]
        return (1/np.sqrt(-uminus*spin**2)) * (2*mm*K - sign_beta*F0 )


def getGphi(    alpha   , 
                beta    ,
                spin    ,
                theta   ,
                mbar     ):
    """
    Returns the phi coordinate quadrature integral Gphi for sources on the equatorial
    plane. For a discussion of how this integral is calculated see Section II.A of the paper

    'Lensing by Kerr Black Holes' by S.Gralla & A.Lupsasca 2020:
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031

    Args:
        alpha (float): screen position in the x direction
        beta  (float): screen position in the y direction
        spin  (float): dimensionless spin a = J/a . Must be between 0 and 1.
        theta (float): observer inclination in radians with respect to the BH axis. 0 corresponds to face-on. pi/2 corresponds to edge-on
        mbar    (int): Number of winding points before photon reaches the observer

    Returns:
        _float_: Angular integral Gphi
    """   
    #If not spinning, use simpler Schwarzchild formula (See Eqs. 26 of [1])
    if spin == 0:

        #Calculate photon conserved quantities
        lam = -alpha * np.sin(theta)
        eta = (alpha**2 - spin**2) * np.cos(theta)**2 + beta**2
        
        #Angular root (see Eq.22 of [1])
        uroot = eta/(eta+lam**2)

        #Get observers cosine
        uobs = np.cos(theta)**2

        #Account for subtle counting of turning points and signn of beta
        mm = mbar if beta*np.cos(theta) < 0 else mbar+1
        sign_beta = 1 if beta >= 0 else -1

        #Calculate Gtheta according to Eq.26b of [1]
        return np.sqrt( uroot / (eta*(1-uroot))) * (np.pi*mm - sign_beta*np.arcsin( np.sqrt((uobs/uroot)*(1-uroot)/(1-uobs))))

    #If spinning, use the more general formulas (See Eqs.32 of [1])
    else:

        #Get angular  roots
        uplus , uminus = getroots_angular(alpha , beta , spin , theta)

        #Get observers cosine and associated Psi variable (See Eq.23 of [1])
        uobs = np.cos(theta)**2         
        Psi_obs_plus = np.arcsin(np.sqrt(uobs/uplus)+0j) 

        #Calculate elliptical integrals 
        Pi    = elliptic_Pi_complete(uplus, uplus/uminus)
        Piobs = elliptic_Pi(uplus, Psi_obs_plus, uplus/uminus)
        
        #Account for subtle counting of turning points and signn of beta
        mm = mbar if beta*np.cos(theta) < 0 else mbar+1
        sign_beta = 1 if beta >= 0 else -1

        #Calculate Gtheta according to Eq.32b of [1]
        return (1/np.sqrt(-uminus*spin**2)) * (  2*mm * Pi - sign_beta*Piobs )

def getGt(      alpha   , 
                beta    ,
                spin    ,
                theta   ,
                mbar     ):
    """
    Returns the t coordinate quadrature integral Gt for sources on the equatorial
    plane. For a discussion of how this integral is calculated see Section II.A of the paper

    'Lensing by Kerr Black Holes' by S.Gralla & A.Lupsasca 2020:
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031

    Args:
        alpha (float): screen position in the x direction
        beta  (float): screen position in the y direction
        spin  (float): dimensionless spin a = J/a . Must be between 0 and 1.
        theta (float): observer inclination in radians with respect to the BH axis. 0 corresponds to face-on. pi/2 corresponds to edge-on
        mbar    (int): Number of winding points before photon reaches the observer

    Returns:
        _float_: Angular integral Gt
    """   

    #If not spinning, use simpler Schwarzchild formula (See Eqs. 26 of [1])
    if spin == 0:

        #Calculate photon conserved quantities
        lam = -alpha * np.sin(theta)
        eta = (alpha**2 - spin**2) * np.cos(theta)**2 + beta**2
        
        #Angular root (see Eq.22 of [1])
        uroot = eta/(eta+lam**2)

        #Get observers cosine
        uobs = np.cos(theta)**2

        #Account for subtle counting of turning points and signn of beta
        mm = mbar if beta*np.cos(theta) < 0 else mbar+1
        sign_beta = 1 if beta >= 0 else -1

        #Calculate Gtheta
        Gtheta = getGtheta(alpha, beta, 0.0, theta, mbar)

        #Return Gt accorrding to Eq. 26c of [1]
        return 0.5 * ( uroot*Gtheta + sign_beta*np.sqrt(uroot/eta) * np.sqrt(uobs*(uroot-uobs))    )

    #If spinning, use the more general formulas (See Eqs.32 of [1])
    else:

        #Get angular  roots
        uplus , uminus = getroots_angular(alpha , beta , spin , theta)

        #Get observers cosine and associated Psi variable (See Eq.23 of [1])
        uobs = np.cos(theta)**2         
        Psi_obs_plus = np.arcsin(np.sqrt(uobs/uplus)+0j) 

        #Calculate elliptical integrals 
        Ep    = elliptic_Ep_complete(uplus/uminus)
        Epobs = elliptic_Ep(Psi_obs_plus, uplus/uminus  )

        #Account for subtle counting of turning points and signn of beta
        mm = mbar if beta*np.cos(theta) < 0 else mbar+1
        sign_beta = 1 if beta >= 0 else -1

        #Calculate Gtheta according to Eq.32b of [1]
        return -(2*uplus/np.sqrt(-uminus*spin**2)) * (  2*mm*Ep - sign_beta*Epobs )

######################
# Radial integrals
######################

def getIr_turn( alpha   , 
                beta    ,
                spin    ,
                theta   ):
    """
    Returns the of the radial integral Ir evaluated between the radial turning point of a 
    photon's trajectory and infinity. For a discussion see Appendix A of the paper

    'Polarized image of qeuatorial emission in the Kerr geometry' by Gelles et. al 2021:    
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031
    
    Args:
        alpha (float): screen position in the x direction
        beta  (float): screen position in the y direction
        spin  (float): dimensionless spin a = J/a . Must be between 0 and 1.
        theta (float): observer inclination in radians with respect to the BH axis. 0 corresponds to face-on. pi/2 corresponds to edge-on
    Returns:
        _float_:  Radial integral Ir evaluated between the radial turning point of a photon's trajectory and infinity
    """    
                
    r1 , r2 , r3 , r4 = getroots_radial(alpha , beta , spin , theta)

    r31 = r3 - r1
    r32 = r3 - r2
    r41 = r4 - r1
    r42 = r4 - r2

    k = (r32 * r41) / (r31 * r42)
    
    return (2 / np.sqrt(r31 * r42))*(elliptic_F( np.arcsin(np.sqrt(r31 / r41)) , k ))
