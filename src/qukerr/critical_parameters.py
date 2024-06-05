
# Note: The implemented functions are taken from a selection of paper
#
#       [1] 'Universal interferometric signatures' of Black Holes photon rings' by Johnson et al.
#            https://www.science.org/doi/10.1126/sciadv.aaz1310
#     
#       Other useful papers are
#
#       [2] 'Lensing by Kerr Black Holes' by Gralla & Lupsasca 2020:
#            https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031
#
#       [3] 'Polarized image of qeuatorial emission in the Kerr geometry' by Gelles et. al 2021:    
#            https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031
#
#       [4] 'Particle motion near high-spin black holes' by D.Kapec and A.Lupsasca 2020:
#            https://iopscience.iop.org/article/10.1088/1361-6382/ab519e
#     

#Import libs
import numpy as np
from lmfit import Minimizer, Parameters

from .special_functions import elliptic_K
from .quadrature_integrals import getGtheta , getIr_turn
from .potentials import Delta

######################
# Radial integrals
######################

def get_photon_shell_angular_roots(spin , r):
    """
    Returns angular roots of a photon shell of a specified radius.
    The expression implemented is taken from Eq.(4) of the paper

    'Universal interferometric signatures' of Black Holes photon rings' by Johnson et al. (2020)
     https://www.science.org/doi/10.1126/sciadv.aaz1310
   
    Args:
        spin (float): dimensionless spin a = J/a . Must be between 0 and 1.
        r    (float): photon's orbital radius

    Returns:
        uplus, uminus: Angular roots
    """

    factor = 2*np.sqrt( Delta(r,spin) * (2*r**3-3*r**2+spin**2))

    u_plus  = r/(spin*(r-1))**2 * (-r**3 + 3*r - 2*spin**2 + factor)
    u_minus = r/(spin*(r-1))**2 * (-r**3 + 3*r - 2*spin**2 - factor)

    return u_plus , u_minus


def get_numerical_impact_parameter(spin , theta , varphi):
    """
    Returns the impact parameter and orbital radius of critical photons from the photon-shell. 
    The value is calculated nummerically by finding the values that satisfy Eq(10a) and Eq(10b) of the paper

    'Universal interferometric signatures' of Black Holes photon rings' by Johnson et al (2020)
     https://www.science.org/doi/10.1126/sciadv.aaz1310
    
    Args:
        spin   (float): dimensionless spin a = J/a . Must be between 0 and 1.
        theta  (float): observer inclination in radians with respect to the BH axis. 0 corresponds to face-on. pi/2 corresponds to edge-on.
        varphi (float): angle on-sky measured from the alpha direction counterclockwise. Spin of the black hole is along the vertical axis (varphi = pi/2) in the direction given by the right hand rule.

    Returns:
        r, rho : critical photon's orbital radius, critical photon's impact parameter.

    """
    
    #Define function whose roots we want to find. See Equation (10a) and (10b) of [1]
    def residuals(params, spin, theta, varphi):

        r      = params['rs']
        rho    = params['rho']

        lam = ((r**2-spin**2) - r*Delta(r,spin))/(spin*(r-1))
        uplus, uminus = get_photon_shell_angular_roots(spin, r)
        
        return (rho-np.sqrt(spin**2 * ((np.cos(theta))**2-uplus*uminus) + lam**2), np.cos(varphi)*rho*np.sin(theta)+lam)
    
    #Photon shell radial bounds from  Eq.(2b) of [1] 
    rmin = 2*(1+np.cos( (2/3) * np.arccos(-spin) ))
    rmax = 2*(1+np.cos( (2/3) * np.arccos( spin) ))

    # Define fitting parameters and find roots
    #
    # Note: Take care if changing the fitting method as 
    #       it may lead to poor convergence or erroneous
    #       function evaluations.

    params = Parameters()
    params.add('rs' , value=3          , min = rmin, max = rmax)
    params.add('rho', value=np.sqrt(27), min = 0)
    
    fitter = Minimizer(residuals, params, fcn_args=(spin, theta, varphi) )
    fitter.minimize(method='cg')
    
    return fitter.result.params['rs'].value, fitter.result.params['rho'].value
    

def get_critical_curve(radius, rho, spin, theta, varphi):
    """
    Returns the perpendicular distance from the critical curve along a speecified on-sky direction for a given photon shell radius.
    The expressions implemented are Eq. (75) and Eq. (76) from the paper
    
    'Lensing by Kerr Black holes' by Gralla & Lupsasca (2020):
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031
    
    Args:
        radius  (float): photon orbital radius
        rho     (float): critical impact parameter
        spin    (float): dimensionless spin a = J/a . Must be between 0 and 1.
        theta   (float): observer inclination in radians with respect to the BH axis. 0 corresponds to face-on. Pi/2 corresponds to edge-on.
        varphi  (float): angle on-sky measured from the alpha direction counterclockwise. Spin of the black hole is along the vertical axis (varphi = pi/2) in the direction given by the right hand rule.

    Returns:
        cplus, cminus : perpendicular distance from critical curve
    """

    #Sky cordinates
    alpha = rho*np.cos(varphi)
    beta  = rho*np.sin(varphi)
    
    #Helper quantities (see paragraph after Eq. (76) )
    delta   = Delta(radius,spin)
    psi     = alpha- (radius+1)/(radius-1) *spin*np.sin(theta)
    chi     = 1-delta/(radius*(radius-1)**2)
    drplus  = (1+np.sqrt(1-spin**2))/radius-1
    
    Q0 = 1 + drplus/chi + drplus**2/(4*chi)
    Q2 = 2*np.sqrt(Q0) / ( Q0 + 1 - drplus**2/(4*chi) )
    
    cplus = ( (1+np.sqrt(chi))/(8*chi) )**2 * delta/(2*radius**4*chi) * np.sqrt(beta**2 + psi**2)
    cminus = -np.sqrt(1-chi)/(1+np.sqrt(chi)) * np.sqrt((1+Q2)/(1-Q2))*cplus
    
    return cplus, cminus

def get_critical_parameters(spin , theta , varphi):
    """
    Returns a list of the following critical parameters (in order)

    r       : critical photon orbital radius
    rho     : critical photoon impact parameter 
    gamma   : Lyapunoov exponent (number of turns associated with given point)
    cminus  : inward distance from critical curve 
    cplus   : outward distance from critical curve

    For a discussion of how these quantities are calculated see Section VII of the paper

    'Lensing by Kerr Black Holes' by Gralla & Lupsasca 2020:
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031

     For the case of a non-spinning black hole, the expressions used are from 
     the paper
    
    'Black hole shadows, photon rings, and lensing rings' by S.Gralla et al. (2019)
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.024018


    Args:
        spin   (float): dimensionless spin a = J/a . Must be between 0 and 1.
        theta  (float): observer inclination in radians with respect to the BH axis. 0 corresponds to face-on. Pi/2 corresponds to edge-on.
        varphi (float): angle on-sky measured from the alpha direction counterclockwise. Spin of the black hole is along the vertical axis (varphi = pi/2) in the direction given by the right hand rule.

    Returns:
        r, rho, gamma, cminus, cplus : critical parameters
    """

    if spin == 0:

        #Photon ring radius and critical impact parameter
        r, rho = 3, np.sqrt(27)

        #Winding number
        gamma = np.pi

        #Lensing ring borders
        cplus = 1944 / (12 + 7 * np.sqrt(3))
        cminus = 648 * (26 * np.sqrt(3) - 45)
        
        return r, rho, gamma, cplus, cminus

    else:

        #Get the impact parameter
        r, rho = get_numerical_impact_parameter(spin, theta, varphi)

        #Get photon shell roots for critical photons
        uplus, uminus = get_photon_shell_angular_roots(spin, r)
        
        #Lyapunov exponent and critical curve distances
        chi   = 1 - Delta(r,spin) /( r*(r-1)**2 ) 
        gamma = 4*r*np.sqrt(chi)/(np.sqrt(-uminus*spin**2))*elliptic_K(uplus/uminus)
        cplus, cminus = get_critical_curve(r, rho, spin, theta, varphi)
        
        return r, rho, gamma, cplus, cminus

def get_sign_pr(b , spin , theta , varphi , mbar):
    """
    Returns the sign of the radial momentum at the source point given it's impact parameter and onsky direction.
    For a discussion of how this parameter is calculated see Appendix B of the paper

    'Polarized image of qeuatorial emission in the Kerr geometry' by Gelles et. al 2021:    
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031


    Args:
        b       (float): Impact parameter of photon
        spin    (float): dimensionless spin a = J/a . Must be between 0 and 1.
        theta  (float): observer inclination in radians with respect to the BH axis. 0 corresponds to face-on. Pi/2 corresponds to edge-on.
        varphi (float): angle on-sky measured from the alpha direction counterclockwise. Spin of the black hole is along the vertical axis (varphi = pi/2) in the direction given by the right hand rule.
        mbar    (float): Turning number associated with image.

    Returns:
        sgn : sign of radial momentum at source
    """

    #Get critical parameters
    if spin == 0:
        bc = np.sqrt(27)
    else:
        bc = get_critical_parameters(spin , theta , varphi)[1]
    
    #If inside the critical curve, the photon must have been eemmited outwards
    if b < bc:
        return 1
    
    #Otherwise we must see if it found a turning point or not.
    else:
        alpha = b*np.cos(varphi)
        beta  = b*np.sin(varphi)
    
        #mm = mbar-1 if beta*np.cos(theta) < 0 else mbar

        Gtheta   = getGtheta( alpha , beta , spin , theta , mbar)
        Ir_turn  = getIr_turn( alpha , beta , spin , theta )
        
        return int(np.sign(np.real(Ir_turn - Gtheta)))
