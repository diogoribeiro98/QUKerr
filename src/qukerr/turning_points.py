#
# Returns the radial and angular turning points to use in the quadrature integrals
#

# The bibliographic references are
#
# [1] 'Lensing by Kerr Black Holes' by Gralla & Lupsasca 2020:
#      https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031
#
# [2] 'Polarized image of qeuatorial emission in the Kerr geometry' by Gelles et. al 2021:    
#      https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031
#   

#Import standard Libs
import numpy as np

####################

def getroots_radial(    alpha       ,
                        beta        ,
                        spin        ,   
                        theta       ):
    """

    Given the screen position (alpha, beta), returns the radial turning 
    points of the photon orbit. Several expressions in the literature 
    exist for these turning points. Here, we chose the expression from 
    Appendix A of 
    
    'Lensing by Kerr Black Holes' by Gralla & Lupsasca 2020:
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031

    An alternative formula can be found in Appendix B of 

    'Polarized image of qeuatorial emission in the Kerr geometry' by Gelles et. al 2021:    
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031
   
    
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
    Returns
    -------
    np.array
        List of radial turning points.
    """

    #Photon conserved quantities (Eq. 58 and 59 of [1])
    lam = -alpha * np.sin(theta)
    eta = (alpha**2 - spin**2) * np.cos(theta)**2 + beta**2

	#ABCPQ coefficients and roots (see Appendix A of [1])
    A = spin**2 - eta - lam**2 
    B = 2 * ( eta + (lam - spin)**2 ) 
    C = -spin**2 * eta 
    P = - (A**2/12) - C
    Q = - (A/3) * ( (A/6)**2 - C ) - (B**2/8) 

    wplus  = ( -Q/2 + np.sqrt((P/3)**3 + (Q/2)**2 + 0j) )**(1/3)
    wminus = ( -Q/2 - np.sqrt((P/3)**3 + (Q/2)**2 + 0j) )**(1/3)

    z = np.sqrt((wplus+wminus)/2 - A/6)

    root_factor_1 = np.sqrt(-A / 2 - z**2 + B / (4*z))
    root_factor_2 = np.sqrt(-A / 2 - z**2 - B / (4*z))
    
    r1 = - z - root_factor_1
    r2 = - z + root_factor_1
    r3 =   z - root_factor_2
    r4 =   z + root_factor_2

    return np.array([r1,r2,r3,r4])


def getroots_angular(   alpha       ,
                        beta        ,
                        spin        ,   
                        theta       ):
    """
    
    Given the screen position (alpha, beta), returns the square cossine of 
    the angular turning points of the photon orbit. For details regarding 
    how these are calculated see section III of:

    'Lensing by Kerr Black Holes' by Gralla & Lupsasca 2020:
     https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.044031

    The expressions above do not hold when spin is set to zero. In that case
    the simplified expression is considered.

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
    Returns
    -------
    np.array
        List of angular turning points.
    """

    lam = -alpha * np.sin(theta)
    eta = (alpha**2 - spin**2) * np.cos(theta)**2 + beta**2

    if spin == 0:
        uroot = np.sqrt(eta/(eta+lam**2))
        return -uroot,uroot
    else:
        if eta < 0:
            raise Exception("ERROR: Vortical motion not implemented.")
        else:
            #Roots (see equation (19) )
            Delta = 0.5 * ( 1 - (eta + lam**2) / spin**2 ) 

            uplus  = Delta + np.sqrt(Delta**2 + eta/spin**2)
            uminus = Delta - np.sqrt(Delta**2 + eta/spin**2)

            return uplus , uminus