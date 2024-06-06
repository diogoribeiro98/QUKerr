#Import libs
import numpy as np
from .potentials import Delta,RadialP
from .critical_parameters import get_sign_pr

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
        [ -gamma*boost*coschi , (gamma-1)*coschi**2+1   ,  (gamma-1)*sinchi*coschi , 0 ],
        [ -gamma*boost*sinchi , (gamma-1)*sinchi*coschi ,  (gamma-1)*sinchi**2+1   , 0 ],
        [          0          ,           0             ,            0             , 1 ]])
	
    return lorentzboost

def get_zamo_tetrad(radius,spin):
    
    #Helper quantities of tetrad (see Eq.(1) and Eq.(2))
    delta = Delta(radius,spin)
    Xi    = (radius**2 + spin**2)**2 - delta * spin**2 
    omega = 2 * spin * radius / Xi
	
	# Zamo frame tetrad (see equation A1) Note, the coordinates here are in the order (t , r , theta , phi)
    emutetrad = np.array([
		[1/radius*np.sqrt(Xi/delta)	,      0		        ,   0	    , omega/radius*np.sqrt(Xi/delta)	], 
		[       0			        , np.sqrt(delta)/radius	,   0	    , 		0				            ], 
		[       0			        ,       0		        ,   0	    ,     radius/np.sqrt(Xi)		    ], 
		[		0			        , 		0		        , -1/radius	, 		0				            ]])

    return emutetrad

def get_polarization(   b , 
	    		        varphi, 
				        rs, 
				        spin, 
				        theta, 
				        boost, 
				        chi, 
				        bvec,			
                        mbar = 0,
                        spectral_index = 1.0,
                        lp = 1.0				
				        ):
    """
    Given an observer sky position (alpha, beta) = (b*cos(varphi) , b*sin(varphi))
    and the parameters associated with the local emission of the photon, returns the
    observed on-sky polarization projected and associated Stokes parameters.

    For a detailed discussion see section III.C and III.D of Gelles et al. (2021)

    For the definition of the Stokes parameters see section VI.A

    Parameters
    ----------
    b : float
        _description_
    varphi : float
        _description_
    rs : float
        Radius of source
    spin : float
        dimensionless spin a = J/a . Must be between 0 and 1.
    theta : float
        observer inclination in radians with respect to the BH axis. 0 corresponds to face-on.
        pi/2 corresponds to edge-on
    boost : float
        Boost magnitude. Must be between 0 and 1.
    chi : float
        Angle with respect to the radial axis measured torwards 
        positive values of phi. Measured in the local frame.
    bvec : np.array
        3d array with the components of the local magnectic field along
        the r,phi, and theta directions. These can be thougt of x,y,z 
        directions where x points in the radial direction and z along the
        spin axis of the black hole.
    mbar : int
        Number of winding points before photon reaches the observer. By default 0.
    spectral_index : float
        Spectral index of the emmited intensity I(frequency) ~ frequency^{-spectral_index}.
        For radio one would use the default value of 1. For infrared one could use the value 0.
        To avoid spectral scalling of the polarization use 3.
    Returns
    -------
    np.array
        Returns in the following order:

        falpha  : polarization vector component along alpha
        fbeta   : polarization vector componetn along beta
        evpa    : Electric field position angle (EVPA) calculated using the vector
        Q       : Stokes parameter Q
        U       : Stokes parameter U
        evpa_qu : Electric field position angle (EVPA) calculated using the QU values

    """
	
    #===================
    # Helper quantities
    #===================

	#Get alpha and beta coordinates
    alpha = b * np.cos(varphi)
    beta  = b * np.sin(varphi)
	
	#Calculate conserved photon quantities
    lam = -alpha * np.sin(theta)
    eta = (alpha**2 - spin**2) * (np.cos(theta))**2 + beta**2
	
	#Helper quantities
    delta = Delta(rs,spin)
    RR    = RadialP(rs,spin,eta,lam)

	# Zamo frame tetrad  and Minkowski metric (see equation A1) 
    # Note, the coordinates here are in the order (t , r , theta , phi)
    emutetrad = get_zamo_tetrad(rs,spin)
    minkmetric = np.diag([-1, 1, 1, 1])

    #===========================
    # Coordinate transformation
    #===========================

    # Boost the ZAMO observer according to Eq.(6)
    # Note: Here we 'unboost' the photon back to the orbiting frame
    boost_matrix = getlorentzboost(-boost, chi)
    
    coordtransform    = np.matmul(np.matmul(minkmetric,boost_matrix), emutetrad)

    #The inverse coordinate transformation is given by Eq.7
    coordtransforminv = np.transpose(np.matmul(boost_matrix, emutetrad))

    #====================================
    # Polarization in the orbiting frame
    #====================================

    #Get appropriate value of turning points
    mm = mbar if beta*np.cos(theta) < 0 else mbar+1

    #Lowered momentum at source (see Eq.23)
    sign_pr  = get_sign_pr(b, spin, theta, varphi, mm)
    plowers = np.array([
		-1, 
		sign_pr * np.sqrt(RR)/delta, 
		np.sign(np.cos(theta))*((-1)**(mm))*np.sqrt(eta), 
		lam])

    #Transform momemtum to local frame (according to Eq.(6))
    pupperfluid = np.matmul(coordtransform, plowers)
	
    #Normalize magnectic field and calculate the polarization vector (see Eq.25)
    bvec = np.asarray(bvec, dtype = float)
    bvec /= np.linalg.norm(bvec)
  
    fupperfluid = np.cross(pupperfluid[1:], bvec)
    fupperfluid = (np.insert(fupperfluid, 0, 0)) / (np.linalg.norm(pupperfluid[1:]))

    #Transform from the orbiting frame back to Kerr coordinates and definr individual coordinates of polaeization vector
    kfuppers = np.matmul(coordtransforminv, fupperfluid)
	
    kft 	= kfuppers[0]
    kfr 	= kfuppers[1]
    kftheta = kfuppers[2]
    kfphi 	= kfuppers[3]

    #============================================
    # Observed polarization on observer's screen
    #============================================

    #Raised momentum at source (see Eq.24) needed to calculate the Penrose Walker constant
    sign_ptheta = np.sign(np.cos(theta))*((-1)**(mbar+1))
    pt      = (1 / rs**2 ) * ( -spin * (spin - lam) + (rs**2 + spin**2) * (rs**2 + spin**2 - spin * lam) / delta)
    pr      = sign_pr * np.sqrt(RR) / rs**2
    ptheta  = sign_ptheta*np.sqrt(eta) / rs**2
    pphi    = 1/(rs**2) * (-(spin - lam) + (spin * (rs**2 + spin**2 - spin * lam)) / delta)

    #Penrose Walker constant see Eq.29 and Eq.30
    AA = (pt * kfr - pr * kft) + spin * (pr * kfphi - pphi * kfr)
    BB = (rs**2 + spin**2) * (pphi * kftheta - ptheta * kfphi) - spin * (pt * kftheta - ptheta * kft)
	
    kappa1 =  rs * AA
    kappa2 = -rs * BB

    #Calculate screen vectors according to Eq.(31)
    mu = -(alpha + spin * np.sin(theta))
    ealpha = (beta * kappa2 - mu * kappa1) / (mu**2 + beta**2)
    ebeta  = (beta * kappa1 + mu * kappa2) / (mu**2 + beta**2)

	#Redshift according to Eq.(33)
    redshift = 1 / (pupperfluid[0])

    #Observed scaling from doopler boost
    scaling = redshift**((3+spectral_index)/2)*np.sqrt(lp)
    
    ealpha *= scaling
    ebeta  *= scaling

    #============================
    # EVPA and Stokes parameters
    #============================

    #EVPA is measured from the beta direction in the counterclockwise direction
    evpa_sky = np.arctan2(-ealpha, ebeta) 
	
    #Define QU parameters
    Q = ebeta**2 - ealpha**2
    U = -2*ealpha*ebeta
    evpa_qu = 0.5*np.arctan2(U,Q)    
    
    return ealpha , ebeta , evpa_sky , Q , U , evpa_qu