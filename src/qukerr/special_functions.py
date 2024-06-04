#Import libs
import numpy as np
import mpmath as mp

# For a description of the functions see:
# https://mpmath.org/doc/current/functions/elliptic.html
#

#
# Elliptic integrals
#

##############
# 1st kind
##############

elliptic_K_temp             = np.frompyfunc(mp.ellipk   , 1, 1)
elliptic_F_temp             = np.frompyfunc(mp.ellipf   , 2, 1)

#Make them return complex values
def elliptic_K(k):
    return complex(elliptic_K_temp(k))

def elliptic_F(phi,k):
    return complex(elliptic_F_temp(phi,k))

##############
# 2nd kind
##############

elliptic_E_complete_temp  = np.frompyfunc(mp.ellipe   , 1, 1)
elliptic_E_temp           = np.frompyfunc(mp.ellipe   , 2, 1)

#Make them return complex values
def elliptic_E_complete(k):
    return complex(elliptic_E_complete_temp(k))

def elliptic_E(phi,k):
    return complex(elliptic_E_temp(phi,k))

def elliptic_Ep_complete(k):
    return (elliptic_E_complete(k) - elliptic_K(k))/(2*k) 

def elliptic_Ep(phi,k):
    return (elliptic_E(phi,k)-elliptic_F(phi,k)) / (2*k)

##############
# 3rd kind
##############

_elliptic_Pi_complete_temp   = np.frompyfunc(mp.ellippi  , 2, 1)
_elliptic_Pi_temp            = np.frompyfunc(mp.ellippi  , 3, 1)

def elliptic_Pi_complete(n,k):
    return complex(_elliptic_Pi_complete_temp(n,k))

def elliptic_Pi(n,phi,k):
    return complex(_elliptic_Pi_temp (n, phi, k))



###########################
# Jacobi elliptic function
###########################

_elliptic_sn_temp = np.frompyfunc(mp.ellipfun , 3, 1)

def elliptic_sn(phi,k):
    return complex(_elliptic_sn_temp( 'sn' , phi , k))


