import numpy as np

def Delta(radius,spin):
    return radius**2 + spin**2 - 2*radius

def RadialP(radius, spin, eta, lam):
    return  (radius**2 + spin**2 - spin*lam)**2 - Delta(radius,spin)*(eta + (lam-spin)**2)

def AngularP(theta,spin,eta,lam):
    return eta + (spin * np.cos(theta))**2 - (lam / np.tan(theta))**2

def get_photon_conserved_quantities(alpha,beta,theta,spin):

    lam = -alpha*np.sin(theta)
    eta = beta**2 + (alpha**2 - spin**2)*np.cos(theta)**2

    return lam,eta 