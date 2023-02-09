## Functions - Optical Front End

import numpy as np
from optic.models import photodiode, pbs

def optical_hybrid(Es, Elo):
    """
    Adapted from optics.models.hybrid_2x4_90deg
    """
    assert Es.shape == (len(Es),), "Es need to have a (N,) shape"
    assert Elo.shape == (len(Elo),), "Elo need to have a (N,) shape"
    assert Es.shape == Elo.shape, "Es and Elo need to have the same (N,) shape"

    # optical hybrid transfer matrix
    T = np.array(
        [
            [1 / 2, 1j / 2, 1j / 2, -1 / 2],
            [1j / 2, -1 / 2, 1 / 2, 1j / 2],
            [1j / 2, 1 / 2, -1j / 2, -1 / 2],
            [-1 / 2, 1j / 2, -1 / 2, 1j / 2],
        ]
    ) * 2 # Compensation in accordance with the reference article 

    Ei = np.array([Es, np.zeros((Es.size,)), np.zeros((Es.size,)), Elo])

    return T @ Ei
    
def singleEnded_PD(Ein, paramPD=[]):
    """
    Single-Ended photodetector

    :param Ein: IQ signal
    :param paramPD: parameters of the photodiodes [struct]
    
    :return: detected analog signals
    """
    i_out = photodiode(Ein, paramPD)
    
    return i_out

def SEReceiver(Es, Elo, paramPD=[]):
    """
    Single polarization single-ended coherent receiver (SER)

    :param Es: received signal field [nparray]
    :param Elo: LO field [nparray]
    :param paramPD: parameters of the photodiodes [struct]

    :return: downconverted signal after single-ended photodetector
    """
    assert Es.shape == (len(Es),), "Es need to have a (N,) shape"
    assert Elo.shape == (len(Elo),), "Elo need to have a (N,) shape"
    assert Es.shape == Elo.shape, "Es and Elo need to have the same (N,) shape"

    # optical 2 x 4 90° hybrid
    Eo = optical_hybrid(Es, Elo)
    
    E1 = Eo[1,:]  # select SER port
    E2 = Eo[2,:]  # select SER port
    
    R1 = singleEnded_PD(E1, paramPD)
    R2 = singleEnded_PD(E2, paramPD)
    
    return R1, R2

def pdmSEReceiver(Es, Elo, θsig=0, paramPD=[]):
    """
    Dual polarization single-ended coherent receiver

    :param Es: received signal field [nparray]
    :param Elo: LO field [nparray]
    :param θsig: input polarization rotation angle [rad]
    :param paramPD: parameters of the photodiodes [struct]

    :return: downconverted signal after single-ended photodetector
    """
    assert len(Es) == len(Elo), "Es and Elo need to have the same length"
    
    Elox, Eloy = pbs(Elo, θ=np.pi/4 )  # split LO into two orth. polarizations
    Esx, Esy   = pbs(Es, θ=θsig)       # split signal into two orth. polarizations

    R1_polx, R2_polx = SEReceiver(Esx, Elox, paramPD)  # coherent detection of pol.X
    R1_poly, R2_poly = SEReceiver(Esy, Eloy, paramPD)  # coherent detection of pol.Y

    return R1_polx, R2_polx, R1_poly, R2_poly