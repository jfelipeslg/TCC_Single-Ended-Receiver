## Functions - Alternatives for SSBI mitigation.

import numpy as np
import cmath
from optic.metrics import signal_power

def clippingComplex(signal, clippingValue):
    """
    Clipping complex signals

    :param signal: signal without clipping
    :param clippingValue: clipping value
    
    :return sigClipped: clipped signal
    """
    
    sigClipped = np.zeros(signal.shape, dtype="complex")
    
    for idx, idxValue in enumerate(signal):
        
        if np.abs(idxValue) > clippingValue:
            absIdx, angIdx = cmath.polar(idxValue)
            sigClipped[idx] = cmath.rect(clippingValue, angIdx)
            
        else:
            sigClipped[idx] = idxValue
        
    return sigClipped


def DFR(R1, R2, sigLO):
    """
    Direct Field Reconstruction (DFR)

    :param R1 and R2: ouput of SER [nparray]
    :param A: local oscillator (LO) [nparray]
    
    :return sigOut: the inphase and quadrature components of the optical field [nparray]
    """
    A = sigLO  # Oscilador local
    
    delta = 4*R1*R2 - (R1 + R2 - 2*A**2)**2    # Delta da função de segundo grau
    
    sigI = - A/2 + 1/(4*A) * (R1 - R2) + 1/(4*A) * np.sqrt(delta) # Cálculo da componente em fase (raiz da função)
    sigQ = - A/2 - 1/(4*A) * (R1 - R2) + 1/(4*A) * np.sqrt(delta) # Cálculo da componente em quadratura (raiz da função)
      
    sigOut = sigI + 1j*sigQ    # Sinal recuperado
    
    return sigOut


def IC(R1, R2, sigWDM, sigLO, N=20, clipping=True):
    """
    Clipped Iterative SSBI Cancellation (CIC)

    :param R1 and R2: ouput of SER [nparray]
    :param sigWDM: received signal [nparray]
    :param sigLO: local oscillator (LO) [nparray]
    :param N: number of iterations [scalar]
    
    :return sigOut: the inphase and quadrature components of the optical field [nparray]
    """
   
    A = sigLO                # Oscilador Local 
    P = signal_power(sigWDM) # Potência do sinal recebido
    
    U1 = (R1 - A**2) / (4*A**2) # Definindo U1 (Mudança de Variável)
    U2 = (R2 - A**2) / (4*A**2) # Definindo U2 (Mudança de Variável)
    
    overline_I = sigWDM.real / (2*A) # Define nova variável I
    overline_Q = sigWDM.imag / (2*A) # Define nova variável Q
        
    overline_I0 = U1 - P / (4*A**2) # Estimativa Inicial de I
    overline_Q0 = U2 - P / (4*A**2) # Estimativa Inicial de Q
    e0 = overline_I0 - overline_I   # Estimativa Inicial do erro estimado

    # Inicializando variáveis de iteração
    overline_In = overline_I0 # I(n) = I(0)
    overline_Qn = overline_Q0 # Q(n) = Q(0)
    error = e0                # ▲I(n) = ▲I(0)
    
    # Cálculo do nível de clipping
    if clipping == True:
        LOSPR = 10*np.log10( signal_power(sigLO) / P )                      # Cálculo da LOSPR
        optimumClip_dB  = LOSPR - 1                                         # Cálculo do valor ótimo de clipping (em dB)
        optimumClip_lin = 10**(optimumClip_dB/10)                           # Cálculo do valor ótimo de clipping (linear)
        clippingValue   = optimumClip_lin * np.sqrt(signal_power(sigWDM))   # Converter o valor de potência para amplitude

    # Iteração do algoritmo
    for nSteps in range(1, N): # Loop de N laços
        reductionSSBI  = (overline_In**2 + overline_Qn**2)      # Cálculo da iteração da SSBI
        
        if clipping == True:
            reductionSSBI  = clippingComplex( reductionSSBI, clippingValue)
        
        overline_Inext = U1 - reductionSSBI                     # Cálculo do valor de I(n+1)
        overline_Qnext = U2 - reductionSSBI                     # Cálculo do valor de Q(n+1)
        
        errorNext = - error * (overline_I + overline_Q + overline_In + overline_Qn) # Cálculo do valor de ▲I(n+1)
                
        overline_In = overline_Inext # Atualiza I(n+1) para I(n)
        overline_Qn = overline_Qnext # Atualiza Q(n+1) para Q(n)
        error = errorNext            # Atualiza ▲I(n+1) para ▲I(n)
    
    overline_sigOut = (overline_In + 1j*overline_Qn) # estimation signal
    sigOut = (2*A) * overline_sigOut                 # recovered signal w/ SSBI cancellation
    
    return sigOut

def gradientDescent(R1, R2, sigWDM, sigLO, mu = 0.05, N=150, clipping=True):
    """
    Gradient Descent (GD)

    :param R1 and R2: ouput of SER [nparray]
    :param sigWDM: received signal [nparray]
    :param sigLO: local oscillator (LO) [nparray]
    :param mu: step size [scalar]
    :param N: number of iterations [scalar]
    
    :return sigOut: the inphase and quadrature components of the optical field [nparray]
    """
    
    A = sigLO                # Oscilador Local 
    P = signal_power(sigWDM) # Potência do sinal recebido
    
    U1 = (R1 - A**2) / (4*A**2) # Definindo U1 (Mudança de Variável)
    U2 = (R2 - A**2) / (4*A**2) # Definindo U2 (Mudança de Variável)
    
    overline_I = sigWDM.real / (2*A) # Define nova variável I
    overline_Q = sigWDM.imag / (2*A) # Define nova variável Q
    
    overline_I0 = U1 - P / (4*A**2) # Estimativa Inicial de I
    overline_Q0 = U2 - P / (4*A**2) # Estimativa Inicial de Q
    
    # Inicializando variáveis de iteração
    overline_In = overline_I0 # I(n) = I(0)
    overline_Qn = overline_Q0 # Q(n) = Q(0)
    
    # Cálculo do nível de clipping
    if clipping == True:
        LOSPR = 10*np.log10( signal_power(sigLO) / signal_power(sigWDM) )       # Cálculo da LOSPR
        optimumClip_dB  = LOSPR + 4                                             # Cálculo do valor ótimo de clipping (em dB)
        optimumClip_lin = 10**(optimumClip_dB/10)                               # Cálculo do valor ótimo de clipping (linear)
        clippingValueI   = optimumClip_lin * np.sqrt(signal_power(sigWDM.real)) # Converter o valor de potência para amplitude
        clippingValueQ   = optimumClip_lin * np.sqrt(signal_power(sigWDM.imag)) # Converter o valor de potência para amplitude
    
    # Iteração do algoritmo
    for nSteps in range(1, N): # Loop de N laços
               
        X_InQn = overline_In**2 + overline_Qn**2 + overline_In - U1 # Cálculo do valor de X(In,Qn)
        Y_InQn = overline_In**2 + overline_Qn**2 + overline_Qn - U2 # Cálculo do valor de Y(In,Qn)
        
        G = X_InQn ** 2 + Y_InQn ** 2
            
        gradientI = (X_InQn * (2*overline_In + 1) + 2 * Y_InQn * overline_In) # Cálculo do gradiente de I
        gradientQ = (X_InQn * overline_Qn + 2 * Y_InQn * (2*overline_Qn + 1)) # Cálculo do gradiente de Q
        
        if clipping == True:
            gradientI  = clippingComplex( gradientI, clippingValueI)
            gradientQ  = clippingComplex( gradientQ, clippingValueQ)
            
        overline_Inext = overline_In - mu * gradientI # Cálculo do valor de I(n+1)
        overline_Qnext = overline_Qn - mu * gradientQ # Cálculo do valor de Q(n+1)
        
        error = 2 * np.sqrt( (overline_In - overline_I)**2 + (overline_Qn - overline_Q)**2 ) # Cálculo do erro normalizado ▲V(n)
        
        overline_In = overline_Inext # Atualiza I(n+1) para I(n)
        overline_Qn = overline_Qnext # Atualiza Q(n+1) para Q(n)
        
    overline_sigOut = (overline_In + 1j*overline_Qn) # estimation signal
    sigOut = (2*A) * overline_sigOut                 # recovered signal w/ SSBI cancellation
    
    return sigOut