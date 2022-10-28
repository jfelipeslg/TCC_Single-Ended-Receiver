import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import pandas as pd

from commpy.modulation import QAMModem, PSKModem

from optic.core import parameters
from optic.tx import simpleWDMTx
from optic.dsp import pulseShape, firFilter, decimate, symbolSync
from optic.carrierRecovery import cpr
from optic.metrics import fastBERcalc, monteCarloGMI, monteCarloMI, signal_power, hardDecision

from optic.plot import pconst
# from optic.models import phaseNoise, coherentReceiver, pdmCoherentReceiver, manakovSSF, hybrid_2x4_90deg, balancedPD, pbs, ssfm
# #from optic.equalization import edc, mimoAdaptEqualizer

#from utils.plot import pconst
from utils.equalization import edc, mimoAdaptEqualizer
from utils.models import phaseNoise, coherentReceiver, pdmCoherentReceiver, manakovSSF, hybrid_2x4_90deg, balancedPD, pbs, ssfm

def singleEnded_PD(Ein, Rd=1):
    """
    Single-Ended photodetector

    :param Ein: IQ signal
    :param R: photodiode responsivity [A/W][scalar, default: 1 A/W]
    
    :return: detected analog signals
    """
    assert Rd > 0, "PD responsivity should be a positive scalar"

    Eout = Rd*Ein*np.conj(Ein)
    
    return Eout

def SEReceiver(Es, Elo, Rd=1):
    """
    Single polarization single-ended coherent receiver (SER)

    :param Es: received signal field [nparray]
    :param Elo: LO field [nparray]
    :param Rd: photodiode responsivity [A/W][scalar, default: 1 A/W]

    :return: downconverted signal after single-ended photodetector
    """
    assert Rd > 0, "PD responsivity should be a positive scalar"
    assert Es.size == Elo.size, "Es and Elo need to have the same size"

    # optical 2 x 4 90° hybrid
    Eo = hybrid_2x4_90deg(Es, Elo)
    
    E1 = 2*Eo[1,:]  # select SER port and compensation
    E2 = 2*Eo[2,:]  # select SER port and compensation
    
    R1 = singleEnded_PD(E1, Rd)
    R2 = singleEnded_PD(E2, Rd)
    
    return R1, R2

def DFR(R1, R2, sigLO):
    """
    Direct Field Reconstruction (DFR)

    :param R1 and R2: ouput of SER [nparray]
    :param A: local oscillator (LO) [nparray]
    
    :return sig: the inphase and quadrature components of the optical field [nparray]
    """
    A = sigLO
    
    delta = 4*R1*R2 - (R1 + R2 - 2*A**2)**2
    
    sigI = - A/2 + 1/(4*A) * (R1 - R2) + 1/(4*A) * np.sqrt(delta)
    sigQ = - A/2 - 1/(4*A) * (R1 - R2) + 1/(4*A) * np.sqrt(delta)
      
    sigOut = sigI + 1j*sigQ
    
    return sigOut

list_withDFR = []
list_withoutDFR = []

rangePotLO = np.arange(1,21,3)

for activateDFR in (0,2):
    
    listPotLO = []
    listSNR = []
    listBER = []
    
    for auxPotLO in rangePotLO:
        # Parâmetros - Transmissor
        paramTx = parameters()
        
        paramTx.M   = 64            # Ordem do formato de modulação
        paramTx.constType = 'qam'   # Formato de modulação ('qam' ou 'psk')
        
        paramTx.Rs  = 100e9         # Taxa de sinalização [baud]
        paramTx.SpS = 4             # Número de amostras por símbolo
        Fs = paramTx.Rs*paramTx.SpS # Taxa de amostragem [Hz]
        
        paramTx.Nbits = 300000  # Número total de bits por polarização
        
        paramTx.pulse    = 'rrc'   # Tipo do filtro formatador de pulso
        paramTx.Ntaps    = 1024    # Número de coeficientes do filtro
        paramTx.alphaRRC = 0.01    # Rolloff do pulso RRC
        
        paramTx.Pch_dBm  = 0       # Potência média por canal WDM [dBm]
        paramTx.Nch      = 1       # Número de canais WDM
        paramTx.freqSpac = 40.0e9  # Espaçamento WDM
        
        paramTx.Fc     = 193.1e12  # Frequência central de operação
        paramTx.Nmodes = 1         # Número de modos de polarização
        
        Tx = ['Parâmetros do Transmissor:',
              '- Formato de Modulação: {}-{}' .format(paramTx.M, paramTx.constType.upper()),
              '- Taxa de Sinalização: {} Gbaud' .format(paramTx.Rs*1e-9),
              '- Taxa de Amostragem: {} GHz' .format(Fs*1e-9),
              '- Potência Média Enviada: {} dBm' .format(paramTx.Pch_dBm),
              '- Frequência de Operação: {} THz' .format(paramTx.Fc*1e-12),
              '- Número de polarizações do sinal: {}\n' .format(paramTx.Nmodes)]
        for i in Tx:
            print(i)
        
        # generate WDM signal
        sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)
        
        # Canal Óptico
        paramCh = parameters()
        
        paramCh.Ltotal = 200     # Distância total do enlace [km]
        paramCh.Lspan  = 20      # Passo da distância [km]
        
        paramCh.alpha = 0.2      # Parâmetro de perda da fibra [dB/km]
        paramCh.D = 16           # Parâmetro de dispersão da fibra [ps/nm/km]
        paramCh.gamma = 1.3      # Parâmetro não-linear da fibra [1/(W.km)]
        
        paramCh.Fc = paramTx.Fc  # Frequência da portadora do sinal WDM [Hz]
        paramCh.hz = 0.1         # Step-size do SSFM [km]
        
        if not paramCh.Ltotal:
            print('back-to-back connection.')
            sigWDM = sigWDM_Tx
            
        else:
            print(f'distance: {paramCh.Ltotal:.2f} km')
            print('symmetric, single-pol.')
            sigWDM, paramCh = ssfm(sigWDM_Tx, Fs, paramCh)
            sigWDM = sigWDM.reshape(len(sigWDM),1)
            
        channel = ['Enlace óptico:',
              '- Distância percorrida: {:.2f} km' .format(paramCh.Ltotal),
              '- Parâmetro de perda da fibra: {} dB/km' .format(paramCh.alpha),
              '- Parâmetro de dispersão da fibra: {} ps/nm/km' .format(paramCh.D),
              '- Parâmetro não-linear da fibra: {} 1/(W.km)\n' .format(paramCh.gamma)]
        
        # Parâmetros - Receptor
        mod = QAMModem(m=paramTx.M)
        
        chIndex  = 0     # Posição do canal WDM a ser demodulado
        Fc = paramTx.Fc
        Ts = 1/Fs
        
        freqGrid = paramTx.freqGrid
        print(f'Demodulating channel #{chIndex} \n',
              f'- fc: {((Fc + freqGrid[chIndex])/1e12):.4f} THz\n',
              f'- λ: {(const.c/(Fc + freqGrid[chIndex])/1e-9):.4f} nm\n')
        
        symbTx = symbTx_[:,:,chIndex]
        
        # Parâmetros - Receptor - Oscilador Local
        π       = np.pi
        FO      = 0*64e6                # Offset de frequência
        Δf_lo   = freqGrid[chIndex]+FO  # Deslocamento de frequência do canal a ser demodulado
        lw      = 0*100e3               # LineWidth
        Plo_dBm = auxPotLO                   # Potência [dBm]
        Plo     = 10**(Plo_dBm/10)*1e-3 # Potência [W]
        ϕ_lo    = 0                     # Fase inicial [rad]    
        
        print(f'Local oscillator\n',
              f'- Power: {Plo_dBm:.2f} dBm\n',
              f'- LineWidth: {(lw/1e3):.2f} kHz\n',
              f'- Frequency offset: {(FO/1e6):.2f} MHz')
        
        # Geração do sinal LO
        t       = np.arange(0, len(sigWDM))*Ts
        ϕ_pn_lo = phaseNoise(lw, len(sigWDM), Ts)
        sigLO   = np.sqrt(Plo)*np.exp(1j*(2*π*Δf_lo*t + ϕ_lo + ϕ_pn_lo))
        
        # Receptor óptico single-ended
        sigWDM = sigWDM.reshape(len(sigWDM),)
        R1, R2 = SEReceiver(sigWDM, sigLO, Rd=1)
        
        withDFR = activateDFR
        if withDFR:
            # Algoritmo de reconstrução de sinal (DFR)
            sigDFR = DFR(R1, R2, sigLO)
            sigDFR = sigDFR - np.mean(sigDFR)
        
            sigRx = sigDFR
        else:
            # Sinal recuperado após os PDs sem DFR
            sigPD = R1 + 1j*R2
            sigPD = sigPD - np.mean(sigPD)
            
            sigRx = sigPD
        
        # Filtro casado
        pulse = pulseShape('rrc', paramTx.SpS, N=paramTx.Ntaps, alpha=paramTx.alphaRRC, Ts=1/paramTx.Rs)
        pulse = pulse/np.max(np.abs(pulse))
        
        sigRx = firFilter(pulse, sigRx)
        
        # Compensação da disperção cromática
        if paramCh.Ltotal:
            sigRx = edc(sigRx, paramCh.Ltotal, paramCh.D, Fc-Δf_lo, Fs)
            
        # Redução de amostras por símbolos
        paramDec = parameters()
        
        paramDec.SpS_in  = paramTx.SpS
        paramDec.SpS_out = 2
        
        sigRx = sigRx.reshape(len(sigRx), 1)
        sigRx = decimate(sigRx, paramDec)
        symbRx = symbolSync(sigRx, symbTx, 2)
        
        x = sigRx
        d = symbRx
        
        x = x.reshape(len(x),1)/np.sqrt(signal_power(x))
        d = d.reshape(len(d),1)/np.sqrt(signal_power(d))
        
        
        # Equalizador
        paramEq = parameters()
        paramEq.nTaps = 15
        paramEq.SpS   = 2
        paramEq.mu    = [3e-3, 4e-3]
        paramEq.numIter = 5
        paramEq.storeCoeff = False
        paramEq.alg   = ['nlms','da-rde']
        paramEq.M     = paramTx.M
        paramEq.L = [int(0.20*len(x)/2), int(0.80*len(x)/2)]
        
        y_EQ, H, errSq, Hiter = mimoAdaptEqualizer(x, dx=d, paramEq=paramEq)
        
        
        # Recuperador de fase
        paramCPR = parameters()
        paramCPR.alg = 'bps'
        paramCPR.M   = paramTx.M
        paramCPR.N   = 35
        paramCPR.B   = 64
        paramCPR.pilotInd = np.arange(0, len(y_EQ), 20) 
        
        y_CPR, θ = cpr(y_EQ, symbTx=d, paramCPR=paramCPR)
        
        y_CPR = y_CPR/np.sqrt(signal_power(y_CPR))
        
        # Métricas
        discard = 5000
        for k in range(y_CPR.shape[1]):
            rot = np.mean(d[:,k]/y_CPR[:,k])
            y_CPR[:,k] = rot*y_CPR[:,k]
        
        y_CPR = y_CPR/np.sqrt(signal_power(y_CPR))
        
        
        ind = np.arange(discard, d.shape[0]-discard)
        BER, SER, SNR = fastBERcalc(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
        GMI,_    = monteCarloGMI(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
        MI       = monteCarloMI(y_CPR[ind,:], d[ind,:], paramTx.M, 'qam')
        
        polx = [SER[0],BER[0],SNR[0],MI[0],GMI[0]]
        results = np.array([polx])
        
        for i in Tx:
            print(i)
        for i in channel:
            print(i)
        
        df2 = pd.DataFrame(results, index=['Pol. X'], columns=['BER', 'SER', 'SNR [dB]', 'GMI [bits]', 'MI [bits]'])
        df2.style.format({"BER": "{:.2e}", "SER": "{:.2e}", "SNR [dB]": "{:.2f}", "GMI [bits]": "{:.2f}", "MI [bits]": "{:.2f}"})
        
        listPotLO.append(Plo_dBm)
        listSNR.append(SNR[0])
        listBER.append(BER[0])
    
    if activateDFR:
        for i in range(len(listPotLO)):
            list_withDFR.append([listPotLO[i], listSNR[i], listBER[i]])
    else:
        for i in range(len(listPotLO)):
            list_withoutDFR.append([listPotLO[i], listSNR[i], listBER[i]])

#
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(np.array(list_withoutDFR)[:,0], np.array(list_withoutDFR)[:,2], 'g-*')
ax2.plot(np.array(list_withoutDFR)[:,0], np.array(list_withoutDFR)[:,1], 'b-*')
ax1.set_xlabel('Pot LO [dBM]')
ax1.set_ylabel('BER', color='g')
ax1.set_yscale('log')
ax2.set_ylabel('SNR', color='b')
plt.title('%.f km | DFR desativado' %(paramCh.Ltotal if paramCh.Ltotal else 0))
plt.grid()
plt.savefig('%.fkm_desativadoDFR.png' %(paramCh.Ltotal if paramCh.Ltotal else 0))
plt.show()

#
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(np.array(list_withDFR)[:,0], np.array(list_withDFR)[:,2], 'g-*')
ax2.plot(np.array(list_withDFR)[:,0], np.array(list_withDFR)[:,1], 'b-*')
ax1.set_xlabel('Pot LO [dBM]')
ax1.set_ylabel('BER', color='g')
ax1.set_yscale('log')
ax2.set_ylabel('SNR', color='b')
plt.title('%.f km | DFR ativado' %(paramCh.Ltotal if paramCh.Ltotal else 0))
plt.grid()
plt.savefig('%.fkm_ativadoDFR.png' %(paramCh.Ltotal if paramCh.Ltotal else 0))
plt.show()

#
plt.figure(figsize=(10,6))
plt.plot(np.array(list_withoutDFR)[:,0], np.array(list_withoutDFR)[:,2], 'g-o', linewidth=2.0, label= 'without DFR')
plt.plot(np.array(list_withDFR)[:,0], np.array(list_withDFR)[:,2], 'b-o',linewidth=2.0, label= 'with DFR')
plt.title('%.f km | Comparação da BER' %(paramCh.Ltotal if paramCh.Ltotal else 0), fontsize=16.0, fontstyle="oblique")
plt.ylabel('BER', fontsize=16.0, fontstyle="oblique")
plt.xlabel('Potência do LO [dBm]', fontsize=16.0, fontstyle="oblique")
plt.legend(loc="upper left", fontsize=12.0)
plt.yscale('log')
plt.grid()
plt.savefig('%.fkm_comparaBER.png' %(paramCh.Ltotal if paramCh.Ltotal else 0))
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(20,4))

ax2 = axs[0].twinx()
axs[0].plot(np.array(list_withoutDFR)[:,0], np.array(list_withoutDFR)[:,2], 'g-*')
ax2.plot(np.array(list_withoutDFR)[:,0], np.array(list_withoutDFR)[:,1], 'b-*')
axs[0].set_xlabel('Pot LO [dBM]')
axs[0].set_ylabel('BER', color='g')
axs[0].set_yscale('log')
ax2.set_ylabel('SNR', color='b')
axs[0].set_title('%.f km | DFR desativado' %(paramCh.Ltotal if paramCh.Ltotal else 0))
axs[0].grid()

ax3 = axs[1].twinx()
axs[1].plot(np.array(list_withDFR)[:,0], np.array(list_withDFR)[:,2], 'g-*')
ax3.plot(np.array(list_withDFR)[:,0], np.array(list_withDFR)[:,1], 'b-*')
axs[1].set_xlabel('Pot LO [dBM]')
axs[1].set_ylabel('BER', color='g')
axs[1].set_yscale('log')
ax3.set_ylabel('SNR', color='b')
axs[1].set_title('%.f km | DFR ativado' %(paramCh.Ltotal if paramCh.Ltotal else 0))
axs[1].grid()

axs[2].plot(np.array(list_withoutDFR)[:,0], np.array(list_withoutDFR)[:,2], 'k-o', linewidth=2.0, label= 'without DFR')
axs[2].plot(np.array(list_withDFR)[:,0], np.array(list_withDFR)[:,2], 'r-o',linewidth=2.0, label= 'with DFR')
axs[2].set_title('%.f km | Comparação da BER' %(paramCh.Ltotal if paramCh.Ltotal else 0), fontsize=12.0, fontstyle="oblique")
axs[2].set_ylabel('BER', fontsize=12.0, fontstyle="oblique")
axs[2].set_xlabel('Potência do LO [dBm]', fontsize=12.0, fontstyle="oblique")
axs[2].legend(loc="best", fontsize=12.0)
axs[2].set_yscale('log')
axs[2].grid()

plt.tight_layout()
plt.savefig('%.f km' %(paramCh.Ltotal if paramCh.Ltotal else 0))



