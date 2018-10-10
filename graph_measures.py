import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
import scipy.signal as sig
import scipy.stats as stats
import statsmodels.tsa.stattools
import scipy.linalg as lin
import math
from scipy.fftpack import fft, fftfreq
import itertools
from scipy.stats import entropy


# plot degree distribution of graph
def degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree().items()], reverse = True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.plot(deg, cnt, '.')

    plt.title("Degree Distribution")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d+0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.show()

# graph similiarity index
def jaccard(A, B):
    andsum = sum([sum([i and j for i,j in zip(a,b)]) for a,b in zip(A,B)])
    orsum = sum([sum([i or j for i,j in zip(a,b)]) for a,b in zip(A,B)])
    if orsum == 0:
        return 1.0
    else: 
        return andsum/orsum

#graph similarity index
def cosine(A,B):
    mag = lambda x: np.sqrt(np.dot(x,x))
    test_z = lambda arr: all(x == 0 for x in itertools.chain(*arr))
    if test_z(A) or test_z(B):
        if test_z(A) and test_z(B):
            return 1.0
        else:
            return 0.0
    else:
        return 1 - (sum(np.nan_to_num([1 - (np.dot(a,b)/(mag(a)*mag(b))) for a,b in zip(A,B)]))/len(A))

#graph similarity index
def distance(A,B):
    dis = sum([sum(x) for x in abs(A-B)])
    n = len(A)
    return max((n**2 - dis - n)/n**2 * n/(n-1) , 0)

#applies function to all undirected pairs of nodes
def cross_func(states, func):
    M = np.zeros((len(states), len(states)))
    for i in range(len(states)):
        M[i,i] = 0
        for j in range(i+1,len(states)):
            val = func(states[i,:],states[j,:])
            M[i,j] = val
            M[j,i] = val
    return M

#applies function to all pairs of nodes
def dir_cross_func(states, func):
    M = np.zeros((len(states), len(states)))
    for i in range(len(states)):
        for j in range(len(states)):
            val = func(states[i,:],states[j,:])
            M[i,j] = val
        M[i,i] = 0
    return M

#phase synchrony measure
def phase_synchrony(X,Y):
    ps = np.mean(abs((X+Y)))/2
    return ps

#correlation
correlation = lambda X,Y: np.real(np.correlate(X,Y)/len(X))

#coherence
coherence = lambda X,Y: np.mean(sig.coherence(X,Y)[1])

#mutual information measure using KNN Kraskov 2004
def kraskov_mi(X, Y, k = 1, est = 1):
    from scipy.special import digamma
    
    n = len(X)
    dx = np.zeros((n,n))
    dy = np.zeros((n,n))
    dz = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            dx[i,j] = np.sqrt((X[i] - X[j])**2)
            dy[i,j] = np.sqrt((Y[i] - Y[j])**2)
            dz[i,j] = max(dx[i,j],dy[i,j])
    
    nx = np.zeros((n,1))
    ny = np.zeros((n,1))
    
    for i in range(n):
        dxi = dx[i,:]
        dyi = dy[i,:]
        dzi = dz[i,:]
        
        epsi = np.sort(dzi)
        
        if est == 1:
            nx[i] = sum([1 for x in dxi if x < epsi[k]])
            ny[i] = sum([1 for y in dyi if y < epsi[k]])
        elif est == 2: 
            nx[i] = sum([1 for x in dxi if x <= epsi[k]])
            ny[i] = sum([1 for y in dyi if y <= epsi[k]])
        else:
            raise ValueError('est must be either 1 or 2')
    
    if est == 1:
        return digamma(k) - sum(digamma(nx) + digamma(ny))/n + digamma(n)
    else:
        return digamma(k) - 1/k - sum(digamma(nx-1) + digamma(ny-1))/n + digamma(n)

#transfer entropy of X onto Y
def kernel_TE(X, Y):
   
    xn = X[:-1]
    yn = Y[:-1]
    yn1 = Y[1:]
    n = len(xn)
    density = np.ptp(xn)*np.ptp(yn)*np.ptp(yn1)/n
    
    gauss_ker = lambda x,o: np.exp(-.5*(x/o)**2)/(o*np.sqrt(2*np.pi))
    oxn = .2*np.std(xn)*n**.2
    oyn = .2*np.std(yn)*n**.2
    oyn1 = .2*np.std(yn1)*n**.2
    TE = 0
    
    xnk = np.array([gauss_ker(xn-xi,oxn) for xi in xn])
    ynk = np.array([gauss_ker(yn-yi,oxn) for yi in yn])
    yn1k = np.array([gauss_ker(yn1-y1i,oxn) for y1i in yn1])
    
    xnk = xnk
    ynk = ynk
    yn1k = yn1k

    for i in range(n):
        pxnynyn1 = sum(xnk[i]*ynk[i]*yn1k[i])/n
        pxnyn = sum(xnk[i]*ynk[i])/n
        pynyn1 = sum(ynk[i]*yn1k[i])/n
        pyn = sum(ynk[i])/n
        TE += pxnynyn1 * np.log2(pxnynyn1*pyn/(pxnyn*pynyn1))
    return TE

#granger causality of X onto Y
def granger_causality(X,Y,ml = 1): 
    results = statsmodels.tsa.stattools.grangercausalitytests(np.stack((Y,X),1), maxlag = ml, verbose = False)
    p_vals = [val[0]['params_ftest'][1] for key,val in results.items()]
    return min(p_vals)

#pearson correlation coefficient
r2 = lambda X,Y: (stats.pearsonr(X,Y)[0])**2

#nonlinear correlation ratio based on kernel estimate of function
def n2(X,Y):
    ind = [i[0] for i in sorted(enumerate(X), key=lambda x:x[1])]
    X = X[ind]
    Y = Y[ind]
    gauss_ker = lambda x,o: np.exp(-((x/o)**2)/2)/(o*np.sqrt(2*np.pi))
    SS_y = sum([(y - np.mean(Y))**2 for y in Y])
    o = .2*np.std(Y)*len(Y)**.2
    SS_res = sum([(y - sum([Y[i]*gauss_ker(X-x,o)[i] for i in range(len(Y))])/sum(gauss_ker(X-x,o)))**2 for x,y in zip(X,Y)])
    n2 = 1 - SS_res/SS_y
    return n2

#partial based methods
def partial_method(X, method):
    nvar = len(X)
    M = np.zeros((nvar, nvar))
    for i in range(nvar):
        M[i, i] = 0
        for j in range(i+1, nvar):
            
            A = np.transpose(np.delete(X,[i,j],0))
            w_i = lin.lstsq(A,X[i])[0]
            w_j = lin.lstsq(A, X[j])[0]
            
            r_j = X[i] - np.dot(A,w_i)
            r_i = X[j] - np.dot(A,w_j)
            val = method(r_i, r_j)
            
            #test for nan
            if val == val:
                M[i, j] = val
                M[j, i] = val
            else:
                M[i, j] = 0
                M[j, i] = 0  
    return M

#return coefficients for MVAR model
def MVAR_fit(X,p):
    v, n = np.shape(X)
    
    cov = np.zeros((p+1, v, v))
    for i in range(p+1):
        cov[i] = np.cov(X[:,0:n-p],X[:,i:n-p+i])[v:,v:]
        
    G = np.zeros((p*v,p*v))
    for i in range(p):
        for j in range(p):
            G[v*i:v*(i+1) , v*j:v*(j+1)] = cov[abs(j-i)]
    
    cov_list = np.concatenate(cov[1:],axis=0)
    phi = lin.lstsq(G, cov_list)[0]
    phi = np.reshape(phi,(p,v,v))
    for k in range(p):
        phi[k] = phi[k].T
    return phi

#spectral density helper function
def spectral_density(A, n_fft=None):
    p, N, N = A.shape
    if n_fft is None:
        n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)
    A2 = np.zeros((n_fft, N, N))
    A2[1:p + 1, :, :] = A  # start at 1 !
    fA = fft(A2, axis=0)
    freqs = fftfreq(n_fft)
    I = np.eye(N)

    for i in range(n_fft):
        fA[i] = lin.inv(I - fA[i])

    return fA, freqs

#directed transfer function
def DTF(A, sigma=None, n_fft=None):
    p, N, N = A.shape

    if n_fft is None:
        n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)

    H, freqs = spectral_density(A, n_fft)
    D = np.zeros((n_fft, N, N))

    if sigma is None:
        sigma = np.ones(N)

    for i in range(n_fft):
        S = H[i]
        V = (S * sigma[None, :]).dot(S.T.conj())
        V = np.abs(np.diag(V))
        D[i] = np.abs(S * np.sqrt(sigma[None, :])) / np.sqrt(V)[:, None]

    return D, freqs

#partial directed coherence
def PDC(A, sigma=None, n_fft=None):
    p, N, N = A.shape

    if n_fft is None:
        n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)

    H, freqs = spectral_density(A, n_fft)
    P = np.zeros((n_fft, N, N))

    if sigma is None:
        sigma = np.ones(N)

    for i in range(n_fft):
        B = H[i]
        B = lin.inv(B)
        V = np.abs(np.dot(B.T.conj(), B * (1. / sigma[:, None])))
        V = np.diag(V)  # denominator squared
        P[i] = np.abs(B * (1. / np.sqrt(sigma))[None, :]) / np.sqrt(V)[None, :]

    return P, freqs

#PAC from Rehman
def CFCfilt(signal,freqForAmp,freqForPhase,fs,passbandRipl):
    numTimeSamples = np.size(signal)
    frqAmpSize = np.size(freqForAmp)
    frqPhaseSize = np.size(freqForPhase)
    oscillations = np.zeros((frqPhaseSize,frqAmpSize,numTimeSamples),dtype=np.complex64)
    Rp = 40*np.log10((1+passbandRipl)/(1-passbandRipl))
    for jj in np.arange(frqPhaseSize):
        for kk in np.arange(frqAmpSize):
            freq = freqForAmp[kk] # Center Frequency
            delf = freqForPhase[jj] # Bandwidth
            if freq > 1.8*delf:
                freqBand = np.array([freq-1.2*delf, freq+1.2*delf])/(fs/2)
                bb, aa = sig.cheby1(3,Rp,freqBand,btype='bandpass')
            else:
                bb, aa = sig.cheby1(3,Rp,(freq+1.2*delf)/(fs/2))
            oscillation = sig.filtfilt(bb,aa,signal)
            oscillations[jj,kk,:] = sig.hilbert(oscillation)
    return oscillations

def preCFCProc(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw,passbandRipl):
    oscilsAmpMod = CFCfilt(sigForAmp,freqForAmp,freqForPhase,fs,passbandRipl)
    oscilsForPhase = CFCfilt(sigForPhase,freqForPhase,np.array([bw]),fs,passbandRipl)
    return oscilsAmpMod, oscilsForPhase

def PhaseLocVal(oscAmpMod,oscForPhase,freqForAmp,freqForPhase):
    PLVs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)))
    frqAmpSize = np.size(freqForAmp)
    frqPhaseSize = np.size(freqForPhase)
    for cc in np.arange(frqAmpSize):
        for rr in np.arange(frqPhaseSize):
            ampOsc = np.abs(oscAmpMod[rr,cc,:])
            phaseOsc = np.angle(oscForPhase[0,rr,:])
            ampOscPhase = np.angle(sig.hilbert(ampOsc))
            PLVs[rr,cc] = np.abs(np.mean(np.exp(1j*(phaseOsc - ampOscPhase))))
            delf = freqForPhase[rr] 
            ctrfreq = freqForAmp[cc]
    MIs = np.arcsin(2*PLVs-1)
    return MIs

def PLVcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=4.5,passbandRipl=0.02):
    oscAmpMod,oscForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=bw,passbandRipl=passbandRipl)
    MIs = PhaseLocVal(oscAmpMod,oscForPhase,freqForAmp,freqForPhase)
    return MIs

def GenLinMod(oscAmpMod,oscForPhase,freqForAmp,freqForPhase):
    ModCorr = np.zeros((np.size(freqForPhase),np.size(freqForAmp)))
    frqAmpSize = np.size(freqForAmp)
    frqPhaseSize = np.size(freqForPhase)
    for cc in np.arange(frqAmpSize):
        for rr in np.arange(frqPhaseSize):
            ampOsc = np.abs(oscAmpMod[rr,cc,:])
            phaseOsc = np.angle(oscForPhase[0,rr,:])
            X = np.matrix(np.column_stack((np.cos(phaseOsc),
                    np.sin(phaseOsc), np.ones(np.size(phaseOsc)))))
            B = np.linalg.inv((np.transpose(X)*X))* \
                    np.transpose(X)*np.transpose(np.matrix(ampOsc))
            ampOscTrend = X*B
            ampOscResid = ampOsc.flatten()-np.array(ampOscTrend).flatten()
            rsq = 1-np.var(ampOscResid)/np.var(ampOsc)
            ModCorr[rr,cc] = np.sqrt(rsq)
            delf = freqForPhase[rr]
            ctrfreq = freqForAmp[cc]
    MIs = np.arctanh(ModCorr)
    return MIs

def GLMcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=4.5,passbandRipl=0.02):
    oscAmpMod,oscForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=bw,passbandRipl=passbandRipl)
    MIs = GenLinMod(oscAmpMod,oscForPhase,freqForAmp,freqForPhase)
    return MIs

def PrinCompAnal(MultChannIn):
    MultChannInCov = np.cov(MultChannIn)
    PrinVals, PrinComps = np.linalg.eig(MultChannInCov)
    return PrinVals, PrinComps

def zScoredMVL(TwoChannIn):
    XPrinVals, XPrinComps = PrinCompAnal(TwoChannIn)
    meanVect = np.array([np.mean(TwoChannIn[0,:]), np.mean(TwoChannIn[1,:])])
    theta = np.arccos(np.dot(meanVect,XPrinComps[0,:])/np.linalg.norm(meanVect))
    R = np.sqrt((np.sqrt(XPrinVals[0])*np.cos(theta))**2+(np.sqrt(XPrinVals[1])*np.sin(theta))**2)
    zScore = np.linalg.norm(meanVect)/R
    return zScore

def zScoredMV_PCA(oscAmpMod,oscForPhase,freqForAmp,freqForPhase):
    MIs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)))
    MVLs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)))
    frqAmpSize = np.size(freqForAmp)
    frqPhaseSize = np.size(freqForPhase)
    for cc in np.arange(frqAmpSize):
        for rr in np.arange(frqPhaseSize):
            ampOsc = np.abs(oscAmpMod[rr,cc,:])
            phaseOsc = np.angle(oscForPhase[0,rr,:])
            phasor = ampOsc*np.exp(1j*phaseOsc)
            MVLs[rr,cc] = np.abs(np.mean(phasor))
            phasorComponents = np.row_stack((np.real(phasor), np.imag(phasor)))
            MIs[rr,cc] = zScoredMVL(phasorComponents)
            delf = freqForPhase[rr]
            ctrfreq = freqForAmp[cc]
    return MIs, MVLs

def zScoreMVcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=4.5,passbandRipl=0.02):
    oscAmpMod,oscForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,
        freqForPhase,fs,bw=bw,passbandRipl=passbandRipl)
    MIs, MVLs = zScoredMV_PCA(oscAmpMod,oscForPhase,freqForAmp,freqForPhase)
    return MIs, MVLs


def KullLeibDiv(P, Q = None):
    if Q == None: 
        KLDiv = np.log(np.size(P)) - entropy(P)
    else:
        KLDiv = entropy(P,Q)
    return KLDiv/np.size(P)

def KullLeibBin(oscAmpMod,oscForPhase,freqForAmp,freqForPhase,n):
    phaseBins = np.linspace(-np.pi,np.pi,n+1)
    highFreqAmplitude = np.zeros(n)
    MIs = np.zeros((np.size(freqForPhase),np.size(freqForAmp)))
    frqAmpSize = np.size(freqForAmp)
    frqPhaseSize = np.size(freqForPhase)
    for cc in np.arange(frqAmpSize):
        for rr in np.arange(frqPhaseSize):
            amplitudes = np.abs(oscAmpMod[rr,cc,:])
            phases = np.angle(oscForPhase[0,rr,:])
            for kk in np.arange(n):
                amps = amplitudes[(phases > phaseBins[kk]) & (phases <= phaseBins[kk+1])]
                highFreqAmplitude[kk] = np.mean(amps)
            MIs[rr,cc] = KullLeibDiv(highFreqAmplitude)
            delf = freqForPhase[rr]
            ctrfreq = freqForAmp[cc]
    return MIs

def KLDivMIcomod(sigForAmp,sigForPhase,freqForAmp,freqForPhase,fs,bw=4.5,passbandRipl=0.02,n=36):
    oscAmpMod,oscForPhase = preCFCProc(sigForAmp,sigForPhase,freqForAmp,
        freqForPhase,fs,bw=bw,passbandRipl=passbandRipl)
    MIs = KullLeibBin(oscAmpMod,oscForPhase,freqForAmp,freqForPhase,n)
    return MIs