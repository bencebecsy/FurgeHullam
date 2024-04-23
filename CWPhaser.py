# -*- coding: utf-8 -*-

import numpy as np
import scipy.special as scs
import scipy.optimize as sco
import scipy.linalg as sl

import numba as nb
from numba import njit,prange
#from numba_scipy import special as scs

from enterprise import constants as const
from enterprise.signals import deterministic_signals, gp_signals, signal_base, utils, parameter

from enterprise_extensions import blocks, deterministic

from scipy.interpolate import CubicSpline
from scipy.interpolate import RegularGridInterpolator
#from fast_interp import interp2d


def get_xCy(Nvec, T, sigmainv, x, y):
    """Get x^T C^{-1} y"""
    TNx = Nvec.solve(x, left_array=T)
    TNy = Nvec.solve(y, left_array=T)
    xNy = Nvec.solve(y, left_array=x)
    print(f"{xNy:g}, {TNx @ sigmainv @ TNy:g}")
    return xNy - TNx @ sigmainv @ TNy


def get_TCy(Nvec, T, y, sigmainv, TNT):
    """Get T^T C^{-1} y"""
    TNy = Nvec.solve(y, left_array=T)
    return TNy - TNT @ sigmainv @ TNy


def innerprod(Nvec, T, sigmainv, TNT, x, y):
    """Get the inner product between x and y"""
    xCy = get_xCy(Nvec, T, sigmainv, x, y)
    TCy = get_TCy(Nvec, T, y, sigmainv, TNT)
    TCx = get_TCy(Nvec, T, x, sigmainv, TNT)
    return xCy #- TCx.T @ sigmainv @ TCy

def innerprod_cho(Nvec, T, cf, x, y):
    TNx = Nvec.solve(x, left_array=T)
    TNy = Nvec.solve(y, left_array=T)
    xNy = Nvec.solve(y, left_array=x)

    expval = sl.cho_solve(cf, TNy)
    return xNy - TNx @ expval

class CWPhaser(object):
    """
    Class for the Fcp-statistic (aka coherent pulsar term F-statistic).

    :param psrs: List of `enterprise` Pulsar instances.
    :param noisedict: Dictionary of white noise parameter values. Default=None
    :param psrTerm: Include the pulsar term in the CW signal model. Default=True
    :param bayesephem: Include BayesEphem model. Default=True
    """

    def __init__(self, psrs, noisedict=None,
                 psrTerm=True, bayesephem=True, pta=None, tnequad=False):

        if pta is None:

            # initialize standard model with fixed white noise
            # and powerlaw red noise
            # uses the implementation of ECORR in gp_signals
            print('Initializing the model...')

            tmin = np.min([p.toas.min() for p in psrs])
            tmax = np.max([p.toas.max() for p in psrs])
            Tspan = tmax - tmin
            self.tref = tmin
            s = gp_signals.TimingModel(use_svd=True)
            s += deterministic.cw_block_circ(amp_prior='log-uniform',
                                             psrTerm=psrTerm, tref=self.tref, name='cw')
            #s += blocks.red_noise_block(prior='log-uniform', psd='powerlaw',
            #                            Tspan=Tspan, components=30)

            log10_A = parameter.Constant()
            gamma = parameter.Constant()

            # define powerlaw PSD and red noise signal
            pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
            s += gp_signals.FourierBasisGP(pl, components=30)

            if bayesephem:
                s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

            # adding white-noise, and acting on psr objects
            models = []
            for p in psrs:
                #if 'NANOGrav' in p.flags['pta']:
                if True: #always add ecorr for now as some of my simulated pulsars have no flags #TODO: fix this later
                    s2 = s + blocks.white_noise_block(vary=False, inc_ecorr=True,
                                                      gp_ecorr=False, tnequad=tnequad)
                                                      #gp_ecorr=True, tnequad=tnequad)
                    models.append(s2(p))
                else:
                    s3 = s + blocks.white_noise_block(vary=False, inc_ecorr=False, select=None, tnequad=tnequad)
                    models.append(s3(p))

            pta = signal_base.PTA(models)

            # set white noise parameters
            if noisedict is None:
                print('No noise dictionary provided!')
            else:
                pta.set_default_params(noisedict)

            self.pta = pta

        else:
            # user can specify their own pta object
            # if ECORR is included, use the implementation in gp_signals
            self.pta = pta
            self.tref = np.min([p.toas.min() for p in psrs])

        self.psrs = psrs
        self.n_psr = len(psrs)
        self.psr_poses = np.array([psr.pos for psr in psrs])
        self.noisedict = noisedict

        # precompute important bits:
        self.phiinvs = self.pta.get_phiinv({})
        #print(self.phiinvs)
        self.TNTs = self.pta.get_TNT({})
        self.Nvecs = self.pta.get_ndiag()
        self.Ts = self.pta.get_basis()
        self.cf_sigmas = [sl.cho_factor(TNT + np.diag(phiinv)) for TNT, phiinv in zip(self.TNTs, self.phiinvs)]
        self.sigmainvs = [np.linalg.pinv(TNT + np.diag(phiinv)) for TNT, phiinv in zip(self.TNTs, self.phiinvs)]

        self.fgw = None
        self.N = None
        self.M = None


    def calculate_M_N(self, fgw):
        self.N = np.zeros((len(self.psrs), 2))
        self.M = np.zeros((len(self.psrs), 2, 2))
        for ii, (psr, Nvec, TNT, T, sigmainv, cf) in enumerate(zip(self.psrs, self.Nvecs, self.TNTs, self.Ts, self.sigmainvs, self.cf_sigmas)):

            ntoa = len(psr.toas)

            A = np.zeros((2, ntoa))
            #A[0, :] = 1 / fgw ** (1 / 3) * np.sin(2 * np.pi * fgw * psr.toas)
            #A[1, :] = 1 / fgw ** (1 / 3) * np.cos(2 * np.pi * fgw * psr.toas)
            A[0, :] = np.sin(2 * np.pi * fgw * (psr.toas-self.tref))
            A[1, :] = np.cos(2 * np.pi * fgw * (psr.toas-self.tref))

            #self.N[ii,0] = innerprod(Nvec, T, sigmainv, TNT, A[0, :], psr.residuals)
            #self.N[ii,1] = innerprod(Nvec, T, sigmainv, TNT, A[1, :], psr.residuals)
            self.N[ii,0] = innerprod_cho(Nvec, T, cf, A[0, :], psr.residuals)
            self.N[ii,1] = innerprod_cho(Nvec, T, cf, A[1, :], psr.residuals)

            # define M matrix M_ij=(A_i|A_j)
            for jj in range(2):
                for kk in range(2):
                    #self.M[ii, jj, kk] = innerprod(Nvec, T, sigmainv, TNT, A[jj, :], A[kk, :])
                    self.M[ii, jj, kk] = innerprod_cho(Nvec, T, cf, A[jj, :], A[kk, :])

        #save fgw so we can keep track what N and M was calculated for
        self.fgw = fgw

    def calculate_M_N_evolve(self, fgw, fpsr):
        self.N = np.zeros((len(self.psrs), 4))
        self.M = np.zeros((len(self.psrs), 4, 4))
        for ii, (psr, Nvec, TNT, T, sigmainv) in enumerate(zip(self.psrs, self.Nvecs, self.TNTs, self.Ts, self.sigmainvs)):

            ntoa = len(psr.toas)

            A = np.zeros((4, ntoa))
            #A[0, :] = 1 / fgw ** (1 / 3) * np.sin(2 * np.pi * fgw * psr.toas)
            #A[1, :] = 1 / fgw ** (1 / 3) * np.cos(2 * np.pi * fgw * psr.toas)
            A[0, :] = np.sin(2 * np.pi * fgw * (psr.toas-self.tref))
            A[1, :] = np.cos(2 * np.pi * fgw * (psr.toas-self.tref))
            A[2, :] = np.sin(2 * np.pi * fpsr * (psr.toas-self.tref))
            A[3, :] = np.cos(2 * np.pi * fpsr * (psr.toas-self.tref))

            self.N[ii,0] = innerprod(Nvec, T, sigmainv, TNT, A[0, :], psr.residuals)
            self.N[ii,1] = innerprod(Nvec, T, sigmainv, TNT, A[1, :], psr.residuals)
            self.N[ii,2] = innerprod(Nvec, T, sigmainv, TNT, A[2, :], psr.residuals)
            self.N[ii,3] = innerprod(Nvec, T, sigmainv, TNT, A[3, :], psr.residuals)

            # define M matrix M_ij=(A_i|A_j)
            for jj in range(4):
                for kk in range(4): #TODO: don't calculate repeating terms multiple times
                    self.M[ii, jj, kk] = innerprod(Nvec, T, sigmainv, TNT, A[jj, :], A[kk, :])

        #save fgw so we can keep track what N and M was calculated for
        self.fgw = fgw

    def set_up_M_N_interpolators(self, fmin, fmax, n_f):
        fff = np.linspace(fmin, fmax, n_f)
        self.fff = fff
        df = fff[1]-fff[0]

        #self.N0_interps = [] #(sin|data)
        #self.N1_interps = [] #(cos|data)
        #self.M00_interps = [] #(sin|sin)
        #self.M11_interps = [] #(cos|cos)
        #self.M01_interps = [] #(sin|cos)
        self.N0s = np.zeros((len(self.psrs),n_f)) #(sin|data)
        self.N1s = np.zeros((len(self.psrs),n_f)) #(cos|data)
        self.M00s = np.zeros((len(self.psrs),n_f,n_f)) #(sin|sin)
        self.M11s = np.zeros((len(self.psrs),n_f,n_f)) #(cos|cos)
        self.M01s = np.zeros((len(self.psrs),n_f,n_f)) #(sin|cos)

        for ii, (psr, Nvec, T, cf) in enumerate(zip(self.psrs, self.Nvecs, self.Ts, self.cf_sigmas)):
            print(ii)
            ntoa = len(psr.toas)
            print(ntoa)

            NN = np.zeros((n_f, 2))
            MM = np.zeros((n_f,n_f, 3))

            for jj in range(n_f):
                S1 = np.sin(2 * np.pi * fff[jj] * (psr.toas-self.tref))
                C1 = np.cos(2 * np.pi * fff[jj] * (psr.toas-self.tref))

                #NN[jj,0] = innerprod(Nvec, T, sigmainv, TNT, S1, psr.residuals)
                #NN[jj,1] = innerprod(Nvec, T, sigmainv, TNT, C1, psr.residuals)
                NN[jj,0] = innerprod_cho(Nvec, T, cf, S1, psr.residuals)
                NN[jj,1] = innerprod_cho(Nvec, T, cf, C1, psr.residuals)

                for kk in range(n_f):
                    S2 = np.sin(2 * np.pi * fff[kk] * (psr.toas-self.tref))
                    C2 = np.cos(2 * np.pi * fff[kk] * (psr.toas-self.tref))

                    if jj>kk:
                        MM[jj,kk,0] = np.copy(MM[kk,jj,0])
                        MM[jj,kk,1] = np.copy(MM[kk,jj,1])
                    else:
                        #MM[jj,kk,0] = innerprod(Nvec, T, sigmainv, TNT, S1, S2) 
                        #MM[jj,kk,1] = innerprod(Nvec, T, sigmainv, TNT, C1, C2)
                        MM[jj,kk,0] = innerprod_cho(Nvec, T, cf, S1, S2)
                        MM[jj,kk,1] = innerprod_cho(Nvec, T, cf, C1, C2)
                    #MM[jj,kk,2] = innerprod(Nvec, T, sigmainv, TNT, S1, C2)
                    MM[jj,kk,2] = innerprod_cho(Nvec, T, cf, S1, C2)

            #self.N0_interps.append(CubicSpline(fff, NN[:,0]))
            #self.N1_interps.append(CubicSpline(fff, NN[:,1]))
            self.N0s[ii,:] = NN[:,0]#.astype('float32')
            self.N1s[ii,:] = NN[:,1]#.astype('float32')

            #self.M00_interps.append(RegularGridInterpolator((fff, fff), MM[:,:,0], method='cubic'))#, bounds_error=True, fill_value=0.0))
            #self.M11_interps.append(RegularGridInterpolator((fff, fff), MM[:,:,1], method='cubic'))#, bounds_error=True, fill_value=0.0))
            #self.M01_interps.append(RegularGridInterpolator((fff, fff), MM[:,:,2], method='cubic'))#, bounds_error=True, fill_value=0.0))
            #TODO: maybe switch to this, which can be ~10 times faster on a 100x100 grid interpolation: https://github.com/dbstein/fast_interp
            #print(MM.dtype)
            #self.M00_interps.append(interp2d([fmin,fmin], [fmax,fmax], [df,df], MM[:,:,0].astype('float32'), k=3,  p=[False,False], e=[1,1]))
            #self.M11_interps.append(interp2d([fmin,fmin], [fmax,fmax], [df,df], MM[:,:,1].astype('float32'), k=3,  p=[False,False], e=[1,1]))
            #self.M01_interps.append(interp2d([fmin,fmin], [fmax,fmax], [df,df], MM[:,:,2].astype('float32'), k=3,  p=[False,False], e=[1,1]))
            self.M00s[ii,:,:] = MM[:,:,0]#.astype('float32')
            self.M11s[ii,:,:] = MM[:,:,1]#.astype('float32')
            self.M01s[ii,:,:] = MM[:,:,2]#.astype('float32')

    def save_N_M_to_file(self, filename):
        np.savez(filename, fff=self.fff, N0s=self.N0s, N1s=self.N1s, M00s=self.M00s, M11s=self.M11s, M01s=self.M01s)

    def load_N_M_from_file(self, filename, single_precision=False):
        npzfile = np.load(filename)
        if single_precision:
            self.fff = npzfile["fff"].astype('float32')
            self.N0s = npzfile["N0s"].astype('float32')
            self.N1s = npzfile["N1s"].astype('float32')
            self.M00s = npzfile["M00s"].astype('float32')
            self.M11s = npzfile["M11s"].astype('float32')
            self.M01s = npzfile["M01s"].astype('float32')
        else:
            self.fff = npzfile["fff"]
            self.N0s = npzfile["N0s"]
            self.N1s = npzfile["N1s"]
            self.M00s = npzfile["M00s"]
            self.M11s = npzfile["M11s"]
            self.M01s = npzfile["M01s"]

    def get_log_L_evolve(self, x):
        """
        x --> (inc, theta, log10_A, log10_f, log10_mc, phhase0, phi, psi, (phases, psr_distances)x[Npsr])
        """

        """
        fpsrs, sigmas = fpsr_sigma_helper(x, self.n_psr, self.psr_poses)
        fgw = 10**x[3]

        Ns = np.zeros((self.n_psr,4))
        Ms = np.zeros((self.n_psr,4,4))
        for ii in range(self.n_psr):
            #Ns[ii,0] = self.N0_interps[ii](fgw)
            #Ns[ii,1] = self.N1_interps[ii](fgw)
            #Ns[ii,2] = self.N0_interps[ii](fpsrs[ii])
            #Ns[ii,3] = self.N1_interps[ii](fpsrs[ii])
            Ns[ii,0::2] = self.N0_interps[ii]([fgw,fpsrs[ii]])
            Ns[ii,1::2] = self.N1_interps[ii]([fgw,fpsrs[ii]])
    
            MM = np.zeros((4,4))
            #MM[0,0] = self.M00_interps[ii]([fgw,fgw])[0]
            #MM[1,1] = self.M11_interps[ii]([fgw,fgw])[0]
            #MM[2,2] = self.M00_interps[ii]([fpsrs[ii],fpsrs[ii]])[0]
            #MM[3,3] = self.M11_interps[ii]([fpsrs[ii],fpsrs[ii]])[0]
            #MM[0,1] = self.M01_interps[ii]([fgw,fgw])[0]
            #MM[2,3] = self.M01_interps[ii]([fpsrs[ii],fpsrs[ii]])[0]
            #MM[0,2] = self.M00_interps[ii]([fgw,fpsrs[ii]])[0]
            #MM[1,3] = self.M11_interps[ii]([fgw,fpsrs[ii]])[0]
            #MM[0,3] = self.M01_interps[ii]([fgw,fpsrs[ii]])[0]
            #MM[1,2] = self.M01_interps[ii]([fpsrs[ii],fgw])[0]
            #M00s = self.M00_interps[ii]([[fgw,fgw],[fpsrs[ii],fpsrs[ii]],[fgw,fpsrs[ii]]])
            #M00s = [self.M00_interps[ii](fgw,fgw), self.M00_interps[ii](fpsrs[ii],fpsrs[ii]), self.M00_interps[ii](fgw,fpsrs[ii])]
            M00s = np.zeros(3)
            self.M00_interps[ii](np.array([fgw,fpsrs[ii],fgw]), np.array([fgw,fpsrs[ii],fpsrs[ii]]), fout=M00s)
            MM[0,0] = M00s[0]
            MM[2,2] = M00s[1]
            MM[0,2] = M00s[2]
            #M11s = self.M11_interps[ii]([[fgw,fgw],[fpsrs[ii],fpsrs[ii]],[fgw,fpsrs[ii]]])
            #M11s = [self.M11_interps[ii](fgw,fgw), self.M11_interps[ii](fpsrs[ii],fpsrs[ii]), self.M11_interps[ii](fgw,fpsrs[ii])]
            M11s = np.zeros(3)
            self.M11_interps[ii](np.array([fgw,fpsrs[ii],fgw]), np.array([fgw,fpsrs[ii],fpsrs[ii]]), fout=M11s)
            MM[1,1] = M11s[0]
            MM[3,3] = M11s[1]
            MM[1,3] = M11s[2]
            #M01s = self.M01_interps[ii]([[fgw,fgw],[fpsrs[ii],fpsrs[ii]],[fgw,fpsrs[ii]],[fpsrs[ii],fgw]])
            #M01s = [self.M01_interps[ii](fgw,fgw), self.M01_interps[ii](fpsrs[ii],fpsrs[ii]), self.M01_interps[ii](fgw,fpsrs[ii]), self.M01_interps[ii](fpsrs[ii],fgw)]
            M01s = np.zeros(4)
            self.M01_interps[ii](np.array([fgw,fpsrs[ii],fgw,fpsrs[ii]]), np.array([fgw,fpsrs[ii],fpsrs[ii],fgw]), fout=M01s)
            MM[0,1] = M01s[0]
            MM[2,3] = M01s[1]
            MM[0,3] = M01s[2]
            MM[1,2] = M01s[3]

            #fill in lower diagonal of MM matrix
            Ms[ii,:,:] = MM + MM.T - np.diag(MM.diagonal())

        return log_L_helper_evolve(sigmas, Ns, Ms, self.n_psr, 0.0, 0.0)
        """
        return log_L_helper_evolve_old(x, self.n_psr, self.psr_poses, self.fff, self.N0s, self.N1s, self.M00s, self.M11s, self.M01s,
                                       0.0, 0.0) #set resres and logdet to 0 for now - TODO:fix them

    def get_log_L(self, fgw, x):
        """
        x --> (inc, theta, log10_A, phhase0, phi, psi, phases[Npsr])
        """
        #if self.N is None or self.fgw!=fgw:
        #    self.calculate_M_N(fgw)
        #assert self.fgw==fgw
        assert np.isclose(self.fgw,fgw)

        return log_L_helper(x, self.fgw, self.n_psr, self.psr_poses, self.N, self.M, 0.0, 0.0) #set resres and logdet to 0 for now - TODO:fix them

    def get_phase_marg_log_L(self, fgw, x_com):
        """
        x_com --> (inc, theta, log10_A, phhase0, phi, psi)
        """
        assert self.fgw==fgw

        return phase_marg_log_L_helper(x_com, self.fgw, self.n_psr, self.psr_poses, self.N, self.M, 0.0, 0.0) #set resres and logdet to 0 for now - TODO:fix them


    def get_incoherent_log_L(self, fgw, x):
        """
        #x --> (log10_A, (As, phases)x[Npsr])
        x --> (log10_As, phases)x[Npsr]
        """
        assert self.fgw==fgw

        return incoherent_log_L_helper(x, self.fgw, self.n_psr, self.N, self.M, 0.0, 0.0) #set resres and logdet to 0 for now - TODO:fix them

    def get_phase_marg_incoherent_log_L(self, fgw, x_com):
        """
        #x_com --> (log10_A, As[Npsr])
        x_com --> (log10_As[Npsr])
        """
        assert self.fgw==fgw

        return incoherent_phase_marg_log_L_helper(x_com, self.fgw, self.n_psr, self.N, self.M, 0.0, 0.0) #set resres and logdet to 0 for now - TODO:fix them

@njit(fastmath=False)
def fpsr_sigma_helper(x, n_psr, psr_poses):
    inc = x[0]
    theta = x[1]
    A = 10**x[2]
    fgw = 10**x[3]
    mc = 10**x[4] * const.Tsun
    phase0 = x[5]
    phi = x[6]
    psi = x[7]
    phases = np.copy(x[8::2])
    pdists = np.copy(x[9::2])

    #print(x)

    amp = A/(2*np.pi*fgw)

    cos_inc = np.cos(inc)
    one_plus_cos_inc_sq = 1+cos_inc**2
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_2psi = np.cos(2*psi)
    sin_2psi = np.sin(2*psi)
    cos_phase0 = np.cos(phase0)
    sin_phase0 = np.sin(phase0)
    cos_phases = np.cos(2*phases+phase0)
    sin_phases = np.sin(2*phases+phase0)

    m = np.array([sin_phi, -cos_phi, 0.0])
    n = np.array([-cos_theta * cos_phi, -cos_theta * sin_phi, sin_theta])
    omhat = np.array([-sin_theta * cos_phi, -sin_theta * sin_phi, -cos_theta])

    fpsrs = np.zeros(n_psr)
    sigmas = np.zeros((n_psr,4))
    for ii in range(n_psr):
    #for ii in prange(n_psr):
        m_pos = 0.
        n_pos = 0.
        cosMu = 0.
        for j in range(0,3):
            m_pos += m[j]*psr_poses[ii,j]
            n_pos += n[j]*psr_poses[ii,j]
            cosMu -= omhat[j]*psr_poses[ii,j]

        F_p = 0.5 * (m_pos ** 2 - n_pos ** 2) / (1 - cosMu)
        F_c = (m_pos * n_pos) / (1 - cosMu)

        p_dist = pdists[ii]*(const.kpc/const.c)

        w0 = np.pi * fgw
        omega_p0 = w0 *(1 + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)

        amp_psr = amp * (w0/omega_p0)**(1.0/3.0)
        #phase0_psr = phases[ii]
        
        cos_phase0_psr = cos_phases[ii]#np.cos(phase0+phase0_psr*2.0)
        sin_phase0_psr = sin_phases[ii]#np.sin(phase0+phase0_psr*2.0)

        sigmas[ii,0] =  amp*(   cos_phase0 * one_plus_cos_inc_sq * (-cos_2psi * F_p + sin_2psi * F_c) +    #Earth term sine
                             2*sin_phase0 *     cos_inc          * (+sin_2psi * F_p + cos_2psi * F_c)   )
        sigmas[ii,1] =  amp*(   sin_phase0 * one_plus_cos_inc_sq * (-cos_2psi * F_p + sin_2psi * F_c) +    #Earth term cosine
                             2*cos_phase0 *     cos_inc          * (-sin_2psi * F_p - cos_2psi * F_c)   )
        sigmas[ii,2] =  -amp_psr*(   cos_phase0_psr * one_plus_cos_inc_sq * (-cos_2psi * F_p + sin_2psi * F_c) +    #Pulsar term sine
                                  2*sin_phase0_psr *     cos_inc          * (+sin_2psi * F_p + cos_2psi * F_c)   )
        sigmas[ii,3] =  -amp_psr*(   sin_phase0_psr * one_plus_cos_inc_sq * (-cos_2psi * F_p + sin_2psi * F_c) +    #Pulsar term cosine
                                  2*cos_phase0_psr *     cos_inc          * (-sin_2psi * F_p - cos_2psi * F_c)   )

        fpsrs[ii] = omega_p0/np.pi#float(omega_p0/np.pi)

    return fpsrs, sigmas

@njit(fastmath=False)
def log_L_helper_evolve(sigmas, Ns, Ms, n_psr, resres, logdet):
    logL = -0.5*resres -0.5*logdet
    for ii in range(n_psr):
        for jj in range(4):
            logL += sigmas[ii,jj] * Ns[ii,jj]
            for kk in range(4):
                logL += -0.5*sigmas[ii,jj]*sigmas[ii,kk] * Ms[ii,jj,kk]

    return logL


@njit(fastmath=False, parallel=False)
def log_L_helper_evolve_old(x, n_psr, psr_poses, fff, NN0s, NN1s, MM00s, MM11s, MM01s, resres, logdet):
    inc = x[0]
    theta = x[1]
    A = 10**x[2]
    fgw = 10**x[3]
    mc = 10**x[4] * const.Tsun
    phase0 = x[5]
    phi = x[6]
    psi = x[7]
    phases = np.copy(x[8::2])
    pdists = np.copy(x[9::2])

    #print(phases)
    #print(pdists)
    #print(x)

    amp = A/(2*np.pi*fgw)

    cos_inc = np.cos(inc)
    one_plus_cos_inc_sq = 1+cos_inc**2
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_2psi = np.cos(2*psi)
    sin_2psi = np.sin(2*psi)
    cos_phase0 = np.cos(phase0)
    sin_phase0 = np.sin(phase0)
    cos_phases = np.cos(2*phases+phase0)
    sin_phases = np.sin(2*phases+phase0)

    m = np.array([sin_phi, -cos_phi, 0.0])
    n = np.array([-cos_theta * cos_phi, -cos_theta * sin_phi, sin_theta])
    omhat = np.array([-sin_theta * cos_phi, -sin_theta * sin_phi, -cos_theta])

    f_min = fff[0]
    df = fff[1]-fff[0]
    n_f = fff.size
    f_lb = f_min - df*(3//2)
    f_ub = fff[-1] + df*(3//2)

    sigma = np.zeros(4)

    logL = -0.5*resres -0.5*logdet
    #for ii in range(n_psr):
    for ii in prange(n_psr):
        m_pos = 0.
        n_pos = 0.
        cosMu = 0.
        for j in range(0,3):
            m_pos += m[j]*psr_poses[ii,j]
            n_pos += n[j]*psr_poses[ii,j]
            cosMu -= omhat[j]*psr_poses[ii,j]

        F_p = 0.5 * (m_pos ** 2 - n_pos ** 2) / (1 - cosMu)
        F_c = (m_pos * n_pos) / (1 - cosMu)

        p_dist = pdists[ii]*(const.kpc/const.c)

        w0 = np.pi * fgw
        omega_p0 = w0 *(1 + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)

        amp_psr = amp * (w0/omega_p0)**(1.0/3.0)
        phase0_psr = phases[ii]

        cos_phase0_psr = np.cos(phase0+phase0_psr*2.0)
        sin_phase0_psr = np.sin(phase0+phase0_psr*2.0)

        sigma[0] =  amp*(   cos_phase0 * (1+cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Earth term sine
                          2*sin_phase0 *     cos_inc    * (+sin_2psi * F_p + cos_2psi * F_c)   )
        sigma[1] =  amp*(   sin_phase0 * (1+cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Earth term cosine
                          2*cos_phase0 *     cos_inc    * (-sin_2psi * F_p - cos_2psi * F_c)   )
        sigma[2] =  -amp_psr*(   cos_phase0_psr * (1+cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Pulsar term sine
                          2*sin_phase0_psr *     cos_inc    * (+sin_2psi * F_p + cos_2psi * F_c)   )
        sigma[3] =  -amp_psr*(   sin_phase0_psr * (1+cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Pulsar term cosine
                          2*cos_phase0_psr *     cos_inc    * (-sin_2psi * F_p - cos_2psi * F_c)   )

        fpsr = omega_p0/np.pi

        Ns = np.zeros(4)
        MM = np.zeros((4,4))

        N0s = np.zeros(2)
        interp1d_k3(NN0s[ii,:], [fgw,fpsr], N0s, f_min, df, n_f, False, 0, f_lb, f_ub)
        N1s = np.zeros(2)
        interp1d_k3(NN1s[ii,:], [fgw,fpsr], N1s, f_min, df, n_f, False, 0, f_lb, f_ub)

        Ns[0] = N0s[0]
        Ns[1] = N1s[0]
        Ns[2] = N0s[1]
        Ns[3] = N1s[1]

        M00s = np.zeros(3)
        interp2d_k3(MM00s[ii,:,:], [fgw,fpsr,fgw], [fgw,fpsr,fpsr], M00s, [f_min,f_min], [df,df], [n_f,n_f], [False,False], [0,0], [f_lb,f_lb], [f_ub,f_ub])
        MM[0,0] = M00s[0]
        MM[2,2] = M00s[1]
        MM[0,2] = M00s[2]

        M11s = np.zeros(3)
        interp2d_k3(MM11s[ii,:,:], [fgw,fpsr,fgw], [fgw,fpsr,fpsr], M11s, [f_min,f_min], [df,df], [n_f,n_f], [False,False], [0,0], [f_lb,f_lb], [f_ub,f_ub])
        MM[1,1] = M11s[0]
        MM[3,3] = M11s[1]
        MM[1,3] = M11s[2]

        M01s = np.zeros(4)
        interp2d_k3(MM01s[ii,:,:], [fgw,fpsr,fgw,fpsr], [fgw,fpsr,fpsr,fgw], M01s, [f_min,f_min], [df,df], [n_f,n_f], [False,False], [0,0], [f_lb,f_lb], [f_ub,f_ub])
        MM[0,1] = M01s[0]
        MM[2,3] = M01s[1]
        MM[0,3] = M01s[2]
        MM[1,2] = M01s[3]

        #fill in lower diagonal of MM matrix
        #Ms = MM + MM.T - np.diag(MM.diagonal())
        Ms = MM + MM.T - np.diag(np.diag(MM))

        #print(fgw, fpsr)
        #print(Ns)
        #print(Ms)

        for jj in range(4):
            logL += sigma[jj] * Ns[jj]
            for kk in range(4):
                logL += -0.5*sigma[jj]*sigma[kk] * Ms[jj,kk]

    return logL

@njit(fastmath=False)
def incoherent_log_L_helper(x, fgw, n_psr, N, M, resres, logdet):
    #A = 10**x[0]
    #As = np.copy(x[1::2])
    #phases = np.copy(x[2::2])
    #Amp = A/(2*np.pi*fgw)

    As = 10**np.copy(x[::2])
    Amps = As/(2*np.pi*fgw)
    phases = np.copy(x[1::2])

    logL = -0.5*resres -0.5*logdet
    for ii in range(n_psr):
        b = np.ones(2)

        #b[0] = Amp*As[ii]*np.cos(phases[ii])
        #b[1] = Amp*As[ii]*np.sin(phases[ii])
        b[0] = Amps[ii]*np.cos(phases[ii])
        b[1] = Amps[ii]*np.sin(phases[ii])

        for kk in range(2):
            logL += b[kk]*N[ii,kk]
            for ll in range(2):
                logL += -0.5*M[ii,kk,ll]*b[kk]*b[ll]

    return logL

@njit(fastmath=False)
def log_L_helper(x, fgw, n_psr, psr_poses, N, M, resres, logdet):
    inc = x[0]
    theta = x[1]
    A = 10**x[2]
    phase0 = x[3]
    phi = x[4]
    psi = x[5]
    phases = np.copy(x[6:])

    #print(x)

    Amp = A/(2*np.pi*fgw)

    cos_inc = np.cos(inc)
    one_plus_cos_inc_sq = 1+cos_inc**2
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_2psi = np.cos(2*psi)
    sin_2psi = np.sin(2*psi)
    cos_phase0 = np.cos(phase0)
    sin_phase0 = np.sin(phase0)
    cos_phases = np.cos(2*phases+phase0)
    sin_phases = np.sin(2*phases+phase0)

    m = np.array([sin_phi, -cos_phi, 0.0])
    n = np.array([-cos_theta * cos_phi, -cos_theta * sin_phi, sin_theta])
    omhat = np.array([-sin_theta * cos_phi, -sin_theta * sin_phi, -cos_theta])

    logL = -0.5*resres -0.5*logdet
    for ii in range(n_psr):
        m_pos = 0.
        n_pos = 0.
        cosMu = 0.
        for j in range(0,3):
            m_pos += m[j]*psr_poses[ii,j]
            n_pos += n[j]*psr_poses[ii,j]
            cosMu -= omhat[j]*psr_poses[ii,j]

        F_p = 0.5 * (m_pos ** 2 - n_pos ** 2) / (1 - cosMu)
        F_c = (m_pos * n_pos) / (1 - cosMu)

        b = np.ones(2)
        #b[0] = Amp* ( one_plus_cos_inc_sq*(-F_p*cos_2psi+F_c*sin_2psi)*(cos_phase0-cos_phases[ii])+
        #                          2*cos_inc*(F_p*sin_2psi+F_c*cos_2psi)*(sin_phase0-sin_phases[ii]) )
        #b[1] = Amp* ( one_plus_cos_inc_sq*(-F_p*cos_2psi+F_c*sin_2psi)*(sin_phase0-sin_phases[ii])+
        #                          2*cos_inc*(-F_p*sin_2psi-F_c*cos_2psi)*(cos_phase0-cos_phases[ii]) )

        cos_sq_antenna_term = one_plus_cos_inc_sq*(-F_p*cos_2psi+F_c*sin_2psi)
        cos_antenna_term = 2*cos_inc*(F_p*sin_2psi+F_c*cos_2psi)
        cos_phase_diff = cos_phase0-cos_phases[ii]
        sin_phase_diff = sin_phase0-sin_phases[ii]
        b[0] = Amp* ( cos_sq_antenna_term * cos_phase_diff+
                      cos_antenna_term    * sin_phase_diff )
        b[1] = Amp* ( cos_sq_antenna_term * sin_phase_diff-
                      cos_antenna_term    * cos_phase_diff )

        #print(b)
        #logLambda += np.dot(b,N[ii,:])
        #logLambda += -0.5*np.dot(b,np.dot(M[ii,:,:],b))
        #print("logL")
        #print(logL)
        for kk in range(2):
            #print(kk)
            logL += b[kk]*N[ii,kk]
            #print(logL, b[kk]*N[ii,kk])
            for ll in range(2):
                #print(ll)
                logL += -0.5*M[ii,kk,ll]*b[kk]*b[ll]
                #print(logL, -0.5*M[ii,kk,ll]*b[kk]*b[ll])

    return logL

@njit(fastmath=False)
def incoherent_phase_marg_log_L_helper(x_com, fgw, n_psr, N, M, resres, logdet):
    #A = 10**x_com[0]
    #As = np.copy(x_com[1:])
    #Amp = A/(2*np.pi*fgw)
    As = 10**np.copy(x_com)
    Amps = As/(2*np.pi*fgw)

    logL_marg = -0.5*resres -0.5*logdet
    for ii in range(n_psr):
        
        #AA = Amp*As[ii]
        AA = Amps[ii]

        #f(phi_psr)=alpha + beta*sin(phi_psr) + gamma*cos(phi_psr) + delta*sin(2*phi_psr) + epsilon*cos(2*phi_psr)
        alpha = -AA**2/4*(M[ii,0,0]+M[ii,1,1])
        beta = AA*N[ii,1]
        gamma = AA*N[ii,0]
        delta = -AA**2/2*M[ii,0,1]
        epsilon = AA**2/4*(M[ii,1,1]-M[ii,0,0])
        
        #convert it into format: a*cos(phi_psr+phase1)+b*cos(2*phi_psr+2*phase2)
        aa = np.sqrt(beta**2+gamma**2)
        bb = np.sqrt(delta**2+epsilon**2)
        phase1 = np.arctan2(-beta,gamma)
        phase2 = np.arctan2(-delta,epsilon) / 2 #divide by 2 because of how we defined phase2


        #print(alpha, beta, gamma, delta, epsilon)

        #print(aa, bb, phase1, phase2)
        integral = integral_solution(aa,bb,phase1, phase2)
        #print(integral)
        logL_marg += alpha + np.log(integral)

    return logL_marg

@njit(fastmath=False)
def phase_marg_log_L_helper(x_com, fgw, n_psr, psr_poses, N, M, resres, logdet):
    inc = x_com[0]
    theta = x_com[1]
    A = 10**x_com[2]
    phase0 = x_com[3]
    phi = x_com[4]
    psi = x_com[5]

    Amp = A/(2*np.pi*fgw)

    cos_inc = np.cos(inc)
    one_plus_cos_inc_sq = 1+cos_inc**2
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_2psi = np.cos(2*psi)
    sin_2psi = np.sin(2*psi)
    cos_phase0 = np.cos(phase0)
    sin_phase0 = np.sin(phase0)
    #cos_2phase0 = np.cos(2*phase0)
    #sin_2phase0 = np.sin(2*phase0)

    m = np.array([sin_phi, -cos_phi, 0.0])
    n = np.array([-cos_theta * cos_phi, -cos_theta * sin_phi, sin_theta])
    omhat = np.array([-sin_theta * cos_phi, -sin_theta * sin_phi, -cos_theta])

    logL_marg = -0.5*resres -0.5*logdet
    for ii in range(n_psr):
        m_pos = 0.
        n_pos = 0.
        cosMu = 0.
        for j in range(0,3):
            m_pos += m[j]*psr_poses[ii,j]
            n_pos += n[j]*psr_poses[ii,j]
            cosMu -= omhat[j]*psr_poses[ii,j]

        F_p = 0.5 * (m_pos ** 2 - n_pos ** 2) / (1 - cosMu)
        F_c = (m_pos * n_pos) / (1 - cosMu)

        #define Fe coefficients
        #a1 =  Amp * (one_plus_cos_inc_sq*cos_2phase0*cos_2psi + 2*cos_inc*sin_2phase0*sin_2psi)
        #a2 = -Amp * (one_plus_cos_inc_sq*sin_2phase0*cos_2psi - 2*cos_inc*cos_2phase0*sin_2psi)
        #a3 =  Amp * (one_plus_cos_inc_sq*cos_2phase0*sin_2psi - 2*cos_inc*sin_2phase0*cos_2psi)
        #a4 = -Amp * (one_plus_cos_inc_sq*sin_2phase0*sin_2psi + 2*cos_inc*cos_2phase0*cos_2psi)
        a1 =  Amp * (-one_plus_cos_inc_sq*cos_phase0*cos_2psi + 2*cos_inc*sin_phase0*sin_2psi)
        a2 = -Amp * ( one_plus_cos_inc_sq*sin_phase0*cos_2psi + 2*cos_inc*cos_phase0*sin_2psi)
        a3 =  Amp * ( one_plus_cos_inc_sq*cos_phase0*sin_2psi + 2*cos_inc*sin_phase0*cos_2psi)
        a4 =  Amp * ( one_plus_cos_inc_sq*sin_phase0*sin_2psi - 2*cos_inc*cos_phase0*cos_2psi)
        #print(a1,a2,a3,a4)
        #define AA and BB
        AA = F_p*a1 + F_c*a3
        BB = F_p*a2 + F_c*a4

        #print(AA, BB)
        #print(M[ii,0,0], M[ii,1,1], M[ii,0,1])
        #print(BB**2/2*M[ii,0,0], -AA**2/2*M[ii,1,1], AA*BB*M[ii,0,1])
        #print(-BB**2*M[ii,0,0])

        #f(phi_psr)=alpha + beta*sin(phi_psr) + gamma*cos(phi_psr) + delta*sin(2*phi_psr) + epsilon*cos(2*phi_psr)
        alpha = AA*N[ii,0] + BB*N[ii,1] - AA*BB*M[ii,0,1] - (BB**2+3*AA**2)/4*M[ii,0,0] - (3*BB**2+AA**2)/4*M[ii,1,1]
        #print(AA*N[ii,0], BB*N[ii,1], - AA*BB*M[ii,0,1], (BB**2-3*AA**2)/4*M[ii,0,0], - (3*BB**2+AA**2)/4*M[ii,1,1])
        #print(AA*N[ii,0], BB*N[ii,1], - AA*BB*M[ii,0,1], AA**2/2*M[ii,0,0], -BB**2/2*M[ii,1,1])
        #alpha = AA*N[ii,0] + BB*N[ii,1] - AA*BB*M[ii,0,1] + AA**2/2*M[ii,0,0] - BB**2/2*M[ii,1,1]
        beta = AA*N[ii,1] - BB*N[ii,0] + AA*BB*(M[ii,0,0]-M[ii,1,1]) - (AA**2-BB**2)*M[ii,0,1]
        gamma = -AA*N[ii,0] - BB*N[ii,1] + AA**2*M[ii,0,0] + BB**2*M[ii,1,1] + 2*AA*BB*M[ii,0,1]
        delta = AA*BB/2*(M[ii,1,1]-M[ii,0,0]) + (AA**2-BB**2)/2*M[ii,0,1]
        epsilon = -(AA**2-BB**2)/4*M[ii,0,0] - (BB**2-AA**2)/4*M[ii,1,1] - AA*BB*M[ii,0,1]
        #convert it into format: a*cos(phi_psr+phase1)+b*cos(2*phi_psr+2*phase2)
        aa = np.sqrt(beta**2+gamma**2)
        bb = np.sqrt(delta**2+epsilon**2)
        phase1 = np.arctan2(-beta,gamma)
        phase2 = np.arctan2(-delta,epsilon) / 2 #divide by 2 because of how we defined phase2
        
        
        #print(alpha, beta, gamma, delta, epsilon)

        #print(aa, bb, phase1, phase2)
        integral = integral_solution(aa,bb,phase1, phase2)
        #print(integral)
        #logL_marg += alpha + np.log(2*np.pi*integral)
        logL_marg += alpha + np.log(integral)

        #integral_pre, integral_exp = integral_solution(aa,bb,phase1, phase2)
        #logL_marg += alpha + integral_exp + np.log(integral_pre)


    return logL_marg


@njit(fastmath=False)
def integral_solution(a,b,phase1, phase2):
    #nmax = 30
    nmax = min(100,int(a+b))
    #print(nmax)
    #Seems to catch very large signals that cause trouble
    #TODO: come up with a better way to handle these
    if nmax==100:
        return 0.0

    #a*cos(x+phase1) = a_p*cos(x)+c_p*sin(x)
    a_p = a*np.cos(phase1-phase2)
    c_p = -a*np.sin(phase1-phase2)
    #print(a_p, c_p)

    res = scs.iv(0.0,a_p)*scs.iv(0.0,b)*scs.iv(0.0,c_p)
    #print(2*np.pi*res)
    for n in range(1, nmax+1):
        n_float = float(n)
        #print("-"*10)
        #print(n)
        res += 2*scs.iv(0.0,c_p) * scs.iv(2*n_float,a_p)*scs.iv(n_float,b)
        #print(2*np.pi*res, 2*np.pi*2*scs.iv(0,c_p) * scs.iv(2*n,a_p)*scs.iv(n,b))
        res += 2*scs.iv(0.0,b)   * scs.iv(2*n_float,a_p)*scs.iv(2*n_float,c_p)                 * (-1)**n
        #print(2*np.pi*res, 2*np.pi*2*scs.iv(0,b)   * scs.iv(2*n,a_p)*scs.iv(2*n,c_p)                 * (-1)**n)
        res += 2*scs.iv(0.0,a_p) * scs.iv(n_float,b)*scs.iv(2*n_float,c_p)                     * (-1)**n
        #print(2*np.pi*res, 2*np.pi*2*scs.iv(0,a_p) * scs.iv(n,b)*scs.iv(2*n,c_p)                     * (-1)**n)
        ###### triple terms
        for k in range(1,int(n/2)+1):
            k_float = float(k)
            #print(k)
            # (n, k, n-k) --> needs n and k even
            if n%2==0 and k%2==0:
                #print(f"({n}, {k}, {n-k})-->({n/2}, {k}, {n-k})")
                res += 2               * scs.iv(n_float/2,b)*scs.iv(k_float,c_p)*scs.iv(n_float-k_float,a_p)     * (-1)**(k/2)
            # (k, n, n-k) --> needs n and k even
            if n%2==0 and (n-k)%2==0:
                #print(f"({k}, {n}, {n-k})-->({k/2}, {n}, {n-k})")
                res += 2               * scs.iv(k_float/2,b)*scs.iv(n_float,c_p)*scs.iv(n_float-k_float,a_p)     * (-1)**(n/2)
            # (k, n-k, n) --> needs k and n-k even
            if n%2==0 and (n-k)%2==0:
                #print(f"({k}, {n-k}, {n})-->({k/2}, {n-k}, {n})")
                res += 2               * scs.iv(k_float/2,b)*scs.iv(n_float-k_float,c_p)*scs.iv(n_float,a_p)     * (-1)**((n-k)/2)
            if k!=(n-k): #only do these if all three indices are distinct, otherwise we double count
                # (n, n-k, k) --> needs n and n-k even
                if n%2==0 and (n-k)%2==0:
                    #print(f"({n}, {n-k}, {k})-->({n/2}, {n-k}, {k})")
                    res += 2               * scs.iv(n_float/2,b)*scs.iv(n_float-k_float,c_p)*scs.iv(k_float,a_p)     * (-1)**((n-k)/2)
                # (n-k, n, k) --> needs n-k and n even
                if n%2==0 and (n-k)%2==0:
                    #print(f"({n-k}, {n}, {k})-->({(n-k)/2}, {n}, {k})")
                    res += 2               * scs.iv((n_float-k_float)/2,b)*scs.iv(n_float,c_p)*scs.iv(k_float,a_p)     * (-1)**(n/2)
                # (n-k, k, n) --> needs n-k and k even
                if n%2==0 and (n-k)%2==0:
                    #print(f"({n-k}, {k}, {n})-->({(n-k)/2}, {k}, {n})")
                    res += 2               * scs.iv((n_float-k_float)/2,b)*scs.iv(k_float,c_p)*scs.iv(n_float,a_p)     * (-1)**(k/2)
        
    return res

#@njit(fastmath=False)
#def integral_solution(a,b,phase1, phase2):
#    #get the integral of np.exp(a*np.cos(x+phase1)+b*np.cos(2*x+2*phase2))
#
#    nmax = 10
#    print(nmax)
#
#    #a*cos(x+phase1) = a_p*cos(x)+c_p*sin(x)
#    #and actually take the phase difference of the two
#    a_p = a*np.cos(phase1-phase2)
#    c_p = -a*np.sin(phase1-phase2)
#    print(a_p, c_p, b)
#
#    res = scs.ive(0.0,a_p)*scs.ive(0.0,b)*scs.ive(0.0,c_p)
#    print(2*np.pi*res)
#    for n in range(1, nmax+1):
#        print(n)
#        n_float = float(n)
#        res += 2*scs.ive(0.0,c_p) * scs.ive(2*n_float,a_p)*scs.ive(n_float,b)
#        print(2*np.pi*res, 2*np.pi*2*scs.ive(0.0,c_p) * scs.ive(2*n_float,a_p)*scs.ive(n_float,b))
#        res += 2*scs.ive(0.0,b)   * scs.ive(2*n_float,a_p)*scs.ive(2*n_float,c_p)                 * (-1)**n
#        print(2*np.pi*res, 2*np.pi*2*scs.ive(0.0,b)   * scs.ive(2*n_float,a_p)*scs.ive(2*n_float,c_p)                 * (-1)**n)
#        res += 2*scs.ive(0.0,a_p) * scs.ive(n_float,b)*scs.ive(2*n_float,c_p)                     * (-1)**n
#        print(2*np.pi*res, 2*np.pi*2*scs.ive(0.0,a_p) * scs.ive(n_float,b)*scs.ive(2*n_float,c_p)                     * (-1)**n)
#        ###### triple terms
#        res += 2               * scs.ive(2*n_float,b)*scs.ive(2*n_float,c_p)*scs.ive(2*n_float,a_p)   * (-1)**n
#        print(2*np.pi*res, 2*np.pi*2               * scs.ive(2*n_float,b)*scs.ive(2*n_float,c_p)*scs.ive(2*n_float,a_p)   * (-1)**n)
#        res += 2               * scs.ive(n_float,b)*scs.ive(2*n_float,c_p)*scs.ive(4*n_float,a_p)     * (-1)**n
#        print(2*np.pi*res, 2*np.pi*2               * scs.ive(n_float,b)*scs.ive(2*n_float,c_p)*scs.ive(4*n_float,a_p)     * (-1)**n)
#        res += 2               * scs.ive(n_float,b)*scs.ive(4*n_float,c_p)*scs.ive(2*n_float,a_p)
#        print(2*np.pi*res, 2*np.pi*2               * scs.ive(n_float,b)*scs.ive(4*n_float,c_p)*scs.ive(2*n_float,a_p))
#    #return res*np.exp(a_p+c_p+b)
#    return res, a_p+c_p+b

#@njit(fastmath=False)
#def integral_solution(a,b,phase1, phase2):
#    #get the integral of np.exp(a*np.cos(x+phase1)+b*np.cos(2*x+2*phase2))
#
#    nmax = 30#4
#    print(nmax)
#
#    #a*cos(x+phase1) = a_p*cos(x)+c_p*sin(x)
#    #and actually take the phase difference of the two
#    a_p = a*np.cos(phase1-phase2)
#    c_p = -a*np.sin(phase1-phase2)
#    print(a_p, c_p, b)
#
#    res = scs.iv(0.0,a_p)*scs.iv(0.0,b)*scs.iv(0.0,c_p)
#    print(2*np.pi*res)
#    for n in range(1, nmax+1):
#        print(n)
#        n_float = float(n)
#        res += 2*scs.iv(0.0,c_p) * scs.iv(2*n_float,a_p)*scs.iv(n_float,b)
#        print(2*np.pi*res, 2*np.pi*2*scs.iv(0.0,c_p) * scs.iv(2*n_float,a_p)*scs.iv(n_float,b))
#        res += 2*scs.iv(0.0,b)   * scs.iv(2*n_float,a_p)*scs.iv(2*n_float,c_p)                 * (-1)**n
#        print(2*np.pi*res, 2*np.pi*2*scs.iv(0.0,b)   * scs.iv(2*n_float,a_p)*scs.iv(2*n_float,c_p)                 * (-1)**n)
#        res += 2*scs.iv(0.0,a_p) * scs.iv(n_float,b)*scs.iv(2*n_float,c_p)                     * (-1)**n
#        print(2*np.pi*res, 2*np.pi*2*scs.iv(0.0,a_p) * scs.iv(n_float,b)*scs.iv(2*n_float,c_p)                     * (-1)**n)
#        ###### triple terms
#        res += 2               * scs.iv(2*n_float,b)*scs.iv(2*n_float,c_p)*scs.iv(2*n_float,a_p)   * (-1)**n
#        print(2*np.pi*res, 2*np.pi*2               * scs.iv(2*n_float,b)*scs.iv(2*n_float,c_p)*scs.iv(2*n_float,a_p)   * (-1)**n)
#        res += 2               * scs.iv(n_float,b)*scs.iv(2*n_float,c_p)*scs.iv(4*n_float,a_p)     * (-1)**n
#        print(2*np.pi*res, 2*np.pi*2               * scs.iv(n_float,b)*scs.iv(2*n_float,c_p)*scs.iv(4*n_float,a_p)     * (-1)**n)
#        res += 2               * scs.iv(n_float,b)*scs.iv(4*n_float,c_p)*scs.iv(2*n_float,a_p)
#        print(2*np.pi*res, 2*np.pi*2               * scs.iv(n_float,b)*scs.iv(4*n_float,c_p)*scs.iv(2*n_float,a_p))
#    return res

#code from https://github.com/dbstein/fast_interp
@njit(fastmath=True, parallel=False)
def interp1d_k3(f, xout, fout, a, h, n, p, o, lb, ub):
    #(NN0s[ii,:], [fgw,], N0s, f_min, df, n_f, False, 0, f_lb, f_ub)
    m = fout.shape[0]
    for mi in prange(m):
        xr = min(max(xout[mi], lb), ub) #fgw if within bounds
        xx = xr - a #fgw-fmin
        ix = int(xx//h) #bin number
        ratx = xx/h - (ix+0.5) #position within bin -0.5 meaning left edge, 0 meaning middle, 0.5 meaning right edge
        asx = np.empty(4)
        asx[0] = -1/16 + ratx*( 1/24 + ratx*( 1/4 - ratx/6))
        asx[1] =  9/16 + ratx*( -9/8 + ratx*(-1/4 + ratx/2))
        asx[2] =  9/16 + ratx*(  9/8 + ratx*(-1/4 - ratx/2))
        asx[3] = -1/16 + ratx*(-1/24 + ratx*( 1/4 + ratx/6))
        #if ratx=-0.5-->asx=[0,1,0,0]
        #if ratx=0.5-->asx=[0,0,1,0]
        ix += o-1
        fout[mi] = 0.0
        for i in range(4):
            ixi = (ix + i) % n if p else ix + i
            fout[mi] += f[ixi]*asx[i]


#code from https://github.com/dbstein/fast_interp
@njit(fastmath=True, parallel=False) #TODO: can potentially be sped up further as some of asx and asy are repetitive (can be seen by printing them out)
def interp2d_k3(f, xout, yout, fout, a, h, n, p, o, lb, ub): #TODO: needs padding to avoid nans close to the edge
    """
        a, b: the lower and upper bounds of the interpolation region
        h:    the grid-spacing at which f is given
        f:    data to be interpolated
        p:    whether the dimension is taken to be periodic
        c:    whether the array should be padded to allow accurate close eval
        if p is True, then f is assumed to given on:
            [a, b)
        if p is False, f is assumed to be given on:
            [a, b]
        For periodic interpolation (p = True)
            this will interpolate accurately for any x
            c is ignored
            e is ignored
        For non-periodic interpolation (p = False)
            if c is True the function is padded to allow accurate eval on:
                [a-e*h, b+e*h]
                (extrapolation is done on [a-e*h, a] and [b, b+e*h], be careful!)
            if c is False, the function evaluates accurately on:
                [a,    b   ] for k = 1
                [a+h,  b-h ] for k = 3
                [a+2h, b-2h] for k = 5
                e is ignored
            c = True requires the allocation of a padded data array, as well
                as a memory copy from f to the padded array and some
                time computing function extrapolations, this setup time is
                quite small and fine when interpolating to many points but
                is significant when interpolating to only a few points
            right now there is no bounds checking; this will probably segfault
            if you provide values outside of the safe interpolation region...
    """
    m = fout.shape[0]
    for mi in prange(m):
        xr = min(max(xout[mi], lb[0]), ub[0])
        yr = min(max(yout[mi], lb[1]), ub[1])
        xx = xr - a[0]
        yy = yr - a[1]
        ix = int(xx//h[0])
        iy = int(yy//h[1])
        ratx = xx/h[0] - (ix+0.5)
        raty = yy/h[1] - (iy+0.5)
        #print(ratx,raty)
        asx = np.empty(4)
        asy = np.empty(4)
        asx[0] = -1/16 + ratx*( 1/24 + ratx*( 1/4 - ratx/6))
        asx[1] =  9/16 + ratx*( -9/8 + ratx*(-1/4 + ratx/2))
        asx[2] =  9/16 + ratx*(  9/8 + ratx*(-1/4 - ratx/2))
        asx[3] = -1/16 + ratx*(-1/24 + ratx*( 1/4 + ratx/6))
        asy[0] = -1/16 + raty*( 1/24 + raty*( 1/4 - raty/6))
        asy[1] =  9/16 + raty*( -9/8 + raty*(-1/4 + raty/2))
        asy[2] =  9/16 + raty*(  9/8 + raty*(-1/4 - raty/2))
        asy[3] = -1/16 + raty*(-1/24 + raty*( 1/4 + raty/6))
        #print(asx)
        #print(asy)
        ix += o[0]-1
        iy += o[1]-1
        fout[mi] = 0.0
        for i in range(4):
            ixi = ix + i
            for j in range(4):
                iyj = iy + j
                #print(f[ixi,iyj])
                fout[mi] += f[ixi,iyj]*asx[i]*asy[j]
