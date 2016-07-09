import numpy as np
import time

from pyscf.pbc.dft.gen_grid import gen_uniform_grids
from pyscf.pbc.dft.numint import eval_ao
from pyscf.pbc import tools
from pyscf.lib import logger
#einsum = pyscf.lib.einsum

#cput0 = (time.clock(), time.time())
#log.timer('CCSD', *cput0)

"""
    (ij|kl) = \int dr1 dr2 i*(r1) j(r1) v(r12) k*(r2) l(r2)
            = (ij|G) v(G) (G|kl)

    i*(r) j(r) = 1/N \sum_G e^{iGr}  (G|ij)
               = 1/N \sum_G e^{-iGr} (ij|G)

    "forward" FFT:
        (G|ij) = \sum_r e^{-iGr} i*(r) j(r) = fft[ i*(r) j(r) ]
    "inverse" FFT:
        (ij|G) = \sum_r e^{iGr} i*(r) j(r) = N * ifft[ i*(r) j(r) ]
               = conj[ \sum_r e^{-iGr} j*(r) i(r) ]
"""

def mo_equal(moi, ki, moj, kj):
    if moi.shape != moj.shape:
        return False
    return np.allclose(moi,moj) and np.allclose(ki,kj)

class _ao2mo_storage:
    def __init__(self):
        self.mop = None
        self.moq = None
        self.mor = None
        self.mos = None
        self.kp = None
        self.kq = None
        self.kr = None
        self.ks = None

        self.mo_pairs12_kG = None

    def ijmade(self, moi, ki, moj, kj):
        if self.mo_pairs12_kG is None:
            return False
        if (self.mop == moi and
            self.moq == moj and
            self.kp == ki and
            self.kq == kj):
            return True
        return False

def general(cell, mo_coeffs, kpts=None, compact=0):
    '''pyscf-style wrapper to get MO 2-el integrals.'''
    assert len(mo_coeffs) == 4
    if kpts is not None:
        assert len(kpts) == 4
    ao2mo_helper = _ao2mo_storage()
    return get_mo_eri(cell, ao2mo_helper, mo_coeffs, kpts)

def get_mo_eri(cell, ao2mo_helper, mo_coeffs, kpts=None):
    '''Convenience function to return MO 2-el integrals.'''
    mo_coeff12 = mo_coeffs[:2]
    mo_coeff34 = mo_coeffs[2:]
    if kpts is None:
        kpts12 = kpts34 = q = None
    else:
        kpts12 = kpts[:2]
        kpts34 = kpts[2:]
        q = kpts12[0] - kpts12[1]
    if q is None:
        q = np.zeros(3)

    ijsame = False
    ijmade = False
    #ijsame = mo_equal(mo_coeff12[0],kpts12[0],mo_coeff12[1],kpts12[1])
    #klsame = mo_equal(mo_coeff34[0],kpts34[0],mo_coeff34[1],kpts34[1])

    #ijmade = ao2mo_helper.ijmade(mo_coeff12[0],kpts12[0],mo_coeff12[1],kpts12[1])

    if ijmade:
        mo_pairs12_kG = ao2mo_helper.mo_pairs12_kG
    else:
        mo_pairs12_kG = get_mo_pairs_G(cell, mo_coeff12, kpts12, ijsame)
        ao2mo_helper.mo_pairs12_kG = mo_pairs12_kG
    mo_pairs34_invkG = get_mo_pairs_invG(cell, mo_coeff34, kpts34, q)
    return assemble_eri(cell, mo_pairs12_kG, mo_pairs34_invkG, q, ijsame)

def get_mo_pairs_G(cell, mo_coeffs, kpts=None, ijsame=False, q=None):
    '''Calculate forward (G|ij) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_G : (ngs, nmoi*nmoj) ndarray
            The FFT of the real-space MO pairs.
    '''
    coords = gen_uniform_grids(cell)
    if kpts is None:
        q = np.zeros(3)
        aoR = eval_ao(cell, coords)
        ngs = aoR.shape[0]

        if np.array_equal(mo_coeffs[0], mo_coeffs[1]):
            nmoi = nmoj = mo_coeffs[0].shape[1]
            moiR = mojR = np.dot(aoR, mo_coeffs[0])
        else:
            nmoi = mo_coeffs[0].shape[1]
            nmoj = mo_coeffs[1].shape[1]
            moiR = np.dot(aoR, mo_coeffs[0])
            mojR = np.dot(aoR, mo_coeffs[1])

    else:
        if q is None:
            q = kpts[1]-kpts[0]
        aoR_ki = eval_ao(cell, coords, kpt=kpts[0])
        aoR_kj = eval_ao(cell, coords, kpt=kpts[1])
        ngs = aoR_ki.shape[0]

        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        moiR = np.dot(aoR_ki, mo_coeffs[0])
        mojR = np.dot(aoR_kj, mo_coeffs[1])

    mo_pairs_G = np.zeros([ngs,nmoi*nmoj], np.complex128)

    expmikr = np.exp(-1j*np.dot(q,coords.T))
    for i in xrange(nmoi):
        for j in xrange(nmoj):
            mo_pairs_R_ij = np.conj(moiR[:,i])*mojR[:,j]
            mo_pairs_G[:,i*nmoj+j] = tools.fftk(mo_pairs_R_ij, cell.gs,
                                                expmikr)
    return mo_pairs_G

def get_mo_pairs_invG(cell, mo_coeffs, kpts=None, q=None):
    '''Calculate "inverse" (ij|G) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_invG : (ngs, nmoi*nmoj) ndarray
            The inverse FFTs of the real-space MO pairs.
    '''
    coords = gen_uniform_grids(cell)
    if kpts is None:
        q = np.zeros(3)
        aoR = eval_ao(cell, coords)
        ngs = aoR.shape[0]

        if np.array_equal(mo_coeffs[0], mo_coeffs[1]):
            nmoi = nmoj = mo_coeffs[0].shape[1]
            moiR = mojR = np.dot(aoR, mo_coeffs[0])
        else:
            nmoi = mo_coeffs[0].shape[1]
            nmoj = mo_coeffs[1].shape[1]
            moiR = np.dot(aoR, mo_coeffs[0])
            mojR = np.dot(aoR, mo_coeffs[1])

    else:
        if q is None:
            q = kpts[1]-kpts[0]
        aoR_ki = eval_ao(cell, coords, kpt=kpts[0])
        aoR_kj = eval_ao(cell, coords, kpt=kpts[1])
        ngs = aoR_ki.shape[0]

        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        moiR = np.dot(aoR_ki, mo_coeffs[0])
        mojR = np.dot(aoR_kj, mo_coeffs[1])

    mo_pairs_invG = np.zeros([ngs,nmoi*nmoj], np.complex128)

    expmikr = np.exp(-1j*np.dot(q,coords.T))
    for i in xrange(nmoi):
        for j in xrange(nmoj):
            mo_pairs_R_ij = np.conj(moiR[:,i])*mojR[:,j]
            mo_pairs_invG[:,i*nmoj+j] = np.conj(tools.fftk(np.conj(mo_pairs_R_ij), cell.gs,
                                                           expmikr.conj()))
    return mo_pairs_invG

def get_mo_pairs_G_old(cell, mo_coeffs, kpts=None, q=None):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_G, mo_pairs_invG : (ngs, nmoi*nmoj) ndarray
            The FFTs of the real-space MO pairs.
    '''
    coords = gen_uniform_grids(cell)
    if kpts is None:
        q = np.zeros(3)
        aoR = eval_ao(cell, coords)
        ngs = aoR.shape[0]

        if np.array_equal(mo_coeffs[0], mo_coeffs[1]):
            nmoi = nmoj = mo_coeffs[0].shape[1]
            moiR = mojR = np.einsum('ri,ia->ra', aoR, mo_coeffs[0])
        else:
            nmoi = mo_coeffs[0].shape[1]
            nmoj = mo_coeffs[1].shape[1]
            moiR = np.einsum('ri,ia->ra', aoR, mo_coeffs[0])
            mojR = np.einsum('ri,ia->ra', aoR, mo_coeffs[1])

    else:
        if q is None:
            q = kpts[1]-kpts[0]
        aoR_ki = eval_ao(cell, coords, kpt=kpts[0])
        aoR_kj = eval_ao(cell, coords, kpt=kpts[1])
        ngs = aoR_ki.shape[0]

        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        moiR = np.einsum('ri,ia->ra', aoR_ki, mo_coeffs[0])
        mojR = np.einsum('ri,ia->ra', aoR_kj, mo_coeffs[1])

    mo_pairs_R = np.einsum('ri,rj->rij', np.conj(moiR), mojR)
    mo_pairs_G = np.zeros([ngs,nmoi*nmoj], np.complex128)
    mo_pairs_invG = np.zeros([ngs,nmoi*nmoj], np.complex128)

    expmikr = np.exp(-1j*np.dot(q,coords.T))
    for i in xrange(nmoi):
        for j in xrange(nmoj):
            mo_pairs_G[:,i*nmoj+j] = tools.fftk(mo_pairs_R[:,i,j], cell.gs,
                                                expmikr)
            mo_pairs_invG[:,i*nmoj+j] = np.conj(tools.fftk(np.conj(mo_pairs_R[:,i,j]), cell.gs,
                                                                   expmikr.conj()))

    return mo_pairs_G, mo_pairs_invG

def assemble_eri(cell, orb_pair_invG1, orb_pair_G2, q=None, ijsame=False, verbose=logger.DEBUG):
    '''Assemble 4-index electron repulsion integrals.

    Returns:
        (nmo1*nmo2, nmo3*nmo4) ndarray

    '''
    log = logger.Logger
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cell.stdout, verbose)

    #log.debug('Performing periodic ERI assembly of (%i, %i) ij,kl pairs',
    #          orb_pair_invG1.shape[1], orb_pair_G2.shape[1])
    if q is None:
        q = np.zeros(3)

    coulqG = tools.get_coulG(cell, -1.0*q)
    ngs = orb_pair_invG1.shape[0]

    Jorb_pair_invG1 = np.einsum('g,gn->gn',coulqG,orb_pair_invG1)*(cell.vol/ngs**2)
    eri = np.dot(Jorb_pair_invG1.T,orb_pair_G2)
    return eri

def get_ao_pairs_G(cell):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all AO pairs.

    Args:
        cell : instance of :class:`Cell`

    Returns:
        ao_pairs_G, ao_pairs_invG : (ngs, nao*(nao+1)/2) ndarray
            The FFTs of the real-space AO pairs.

    '''
    coords = gen_uniform_grids(cell)
    aoR = eval_ao(cell, coords) # shape = (coords, nao)
    ngs, nao = aoR.shape
    npair = nao*(nao+1)/2
    ao_pairs_G = np.zeros([coords.shape[0], npair], np.complex128)
    ao_pairs_invG = np.zeros([coords.shape[0], npair], np.complex128)
    ij = 0
    for i in range(nao):
        for j in range(i+1):
            ao_ij_R = np.einsum('r,r->r', np.conj(aoR[:,i]), aoR[:,j])
            ao_pairs_G[:,ij] = tools.fft(ao_ij_R, cell.gs)
            ao_pairs_invG[:,ij] = ngs*tools.ifft(ao_ij_R, cell.gs)
            ij += 1
    return ao_pairs_G, ao_pairs_invG

def get_ao_eri(cell):
    '''Convenience function to return AO 2-el integrals.'''

    ao_pairs_G, ao_pairs_invG = get_ao_pairs_G(cell)
    return assemble_eri(cell, ao_pairs_invG, ao_pairs_G)

"""

def get_mo_pairs_G_kpts(cell, mo_coeff_kpts):
    nkpts = mo_coeff_kpts.shape[0]
    ngs = cell.Gv.shape[0]
    nmo = mo_coeff_kpts.shape[2]

    mo_pairs_G_kpts = np.zeros([nkpts, nkpts, ngs, nmo*nmo], np.complex128)
    mo_pairs_invG_kpts = np.zeros([nkpts, nkpts, ngs, nmo*nmo], np.complex128)

    for K in range(nkpts):
        for L in range(nkpts):
            mo_pairs_G_kpts[K,L], mo_pairs_invG_kpts[K,L] = \
            get_mo_pairs_G(cell, [mo_coeff_kpts[K,:,:], mo_coeff_kpts[L,:,:]])

    return mo_pairs_G_kpts, mo_pairs_invG_kpts


def get_mo_eri_kpts(cell, kpts, mo_coeff_kpts):
    '''Assemble *all* MO integrals across kpts'''
    nkpts = mo_coeff_kpts.shape[0]
    nmo = mo_coeff_kpts.shape[2]
    eris=np.zeros([nkpts,nkpts,nkpts,nmo*nmo,nmo*nmo], np.complex128)

    KLMN = tools.get_KLMN(kpts)

    mo_pairs_G_kpts, mo_pairs_invG_kpts = get_mo_pairs_G_kpts(cell, mo_coeff_kpts)

    for K in range(nkpts):
        for L in range(nkpts):
            for M in range(nkpts):
                N = KLMN[K, L, M]
                eris[K,L,M, :, :]=\
                assemble_eri(cell, mo_pairs_G_kpts[K, L],
                             mo_pairs_invG_kpts[M, N])
    return eris

"""
