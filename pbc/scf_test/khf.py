'''
Hartree-Fock for periodic systems with k-point sampling

See Also:
    hf.py : Hartree-Fock for periodic systems at a single k-point
'''

import time
import numpy as np
import scipy.special
import pyscf.dft
import pyscf.pbc.scf_test.conv1D as conv1D
import pyscf.pbc.dft
import pyscf.pbc.scf_test.hf as pbchf
#from pyscf.pbc.df import mdf
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.scf_test import scfint
import pyscf.pbc.cc.symm as symm


def get_ovlp(mf, cell, kpts):
    '''Get the overlap AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        ovlp_kpts : (nkpts, nao, nao) ndarray
    '''
    nkpts = len(kpts)
    nao = cell.nao_nr()
    ovlp_kpts = np.zeros((nkpts,nao,nao), np.complex128)
    for k in range(nkpts):
        kpt = kpts[k,:]
        if mf.analytic_int:
            ovlp_kpts[k,:,:] = scfint.get_ovlp(cell, kpt)
        else:
            ovlp_kpts[k,:,:] = pbchf.get_ovlp(cell, kpt)
    return ovlp_kpts


def get_hcore(mf, cell, kpts):
    '''Get the core Hamiltonian AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        hcore : (nkpts, nao, nao) ndarray
    '''
    nao = cell.nao_nr()
    nkpts = len(kpts)
    hcore = np.zeros((nkpts,nao,nao), np.complex128)
    for k in range(nkpts):
        kpt = kpts[k,:]
        if mf.analytic_int:
            hcore[k,:,:] = scfint.get_hcore(cell, kpt)
        else:
            hcore[k,:,:] = pbchf.get_hcore(cell, kpt, mf=mf)
    return hcore


def get_j(mf, cell, dm_kpts, kpts, kpt_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpt_band : (3,) ndarray
            An arbitrary "band" k-point at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    nkpts = len(kpts)
    ngs = len(coords)
    nao = cell.nao_nr()

    aoR_kpts = np.zeros((nkpts,ngs,nao), np.complex128)
    for k in range(nkpts):
        kpt = kpts[k,:]
        aoR_kpts[k,:,:] = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)

    vjR = get_vjR_(cell, dm_kpts, aoR_kpts, mf)
    if kpt_band is not None:
        aoR_kband = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt_band)
        vj_kpts = cell.vol/ngs * np.dot(aoR_kband.T.conj(),
                                        vjR.reshape(-1,1)*aoR_kband)
    else:
        vj_kpts = np.zeros((nkpts,nao,nao), np.complex128)
        for k in range(nkpts):
            vj_kpts[k,:,:] = cell.vol/ngs * np.dot(aoR_kpts[k,:,:].T.conj(),
                                                   vjR.reshape(-1,1)*aoR_kpts[k,:,:])

    return vj_kpts


def get_jk(mf, cell, dm_kpts, kpts, kpt_band=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpt_band : (3,) ndarray
            An arbitrary "band" k-point at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    nkpts = len(kpts)
    ngs = len(coords)
    nao = cell.nao_nr()

    aoR_kpts = np.zeros((nkpts,ngs,nao), np.complex128)
    for k in range(nkpts):
        kpt = kpts[k,:]
        aoR_kpts[k,:,:] = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)

    vjR = get_vjR_(cell, dm_kpts, aoR_kpts, mf)
    if kpt_band is not None:
        aoR_kband = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt_band)
        vj_kpts = cell.vol/ngs * np.dot(aoR_kband.T.conj(),
                                        vjR.reshape(-1,1)*aoR_kband)
        vk_kpts = np.zeros((nao,nao), np.complex128)
        for k2 in range(nkpts):
            kpt2 = kpts[k2,:]
            vkR_k1k2 = pbchf.get_vkR_(mf, cell, aoR_kband, aoR_kpts[k2,:,:],
                                      kpt_band, kpt2)
            aoR_dm_k2 = np.dot(aoR_kpts[k2,:,:], dm_kpts[k2,:,:])
            tmp_Rq = np.einsum('Rqs,Rs->Rq', vkR_k1k2, aoR_dm_k2)
            vk_kpts += 1./nkpts * (cell.vol/ngs) \
                                * np.dot(aoR_kband.T.conj(), tmp_Rq)
            #vk_kpts += 1./nkpts * (cell.vol/ngs) * np.einsum('rs,Rp,Rqs,Rr->pq',
            #            dm_kpts[k2,:,:], aoR_kband.conj(),
            #            vkR_k1k2, aoR_kpts[k2,:,:])
    else:
        vj_kpts = np.zeros((nkpts,nao,nao), np.complex128)
        for k in range(nkpts):
            vj_kpts[k,:,:] = cell.vol/ngs * np.dot(aoR_kpts[k,:,:].T.conj(),
                                                   vjR.reshape(-1,1)*aoR_kpts[k,:,:])
        aoR_dm_kpts = np.zeros((nkpts,ngs,nao), np.complex128)
        for k in range(nkpts):
            aoR_dm_kpts[k,:,:] = np.dot(aoR_kpts[k,:,:], dm_kpts[k,:,:])
        vk_kpts = np.zeros((nkpts,nao,nao), np.complex128)
        for k1 in range(nkpts):
            kpt1 = kpts[k1,:]
            for k2 in range(nkpts):
                kpt2 = kpts[k2,:]
                vkR_k1k2 = pbchf.get_vkR_(mf, cell, aoR_kpts[k1,:,:], aoR_kpts[k2,:,:],
                                          kpt1, kpt2)
                tmp_Rq = np.einsum('Rqs,Rs->Rq', vkR_k1k2, aoR_dm_kpts[k2,:,:])
                vk_kpts[k1,:,:] += 1./nkpts * (cell.vol/ngs) \
                                    * np.dot(aoR_kpts[k1,:,:].T.conj(), tmp_Rq)
                #vk_kpts[k1,:,:] += 1./nkpts * (cell.vol/ngs) * np.einsum('rs,Rp,Rqs,Rr->pq',
                #                    dm_kpts[k2,:,:], aoR_kpts[k1,:,:].conj(),
                #                    vkR_k1k2, aoR_kpts[k2,:,:])

    return vj_kpts, vk_kpts


def get_vjR_(cell, dm_kpts, aoR_kpts, mf=None):
    '''Get the real-space Hartree potential of the k-point sampled density matrix.

    Returns:
        vR : (ngs,) ndarray
            The real-space Hartree potential at every grid point.
    '''
    nkpts, ngs, nao = aoR_kpts.shape
    coulG = tools.get_coulG(cell, mf=mf)

    rhoR = np.zeros(ngs)
    for k in range(nkpts):
        rhoR += 1./nkpts*pyscf.pbc.dft.numint.eval_rho(cell, aoR_kpts[k,:,:], dm_kpts[k,:,:])
    rhoG = tools.fft(rhoR, cell.gs)

    vG = coulG*rhoG
    vR = tools.ifft(vG, cell.gs)
    return vR


def get_fock_(mf, h1e_kpts, s1e_kpts, vhf_kpts, dm_kpts, cycle=-1, adiis=None,
              diis_start_cycle=0, level_shift_factor=0, damp_factor=0):
    '''Get the Fock matrices at sampled k-points.

    This is a k-point version of pyscf.scf.hf.get_fock_

    Returns:
       fock : (nkpts, nao, nao) ndarray
    '''
    fock = np.zeros_like(h1e_kpts)
    # By inheritance, this is just pyscf.scf.hf.get_fock_
    fock = pbchf.RHF.get_fock_(mf, h1e_kpts, s1e_kpts,
                               vhf_kpts, dm_kpts,
                               cycle, adiis, diis_start_cycle,
                               level_shift_factor, damp_factor)
    return fock


def make_rdm1(mo_coeff_kpts, mo_occ_kpts):
    nkpts = len(mo_occ_kpts)
    dm_kpts = np.zeros_like(mo_coeff_kpts)
    for k in range(nkpts):
        dm_kpts[k,:,:] = pyscf.scf.hf.make_rdm1(mo_coeff_kpts[k,:,:],
                                                mo_occ_kpts[k,:]).T.conj()
    return dm_kpts


#FIXME: project initial guess for k-point
def init_guess_by_chkfile(cell, chkfile_name, project=True):
    '''Read the KHF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, 3D ndarray
    '''
    #from pyscf.pbc.scf import addons
    mo = pyscf.pbc.scf.chkfile.load(chkfile_name, 'scf/mo_coeff')
    mo_occ = pyscf.pbc.scf.chkfile.load(chkfile_name, 'scf/mo_occ')

    #def fproj(mo):
    #    if project:
    #        return addons.project_mo_nr2nr(chk_cell, mo, cell)
    #    else:
    #        return mo
    dm = make_rdm1(mo, mo_occ)
    return dm


class KRHF(pbchf.RHF):
    '''RHF class with k-point sampling.

    Compared to molecular SCF, some members such as mo_coeff, mo_occ
    now have an additional first dimension for the k-points,
    e.g. mo_coeff is (nkpts, nao, nao) ndarray

    Attributes:
        kpts : (nks,3) ndarray
            The sampling k-points in Cartesian coordinates, in units of 1/Bohr.
    '''
    def __init__(self, cell, kpts, symmetry=None, exxdiv='ewald'):
        pbchf.RHF.__init__(self, cell, exxdiv=exxdiv)
        self.symmetry = symmetry
        if kpts is None:
            self.kpts = np.zeros((1,3))
        else:
            self.kpts = kpts
        if len(self.kpts) == 1 and np.allclose(self.kpts[0], np.zeros(3)):
            self._dtype = np.float64
        else:
            self._dtype = np.complex128
        self.mo_occ = []
        self.mo_coeff_kpts = []

        if cell.ke_cutoff is not None:
            raise RuntimeError("ke_cutoff not supported with K pts yet")

        self.exx_built = False
        if self.exxdiv == 'vcut_ws':
            self.precompute_exx()
        if cell.dimension == 1:
            self.precompute_exx1D()

    def dump_flags(self):
        pbchf.RHF.dump_flags(self)
        if self.exxdiv == 'vcut_ws':
            if self.exx_built is False:
                self.precompute_exx()
            logger.info(self, 'WS alpha = %s', self.exx_alpha)

    def precompute_exx1D(self):
        print "# Precomputing Wigner-Seitz EXX kernel"
        from pyscf.pbc import gto as pbcgto
        Nk = tools.get_monkhorst_pack_size(self.cell, self.kpts)
        print "# Nk =", Nk
        kcell = pbcgto.Cell()
        kcell.atom = 'H 0. 0. 0.'
        kcell.spin = 1
        kcell.unit = 'B'
        kcell.dimension = 3
        kcell.h = self.cell._h * Nk
        Lc = 1.0/np.linalg.norm(np.linalg.inv(kcell.h.T), axis=0)
        print "# Lc =", Lc
        Rin = Lc[:2].min() / 2.0
        print "# Rin =", Rin
        # ASE:
        alpha = 5./Rin # sqrt(-ln eps) / Rc, eps ~ 10^{-11}
        #alpha = 4./Rin # sqrt(-ln eps) / Rc, eps ~ 10^{-11}
        print "alpha   = ", alpha
        kcell.gs = np.array([2*int(L*alpha*3.0) for L in Lc])
        # Set the resolution to be equal to or smaller than
        # that prescribed by the planewave basis
        kcell.gs = np.array([max(kcell.gs[i],self.cell.gs[i]) for i in range(3)])
        # JM commented out
        kcell.gs[2] = 0
        # QE:
        #alpha = 3./Rin * np.sqrt(0.5)
        #kcell.gs = (4*alpha*np.linalg.norm(kcell.h,axis=0)).astype(int)
        print "# kcell.gs FFT =", kcell.gs
        kcell.build(False,False)
        vR = tools.ifft( tools.get_coulG(kcell), kcell.gs )
        kngs = len(vR)
        print "# kcell kngs =", kngs
        rs = pyscf.pbc.dft.gen_grid.gen_uniform_grids(kcell)

        # minimum image convention for a unit cell with orthogonal lattice vectors
        #

        # Uncomment the following to get the minimum image points over blocks
        #

        #mem = 200
        #blksize = int(mem * 1024 * 1024. / (4*8))
        #rhoValsMin = 999.
        #rhoValsMax = 0.
        #rsnorm = np.zeros(len(rs))
        ## calculating (x^2 + y^2) ** 0.5
        ## doing this over blocks so we don't run into memory issues with the np.dot
        #for i in range(int(np.ceil(rs.shape[0]*1./blksize))):
        #    lower = i*blksize
        #    upper = min((i+1)*blksize,rs.shape[0])
        #    red_rs = np.dot(rs[lower:upper],np.linalg.inv(kcell._h.T)) # getting relative positions
        #    rs[lower:upper] = rs[lower:upper] - np.dot(( red_rs >= 0.5 ),kcell._h.T) # minimum-image convention
        #    rhoVals = np.sqrt(np.einsum('ij,ij->i',rs[lower:upper,:2],rs[lower:upper,:2])) # (x^2+y^2)**(0.5)
        #    rsnorm[lower:upper] = rhoVals.copy()
        #    rhoValsMin = min(min(rhoVals),rhoValsMin)
        #    rhoValsMax = max(max(rhoVals),rhoValsMax)

        red_rs = np.dot(rs,np.linalg.inv(kcell._h.T))   # getting relative coordinates
        rs = rs - np.dot( (red_rs >= 0.5 ), kcell._h.T) # minimum image convention for orthorhombic cell
        rsnorm = np.sqrt(np.einsum('ij,ij->i',rs[:,:2],rs[:,:2]))
        rhoValsMin = min(rsnorm)
        rhoValsMax = max(rsnorm)
        rhoValsMax *= 1.1
        dx = 0.01
        nrhoVals = int((rhoValsMax - rhoValsMin)/dx)
        print "rho min = ", rhoValsMin
        print "rho max = ", rhoValsMax
        print "rho dx  = ", dx
        print "rho nv  = ", nrhoVals

        # Builds the C^{alpha}_{Gz+k}( (x^2+y^2)**(0.5) ) for each unique |Gz+k| in the list self.cell.Gv + {k},
        # interpolating over values of (x^2+y^2)**(0.5)
        #
        c1f = conv1D.conv1Dfunc(alpha, self.cell.Gv, np.linspace(rhoValsMin,rhoValsMax,nrhoVals),
              kpts=np.append(self.kpts,[[0,0,0]],axis=0))

        gz = c1f.gz
        fftgz = np.zeros((len(gz),kngs),dtype=vR.dtype)
        area = kcell._h[0,0]*kcell._h[1,1]
        for igz,cgz in enumerate(gz):
            fftgz[igz] = (area/kngs) * tools.fft(c1f.getCmatrixAtGz(cgz,rsnorm),kcell.gs)
        vG = fftgz

        self.c1f = c1f
        self.exx_alpha = alpha
        self.exx_kcell = kcell
        self.exx_q = kcell.Gv
        self.exx_gz = gz
        self.exx_vq = vG
        self.exx_built = True

        #mdf._erfc_nuc(cell, eta, kpts)
        print "# Finished precomputing"

    def precompute_exx(self):
        print "# Precomputing Wigner-Seitz EXX kernel"
        from pyscf.pbc import gto as pbcgto
        Nk = tools.get_monkhorst_pack_size(self.cell, self.kpts)
        print "# Nk =", Nk
        kcell = pbcgto.Cell()
        kcell.atom = 'H 0. 0. 0.'
        kcell.spin = 1
        kcell.unit = 'B'
        kcell.h = self.cell._h * Nk
        Lc = 1.0/np.linalg.norm(np.linalg.inv(kcell.h.T), axis=0)
        print "# Lc =", Lc
        Rin = Lc.min() / 2.0
        print "# Rin =", Rin
        # ASE:
        alpha = 5./Rin # sqrt(-ln eps) / Rc, eps ~ 10^{-11}
        kcell.gs = np.array([2*int(L*alpha*3.0) for L in Lc])
        # QE:
        #alpha = 3./Rin * np.sqrt(0.5)
        #kcell.gs = (4*alpha*np.linalg.norm(kcell.h,axis=0)).astype(int)
        print "# kcell.gs FFT =", kcell.gs
        kcell.build(False,False)
        vR = tools.ifft( tools.get_coulG(kcell), kcell.gs )
        kngs = len(vR)
        print "# kcell kngs =", kngs
        rs = pyscf.pbc.dft.gen_grid.gen_uniform_grids(kcell)
        corners = np.dot(np.indices((2,2,2)).reshape((3,8)).T, kcell._h.T)
        for i, rv in enumerate(rs):
            # Minimum image convention to corners of kcell parallelepiped
            r = np.linalg.norm(rv-corners, axis=1).min()
            if np.isclose(r, 0.):
                vR[i] = 2*alpha / np.sqrt(np.pi)
            else:
                vR[i] = scipy.special.erf(alpha*r) / r
        vG = (kcell.vol/kngs) * tools.fft(vR, kcell.gs)
        self.exx_alpha = alpha
        self.exx_kcell = kcell
        self.exx_q = kcell.Gv
        self.exx_vq = vG
        self.exx_built = True
        print "# Finished precomputing"

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell

        if key.lower() == '1e':
            return self.init_guess_by_1e(cell)
        elif key.lower() == 'chkfile':
            return self.init_guess_by_chkfile()
        else:
            dm = pyscf.scf.hf.get_init_guess(cell, key)
            nao = cell.nao_nr()
            nkpts = len(self.kpts)
            dm_kpts = np.zeros((nkpts,nao,nao), np.complex128)

            # Use the molecular "unit cell" dm for each k-point
            for k in range(nkpts):
                dm_kpts[k,:,:] = dm

        return dm_kpts

    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        return self._safe_cast(get_hcore(self, cell, kpts))

    def get_ovlp(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        return self._safe_cast(get_ovlp(self, cell, kpts))

    def get_j(self, cell=None, dm_kpts=None, hermi=1, kpt=None, kpt_band=None):
        # Must use 'kpt' kwarg
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpts
        kpts = kpt
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (time.clock(), time.time())
        vj = get_j(self, cell, dm_kpts, kpts, kpt_band)
        logger.timer(self, 'vj', *cpu0)
        return self._safe_cast(vj)

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpt=None, kpt_band=None):
        # Must use 'kpt' kwarg
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpts
        kpts = kpt
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (time.clock(), time.time())
        vj, vk = get_jk(self, cell, dm_kpts, kpts, kpt_band)
        logger.timer(self, 'vj and vk', *cpu0)
        return self._safe_cast(vj), self._safe_cast(vk)

    def get_fock_(self, h1e_kpts, s1e, vhf, dm_kpts, cycle=-1, adiis=None,
                  diis_start_cycle=None, level_shift_factor=None, damp_factor=None):

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if damp_factor is None:
            damp_factor = self.damp

        f = get_fock_(self, h1e_kpts, s1e, vhf, dm_kpts, cycle, adiis,
                      diis_start_cycle, level_shift_factor, damp_factor)
        if self.symmetry is not None:
            cart_kpts = self.kpts
            ibzk, ibzk_weight, sym_k, time_reversal_k, irrep = self.symmetry.reduce(np.array(cart_kpts))
            symm.unpack_1e(self.symmetry, cart_kpts, ibzk, ibzk_weight, sym_k, time_reversal_k, irrep, f)
        return self._safe_cast(f)

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpt_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpts is None: kpts = self.kpts
        # TODO: Check incore, direct_scf, _eri's, etc
        vj, vk = self.get_jk(cell, dm, hermi, kpts, kpt_band)
        if self.symmetry is not None:
            cart_kpts = self.kpts
            ibzk, ibzk_weight, sym_k, time_reversal_k, irrep = self.symmetry.reduce(np.array(cart_kpts))
            #symm.unpack_1e(self.symmetry, cart_kpts, ibzk, ibzk_weight, sym_k, time_reversal_k, irrep, vj)
            #symm.unpack_1e(self.symmetry, cart_kpts, ibzk, ibzk_weight, sym_k, time_reversal_k, irrep, vk)
        return vj - vk * .5

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        '''
        returns 1D array of gradients, like non K-pt version
        note that occ and virt indices of different k pts now occur
        in sequential patches of the 1D array
        '''
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)

        nkpts = len(self.kpts)

        # make this closer to the non-kpt one
        grad_kpts = np.empty(0,)

        for k in range(nkpts):
            grad = pyscf.scf.hf.RHF.get_grad(self,
                        mo_coeff_kpts[k,:,:], mo_occ_kpts[k,:], fock[k,:,:])
            grad_kpts = np.hstack((grad_kpts, grad))
        return self._safe_cast(grad_kpts)

    def eig(self, h_kpts, s_kpts):
        nkpts = len(h_kpts)
        nao = h_kpts.shape[1]
        eig_kpts = np.zeros((nkpts,nao))
        mo_coeff_kpts = np.zeros_like(h_kpts)

        # TODO: should use superclass eig fn here?
        for k in range(nkpts):
            eig_kpts[k,:], mo_coeff_kpts[k,:,:] = pyscf.scf.hf.eig(h_kpts[k,:,:], s_kpts[k,:,:])

        print "mo_coeff"
        print mo_coeff_kpts
        print "eigenvalues"
        print eig_kpts
        return eig_kpts, mo_coeff_kpts

    def get_occ(self, mo_energy_kpts, mo_coeff_kpts):
        '''Label the occupancies for each orbital for sampled k-points.

        This is a k-point version of scf.hf.SCF.get_occ
        '''
        if mo_energy_kpts is None: mo_energy_kpts = self.mo_energy
        mo_occ_kpts = np.zeros_like(mo_energy_kpts)

        nkpts, nao = mo_coeff_kpts.shape[:2]
        nocc = (self.cell.nelectron * nkpts) // 2

        # Sort eigs in each kpt
        mo_energy = np.reshape(mo_energy_kpts, [nkpts*nao])
        # TODO: store mo_coeff correctly (for later analysis)
        #self.mo_coeff = np.reshape(mo_coeff_kpts, [nao, nao*nkpts])
        mo_idx = np.argsort(mo_energy)
        mo_energy = mo_energy[mo_idx]
        for ix in mo_idx[:nocc]:
            k, ikx = divmod(ix, nao)
            # TODO: implement Fermi smearing
            mo_occ_kpts[k, ikx] = 2

        if nocc < mo_energy.size:
            logger.info(self, 'HOMO = %.12g  LUMO = %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
            if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                logger.warn(self, '!! HOMO %.12g == LUMO %.12g',
                            mo_energy[nocc-1], mo_energy[nocc])
        else:
            logger.info(self, 'HOMO = %.12g', mo_energy[nocc-1])
        if self.verbose >= logger.DEBUG:
            np.set_printoptions(threshold=len(mo_energy))
            logger.debug(self, '  mo_energy = %s', mo_energy)
            np.set_printoptions()

        self.mo_energy = mo_energy_kpts
        self.mo_occ = mo_occ_kpts

        return mo_occ_kpts

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None):
        '''One particle density matrix at each k-point.

        Returns:
            dm_kpts : (nkpts, nao, nao) ndarray
        '''
        if mo_coeff_kpts is None:
            # Note: this is actually "self.mo_coeff_kpts"
            # which is stored in self.mo_coeff of the scf.hf.RHF superclass
            mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None:
            # Note: this is actually "self.mo_occ_kpts"
            # which is stored in self.mo_occ of the scf.hf.RHF superclass
            mo_occ_kpts = self.mo_occ

        return self._safe_cast(make_rdm1(mo_coeff_kpts, mo_occ_kpts))

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        '''Following pyscf.scf.hf.energy_elec()
        '''
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if h1e_kpts is None: h1e_kpts = self.get_hcore()
        if vhf_kpts is None: vhf_kpts = self.get_veff(self.cell, dm_kpts)

        nkpts = len(dm_kpts)
        e1 = e_coul = 0.
        for k in range(nkpts):
            e1 += 1./nkpts * np.einsum('ij,ji', dm_kpts[k,:,:], h1e_kpts[k,:,:])
            e_coul += 1./nkpts * 0.5 * np.einsum('ij,ji', dm_kpts[k,:,:], vhf_kpts[k,:,:])
        if abs(e_coul.imag > 1.e-8):
            raise RuntimeError("Coulomb energy has imaginary part, "
                               "something is wrong!", e_coul.imag)
        e1 = e1.real
        e_coul = e_coul.real
        logger.debug(self, 'e1     = %.15g', e1    )
        logger.debug(self, 'E_coul = %.15g', e_coul)
        return e1+e_coul, e_coul

    def get_bands(self, kpt_band, cell=None, dm_kpts=None, kpts=None):
        '''Get energy bands at a given (arbitrary) 'band' k-point.

        Returns:
            mo_energy : (nao,) ndarray
                Bands energies E_n(k)
            mo_coeff : (nao, nao) ndarray
                Band orbitals psi_n(k)
        '''
        if cell is None: cell = self.cell
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if kpts is None: kpts = self.kpts

        fock = pbchf.get_hcore(cell, kpt_band) \
                + self.get_veff(kpts=kpts, kpt_band=kpt_band)
        s1e = pbchf.get_ovlp(cell, kpt_band)
        fock = self._safe_cast(fock)
        s1e = self._safe_cast(s1e)
        mo_energy, mo_coeff = pyscf.scf.hf.eig(fock, s1e)
        return mo_energy, mo_coeff

    def init_guess_by_chkfile(self, chk=None, project=True):
        if chk is None: chk = self.chkfile
        return init_guess_by_chkfile(self.cell, chk, project)
