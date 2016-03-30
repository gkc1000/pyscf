import sys
import time
import tempfile
import numpy
import h5py
import kpoint_helper
import mpi_load_balancer
from mpi4py import MPI

import pyscf.pbc.tools.pbc as tools
from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
from pyscf.pbc import lib as pbclib
import pyscf.cc
import pyscf.cc.ccsd
from pyscf.cc.ccsd import _cp
from pyscf.pbc.cc import kintermediates_rhf as imdk
from pyscf.pbc.lib.linalg_helper import eigs

#einsum = numpy.einsum
einsum = pbclib.einsum


# This is restricted (R)CCSD
# following Hirata, ..., Barlett, J. Chem. Phys. 120, 2581 (2004)

def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           max_memory=2000, verbose=logger.INFO):
    """Exactly the same as pyscf.cc.ccsd.kernel, which calls a
    *local* energy() function."""
    comm = cc.comm
    rank = cc.rank
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)


    if t1 is None and t2 is None:
        t1, t2 = cc.init_amps(eris)[1:]
    elif t1 is None:
        nocc = cc.nocc()
        nvir = cc.nmo() - nocc
        t1 = numpy.zeros((nocc,nvir), eris.dtype)
    elif t2 is None:
        t2 = cc.init_amps(eris)[2]

    cput1 = cput0 = (time.clock(), time.time())
    nkpts, nocc, nvir = t1.shape
    eold = 0.0 + 1j*0.0
    eccsd = 0.0 + 1j*0.0
    if cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1,t2)

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris, max_memory)
        normt = numpy.array(0.0,dtype=numpy.float64)
        normt += numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        comm.Allreduce(MPI.IN_PLACE, normt, op=MPI.SUM)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, energy(cc, t1, t2, eris)
        if cc.rank == 0:
            log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                     istep, eccsd, eccsd - eold, normt)
            cput1 = log.timer('CCSD iter', *cput1)
        #if istep == 2:
        #    eccsd = eold = normt = 0.0
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    if cc.rank == 0:
        log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2

@profile
def update_amps(cc, t1, t2, eris, max_memory=2000):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:,:nocc,nocc:]
    foo = fock[:,:nocc,:nocc]
    fvv = fock[:,nocc:,nocc:]

    #mo_e = eris.fock.diagonal()
    #eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    #eijab = lib.direct_sum('ia,jb->ijab',eia,eia)

    ds_type = t1.dtype

    Foo = imdk.cc_Foo(cc,t1,t2,eris)
    Fvv = imdk.cc_Fvv(cc,t1,t2,eris)
    Fov = imdk.cc_Fov(cc,t1,t2,eris)
    Loo = imdk.Loo(cc,t1,t2,eris)
    Lvv = imdk.Lvv(cc,t1,t2,eris)

    #Woooo = imdk.cc_Woooo(cc,t1,t2,eris)
    #Wvvvv = imdk.cc_Wvvvv(cc,t1,t2,eris)
    #Wvoov = imdk.cc_Wvoov(cc,t1,t2,eris)
    #Wvovo = imdk.cc_Wvovo(cc,t1,t2,eris)

    print "setting up file..."
    _tmpfile2_name = None
    if cc.rank == 0:
        _tmpfile2_name = "cc_intermediates.hdf5"
    _tmpfile2_name = cc.comm.bcast(_tmpfile2_name, root=0)
    feri2 = h5py.File(_tmpfile2_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)

    tau1      = feri2.create_dataset('tau1',      (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=ds_type)
    tau1_ooVv = feri2.create_dataset('tau1_ooVv', (nkpts,nkpts,nkpts*nvir,nocc,nocc,nvir), dtype=ds_type)
    tau2_Oovv = feri2.create_dataset('tau2_Oovv', (nkpts,nkpts,nkpts*nocc,nocc,nvir,nvir), dtype=ds_type)
    Woooo = feri2.create_dataset('Woooo', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=ds_type)
    Wvvvv = feri2.create_dataset('Wvvvv', (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=ds_type)
    Wvoov = feri2.create_dataset('Wvoov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
    Wvovo = feri2.create_dataset('Wvovo', (nkpts,nkpts,nkpts,nvir,nocc,nvir,nocc), dtype=ds_type)

    print "making taus..."
    imdk.cc_tau1(cc,t1,t2,eris,feri2)
    feri2.close()
    feri2 = h5py.File(_tmpfile2_name, 'r+', driver='mpio', comm=MPI.COMM_WORLD)

    print "making Woooo..."
    imdk.cc_Woooo(cc,t1,t2,eris,feri2)
    feri2.close()
    feri2 = h5py.File(_tmpfile2_name, 'r+', driver='mpio', comm=MPI.COMM_WORLD)

    print "making Wvvvv..."
    imdk.cc_Wvvvv(cc,t1,t2,eris,feri2)
    feri2.close()
    feri2 = h5py.File(_tmpfile2_name, 'r+', driver='mpio', comm=MPI.COMM_WORLD)

    print "making Wvoov..."
    imdk.cc_Wvoov(cc,t1,t2,eris,feri2)
    feri2.close()
    feri2 = h5py.File(_tmpfile2_name, 'r+', driver='mpio', comm=MPI.COMM_WORLD)

    print "making Wvovo..."
    imdk.cc_Wvovo(cc,t1,t2,eris,feri2)
    feri2.close()
    feri2 = h5py.File(_tmpfile2_name, 'r+', driver='mpio', comm=MPI.COMM_WORLD)

    Woooo = feri2['Woooo']
    Wvvvv = feri2['Wvvvv']
    Wvoov = feri2['Wvoov']
    Wvovo = feri2['Wvovo']
    tau1_ooVv = feri2['tau1_ooVv']

    print "done making intermediates..."
    # Move energy terms to the other side
    Foo -= foo
    Fvv -= fvv
    Loo -= foo
    Lvv -= fvv

    kconserv = cc.kconserv

    # T1 equation
    # TODO: Check this conj(). Hirata and Bartlett has
    # f_{vo}(a,i), which should be equal to f_{ov}^*(i,a)
    t1new = numpy.empty((nkpts,nocc,nvir),dtype=t1.dtype)
    t1new[:] = fov[:].conj().copy()
    for ka in range(nkpts):
        ki = ka
        # kc == ki; kk == ka
        t1new[ka] += -2.*einsum('kc,ka,ic->ia',fov[ki],t1[ka],t1[ki])
        t1new[ka] += einsum('ac,ic->ia',Fvv[ka],t1[ki])
        t1new[ka] += -einsum('ki,ka->ia',Foo[ki],t1[ka])

        tau_term = numpy.empty((nkpts,nocc,nocc,nvir,nvir),dtype=t1.dtype)
        for kk in range(nkpts):
            tau_term[kk] = 2*t2[kk,ki,kk] - t2[ki,kk,kk].transpose(1,0,2,3)
        tau_term[ka] += einsum('ic,ka->kica',t1[ki],t1[ka])

        for kk in range(nkpts):
            kc = kk
            t1new[ka] += einsum('kc,kica->ia',Fov[kc],tau_term[kk])

            t1new[ka] +=  einsum('akic,kc->ia',2*eris.ovvo[kk,ka,kc].transpose(1,0,3,2),t1[kc])
            t1new[ka] +=  einsum('akci,kc->ia', -eris.ovov[kk,ka,ki].transpose(1,0,3,2),t1[kc])

            for kc in range(nkpts):
                kd = kconserv[ka,kc,kk]

                Svovv = 2*eris.ovvv[kk,ka,kd].transpose(1,0,3,2) - eris.ovvv[kk,ka,kc].transpose(1,0,2,3)
                tau_term_1 = t2[ki,kk,kc].copy()
                tau1_ooVv = feri2['tau1_ooVv']
                if ki == kc and kk == kd:
                    tau_term_1 += einsum('ic,kd->ikcd',t1[ki],t1[kk])
                t1new[ka] += einsum('akcd,ikcd->ia',Svovv,tau_term_1)
            #t1new[ka] += einsum('akcd,ikcd->ia',Svoov[k

            for kc in range(nkpts):
                # kk - ki + kl = kc
                #  => kl = ki - kk + kc
                kl = kconserv[ki,kk,kc]
                Sooov = 2*eris.ooov[kk,kl,ki] - eris.ooov[kl,kk,ki].transpose(1,0,2,3)
                tau_term_1 = t2[kk,kl,ka].copy()
                if kk == ka and kl == kc:
                    tau_term_1 += einsum('ka,lc->klac',t1[ka],t1[kc])
                t1new[ka] += -einsum('klic,klac->ia',Sooov,tau_term_1)

    # T2 equation
    # For conj(), see Hirata and Bartlett, Eq. (36)
    t2new = numpy.array(eris.oovv, copy=True).conj()
    for ki in range(nkpts):
      for kj in range(nkpts):
        for ka in range(nkpts):
            # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
            kb = kconserv[ki,ka,kj]

            for kl in range(nkpts):
                # kk - ki + kl = kj
                # => kk = kj - kl + ki
                kk = kconserv[kj,kl,ki]
                t2new[ki,kj,ka] += einsum('klij,klab->ijab',Woooo[kk,kl,ki],t2[kk,kl,ka])
                if kl == kb and kk == ka:
                    t2new[ki,kj,ka] += einsum('klij,ka,lb->ijab',Woooo[ka,kb,ki],t1[ka],t1[kb])

            for kc in range(nkpts):
                kd = kconserv[ka,kc,kb]
                tau_term = t2[ki,kj,kc].copy()
                if ki == kc and kj == kd:
                    tau_term += einsum('ic,jd->ijcd',t1[ki],t1[kj])
                t2new[ki,kj,ka] += einsum('abcd,ijcd->ijab',Wvvvv[ka,kb,kc],tau_term)

            t2new[ki,kj,ka] += einsum('ac,ijcb->ijab',Lvv[ka],t2[ki,kj,ka])
            #P(ij)P(ab)
            t2new[ki,kj,ka] += einsum('bc,jica->ijab',Lvv[kb],t2[kj,ki,kb])

            t2new[ki,kj,ka] += einsum('ki,kjab->ijab',-Loo[ki],t2[ki,kj,ka])
            #P(ij)P(ab)
            t2new[ki,kj,ka] += einsum('kj,kiba->ijab',-Loo[kj],t2[kj,ki,kb])

            tmp2 = eris.ovvv[ki,kj,ka].transpose(2,3,0,1).conj() - einsum('kbic,ka->abic',eris.ovov[ka,kb,ki],t1[ka])
            tmp  = einsum('abic,jc->ijab',tmp2,t1[kj])
            t2new[ki,kj,ka] += tmp
            #P(ij)P(ab)
            tmp2 = eris.ovvv[kj,ki,kb].transpose(2,3,0,1).conj() - einsum('kajc,kb->bajc',eris.ovov[kb,ka,kj],t1[kb])
            tmp  = einsum('bajc,ic->ijab',tmp2,t1[ki])
            t2new[ki,kj,ka] += tmp

            # ka - ki + kk = kj
            # => kk = ki - ka + kj
            kk = kconserv[ki,ka,kj]
            tmp2 = eris.ooov[kj,ki,kb].transpose(3,2,1,0).conj() + einsum('akic,jc->akij',eris.ovvo[kk,ka,kj].transpose(1,0,3,2),t1[kj])
            tmp  = einsum('akij,kb->ijab',tmp2,t1[kb])
            t2new[ki,kj,ka] -= tmp
            #P(ij)P(ab)
            kk = kconserv[kj,kb,ki]
            tmp2 = eris.ooov[ki,kj,ka].transpose(3,2,1,0).conj() + einsum('bkjc,ic->bkji',eris.ovvo[kk,kb,ki].transpose(1,0,3,2),t1[ki])
            tmp  = einsum('bkji,ka->ijab',tmp2,t1[ka])
            t2new[ki,kj,ka] -= tmp

            for kk in range(nkpts):
                kc = kconserv[ka,ki,kk]
                tmp_voov = 2*numpy.array(Wvoov[ka,kk,ki]) - numpy.array(Wvovo[ka,kk,kc]).transpose(0,1,3,2)
                tmp = einsum('akic,kjcb->ijab',tmp_voov,t2[kk,kj,kc])
                #tmp = 2*einsum('akic,kjcb->ijab',Wvoov[ka,kk,ki],t2[kk,kj,kc]) - \
                #        einsum('akci,kjcb->ijab',Wvovo[ka,kk,kc],t2[kk,kj,kc])
                t2new[ki,kj,ka] += tmp
                #P(ij)P(ab)
                kc = kconserv[kb,kj,kk]
                tmp_voov = 2*numpy.array(Wvoov[kb,kk,kj]) - numpy.array(Wvovo[kb,kk,kc]).transpose(0,1,3,2)
                tmp = einsum('bkjc,kica->ijab',tmp_voov,t2[kk,ki,kc])
                #tmp = 2*einsum('bkjc,kica->ijab',Wvoov[kb,kk,kj],t2[kk,ki,kc]) - \
                #        einsum('bkcj,kica->ijab',Wvovo[kb,kk,kc],t2[kk,ki,kc])
                t2new[ki,kj,ka] += tmp

                kc = kconserv[ka,ki,kk]
                tmp = einsum('akic,kjbc->ijab',Wvoov[ka,kk,ki],t2[kk,kj,kb])
                t2new[ki,kj,ka] -= tmp
                #P(ij)P(ab)
                kc = kconserv[kb,kj,kk]
                tmp = einsum('bkjc,kiac->ijab',Wvoov[kb,kk,kj],t2[kk,ki,ka])
                t2new[ki,kj,ka] -= tmp

                kc = kconserv[kk,ka,kj]
                tmp = einsum('bkci,kjac->ijab',Wvovo[kb,kk,kc],t2[kk,kj,ka])
                t2new[ki,kj,ka] -= tmp
                #P(ij)P(ab)
                kc = kconserv[kk,kb,ki]
                tmp = einsum('akcj,kibc->ijab',Wvovo[ka,kk,kc],t2[kk,ki,kb])
                t2new[ki,kj,ka] -= tmp

    eia = numpy.zeros(shape=t1new.shape, dtype=t1new.dtype)
    for ki in range(nkpts):
        for i in range(nocc):
            for a in range(nvir):
                eia[ki,i,a] = foo[ki,i,i] - fvv[ki,a,a]
        t1new[ki] /= eia[ki]

    for ki in range(nkpts):
      for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[ki,ka,kj]
            eia = numpy.diagonal(foo[ki]).reshape(-1,1) - numpy.diagonal(fvv[ka])
            ejb = numpy.diagonal(foo[kj]).reshape(-1,1) - numpy.diagonal(fvv[kb])
            eijab = pyscf.lib.direct_sum('ia,jb->ijab',eia,ejb)
            t2new[ki,kj,ka] /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)
#    sys.exit("exiting for testing...")

    cc.comm.Barrier()
    feri2.close()

    return t1new, t2new


def energy(cc, t1, t2, eris):
    comm = cc.comm
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    fock = eris.fock
    e = numpy.array(0.0,dtype=numpy.complex128)
    for ki in range(nkpts):
        e += 2*einsum('ia,ia', fock[ki,:nocc,nocc:], t1[ki])
    t1t1 = numpy.zeros(shape=t2.shape,dtype=t2.dtype)
    for ki in range(nkpts):
        ka = ki
        for kj in range(nkpts):
            #kb = kj
            t1t1[ki,kj,ka] = einsum('ia,jb->ijab',t1[ki],t1[kj])
    tau = t2 + t1t1
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                e += einsum('ijab,ijab', 2*tau[ki,kj,ka], eris.oovv[ki,kj,ka])
                e += einsum('ijab,ijba',  -tau[ki,kj,ka], eris.oovv[ki,kj,kb])
    comm.Barrier()
    e /= nkpts
    return e.real


class RCCSD(pyscf.cc.ccsd.CCSD):

    def __init__(self, mf, abs_kpts, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        ####################################################
        # MPI data                                         #
        ####################################################
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
        self.comm = MPI.COMM_WORLD

        self._kpts = abs_kpts
        self.nkpts = len(self._kpts)
        self.kconserv = tools.get_kconserv(mf.cell, abs_kpts)
        self.khelper  = kpoint_helper.unique_pqr_list(mf.cell, abs_kpts)
        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)

    def dump_flags(self):
        pyscf.cc.ccsd.CCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** EOM CC flags ********')

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        rank = self.rank
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = len(self._kpts)
        t1 = numpy.zeros((nkpts,nocc,nvir), dtype=numpy.complex128)
        t2 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=numpy.complex128)
        local_mp2 = numpy.array(0.0,dtype=numpy.complex128)
        #woovv = numpy.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=numpy.complex128)
        self.emp2 = 0
        foo = eris.fock[:,:nocc,:nocc].copy()
        fvv = eris.fock[:,nocc:,nocc:].copy()
        #eris_oovv = numpy.asarray(eris.oovv).copy()
        eia = numpy.zeros((nocc,nvir))
        eijab = numpy.zeros((nocc,nocc,nvir,nvir))

        kconserv = self.kconserv
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(2,2,nkpts,))
        loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

        cput1 = time.clock(), time.time()
        good2go = True
        while(good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
            for ki in ranges0:
                for kj in ranges1:
                    for ka in ranges2:
                        kb = kconserv[ki,ka,kj]
                        eia = numpy.diagonal(foo[ki]).reshape(-1,1) - numpy.diagonal(fvv[ka])
                        ejb = numpy.diagonal(foo[kj]).reshape(-1,1) - numpy.diagonal(fvv[kb])
                        eijab = pyscf.lib.direct_sum('ia,jb->ijab',eia,ejb)
                        oovv_ijab = numpy.array(eris.oovv[ki,kj,ka])
                        oovv_ijba = numpy.array(eris.oovv[ki,kj,kb]).transpose(0,1,3,2)
                        woovv = 2.*oovv_ijab - oovv_ijba
                        #woovv = (2*eris_oovv[ki,kj,ka] - eris_oovv[ki,kj,kb].transpose(0,1,3,2))
                        #t2[ki,kj,ka] = numpy.conj(eris_oovv[ki,kj,ka] / eijab)
                        t2[ki,kj,ka] = numpy.conj(oovv_ijab / eijab)
                        local_mp2 += numpy.dot(t2[ki,kj,ka].flatten(),woovv.flatten())
            loader.slave_finished()

        self.comm.Allreduce(MPI.IN_PLACE, local_mp2, op=MPI.SUM)
        self.comm.Allreduce(MPI.IN_PLACE, t2, op=MPI.SUM)
        self.emp2 = local_mp2.real
        self.emp2 /= nkpts

        if rank == 0:
            logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
            logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def nocc(self):
        # Spin orbitals
        # TODO: Possibly change this to make it work with k-points with frozen
        #       As of right now it works, but just not sure how the frozen list will work
        #       with it
        self._nocc = pyscf.cc.ccsd.CCSD.nocc(self)
        self._nocc = (self._nocc // len(self._kpts))
        return self._nocc

    def nmo(self):
        # TODO: Change this for frozen at k-points, seems like it should work
        if isinstance(self.frozen, (int, numpy.integer)):
            self._nmo = len(self.mo_energy[0]) - self.frozen
        else:
            if len(self.frozen) > 0:
                self._nmo = len(self.mo_energy[0]) - len(self.frozen[0])
            else:
                self._nmo = len(self.mo_energy[0])
        return self._nmo

    def ccsd(self, t1=None, t2=None, mo_coeff=None, eris=None):
        if eris is None: eris = self.ao2mo(mo_coeff)
        self.eris = eris
        self._conv, self.ecc, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol,
                       tolnormt=self.conv_tol_normt,
                       max_memory=self.max_memory, verbose=self.verbose)
        if self._conv:
            logger.info(self, 'CCSD converged')
        else:
            logger.info(self, 'CCSD not converge')
        if self._scf.e_tot == 0:
            logger.info(self, 'E_corr = %.16g', self.ecc)
        else:
            logger.info(self, 'E(CCSD) = %.16g  E_corr = %.16g',
                        self.ecc+self._scf.e_tot, self.ecc)
        return self.ecc, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def update_amps(self, t1, t2, eris, max_memory=2000):
        return update_amps(self, t1, t2, eris, max_memory)

    def ipccsd(self, nroots=2*4):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size =  nocc + nkpts*nkpts*nocc*nocc*nvir
        for kshift in range(nkpts):
            self.kshift = kshift
            evals, evecs = eigs(self.ipccsd_matvec, size, nroots)
            numpy.set_printoptions(precision=16)
            print "kshift evals : ", evals[:nroots]
        return evals.real[:nroots], evecs

    def ipccsd_matvec(self, vector):
    ########################################################
    # FOLLOWING:                                           #
    # Z. Tu, F. Wang, and X. Li                            #
    # J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)        #
    ########################################################
        r1,r2 = self.vector_to_amplitudes_ip(vector)

        t1,t2,eris = self.t1, self.t2, self.eris
        nkpts = self.nkpts
        kshift = self.kshift
        kconserv = self.kconserv

        Lvv = imdk.Lvv(self,t1,t2,eris)
        Loo = imdk.Loo(self,t1,t2,eris)
        Fov = imdk.cc_Fov(self,t1,t2,eris)
        Wooov = imdk.Wooov(self,t1,t2,eris)
        Wovvo = imdk.Wovvo(self,t1,t2,eris)
        Wovoo = imdk.Wovoo(self,t1,t2,eris)
        Woooo = imdk.Woooo(self,t1,t2,eris)
        Wovov = imdk.Wovov(self,t1,t2,eris)
        Woovv = eris.oovv

        Hr1 = -einsum('ki,k->i',Loo[kshift],r1)
        for kl in range(nkpts):
            Hr1 += 2.*einsum('ld,ild->i',Fov[kl],r2[kshift,kl])
            Hr1 +=   -einsum('ld,lid->i',Fov[kl],r2[kl,kshift])
            for kk in range(nkpts):
                kd = kconserv[kk,kshift,kl]
                Hr1 += -2.*einsum('klid,kld->i',Wooov[kk,kl,kshift],r2[kk,kl])
                Hr1 +=     einsum('lkid,kld->i',Wooov[kl,kk,kshift],r2[kk,kl])

        Hr2 = numpy.zeros(r2.shape,dtype=t1.dtype)
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki,kshift,kj]
                Hr2[ki,kj] += einsum('bd,ijd->ijb',Lvv[kb],r2[ki,kj])
                Hr2[ki,kj] -= einsum('li,ljb->ijb',Loo[ki],r2[ki,kj])
                Hr2[ki,kj] -= einsum('lj,ilb->ijb',Loo[kj],r2[ki,kj])
                Hr2[ki,kj] -= einsum('kbij,k->ijb',Wovoo[kshift,kb,ki],r1)
                for kl in range(nkpts):
                    kk = kconserv[ki,kl,kj]
                    Hr2[ki,kj] += einsum('klij,klb->ijb',Woooo[kk,kl,ki],r2[kk,kl])
                    kd = kconserv[kl,kj,kb]
                    Hr2[ki,kj] += 2.*einsum('lbdj,ild->ijb',Wovvo[kl,kb,kd],r2[ki,kl])
                    Hr2[ki,kj] += -einsum('lbdj,lid->ijb',Wovvo[kl,kb,kd],r2[kl,ki])
                    Hr2[ki,kj] += -einsum('lbjd,ild->ijb',Wovov[kl,kb,kj],r2[ki,kl]) #typo in nooijen's paper
                    kd = kconserv[kl,ki,kb]
                    Hr2[ki,kj] += -einsum('lbid,ljd->ijb',Wovov[kl,kb,ki],r2[kl,kj])
                    for kk in range(nkpts):
                        kc = kshift
                        kd = kconserv[kl,kc,kk]
                        tmp = ( 2.*einsum('lkdc,kld->c',Woovv[kl,kk,kd],r2[kk,kl])
                                  -einsum('kldc,kld->c',Woovv[kk,kl,kd],r2[kk,kl]) )
                        Hr2[ki,kj] += -einsum('c,ijcb->ijb',tmp,t2[ki,kj,kshift])

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ip(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts

        r1 = vector[:nocc].copy()
        r2 = numpy.zeros((nkpts,nkpts,nocc,nocc,nvir), vector.dtype)
        index = nocc
        for ki in range(nkpts):
            for kj in range(nkpts):
                for i in range(nocc):
                    for j in range(nocc):
                        for a in range(nvir):
                            r2[ki,kj,i,j,a] =  vector[index]
                            index += 1
        return [r1,r2]

    def amplitudes_to_vector_ip(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size = nocc + nkpts*nkpts*nocc*nocc*nvir

        vector = numpy.zeros((size), r1.dtype)
        vector[:nocc] = r1.copy()
        index = nocc
        for ki in range(nkpts):
            for kj in range(nkpts):
                for i in range(nocc):
                    for j in range(nocc):
                        for a in range(nvir):
                            vector[index] = r2[ki,kj,i,j,a]
                            index += 1
        return vector

    def eaccsd(self, nroots=2*4):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size =  nvir + nkpts*nkpts*nocc*nvir*nvir
        for kshift in range(nkpts):
            self.kshift = kshift
            evals, evecs = eigs(self.eaccsd_matvec, size, nroots)
            numpy.set_printoptions(precision=16)
            print "kshift evals : ", evals[:nroots]
        return evals.real[:nroots], evecs

    def eaccsd_matvec(self, vector):
    ########################################################
    # FOLLOWING:                                           #
    # M. Nooijen and R. J. Bartlett,                       #
    # J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)        #
    ########################################################
        r1,r2 = self.vector_to_amplitudes_ea(vector)

        t1,t2,eris = self.t1, self.t2, self.eris
        nkpts = self.nkpts
        kshift = self.kshift
        kconserv = self.kconserv

        Lvv = imdk.Lvv(self,t1,t2,eris)
        Loo = imdk.Loo(self,t1,t2,eris)
        Fov = imdk.cc_Fov(self,t1,t2,eris)
        Wvovv = imdk.Wvovv(self,t1,t2,eris)
        Wvvvo = imdk.Wvvvo(self,t1,t2,eris)
        Wovvo = imdk.Wovvo(self,t1,t2,eris)
        Wvvvv = imdk.Wvvvv(self,t1,t2,eris)
        Woovv = eris.oovv
        Wovov = imdk.Wovov(self,t1,t2,eris)

        Hr1 = einsum('ac,c->a',Lvv[kshift],r1)
        for kl in range(nkpts):
            Hr1 += 2.*einsum('ld,lad->a',Fov[kl],r2[kl,kshift])
            Hr1 +=   -einsum('ld,lda->a',Fov[kl],r2[kl,kl])
            for kc in range(nkpts):
                kd = kconserv[kshift,kc,kl]
                Hr1 +=  2.*einsum('alcd,lcd->a',Wvovv[kshift,kl,kc],r2[kl,kc])
                Hr1 +=    -einsum('aldc,lcd->a',Wvovv[kshift,kl,kd],r2[kl,kc])

        Hr2 = numpy.zeros(r2.shape,dtype=t1.dtype)
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift,ka,kj]
                Hr2[kj,ka] += einsum('abcj,c->jab',Wvvvo[ka,kb,kshift],r1)
                Hr2[kj,ka] -= einsum('lj,lab->jab',Loo[kj],r2[kj,ka])
                Hr2[kj,ka] += einsum('ac,jcb->jab',Lvv[ka],r2[kj,ka])
                Hr2[kj,ka] += einsum('bd,jad->jab',Lvv[kb],r2[kj,ka])
                for kd in range(nkpts):
                    kc = kconserv[ka,kd,kb]
                    Hr2[kj,ka] += einsum('abcd,jcd->jab',Wvvvv[ka,kb,kc],r2[kj,kc])
                    kl = kconserv[kd,kb,kj]
                    Hr2[kj,ka] += 2.*einsum('lbdj,lad->jab',Wovvo[kl,kb,kd],r2[kl,ka])
                    #Wvovo[kb,kl,kd,kj] <= Wovov[kl,kb,kj,kd].transpose(1,0,3,2)
                    Hr2[kj,ka] += -einsum('bldj,lad->jab',Wovov[kl,kb,kj].transpose(1,0,3,2),r2[kl,ka])
                    #Wvoov[kb,kl,kj,kd] <= Wovvo[kl,kb,kd,kj].transpose(1,0,3,2)
                    Hr2[kj,ka] += -einsum('bljd,lda->jab',Wovvo[kl,kb,kd].transpose(1,0,3,2),r2[kl,kd])
                    kl = kconserv[kd,ka,kj]
                    #Wvovo[ka,kl,kd,kj] <= Wovov[kl,ka,kj,kd].transpose(1,0,3,2)
                    Hr2[kj,ka] += -einsum('aldj,ldb->jab',Wovov[kl,ka,kj].transpose(1,0,3,2),r2[kl,kd])
                    for kc in range(nkpts):
                        kk = kshift
                        kl = kconserv[kc,kk,kd]
                        tmp = ( 2.*einsum('klcd,lcd->k',Woovv[kk,kl,kc],r2[kl,kc])
                                  -einsum('kldc,lcd->k',Woovv[kk,kl,kd],r2[kl,kc]) )
                        Hr2[kj,ka] += -einsum('k,kjab->jab',tmp,t2[kshift,kj,ka])

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ea(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts

        r1 = vector[:nvir].copy()
        r2 = numpy.zeros((nkpts,nkpts,nocc,nvir,nvir), vector.dtype)
        index = nvir
        for kj in range(nkpts):
            for ka in range(nkpts):
                for j in range(nocc):
                    for a in range(nvir):
                        for b in range(nvir):
                            r2[kj,ka,j,a,b] = vector[index]
                            index += 1
        return [r1,r2]

    def amplitudes_to_vector_ea(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size = nvir + nkpts*nkpts*nocc*nvir*nvir

        vector = numpy.zeros((size), r1.dtype)
        vector[:nvir] = r1.copy()
        index = nvir
        for kj in range(nkpts):
            for ka in range(nkpts):
                for j in range(nocc):
                    for a in range(nvir):
                        for b in range(nvir):
                            vector[index] = r2[kj,ka,j,a,b]
                            index += 1
        return vector

class _ERIS:
    #@profile
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=pyscf.ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        moidx = numpy.ones(cc.mo_energy.shape, dtype=numpy.bool)
        rank = cc.rank
        nkpts = len(cc._kpts)
        nmo = cc.nmo()
        #TODO check that this and kccsd work for frozen...
        if isinstance(cc.frozen, (int, numpy.integer)):
            moidx[:,:cc.frozen] = False
        elif len(cc.frozen) > 0:
            moidx[:,numpy.asarray(cc.frozen)] = False
        if mo_coeff is None:
            self.mo_coeff = numpy.zeros((nkpts,nmo,nmo),dtype=cc.mo_coeff.dtype)
            for kp in range(nkpts):
                self.mo_coeff[kp] = cc.mo_coeff[kp][:,moidx[kp]]
            mo_coeff = self.mo_coeff
            self.fock = numpy.zeros((nkpts,nmo,nmo),dtype=cc.mo_coeff.dtype)
            for kp in range(nkpts):
                self.fock[kp] = numpy.diag(cc.mo_energy[kp][moidx[kp]]).astype(mo_coeff.dtype)
        else:  # If mo_coeff is not canonical orbital
            self.mo_coeff = mo_coeff = mo_coeff[:,:,moidx]
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
            self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc()
        nmo = cc.nmo()
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = pyscf.cc.ccsd._mem_usage(nocc, nvir)
        mem_now = pyscf.lib.current_memory()[0]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and False and (mem_incore+mem_now < cc.max_memory)
            or cc.mol.incore_anyway):
            kconserv = cc.kconserv
            khelper = cc.khelper #kpoint_helper.unique_pqr_list(cc._scf.cell,cc._kpts)

            unique_klist = khelper.get_uniqueList()
            nUnique_klist = khelper.nUnique
            loader = mpi_load_balancer.load_balancer(BLKSIZE=(2,2,nkpts,))
            loader.set_ranges((range(nUnique_klist),))

            eri = numpy.zeros((nkpts,nkpts,nkpts,nmo,nmo,nmo,nmo), dtype=numpy.complex128)

            good2go = True
            while(good2go):
                good2go, data = loader.slave_set()
                if good2go is False:
                    break
                index = 0
                block = data[index]
                ranges = loader.outblocks[index][block]
                for indices in ranges:
                    kp, kq, kr = unique_klist[indices]
                    ks = kconserv[kp,kq,kr]
                    eri_kpt = pyscf.pbc.ao2mo.general(cc._scf.cell,
                                (mo_coeff[kp,:,:],mo_coeff[kq,:,:],mo_coeff[kr,:,:],mo_coeff[ks,:,:]),
                                (cc._kpts[kp],cc._kpts[kq],cc._kpts[kr],cc._kpts[ks]))
                    eri_kpt = eri_kpt.reshape(nmo,nmo,nmo,nmo)
                    eri[kp,kq,kr] = eri_kpt.copy()
                    loader.slave_finished()

            cc.comm.Barrier()
            rank = cc.rank
            cc.comm.Allreduce(MPI.IN_PLACE, eri, op=MPI.SUM)
            cc.comm.Barrier()

            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ikp, ikq, ikr = khelper.get_irrVec(kp,kq,kr)
                        irr_eri = eri[ikp,ikq,ikr]
                        eri[kp,kq,kr] = khelper.transform_irr2full(irr_eri,kp,kq,kr)
            cc.comm.Barrier()

            # Chemist -> physics notation
            eri = eri.transpose(0,2,1,3,5,4,6)

            self.dtype = eri.dtype
            self.oooo = eri[:,:,:,:nocc,:nocc,:nocc,:nocc].copy() / nkpts
            self.ooov = eri[:,:,:,:nocc,:nocc,:nocc,nocc:].copy() / nkpts
            self.ovoo = eri[:,:,:,:nocc,nocc:,:nocc,:nocc].copy() / nkpts
            self.oovv = eri[:,:,:,:nocc,:nocc,nocc:,nocc:].copy() / nkpts
            self.ovov = eri[:,:,:,:nocc,nocc:,:nocc,nocc:].copy() / nkpts
            self.ovvv = eri[:,:,:,:nocc,nocc:,nocc:,nocc:].copy() / nkpts
            self.vvvv = eri[:,:,:,nocc:,nocc:,nocc:,nocc:].copy() / nkpts
            #ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            #self.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
            #for i in range(nocc):
            #    for j in range(nvir):
            #        self.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
            #self.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)

            # TODO: Avoid this.
            # Store all for now, while DEBUGGING
            self.voov = eri[:,:,:,nocc:,:nocc,:nocc,nocc:].copy() / nkpts
            self.vovo = eri[:,:,:,nocc:,:nocc,nocc:,:nocc].copy() / nkpts
            self.vovv = eri[:,:,:,nocc:,:nocc,nocc:,nocc:].copy() / nkpts
            self.oovo = eri[:,:,:,:nocc,:nocc,nocc:,:nocc].copy() / nkpts
            self.vvov = eri[:,:,:,nocc:,nocc:,:nocc,nocc:].copy() / nkpts
            self.vooo = eri[:,:,:,nocc:,:nocc,:nocc,:nocc].copy() / nkpts
        else:
            print "*** Using HDF5 ERI storage ***"
            _tmpfile1_name = None
            _tmpfile2_name = None
            _tmpfile3_name = None
            _tmpfile4_name = None
            _tmpfile5_name = None
            _tmpfile6_name = None
            _tmpfile7_name = None
            if cc.rank == 0:
                _tmpfile1_name = "eris1.hdf5"
                _tmpfile2_name = "eris2.hdf5"
                _tmpfile3_name = "eris3.hdf5"
                _tmpfile4_name = "eris4.hdf5"
                _tmpfile5_name = "eris5.hdf5"
                _tmpfile6_name = "eris6.hdf5"
                _tmpfile7_name = "eris7.hdf5"
            _tmpfile1_name = cc.comm.bcast(_tmpfile1_name, root=0)
            _tmpfile2_name = cc.comm.bcast(_tmpfile2_name, root=0)
            _tmpfile3_name = cc.comm.bcast(_tmpfile3_name, root=0)
            _tmpfile4_name = cc.comm.bcast(_tmpfile4_name, root=0)
            _tmpfile5_name = cc.comm.bcast(_tmpfile5_name, root=0)
            _tmpfile6_name = cc.comm.bcast(_tmpfile6_name, root=0)
            _tmpfile7_name = cc.comm.bcast(_tmpfile7_name, root=0)
            print _tmpfile1_name
######
            #self.feri1 = h5py.File(_tmpfile1_name, 'r', driver='mpio', comm=MPI.COMM_WORLD)
            #self.oooo  = self.feri1['oooo']
            #self.ooov  = self.feri1['ooov']
            #self.ovoo  = self.feri1['ovoo']
            #self.oovv  = self.feri1['oovv']
            #self.ooVv  = self.feri1['ooVv']
            #self.oOvv  = self.feri1['oOvv']
            #self.ovov  = self.feri1['ovov']
            #self.ovvo  = self.feri1['ovvo']
            #self.ovvv  = self.feri1['ovvv']
            #self.vvvv  = self.feri1['vvvv']
            #self.Soovv = self.feri1['Soovv']
            #self.SoOvv = self.feri1['SoOvv']
            #self.oovv_new  = self.feri1['oovv_new']
            #return
######
            self.feri1 = h5py.File(_tmpfile1_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            self.feri2 = h5py.File(_tmpfile2_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            self.feri3 = h5py.File(_tmpfile3_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            self.feri4 = h5py.File(_tmpfile4_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            self.feri5 = h5py.File(_tmpfile5_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            self.feri6 = h5py.File(_tmpfile6_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            self.feri7 = h5py.File(_tmpfile7_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)

            ds_type = mo_coeff.dtype

            self.oooo  = self.feri1.create_dataset('oooo',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=ds_type)
            self.ooov  = self.feri5.create_dataset('ooov',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
            self.ovoo  = self.feri6.create_dataset('ovoo',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=ds_type)
            self.oovv  = self.feri7.create_dataset('oovv',  (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=ds_type)

            self.ooVv  = self.feri2.create_dataset('ooVv',  (nkpts,nkpts,nkpts*nvir,nocc,nocc,nvir), dtype=ds_type)
            self.oOvv  = self.feri2.create_dataset('oOvv',  (nkpts,nkpts,nkpts*nocc,nocc,nvir,nvir), dtype=ds_type)

            self.ovov  = self.feri3.create_dataset('ovov',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
            self.ovvo  = self.feri3.create_dataset('ovvo',  (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
            self.ovvv  = self.feri3.create_dataset('ovvv',  (nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=ds_type)
            self.vvvv  = self.feri3.create_dataset('vvvv',  (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=ds_type)

            self.Soovv = self.feri4.create_dataset('Soovv', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=ds_type)
            self.SoOvv = self.feri4.create_dataset('SoOvv', (nkpts,nkpts,nkpts*nocc,nocc,nvir,nvir), dtype=ds_type)
            self.oovv_new  = self.feri4.create_dataset('oovv_new',  (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=ds_type)

            #######################################################
            ## Setting up permutational symmetry and MPI stuff    #
            #######################################################
            kconserv = cc.kconserv
            khelper = cc.khelper #kpoint_helper.unique_pqr_list(cc._scf.cell,cc._kpts)
            unique_klist = khelper.get_uniqueList()
            nUnique_klist = khelper.nUnique
            print "ints = ", nUnique_klist

            BLKSIZE = (1,1,nkpts,)
            loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
            loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

            tmp_block_shape = BLKSIZE + (nocc,nocc,nmo,nmo)
            tmp_block = numpy.empty(shape=tmp_block_shape,dtype=ds_type)
            cput1 = time.clock(), time.time()
            good2go = True
            print "performing oopq transformation"
            while(good2go):
                good2go, data = loader.slave_set()
                if good2go is False:
                    break
                ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
                rslice = [slice(0,max(x)-min(x)) for x in ranges0,ranges1,ranges2]
                for kp in ranges0:
                    for kq in ranges2:
                        for kr in ranges1:
                            ks = kconserv[kp,kq,kr]
                            orbo_p = mo_coeff[kp,:,:nocc]
                            orbo_r = mo_coeff[kr,:,:nocc]
                            eri_kpt = pyscf.pbc.ao2mo.general(cc._scf.cell,
                                        (orbo_p,mo_coeff[kq,:,:],orbo_r,mo_coeff[ks,:,:]),
                                        (cc._kpts[kp],cc._kpts[kq],cc._kpts[kr],cc._kpts[ks]))
                            eri_kpt = eri_kpt.reshape(nocc,nmo,nocc,nmo)
                            eri_kpt = eri_kpt.transpose(0,2,1,3) / nkpts
                            tmp_block[kp-ranges0[0],kr-ranges1[0],kq-ranges2[0]] = eri_kpt
                            #######################################################
                            ## Storing in physics notation... note it's kp,kr,kq  #
                            ## and not kp,kq,kr...                                #
                            #######################################################
                            #self.oooo[kp,kr,kq]  = eri_kpt[:,:,:nocc,:nocc]
                            #self.ooov[kp,kr,kq]  = eri_kpt[:,:,:nocc,nocc:]
                            #self.oovv[kp,kr,kq]  = eri_kpt[:,:,nocc:,nocc:]
                            ## The -<pr|sq> will be added below... just initializing for now
                            ## This will be stored in a special manner, and in order to get
                            ## <pq|rs>, you will need to call Soov[kr,kq,kp]
                            ##self.Soovv[kq,kr,kp] = 2.*eri_kpt[:,:,nocc:,nocc:]
                            #self.Soovv[kp,kq,kr] = 2.*eri_kpt[:,:,nocc:,nocc:]
                            #self.oovv_new[kp,kq,kr]  = eri_kpt[:,:,nocc:,nocc:]
                #self.oooo[ranges0,ranges1,ranges2] = tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,:nocc]
                #self.ooov[ranges0,ranges1,ranges2] = tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:]
                #self.oovv[ranges0,ranges1,ranges2] = tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:]
                #self.Soovv[ranges0,ranges2,ranges1] = 2*tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:]
                #self.oovv_new[ranges0,ranges2,ranges1] = tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:]
                ############################################################################
                # Unfortunately h5py has some problems with some more advanced indexing... #
                # Currently if you're not careful, can be prone to mistakes when writing   #
                # code.  Look into a better way to write this TODO                         #
                ############################################################################
                self.oooo    [min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,:nocc]
                self.ooov    [min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:]
                self.oovv    [min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:]
                self.Soovv   [min(ranges0):max(ranges0)+1,min(ranges2):max(ranges2)+1,min(ranges1):max(ranges1)+1] = \
                        2*tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:].transpose(0,2,1,3,4,5,6)
                self.oovv_new[min(ranges0):max(ranges0)+1,min(ranges2):max(ranges2)+1,min(ranges1):max(ranges1)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:].transpose(0,2,1,3,4,5,6)

                r0 = ranges0
                r1 = ranges1
                r2 = ranges2
                # [ki,ka,kj*j,i,a,b] = (2*[ki,kj,ka,i,j,a,b] - [ki,kj,kb,i,j,b,a]).transpose(0,2,1,4,3,5,6)
                self.SoOvv[min(r0):max(r0)+1,min(r2):max(r2)+1,min(r1)*nocc:nocc*(max(r1)+1)] = \
                        2*tmp_block[:len(r0),:len(r1),:len(r2),:,:,nocc:,nocc:].transpose(0,2,1,4,3,5,6).reshape(len(r0),len(r2),len(r1)*nocc,nocc,nvir,nvir)
                self.oOvv[min(r0):max(r0)+1,min(r2):max(r2)+1,min(r1)*nocc:nocc*(max(r1)+1)] = \
                        tmp_block[:len(r0),:len(r1),:len(r2),:,:,nocc:,nocc:].transpose(0,2,1,4,3,5,6).reshape(len(r0),len(r2),len(r1)*nocc,nocc,nvir,nvir)

                loader.slave_finished()

            cc.comm.Barrier()
            cput1 = log.timer_debug1('transforming oopq', *cput1)

            loaderS = mpi_load_balancer.load_balancer(BLKSIZE=(1,1,nkpts,))
            loaderS.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

            tmp_block = numpy.empty(shape=(1,1,nkpts,nocc,nocc,nvir,nvir),dtype=ds_type)
            cput1 = time.clock(), time.time()
            good2go = True
            while(good2go):
                good2go, data = loaderS.slave_set()
                if good2go is False:
                    break
                r0,r1,r2 = loaderS.get_blocks_from_data(data)
                rslice = [slice(0,len(x)) for x in r0,r1,r2]
                for kp in r0:
                    for kr in r1:
                        for kq in r2:
                            # <pq||rs> = <pq|rs> - <pq|sr>
                            ks = kconserv[kp,kr,kq]
                            tmp_block[kp-r0[0],kr-r1[0],kq-r2[0]] = -self.oovv[kp,kq,ks].transpose(0,1,3,2)
                self.ooVv    [min(r0):max(r0)+1,min(r1):max(r1)+1,min(r2)*nvir:nvir*(max(r2)+1)] = \
                        self.oovv[min(r0):max(r0)+1,min(r1):max(r1)+1,min(r2):max(r2)+1].transpose(0,1,2,5,3,4,6).reshape(len(r0),len(r1),len(r2)*nvir,nocc,nocc,nvir)
                self.Soovv[min(r0):max(r0)+1,min(r1):max(r1)+1,min(r2):max(r2)+1] += tmp_block[rslice]

                #self.SoOvv[min(r0):max(r0)+1,min(r2):max(r2)+1,nocc*min(r1):nocc*(max(r1)+1)] += \

                # Note: we don't have to do a .transpose(0,2,1) because we already stored in reverse order in tmp block!
                self.SoOvv[min(r0):max(r0)+1,min(r1):max(r1)+1,min(r2)*nocc:nocc*(max(r2)+1)] += \
                        tmp_block[:len(r0),:len(r1),:len(r2)].reshape(len(r0),len(r1),len(r2)*nocc,nocc,nvir,nvir)
                        #tmp_block[:len(r0),:len(r1),:len(r2),:,:,nocc:,nocc:].transpose(0,2,1,4,3,5,6).reshape(len(r0),len(r2),len(r1)*nocc,nocc,nvir,nvir)
                        #self.oovv[:len(r1),:len(r0),:len(r2),:,:,:,:].transpose(1,0,2,4,3,5,6).reshape(len(r0),len(r2),len(r1)*nocc,nocc,nvir,nvir)
                loaderS.slave_finished()

            cc.comm.Barrier()
            cput1 = log.timer_debug1('transforming Soovv', *cput1)

            BLKSIZE = (1,1,nkpts,)
            loader1 = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
            loader1.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

            tmp_block_shape = BLKSIZE + (nocc,nvir,nmo,nmo)
            tmp_block = numpy.empty(shape=tmp_block_shape,dtype=ds_type)

            cput1 = time.clock(), time.time()
            good2go = True
            while(good2go):
                good2go, data = loader1.slave_set()
                if good2go is False:
                    break
                ranges0, ranges1, ranges2 = loader1.get_blocks_from_data(data)
                rslice = [slice(0,len(x)) for x in ranges0,ranges1,ranges2]
                for kp in ranges0:
                    for kq in ranges2:
                       for kr in ranges1:
                            ks = kconserv[kp,kq,kr]
                            orbo_p = mo_coeff[kp,:,:nocc]
                            orbv_r = mo_coeff[kr,:,nocc:]
                            eri_kpt = pyscf.pbc.ao2mo.general(cc._scf.cell,
                                        (orbo_p,mo_coeff[kq,:,:],orbv_r,mo_coeff[ks,:,:]),
                                        (cc._kpts[kp],cc._kpts[kq],cc._kpts[kr],cc._kpts[ks]))
                            eri_kpt = eri_kpt.reshape(nocc,nmo,nvir,nmo)
                            eri_kpt = eri_kpt.transpose(0,2,1,3) / nkpts
                            tmp_block[kp-ranges0[0],kr-ranges1[0],kq-ranges2[0]] = eri_kpt
                            ######################################################
                            # Storing in physics notation... note it's kp,kr,kq  #
                            # and not kp,kq,kr...                                #
                            ######################################################
                            #self.ovoo[kp,kr,kq] = eri_kpt[:,:,:nocc,:nocc]
                            #self.ovov[kp,kr,kq] = eri_kpt[:,:,:nocc,nocc:]
                            #self.ovvo[kp,kr,kq] = eri_kpt[:,:,nocc:,:nocc]
                            #self.ovvv[kp,kr,kq] = eri_kpt[:,:,nocc:,nocc:]
                self.ovoo[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                                                            tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,:nocc]
                self.ovov[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                                                            tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:]
                self.ovvo[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                                                            tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,:nocc]
                self.ovvv[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                                                            tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:]
                loader1.slave_finished()

            cc.comm.Barrier()
            cput1 = log.timer_debug1('transforming ovpq', *cput1)

            #######################################################
            # Here we can exploit the full 4-permutational symm.  #
            # for 'vvvv' unlike in the cases above.               #
            #######################################################
            BLKSIZE = (nkpts,)
            loader2 = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
            loader2.set_ranges((range(nUnique_klist),))

            tmp_block_shape = BLKSIZE + (nvir,nvir,nvir,nvir)
            tmp_block = numpy.empty(shape=tmp_block_shape,dtype=ds_type)

            good2go = True
            while(good2go):
                good2go, data = loader2.slave_set()
                if good2go is False:
                    break
                ranges = loader2.get_blocks_from_data(data)
                chkpts = [int(numpy.ceil(nUnique_klist/10))*i for i in range(10)]
                for indices in ranges:
                    if indices in chkpts:
                        print ":: %4.2f percent complete" % (1.*indices/nUnique_klist*100)
                    kp, kq, kr = unique_klist[indices]
                    ks = kconserv[kp,kq,kr]
                    orbva_p = mo_coeff[kp,:,nocc:]
                    orbv = mo_coeff[:,:,nocc:]
                    eri_kpt = pyscf.pbc.ao2mo.general(cc._scf.cell,
                                (orbva_p,orbv[kq],orbv[kr],orbv[ks]),
                                (cc._kpts[kp],cc._kpts[kq],cc._kpts[kr],cc._kpts[ks]))
                    eri_kpt = eri_kpt.reshape(nvir,nvir,nvir,nvir)
                    eri_kpt = eri_kpt.transpose(0,2,1,3) / nkpts
                    #tmp_block[kp,kr,kq] =
                    ######################################################
                    # Storing in physics notation... note it's kp,kr,kq  #
                    # and not kp,kq,kr...                                #
                    ######################################################
                    self.vvvv[kp,kr,kq] = eri_kpt.copy()
                    ######################################################
                    # Storing all permutations                           #
                    ######################################################
                    self.vvvv[kr,kp,ks] = eri_kpt.transpose(1,0,3,2).copy()
                    self.vvvv[kq,ks,kp] = eri_kpt.transpose(2,3,0,1).conj().copy()
                    self.vvvv[ks,kq,kr] = eri_kpt.transpose(3,2,1,0).conj().copy()
                loader2.slave_finished()

            #BLKSIZE = (1,1,nkpts,)
            #loader1 = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
            #loader1.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

            #tmp_block_shape = BLKSIZE + (nvir,nvir,nvir,nvir)
            #tmp_block = numpy.empty(shape=tmp_block_shape,dtype=ds_type)

            #cput1 = time.clock(), time.time()
            #good2go = True
            #while(good2go):
            #    good2go, data = loader1.slave_set()
            #    if good2go is False:
            #        break
            #    ranges0, ranges1, ranges2 = loader1.get_blocks_from_data(data)
            #    rslice = [slice(0,len(x)) for x in ranges0,ranges1,ranges2]
            #    orbv = mo_coeff[:,:,nocc:]
            #    for kp in ranges0:
            #        for kq in ranges2:
            #            for kr in ranges1:
            #                ks = kconserv[kp,kq,kr]
            #                orbva_p = mo_coeff[kp,:,nocc:]
            #                eri_kpt = pyscf.pbc.ao2mo.general(cc._scf.cell,
            #                            (orbva_p,orbv[kq],orbv[kr],orbv[ks]),
            #                            (cc._kpts[kp],cc._kpts[kq],cc._kpts[kr],cc._kpts[ks]))
            #                eri_kpt = eri_kpt.reshape(nvir,nvir,nvir,nvir)
            #                eri_kpt = eri_kpt.transpose(0,2,1,3) / nkpts
            #                tmp_block[kp-ranges0[0],kr-ranges1[0],kq-ranges2[0]] = eri_kpt
            #                ######################################################
            #                # Storing in physics notation... note it's kp,kr,kq  #
            #                # and not kp,kq,kr...                                #
            #                ######################################################
            #    self.vvvv[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
            #                                                tmp_block[:len(ranges0),:len(ranges1),:len(ranges2)]
            #    loader1.slave_finished()

            cc.comm.Barrier()
            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)

    def __del__(self):
        if hasattr(self, 'feri1'):
            for key in self.feri1.keys(): del(self.feri1[key])
            self.feri1.close()
            self.feri2.close()
            self.feri3.close()
            self.feri4.close()

def print_james_header():
    print ""
    print " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ "
    print " You are now about to use the kpointified/restrictified version "
    print " of eom-ccsd                                                    "
    print "                                           -James               "
    print " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ "
    print ""
    return

def print_james_ascii():
    print "\
NmmmmmmddddddddddddmmmmmNNNNNNNNNNNNNNNNNNmmmmmddddddddddddmmmmmddhhhyyyyyyyyyyyyyyyyyhhmmNmmddmmmmm\n\
mmmmmmmddddddddmmmmmmmmNNNNNNNNNNNNNmmmmddhhhhhhhhhhhdddddddddhhhhyyyyyyyyyyyyyyyyyyyyhhmmmmmmdddddd\n\
mmmmmmmdddddddmmmmmmmmmmmmNNNNNNNNNmmdhysyyyyyhhhhddddddddddhhhyyyyyyyyyyyyyyyyyyyyyyyhhmmmmdddddddm\n\
mmmmmmmmmddddddmmmmmmmmmmmmmmNNNNmddyyssyyyhhhhdddmmmmmmmmmmddddhhhhhhhhhyyyyyyyyyyyyyhhmmmddddddddd\n\
mmmmmmmmmmddddmmmmmmmmmmmmNNNNNmmhysssyyhhhhhddddmmmNmmmdmmmmmmmmmmmmddddhhhyyyyyyyyyyhhmmmddhhddddd\n\
mmmmmmmdddmmmmmmmmmmmmmmmmNNNNmdhyysyyhhhhhdddddmmmmmmmmmmmmmmmmmNNmmmmmmddhhyyyyyyyyyhhmmmmdddddddd\n\
mmdddddmmmddmmmmmmmmmmmmmmNNmddyyyyyhhhhhhhdddmmmmmmmmmmmmmmmmmmmmNmmmmmmmmdhhhyyyyyyyhhmmmmmddddddd\n\
ddhhhddmmmmmmmmmmmmmmmmmmmmdhyyyyyyyhhddhhddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmdhhyyhhhyyhhmmmmdddddddd\n\
ddhddddmmmmmmmmmmmmmmmddddhhyyyyyyhhhhdhhhdddddddddddddddddddddmmmmmmmmmmmmmddhhyyyyyyhhmmmddddddddd\n\
dddddddmmmmmmmmmmmmdddddhhyyyssyyhhdddhhhhhddddhhhhhdhhhhhhyyyhhdddmmmmmmmNNmmdhhhhhyyhhmmmmdddddddd\n\
ddddddmmmmmmmmmmmddddhhhyyyyyyyhhhdddhhhhhhdhhhyyyyyyhhhhhhhhhhhdddmmmmmmmNNmmmddhhhhyhhmmNmmmmmmmmd\n\
ddhhhddmmmmmmmmddddddhhyhhyhhhhhhhhhhhhhhhhhhyyyyyyyyyyyhhhyyyhhhhhddddmmmmmmmmmdddddhhhmmNmmmmmmmmm\n\
hhdhhdmmmmmmdddddddhhhhhyyhhhhhhhhhhhhyhhyyyyyssssssyyyssssoooooooosssyyhhddmmmmmdddddddmmmmmddddddd\n\
yhhhhddmmddmddhhhhhhhhhhyydhhhhhhyyyyyyyyssysssoooossssoo++++++++++++ooossyhddddddddddddmmmmmmdddddd\n\
oossohdmmmddhddddhhdhhhhhhdhhhyyyyysyyyyssssssooosooso+++//////////+++++oossyhddddddddddmmmmmddddddd\n\
...--oyddddhddddhhhhhhhhhhhhyyysssyyyyyhhhyyysooooo++////://////////////++oossyhdddddddddddddddddddd\n\
     +sddddhddhhhddhhhhhhhyyyyyyssyyhhhhhhhysooo++////::::::::::://:://///+++osyydddddmmmmdddhhhdddd\n\
     +odhhhhhhhhdddhhyhhyyyyyyyyysyyhddddhyoo++/////::::::::::::::::::::////++osshhdddddmmmdhhhhhhhh\n\
     /ohhhhhhhhhdddhhyyyssyyhhhyyyyhddddyso+//:---/:-----::::::::::::::://///++ooyyhhdddmmmddhhhhhhh\n\
     /ohhhhhhhhhhddhysssyyhhdddyysyhddys+++//:----:--------::::::::::::://////++ossyyhddmmmddhhhhhhh\n\
    `/ohhhhhhhhhhddyyoosyhdddhhsshhdys///++::.--------------:::::::::::::://///++ooosyhdmmddhhhhhhhh\n\
 ``..+ohhdddhhhhyhhysosshhdddyyyhdys/:////:---------.--------::::::::::::://///++ooooshdmmmddhhhhhhh\n\
`....+ohddddhhhyyyhsssyhdddhhhhhhs+/:://:::-------::--------::::::::::::://///++ossosshdmmmmmhhhhhhh\n\
````.+oddddddhyysyyssyhdddyyhhhso//:://::::------::::-------::::::::::::::////+oossssyhdmmNmmhhhhhhh\n\
--.-:osddddddhyysssssyhdhhsyyss+/::::::::::---------------:::::::::::::://////+oossyyyhdmmNmmddhhhhh\n\
///++yhmmmddhhyyyyysyyhhyyooooo//////::::::::-------------::::--::::://///////+oossyyhddmmNmmddhhhdd\n\
hhhhhddddmdhhhyyyyysyyyyoo++o++/////////://::::-----------------::://+++++oo++ooossyhhddmmmmmddhdddd\n\
ddhhhdddddhhhhyyyyysssso++++++++++oooooooooooo++//:::--------://++oosssyyyyyyyyyysyyhdddddmmmmdddddd\n\
ssooshdmddhyhhyyhyyssso++++oosssssssoossosssssssso++//::---::/++osssssssooooossyyyyyhdhhhhddmddddddd\n\
ooooshdddhhhddhhhhhyyso++++osssoo++++++++++oooooooo++//:::::/+ooooooooo+++++/++osyyyyhhyhhdddhhyyyyy\n\
ooooohdmddhhddhhdhhhyso+++ooo+++++++++oooooooooo+++++//::::/++ooooooooooooooooooossyyhhhddmddhhyyyhh\n\
ssssshddddddddddhhhhyso+++++///++ooooosyhys//++o+++++//:::/+oooo+++++yyhso++sss++oosyhhdmmmmdhhhhhhh\n\
sssssdddddddmmdhyhhhyoo++///::/++ooo++osyo+::/+++/++++/:::/+oooo+//::ssso+++ssso+++oyhddNNNmddhhhhhh\n\
ssssshddmmmddddhhyyhyso+///::::///++//////////++//++++//:://+ooo++////////++++++///+syddNNNmdddddddh\n\
ssosshdmmdddsssydyyhyso+++/::::::::/::::::/://///+++++//:::/++oo++/////////////////+syddNmmddddddddd\n\
ooooohdmddhy//+shyyhyyoo++/:::::::::::::::::::///++++///:::/++ooo++///:::::::://///oshddmmmddddddddd\n\
oosoohdmddyy:::+oyhhhyso++//::::::::::::::::::///+/++////://++oo++///:::::::::::://oshddmmmddddddddd\n\
ooooohdmmdhh//://syhhhyso+//::::::::::-----::::///++++//////++ooo+///:::::::::::://oyhdmmmmdddddddhh\n\
oossshdmmdddo+///ssddhyyoo///::::::--------::://++++++///:::/+oo++//::::::::::::://oyhmmmmmddddddhhh\n\
ooossddmmdmmys///+odhhysoo///:::::----------://++++++/:::-:://+o+++::::::::::::://+oydddmmmdhddddhhh\n\
//+ooddmmdmmhy+//++hhyssoo///:::::----------://++++++//::::://+++++::::---::::::/++syyyhmmmddddddddd\n\
///++yhmmmmmdhoo+//osysooo////:::::-----::-:://++++++++++///+++++++/::----::::://+ossoyymmmddddddddd\n\
/////+ohdmmmddhys++osyssoo+////:::::--:::::////++++++++++++++++++++//::--:::::///+osooyhNmmddddddddd\n\
//////+sydmmmmmddhhddhysoo/////::::-:://:////////+///////////+++++++//:::::::://+ooyyyhdNmmddddddddd\n\
++++++++shdmmmmmmmmNmdhyoo/////:::::://///+++///////:///////////++++++//::::::/++oshhhhdNmmddddddddd\n\
ooooooooshdmmmmmmmmNmmhyoo/////::-:::///+//////////:::///////////++++++/////://++oshhhhdmmmddddddddd\n\
oooooooosydmmmmmmmmmmmdhso+////::--::://++++++++++///////////////++++++////////++sshhhhdmmmddddddhdd\n\
oooooooosydmmmmmmmmmmmdhss++/////:--:::///+oooooooooo++++++++++oooooo+////////+++syhhhhdmmmddddhhhhh\n\
ooooooo+sddmmmmmmmmmmmddyso+/////::---::://++o+++++++////+++++ooooooo////:://++ooyyhhhhdmmddhhhsyyyy\n\
ooooooooymmmmmmmmmmmmNmdysso+++//:::--:::///////////::::::////++++++///:::///+ooohhhhhhdmmddhhhyyhhh\n\
oooooooyhmmmmmmmmmmmmNmdsssso++/////:://////////////:::::///////+++/////////++osyhhhhhddmmmddhhhhdhh\n\
ooooossddmmmmmmmmmmdmmmdsossooo++/////////////:://///////////////////++/++++oosyhhhhhhhdmmmddddddddd\n\
ooooohhmmmmmmmmmmmmmmmdhoo+osss++++///++//////::::::///////////////++++++++osyyyyhhhyyhdmmdhyhhhhhhh\n\
ooossdmmmmmmmmmmmmmdhsoooo/+oooooo++++++////////::::::///////////+++++oooooosyyyyhhyyyhdmdhyshhyyyyy\n\
oossydmddmmmmmdmmddo+:+ooo///++oosoo+++++++///////:::::::://////++++++osssso//+yhhhhhhhhmmdhhhhyyyyy\n\
ssyyydmmmmmmmmmmdys+++syo++////++oooooooo+++//////::::::://///++++ooosssssso::/oshhhhhhhmmmdhdhyyyyy\n\
yyyyydmmmmmdmmmdysooossso+////////+oossoooo++++///////////////+++ooosssooooooo++oyyhhyhhmmmdhddhhhhh\n\
yyyyydmmmmmmmmdyo++osyss++//////////+ooooooooo+++/////+++++++++ooosssoo++ooosso+/ssyyyhhmmmddddhdddd\n\
yyyyydmmmmmmmmdy+++oosss++///////:////++++oooooooooo+oooooooooossssso+++++ooyss++ooyyyhhmmmddddddddd\n\
yyyyymmmmdddmdhs///ooooo///////::::///+///++++++oossssssssoooooossso+//+++oossso++osyyhhmmmddddddddd\n\
    "
    return

def print_tim_ascii():
    print "\n\
                                       .-/+ossyyyyssssooo+/-.`                                      \n\
                                 ./oydmmmmmdhddhhhddddhdmNNNNNmyo:`                                 \n\
                             -+ymNNNNmmddhhhdhyhhhhhhdmNNNmmNNNNNdso/:-                             \n\
                         `-omNMMNNmmmdhddddddmmNNNNNNMMMMNNNNNNNmmysosmNy/`                         \n\
                       -+syymmmdmmmddddmNNNNNmmNMMMMMMMMMMMMMMNNNmdyssdMMNNy:                       \n\
                    `/yhdddmmmmmmmmddmmNNMMNmmmNMMMMMMMMMMMMMMMMMNNmdyhNMMMMMm+`                    \n\
                  `/shdNNmddddddmmmNNNNNNNNmmmNNNNMNNMMMMMMMMNNNNNNNNmmmNMMMMMMNs.                  \n\
                `+mNNNNNmmdddmmddmmmmNNNNNNNmNNNNNNNNMMMMMNNNNNNNNNNNNNmmNNMMMMMMNo`                \n\
               .sNMMNNNmmddddmmmmddddmmmNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNmmmNMMMMMMMm:               \n\
             `/oyNMMNNmddhdmmddmddddmmmmNNNNNNNNNNNNNNNNNNNmmNNNNNNNNNmmNmmNMMMMMMMMNs`             \n\
            .sysymMMmdyyhhhdmmmmddmmmNNNmmmmNmmNmmmmmdmdddddmmNNNNNNNmmmmNmmNMMMMMMMMMd-            \n\
           -mdysydMMmhyyyhddddmmmmmmmNNNmmddddddmmmmhysssssshmNNNNNNmNmmmmmmNMMMMMMMMMMm:           \n\
          -mNmhyyhMMmhyyyhddddddmmmmmmdhyssoooosyyyyoo++++oooydmmmmmmmNmmmmNNNMMMMMMMMMMN:          \n\
         .hmdddhhydyshhhdddddhhddddhyso+++////++++o++++++++++oyddmmmmmmNmmNNNNMMMMMMMMMMMN-         \n\
        `sddhso+/:::+yydddddhyhysoo+++/////////://////////++/+oshdmmmmmmmmmNNNMMMMMMMMMMMMm`        \n\
        /hhhyso+//:::/sdmmmhyss+//+///://:::::-:://///////////+oshdmmmmmmmNNNNMMMMMMMMMMMMMy        \n\
       .hdddhhhhdddhysydmmmyso+/////::::::::::::::::::////////++oyhmmmmmNNNNNNdNMMMMMMMMMMMN:       \n\
       smddddhhhdmmdddddmmmyo+//////:::::::::::::::::::////////++oydmmmmNNNNdyosdMMMMMMMMMMMh       \n\
      .mmmmmmddyyyyyyyhhhmms++////////:::::::::::::::::://///////++shmmNNNmdyoooyMMMMMMMMMMMN.      \n\
      :NNNNNNmmysoooooosyddo++////::::::::::::----::::::///////////+ohmNNmdyo+++yMMMMMMMMMMMM+      \n\
      oNmmmmmmmmdhhysssyydms++//:::---:::::::::-:://++ooo++/////////+shmNmys+o++hMMMMMMMMMMMMy      \n\
      ymmmmmmmmmmmNmmmmmmmmy++/:::::::::::://+osyhhhyyssssoo++//////+oyhmd+o/+/omMMMMMMMMMMMMd      \n\
      ymmmmmmmdddhdNNNNNNNNmoooosssoo++//++osyyhddddyyyyysoo++++++/+++oydh/+//+yMMMMMMMMMMMMMd      \n\
      sddhhyssoooohNNNNmmmdNyhhhhhddhyysso+ossyyhdddyssysso++//+++++++oydh+//+sNMMMMMMMMMNNNNh      \n\
      :oo++++++++odNMMNmdhhmmssshdmmyoshho:/ossssssssooso+//:://///+++osyyo+sdNMMMMMMMMNNMMMMs      \n\
      .+++++++oooshNMMMmdmNNNsosyyyysssss/::/+oooo++++++/::::::////++ooosyyymMMMMMMMMMMMMMMMM/      \n\
       +ossyhddmmdmMMMMMNMNMNdo+++oooo++o/:::///++/////:::--::////+++oossyyymMMMMMMMMNNNNNMMN`      \n\
       +mmmdhyssoohNMMMMMMMMMNo////////++/:::///+++ooo++//::////+++++oosyysydNNMMMMMMMmNNNMMs       \n\
       `oooo+++++odMMMMMMMMNNNy//::///+o+/::///////oooooooo++++++++++oosyyssyyhdmMMMMMMNNMMm.       \n\
        .+oooooosydMMMNNmdhyyNd////+++o+////://++osso++++ooooo++++++oosyyyssyo:/sdNMMMMMMMN+        \n\
         /yyhdmNNNmddhyyssssymNs++++oosso+++++sssso+/+++oosysso+++++ossyyysssy//::/sdmmmNNy         \n\
         `hNNmdhyyysyysssssssyhy++ooossssoooosssssooo+sosyhoooo++++oossyyoosss/://:-.-///+`         \n\
          `osssssssssssssssssssssoooosssssyhsssoo///+osyys+++oo++ooosyyyoooss+-:://::-.--           \n\
           `/ssssooosssssyyyyyyyhyo++ooosshdhso+ooossysso++++o+oosssyyyoooosso:-::/::::.            \n\
             :ossssyyyyyyyyhhhhhhhyo++++ooosyysyssssoo+++++++oosssyyhysoooosso::-::/::.             \n\
              -oyyyyyyhhhhhhhhhhhy+--:+oo+++oossssoo++++++++oosssyyyysoooooooo::::-::.              \n\
               `/yhhhhhhhhhhhhhho----``-/+++++++++++/////+++osyyyyyysoo+ooooso+-::-.                \n\
                 .ohhhhhhhhhyo+:.---:.```.-++++/////////+oosyyhyysssooooooooooo--.`                 \n\
                   .ohhhs+:......-:--.`````/oo++//++++++osyhhyyssoooooooooooooo-`                   \n\
                     .-.....``..------..`.`.+yooooossssyyhhyyssoooooooooosoo+:`                     \n\
                         ````..--------..`..-sdyssssyssssssssooooooooo++oo/.                        \n\
                            ``.:-:-.-.---..``:sdysooooooooooosssoooo+++:-`                          \n\
                               `.----..---.`-//+so+osoo++++++ooooo+:-`                              \n\
                                   ```...-./o//::--:/yyo++++//:-.`                                  \n\
                                           .....```.`.:.```                                         "
    return
