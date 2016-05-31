import sys
import time
import tempfile
import os.path
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
#from pyscf.pbc.lib.davidson import eigs

def barrier(comm, tag=0, sleep=0.01):
    size = comm.Get_size()
    if size == 1:
        return
    rank = comm.Get_rank()
    mask = 1
    while mask < size:
        dst = (rank + mask) % size
        src = (rank - mask + size) % size
        req = comm.isend(None, dst, tag)
        while not comm.Iprobe(src, tag):
            time.sleep(sleep)
        comm.recv(None, src, tag)
        req.Wait()
        mask <<= 1

#einsum = numpy.einsum
einsum = pbclib.einsum

# This is restricted (R)CCSD
# following Hirata, ..., Barlett, J. Chem. Phys. 120, 2581 (2004)

def tril_index(ki,kj):
    assert ki <= kj
    return (kj*(kj+1))/2 + ki

#@profile
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

    #print_james_ascii()
    if t1 is None and t2 is None:
        t1, t2 = cc.init_amps(eris)[1:]
        #t1, t2_tril = cc.init_amps_tril(eris)[1:]
    elif t1 is None:
        nocc = cc.nocc()
        nvir = cc.nmo() - nocc
        t1 = numpy.zeros((nocc,nvir), eris.dtype)
    elif t2 is None:
        t2 = cc.init_amps(eris)[2]
        #t2 = cc.init_amps_tril(eris)[2]

    nkpts, nocc, nvir = t1.shape

    #################################
    # Reading t amplitudes to file  #
    #################################
    _tmpfile3_name = None
    if cc.rank == 0:
        _tmpfile3_name = "t_amplitudes.hdf5"
    _tmpfile3_name = cc.comm.bcast(_tmpfile3_name, root=0)
    if os.path.isfile(_tmpfile3_name):
        print "reading t amplitudes from file..."
        feri4 = h5py.File(_tmpfile3_name, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        in_t1 = feri4['t1']
        in_t2 = feri4['t2']
        for ki in range(nkpts):
            t1[ki] = in_t1[ki]
            for kj in range(nkpts):
                for ka in range(nkpts):
                    t2[ki,kj,ka] = in_t2[ki,kj,ka]
        feri4.close()
    #################################

    cput1 = cput0 = (time.clock(), time.time())
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

        print "calculating norm..."
        normt = numpy.array(0.0,dtype=numpy.float64)
        normt += numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        comm.Allreduce(MPI.IN_PLACE, normt, op=MPI.SUM)

        print "setting old t1,t2 to t1,t2new"
        t1, t2 = t1new, t2new
        print "deleting old t1new,t2new"
        t1new = t2new = None
        print "cc.diis begin"
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        print "cc.diis end"

        print "writing to file..."
        #################################
        # Writing t amplitudes to file  #
        #################################
        _tmpfile3_name = None
        if cc.rank == 0:
            _tmpfile3_name = "t_amplitudes.hdf5"
        _tmpfile3_name = cc.comm.bcast(_tmpfile3_name, root=0)
        feri4 = h5py.File(_tmpfile3_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
        ds_type = t2.dtype
        out_t1  = feri4.create_dataset('t1', (nkpts,nocc,nvir), dtype=ds_type)
        out_t2  = feri4.create_dataset('t2', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=ds_type)
        for ki in range(nkpts):
            out_t1[ki] = t1[ki].copy()
            for kj in range(nkpts):
                for ka in range(nkpts):
                    out_t2[ki,kj,ka] = t2[ki,kj,ka].copy()
        feri4.close()
        #################################

        eold, eccsd = eccsd, energy(cc, t1, t2, eris)
        if cc.rank == 0:
            log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                     istep, eccsd, eccsd - eold, normt)
            cput1 = log.timer('CCSD iter', *cput1)
        #if istep == 1:
        #    eccsd = eold = normt = 0.0
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    if cc.rank == 0:
        log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2

#@profile
def update_t1(cc,t1,t2,eris,ints1e):
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:,:nocc,nocc:]
    foo = fock[:,:nocc,:nocc]
    fvv = fock[:,nocc:,nocc:]

    ds_type = t1.dtype

    Foo,Fvv,Fov,Loo,Lvv = ints1e

    kconserv = cc.kconserv
    # T1 equation
    # TODO: Check this conj(). Hirata and Bartlett has
    # f_{vo}(a,i), which should be equal to f_{ov}^*(i,a)
    t1new = numpy.zeros((nkpts,nocc,nvir),dtype=t1.dtype)

####
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(numpy.floor(numpy.sqrt(int(numpy.floor(mem/pre))))),1),nkpts)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts_blksize,))
    loader.set_ranges((range(nkpts),))
####

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0 = loader.get_blocks_from_data(data)

        s0 = slice(min(ranges0),max(ranges0)+1)

        eris_voov_aXi = _cp(eris.voov[s0,:,s0])
        eris_ovov_Xai = _cp(eris.ovov[:,s0,s0])

        for iterka,ka in enumerate(ranges0):
            ki = ka
            iterki = iterka
            # kc == ki; kk == ka
            t1new[ka] = fov[ka].conj().copy()
            t1new[ka] += -2.*einsum('kc,ka,ic->ia',fov[ki],t1[ka],t1[ki])
            t1new[ka] += einsum('ac,ic->ia',Fvv[ka],t1[ki])
            t1new[ka] += -einsum('ki,ka->ia',Foo[ki],t1[ka])

            tau_term = numpy.empty((nkpts,nocc,nocc,nvir,nvir),dtype=t1.dtype)
            for kk in range(nkpts):
                tau_term[kk] = 2*t2[kk,ki,kk] - t2[ki,kk,kk].transpose(1,0,2,3)
            tau_term[ka] += einsum('ic,ka->kica',t1[ki],t1[ka])

            t1new[ka] += einsum('kc,kica->ia',Fov[:].reshape(nocc*nkpts,nvir),tau_term[:].reshape(nocc*nkpts,nocc,nvir,nvir))

            t1new[ka] += einsum('akic,kc->ia',eris_voov_aXi[iterka,:,iterki].transpose(1,0,2,3,4).reshape(nvir,nocc*nkpts,nocc,nvir),
                                              2*t1[:].reshape(nocc*nkpts,nvir))
            t1new[ka] += einsum('kaic,kc->ia',eris_ovov_Xai[:,iterka,iterki].reshape(nocc*nkpts,nvir,nocc,nvir),
                                               -t1[:].reshape(nocc*nkpts,nvir))
        loader.slave_finished()

    cc.comm.Barrier()

####
    mem = 0.5e9
    pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(numpy.floor(mem/(nkpts_blksize*pre))),1),nkpts)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts_blksize,nkpts_blksize2,))
    loader.set_ranges((range(nkpts),range(nkpts),))
####
    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0,ranges1 = loader.get_blocks_from_data(data)

        s0,s1= [slice(min(x),max(x)+1) for x in ranges0,ranges1]

        eris_ovvv_kaX = _cp(eris.ovvv[s1,s0,:])
        eris_ooov_kXi = _cp(eris.ooov[s1,:,s0])
        eris_ooov_Xki = _cp(eris.ooov[:,s1,s0])

        for iterka,ka in enumerate(ranges0):
            ki = ka
            iterki = iterka
            for iterkk,kk in enumerate(ranges1):
                kd_list = _cp(kconserv[ka,range(nkpts),kk])
                kc_list = _cp(range(nkpts))
                Svovv = 2*eris_ovvv_kaX[iterkk,iterka,kd_list].transpose(0,2,1,4,3) - eris_ovvv_kaX[iterkk,iterka,kc_list].transpose(0,2,1,3,4)
                tau_term_1 = t2[ki,kk,:].copy()
                tau_term_1[ki] += einsum('ic,kd->ikcd',t1[ki],t1[kk])
                t1new[ka] += einsum('ak,ik->ia',Svovv.transpose(1,2,0,3,4).reshape(nvir,-1),
                                                tau_term_1.transpose(1,2,0,3,4).reshape(nocc,-1))

                kl_list = _cp(kconserv[ki,kk,range(nkpts)])
                # ooov_kXi and ooov_Xki can have overlap, so different processors could call the same ooov
                # block from disk, causing some blocking and slowing code.  Maybe change the first one to call ovoo
                Sooov = 2*eris_ooov_kXi[iterkk,kl_list,iterki] - eris_ooov_Xki[kl_list,iterkk,iterki].transpose(0,2,1,3,4)
                tau_term_1 = t2[kk,kl_list,ka].copy()
                if kk == ka:
                    tau_term_1[kc_list==kl_list] += einsum('ka,xlc->xklac',t1[ka],t1[kc_list==kl_list])
                t1new[ka] += -einsum('ki,ka->ia',Sooov.transpose(0,1,2,4,3).reshape(-1,nocc),
                                     tau_term_1.transpose(0,1,2,4,3).reshape(-1,nvir))

        loader.slave_finished()

    cc.comm.Allreduce(MPI.IN_PLACE, t1new, op=MPI.SUM)
    return t1new

#@profile
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

    #Woooo = imdk.cc_Woooo(cc,t1,t2,eris)
    #Wvvvv = imdk.cc_Wvvvv(cc,t1,t2,eris)
    #Wvoov = imdk.cc_Wvoov(cc,t1,t2,eris)
    #Wvovo = imdk.cc_Wvovo(cc,t1,t2,eris)

    Foo = imdk.cc_Foo(cc,t1,t2,eris)
    Fvv = imdk.cc_Fvv(cc,t1,t2,eris)
    Fov = imdk.cc_Fov(cc,t1,t2,eris)
    Loo = imdk.Loo(cc,t1,t2,eris)
    Lvv = imdk.Lvv(cc,t1,t2,eris)

    print "done making intermediates..."
    # Move energy terms to the other side
    Foo -= foo
    Fvv -= fvv
    Loo -= foo
    Lvv -= fvv

    kconserv = cc.kconserv

    print "t1 equation..."
    # T1 equation
    # TODO: Check this conj(). Hirata and Bartlett has
    # f_{vo}(a,i), which should be equal to f_{ov}^*(i,a)
    t1new = update_t1(cc,t1,t2,eris,[Foo,Fvv,Fov,Loo,Lvv])

    print "t2 equation..."
    # T2 equation
    # For conj(), see Hirata and Bartlett, Eq. (36)
    #t2new = numpy.array(eris.oovv, copy=True).conj()
    t2new = numpy.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir),dtype=ds_type)
    #tril_shape = ((nkpts)*(nkpts+1))/2
    #t2new_tril = numpy.zeros((tril_shape,nkpts,nocc,nocc,nvir,nvir),dtype=ds_type)

    cput1 = time.clock(), time.time()
    cput2 = time.clock(), time.time()
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(1,nkpts,nkpts,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    #
    #
    # Figuring out number of kpoints we can have in our oovv terms below
    # TODO : clean this up- just temporary
    #
    #

    mem = 0.5e9
    pre = 1.*nkpts*nkpts*nocc*nocc*nvir*nvir*16
    nkpts_blksize = max(int(numpy.floor(mem/pre)),1)
    BLKSIZE2 = min(nkpts,nkpts_blksize)
    BLKSIZE2_ranges = [(BLKSIZE2*i,min(nkpts,BLKSIZE2*(i+1))) for i in range(int(numpy.ceil(1.*nkpts/BLKSIZE2)))]

    #######################################################
    # Making Woooo terms...
    #######################################################
    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        if min(ranges0) > max(ranges1): #continue if ki > kj
            loader.slave_finished()
            continue

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in ranges0,ranges1,ranges2]
        eris_oovv = _cp(eris.oovv[s0,s1,s2])

        eris_oooo = _cp(eris.oooo[s1,s0])
        eris_ovoo_ij = _cp(eris.ovoo[s0,s1])
        eris_ovoo_ji = _cp(eris.ovoo[s1,s0])

        for iterki,ki in enumerate(ranges0):
            for iterkj,kj in enumerate(ranges1):
                if ki <= kj:
                    t2new[ki,kj,ranges2] += _cp(eris_oovv[iterki,iterkj,s2]).conj()
                    #t2new_tril[tril_index(ki,kj),ranges2] += _cp(eris_oovv[iterki,iterkj,s2]).conj()

        for kblock in BLKSIZE2_ranges:
            kl_block_size = kblock[1]-kblock[0]
            #
            #  Find out how large of a block_size we need...
            #
            kklist = []
            for iterkl,kl in enumerate(range(kblock[0],kblock[1])):
                for iterki,ki in enumerate(ranges0):
                    for iterkj,kj in enumerate(ranges1):
                        kk = kconserv[kj,kl,ki]
                        kklist.append(kk)
                        iterkk = numpy.where(numpy.asarray(kklist)==kk)[0]
                        if len(iterkk)==0: #if not found, append
                            kklist.append(kk)

            kk_block_size = len(kklist)
            eris_oovv1 = numpy.empty((kk_block_size,kl_block_size,nkpts,nocc,nocc,nvir,nvir),
                                  dtype=t2.dtype)
            #
            #  Now fill in the matrix elements...
            #
            for iterkl,kl in enumerate(range(kblock[0],kblock[1])):
                for iterki,ki in enumerate(ranges0):
                    for iterkj,kj in enumerate(ranges1):
                        kk = kconserv[kj,kl,ki]
                        iterkk = numpy.where(numpy.asarray(kklist)==kk)[0][0]
                        eris_oovv1[iterkk,iterkl,:] = _cp(eris.oovv[kk,kl,:])

            kl_slice = slice(kblock[0],kblock[1])

            for iterkl,kl in enumerate(range(kblock[0],kblock[1])):
                for iterki,ki in enumerate(ranges0):
                    for iterkj,kj in enumerate(ranges1):
                        if ki <= kj:
                            kk = kconserv[kj,kl,ki]
                            iterkk = numpy.where(kklist==kk)[0][0]

                            #wOOoo = numpy.empty((nkpts,nocc,nocc,nocc,nocc),dtype=t2.dtype)
                            tau1_ooVV = t2[ki,kj,:].copy()
                            tau1_ooVV[ki] += einsum('ic,jd->ijcd',t1[ki],t1[kj])

                            # TODO read only packed oovv terms and unpack after reading
                            wOOoo = _cp(eris_oooo[iterkj,iterki,kl].transpose(3,2,1,0)).conj()
                            wOOoo += einsum('klic,jc->klij',eris_ovoo_ij[iterki,iterkj,kk].transpose(2,3,0,1).conj(),t1[kj])
                            wOOoo += einsum('lkjc,ic->klij',eris_ovoo_ji[iterkj,iterki,kl].transpose(2,3,0,1).conj(),t1[ki])
                            wOOoo += einsum('klcd,ijcd->klij',eris_oovv1[iterkk,iterkl,:].transpose(1,2,0,3,4).reshape(nocc,nocc,nkpts*nvir,nvir),
                                                              tau1_ooVV.transpose(1,2,0,3,4).reshape(nocc,nocc,nkpts*nvir,nvir))

                            for iterka,ka in enumerate(ranges2):
                                # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
                                kb = kconserv[ki,ka,kj]
                                kn = kconserv[kj,kl,ki]
                                tau1_OOvv = t2[kn,kl,ka].copy()
                                if ka == kk and kl == kb:
                                    tau1_OOvv += einsum('ka,lb->klab',t1[ka],t1[kb])
                                tmp = einsum('klij,klab->ijab',wOOoo,tau1_OOvv) #kl combined into one
                                #t2new_tril[tril_index(ki,kj),ka] += tmp
                                t2new[ki,kj,ka] += tmp

        loader.slave_finished()
    cc.comm.Barrier()
    cput2 = log.timer_debug1('transforming Woooo', *cput2)

    cput2 = time.clock(), time.time()
####
    mem = 0.5e9
    pre = 1.*nvir*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)
    print "vvvv blocksize = ", nkpts_blksize
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,1,nkpts_blksize,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
####

    #######################################################
    # Making Wvvvv terms... notice the change of for loops
    #######################################################
    #@profile
    def func3():
        good2go = True
        while(good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
            if min(ranges1) > max(ranges2): #continue if ka > kb
                loader.slave_finished()
                continue

            s0,s1,s2 = [slice(min(x),max(x)+1) for x in ranges0,ranges1,ranges2]

            eris_ovvv_ab = _cp(eris.ovvv[s1,s2])
            eris_vovv_ab = _cp(eris.vovv[s1,s2])
            eris_vvvv_ab = _cp(eris.vvvv[s1,s2])

            for iterka,ka in enumerate(ranges1):
                for iterkb,kb in enumerate(ranges2):
                    if ka <= kb:
                        ###################################
                        # Wvvvv term ...
                        ###################################
                        ovVV = eris_ovvv_ab[iterka,iterkb,:].transpose(1,2,0,3,4).reshape(nocc,nvir,-1)
                        voVV = eris_vovv_ab[iterka,iterkb,:].transpose(1,2,0,3,4).reshape(nvir,nocc,-1)
                        wvvVV = einsum('akd,kb->abd',voVV,-t1[kb])
                        wvvVV += einsum('ak,kbd->abd',-t1[ka].T,ovVV)
                        wvvVV += eris_vvvv_ab[iterka,iterkb].transpose(1,2,0,3,4).reshape(nvir,nvir,-1)
                        wvvVV = wvvVV.transpose(2,0,1)

                        kj_list = kconserv[kb,ranges0,ka]
                        tau1_ooVV = numpy.zeros((len(ranges0),nkpts,nocc,nocc,nvir,nvir),dtype=t2.dtype)
                        for iterki,ki in enumerate(ranges0):
                            kj = kj_list[iterki]
                            tau1_ooVV[iterki]    += t2[ki,kj,:]
                            tau1_ooVV[iterki,ki] += einsum('ic,jd->ijcd',t1[ki],t1[kj])
                        tau1_ooVV = tau1_ooVV.transpose(0,2,3,1,4,5).reshape(len(ranges0),nocc,nocc,-1)
                        tmp = einsum('kijd,dab->kijab',tau1_ooVV,wvvVV)

                        for iterki,ki in enumerate(ranges0):
                            kj = kj_list[iterki]
                            if ki == kj:
                                #t2new_tril[tril_index(ki,kj),ka] += tmp[iterki]
                                t2new[ki,kj,ka] += tmp[iterki]
                                if ka < kb:
                                    #t2new_tril[tril_index(kj,ki),kb] += tmp[iterki].transpose(1,0,3,2)
                                    t2new[kj,ki,kb] += tmp[iterki].transpose(1,0,3,2)
                            elif ki < kj:
                                #t2new_tril[tril_index(ki,kj),ka] += tmp[iterki]
                                t2new[ki,kj,ka] += tmp[iterki]
                            elif ki > kj:
                                if ka < kb:
                                    #t2new_tril[tril_index(kj,ki),kb] += tmp[iterki].transpose(1,0,3,2)
                                    t2new[kj,ki,kb] += tmp[iterki].transpose(1,0,3,2)
            loader.slave_finished()
    func3()
    cc.comm.Barrier()
    cput2 = log.timer_debug1('transforming Wvvvv', *cput2)

    #######################################################
    # Making Wvoov and Wovov terms... (part 1/2)
    #######################################################
    cput2 = time.clock(), time.time()
####
    mem = 0.5e9
    pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)
    BLKSIZE = (nkpts_blksize,nkpts,1,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts_blksize,nkpts,1,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
####

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        ranges0, ranges1, ranges2 = _cp(ranges0), _cp(ranges1), _cp(ranges2)
        if min(ranges0) > max(ranges1): #continue if ki > kj
            loader.slave_finished()
            continue

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in ranges0,ranges1,ranges2]
        # TODO this can sometimes not be optimal for ooov, calls for all kb, but in most block set-ups you only need 1 index
        eris_ooov_ji = _cp(eris.ooov[s1,s0])

        eris_voovR1_aXi = _cp(eris.voovR1[s0,s2,:])
        eris_ooovR1_aXi = _cp(eris.ooovR1[s0,s2,:])
        eris_vovvR1_aXi = _cp(eris.vovvR1[s0,s2,:])

        eris_ovovRev_Xai = _cp(eris.ovovRev[s0,s2,:])
        eris_ooovRev_Xai = _cp(eris.ooovRev[s0,s2,:])
        eris_ovvvRev_Xai = _cp(eris.ovvvRev[s0,s2,:])

        for iterki,ki in enumerate(ranges0):
            for iterkj,kj in enumerate(ranges1):
                if ki <= kj:
                    for iterka,ka in enumerate(ranges2):
                        kb = kconserv[ki,ka,kj]
                        ####################################
                        # t2 with 1-electron terms ... (1/2)
                        ####################################
                        tmp = einsum('ac,ijcb->ijab',Lvv[ka],t2[ki,kj,ka])
                        tmp += einsum('ki,kjab->ijab',-Loo[ki],t2[ki,kj,ka])
                        ####################################
                        # t1 with ooov terms ...       (1/2)
                        ####################################
                        tmp2 = eris_ooov_ji[iterkj,iterki,kb].transpose(3,2,1,0).conj() + \
                                einsum('akic,jc->akij',eris_voovR1_aXi[iterki,iterka,kb],t1[kj]) #ooov[kj,ki,kb,ka] ovvo[kb,ka,kj,ki]
                        tmp -= einsum('akij,kb->ijab',tmp2,t1[kb])
                        if ki == kj:
                            #t2new_tril[tril_index(ki,kj),ka] += tmp
                            #t2new_tril[tril_index(ki,kj),kb] += tmp.transpose(1,0,3,2)
                            t2new[ki,kj,ka] += tmp
                            t2new[ki,kj,kb] += tmp.transpose(1,0,3,2)
                        else:
                            #t2new_tril[tril_index(ki,kj),ka] += tmp
                            t2new[ki,kj,ka] += tmp

        for kblock in BLKSIZE2_ranges:
            kk_block_size = kblock[1]-kblock[0]
            kk_slice = slice(kblock[0],kblock[1])
            kk_range = range(kblock[0],kblock[1])

            oOVv   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)
            oOvV   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)

            for iterki,ki in enumerate(ranges0):
                for iterka,ka in enumerate(ranges2):
                    kc_list = kconserv[kk_slice,ki,ka]
                    #########################################################################################
                    # Wvoov term (ka,kk,ki,kc)
                    #    a) the Soovv and oovv contribution to Wvoov is done after the Wovov term, where
                    #        Soovv = 2*oovv[l,k,c,d] - oovv[l,k,d,c]
                    #########################################################################################
                    _WvOoV  = _cp(eris_voovR1_aXi[iterki,iterka,kk_slice]).transpose(1,3,0,2,4).reshape(nvir,nocc,-1)                               #voov[ka,*,ki,*]
                    _WvOoV -= einsum('lic,la->aic',eris_ooovR1_aXi[iterki,iterka,kk_slice].transpose(1,3,0,2,4).reshape(nocc,nocc,-1),t1[ka])       #ooov[ka,*,ki,*]
                    _WvOoV += einsum('adc,id->aic',eris_vovvR1_aXi[iterki,iterka,kk_slice].transpose(1,3,0,2,4).reshape(nvir,nvir,-1),t1[ki])       #vovv[ka,*,ki,*]
                    ###################################
                    # Wovov term (kk,ka,ki,kc)
                    ###################################
                    _WOvoV = _cp(eris_ovovRev_Xai[iterki,iterka,kk_slice]).transpose(2,3,0,1,4).reshape(nvir,nocc,-1)                          #ovov[*,ka,ki,*]
                    _WOvoV -= einsum('lic,la->aic',eris_ooovRev_Xai[iterki,iterka,kk_slice].transpose(2,3,0,1,4).reshape(nocc,nocc,-1),t1[ka]) #ooov[*,ka,ki,*]
                    _WOvoV += einsum('adc,id->aic',eris_ovvvRev_Xai[iterki,iterka,kk_slice].transpose(2,3,0,1,4).reshape(nvir,nvir,-1),t1[ki]) #ovvv[*,ka,ki,*]
                    #
                    # Filling in the oovv terms...
                    #
                    for iterkk,kk in enumerate(kk_range):
                        oOVv[:,iterkk] = _cp(eris.oovv[:,kk,kc_list[iterkk]])
                        oOvV[:,iterkk] = _cp(eris.oovv[kk,:,kc_list[iterkk]])
                    oOVv_f = oOVv.transpose(0,2,5,1,3,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)
                    oOvV_f = oOvV.transpose(0,3,5,1,2,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)

                    tau2_OovV  = t2[:,ki,ka].copy()
                    tau2_OovV[ka] += 2*einsum('id,la->liad',t1[ki],t1[ka])
                    tau2_OovV = tau2_OovV.transpose(2,3,0,1,4).reshape(nocc,nvir,-1)

                    _WvOoV -= 0.5*einsum('dc,iad->aic',oOvV_f,tau2_OovV) # kc consolidated into c, ld consolidated into d
                    _WOvoV -= 0.5*einsum('dc,iad->aic',oOVv_f,tau2_OovV)
                    _WvOoV += 0.5*einsum('dc,iad->aic',2*oOvV_f-oOVv_f,t2[ki,:,ka].transpose(1,3,0,2,4).reshape(nocc,nvir,-1))

                    for iterkj,kj in enumerate(ranges1):
                        if ki <= kj:
                            kb = kconserv[ki,ka,kj]
                            tmp = einsum('aic,jbc->ijab',(2*_WvOoV-_WOvoV),t2[kj,kk_slice,kb].transpose(1,3,0,2,4).reshape(nocc,nvir,-1))
                            tmp -= einsum('aic,jbc->ijab',_WvOoV,t2[kk_slice,kj,kb].transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            if ki == kj:
                                #t2new_tril[tril_index(ki,kj),ka] += tmp
                                #t2new_tril[tril_index(ki,kj),kb] += tmp.transpose(1,0,3,2)
                                t2new[ki,kj,ka] += tmp
                                t2new[ki,kj,kb] += tmp.transpose(1,0,3,2)
                            else:
                                #t2new_tril[tril_index(ki,kj),ka] += tmp
                                t2new[ki,kj,ka] += tmp
                    #kj_ranges = ranges1[ranges1 >= ki]
                    #nkj = kj_ranges.shape[0]
                    #kb_ranges = kconserv[ki,ka,kj_ranges]
                    #t2new[ki,kj_ranges,ka] += einsum('aic,xjbc->xijab',
                    #                                 (2*_WvOoV-_WOvoV),
                    #                                 t2[kj_ranges,kk_slice,kb_ranges].transpose(0,2,4,1,3,5).reshape(nkj,nocc,nvir,-1))
                    #t2new[ki,kj_ranges,ka] -= einsum('aic,xjbc->xijab',_WvOoV,t2[kk_slice,kj_ranges,kb_ranges].transpose(1,3,4,0,2,5).reshape(nkj,nocc,nvir,-1))
        loader.slave_finished()
    cc.comm.Barrier()
    cput2 = log.timer_debug1('transforming Wvoov (ai)', *cput2)

    #######################################################
    # Making Wvoov and Wovov terms... (part 2/2)
    #######################################################

    cput2 = time.clock(), time.time()
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,1,nkpts_blksize,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        if min(ranges0) >= max(ranges1): #continue if ki >= kj
            loader.slave_finished()
            continue

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in ranges0,ranges1,ranges2]

        # TODO this is not optimal for ooov, calls for all ka, but in most block set-ups you only need 1 index
        eris_ooov_ij = _cp(eris.ooov[:,s1])

        eris_voovR1_bXj = _cp(eris.voovR1[s1,s2,:])
        eris_ooovR1_bXj = _cp(eris.ooovR1[s1,s2,:])
        eris_vovvR1_bXj = _cp(eris.vovvR1[s1,s2,:])

        eris_ovovRev_Xbj = _cp(eris.ovovRev[s1,s2,:])
        eris_ooovRev_Xbj = _cp(eris.ooovRev[s1,s2,:])
        eris_ovvvRev_Xbj = _cp(eris.ovvvRev[s1,s2,:])

        for iterki,ki in enumerate(ranges0):
            for iterkj,kj in enumerate(ranges1):
                if ki < kj:
                    for iterkb,kb in enumerate(ranges2):
                        ka = kconserv[ki,kb,kj]
                        ####################################
                        # t2 with 1-electron terms ... (2/2)
                        ####################################
                        tmp = einsum('bc,jica->ijab',Lvv[kb],t2[kj,ki,kb])
                        tmp += einsum('kj,kiba->ijab',-Loo[kj],t2[kj,ki,kb])
                        ####################################
                        # t1 with ooov terms ...       (2/2)
                        ####################################
                        tmp2 = eris_ooov_ij[iterki,iterkj,ka].transpose(3,2,1,0).conj() + \
                                einsum('bkjc,ic->bkji',eris_voovR1_bXj[iterkj,iterkb,ka],t1[ki]) #ooov[ki,kj,ka,kb] ovvo[ka,kb,ki,kj]
                        tmp -= einsum('bkji,ka->ijab',tmp2,t1[ka])
                        #t2new_tril[tril_index(ki,kj),ka] += tmp
                        t2new[ki,kj,ka] += tmp

        for kblock in BLKSIZE2_ranges:
            kk_block_size = kblock[1]-kblock[0]
            kk_slice = slice(kblock[0],kblock[1])
            kk_range = range(kblock[0],kblock[1])

            oOVv   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)
            oOvV   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)

            for iterkj,kj in enumerate(ranges1):
                for iterkb,kb in enumerate(ranges2):
                    kc_list = kconserv[kk_slice,kj,kb]
                    ###################################
                    # Wvoov term (kb,kk,kj,kc)
                    ###################################
                    _WvOoV  = _cp(eris_voovR1_bXj[iterkj,iterkb,kk_slice]).transpose(1,3,0,2,4).reshape(nvir,nocc,-1)                          #voov[kb,*,kj,*]
                    _WvOoV -= einsum('ljc,lb->bjc',eris_ooovR1_bXj[iterkj,iterkb,kk_slice].transpose(1,3,0,2,4).reshape(nocc,nocc,-1),t1[kb])  #ooov[kb,*,kj,*]
                    _WvOoV += einsum('bdc,jd->bjc',eris_vovvR1_bXj[iterkj,iterkb,kk_slice].transpose(1,3,0,2,4).reshape(nvir,nvir,-1),t1[kj])  #vovv[kb,*,kj,*]
                    ###################################
                    # Wovov term (kk,kb,kj,kc)
                    ##################################
                    _WOvoV = _cp(eris_ovovRev_Xbj[iterkj,iterkb,kk_slice]).transpose(2,3,0,1,4).reshape(nvir,nocc,-1)                          #ovov[*,kb,kj,*]
                    _WOvoV -= einsum('ljc,lb->bjc',eris_ooovRev_Xbj[iterkj,iterkb,kk_slice].transpose(2,3,0,1,4).reshape(nocc,nocc,-1),t1[kb]) #ooov[*,kb,kj,*]
                    _WOvoV += einsum('bdc,jd->bjc',eris_ovvvRev_Xbj[iterkj,iterkb,kk_slice].transpose(2,3,0,1,4).reshape(nvir,nvir,-1),t1[kj]) #ovvv[*,kb,kj,*]
                    #
                    # Filling in the oovv terms...
                    #
                    for iterkk,kk in enumerate(kk_range):
                        oOVv[:,iterkk] = _cp(eris.oovv[:,kk,kc_list[iterkk]])
                        oOvV[:,iterkk] = _cp(eris.oovv[kk,:,kc_list[iterkk]])
                    oOVv_f = oOVv.transpose(0,2,5,1,3,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)
                    oOvV_f = oOvV.transpose(0,3,5,1,2,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)

                    tau2_OovV  = t2[:,kj,kb].copy()
                    tau2_OovV[kb] += 2*einsum('jd,lb->ljbd',t1[kj],t1[kb])
                    tau2_OovV = tau2_OovV.transpose(2,3,0,1,4).reshape(nocc,nvir,-1)

                    _WvOoV -= 0.5*einsum('dc,jbd->bjc',oOvV_f,tau2_OovV) # kc consolidated into c, ld consolidated into d
                    _WOvoV -= 0.5*einsum('dc,jbd->bjc',oOVv_f,tau2_OovV)
                    _WvOoV += 0.5*einsum('dc,jbd->bjc',2*oOvV_f-oOVv_f,t2[kj,:,kb].transpose(1,3,0,2,4).reshape(nocc,nvir,-1))

                    for iterki,ki in enumerate(ranges0):
                        if ki < kj:
                            ka = kconserv[ki,kb,kj]
                            tmp = einsum('bjc,iac->ijab',(2*_WvOoV-_WOvoV),t2[ki,kk_slice,ka].transpose(1,3,0,2,4).reshape(nocc,nvir,-1))
                            tmp -= einsum('bjc,iac->ijab',_WvOoV,t2[kk_slice,ki,ka].transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            #t2new_tril[tril_index(ki,kj),ka] += tmp
                            t2new[ki,kj,ka] += tmp
        loader.slave_finished()
    cc.comm.Barrier()
    cput2 = log.timer_debug1('transforming Wvoov (bj)', *cput2)

    cput2 = time.clock(), time.time()
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(1,nkpts,nkpts_blksize,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    #######################################################
    # Making last of the Wovov terms... (part 1/2)
    #######################################################

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        if min(ranges0) > max(ranges1): #continue if ki > kj
            loader.slave_finished()
            continue

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in ranges0,ranges1,ranges2]

        eris_ovovRev_Xbi = _cp(eris.ovovRev[s0,s2,:])
        eris_ooovRev_Xbi = _cp(eris.ooovRev[s0,s2,:])
        eris_ovvvRev_Xbi = _cp(eris.ovvvRev[s0,s2,:])

        eris_vovvL1_jib = _cp(eris.vovvL1[s0,s2,:])

        for iterki,ki in enumerate(ranges0):
            for iterkj,kj in enumerate(ranges1):
                if ki <= kj:
                    for iterkb,kb in enumerate(ranges2):
                        ka = kconserv[ki,kb,kj]
                        ###################################
                        # t1 with ovvv terms ... (part 1/2)
                        ###################################
                        tmp2 = eris_vovvL1_jib[iterki,iterkb,iterkj].transpose(3,2,1,0).conj() - \
                                einsum('kbic,ka->abic',eris_ovovRev_Xbi[iterki,iterkb,ka],t1[ka]) #ovvv[ki,kj,ka,kb]  ovov[ka,kb,ki,kj]
                        tmp  = einsum('abic,jc->ijab',tmp2,t1[kj])
                        if ki == kj:
                            #t2new_tril[tril_index(ki,kj),ka] += tmp
                            #t2new_tril[tril_index(ki,kj),kb] += tmp.transpose(1,0,3,2)
                            t2new[ki,kj,ka] += tmp
                            t2new[ki,kj,kb] += tmp.transpose(1,0,3,2)
                        else:
                            #t2new_tril[tril_index(ki,kj),ka] += tmp
                            t2new[ki,kj,ka] += tmp

        for kblock in BLKSIZE2_ranges:
            kk_block_size = kblock[1]-kblock[0]
            kk_slice = slice(kblock[0],kblock[1])
            kk_range = range(kblock[0],kblock[1])

            oOVv   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)

            for iterki,ki in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges2):
                    kc_list = kconserv[kk_slice,ki,kb]
                    ###################################
                    # Wovov term (kk,kb,ki,kc)
                    ###################################
                    _WOvoV = _cp(eris_ovovRev_Xbi[iterki,iterkb,kk_slice]).transpose(2,3,0,1,4).reshape(nvir,nocc,-1)                          #ovov[*,kb,ki,*]
                    _WOvoV -= einsum('lic,lb->bic',eris_ooovRev_Xbi[iterki,iterkb,kk_slice].transpose(2,3,0,1,4).reshape(nocc,nocc,-1),t1[kb]) #ooov[*,kb,ki,*]
                    _WOvoV += einsum('bdc,id->bic',eris_ovvvRev_Xbi[iterki,iterkb,kk_slice].transpose(2,3,0,1,4).reshape(nvir,nvir,-1),t1[ki]) #ovvv[*,kb,ki,*]
                    #
                    # Filling in the oovv terms...
                    #
                    for iterkk,kk in enumerate(kk_range):
                        oOVv[:,iterkk] = _cp(eris.oovv[:,kk,kc_list[iterkk]])
                    oOVv_f = oOVv.transpose(0,2,5,1,3,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)

                    tau2_OovV  = t2[:,ki,kb].copy()
                    tau2_OovV[kb] += 2*einsum('id,lb->libd',t1[ki],t1[kb])
                    _WOvoV -= 0.5*einsum('dc,ibd->bic',oOVv_f,tau2_OovV.transpose(2,3,0,1,4).reshape(nocc,nvir,-1))

                    for iterkj,kj in enumerate(ranges1):
                        if ki <= kj:
                            ka = kconserv[ki,kb,kj]
                            tmp = einsum('bic,jac->ijab',_WOvoV,t2[kk_slice,kj,ka].transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            if ki == kj:
                                #t2new_tril[tril_index(ki,kj),ka] -= tmp
                                #t2new_tril[tril_index(ki,kj),kb] -= tmp.transpose(1,0,3,2)
                                t2new[ki,kj,ka] -= tmp
                                t2new[ki,kj,kb] -= tmp.transpose(1,0,3,2)
                            else:
                                #t2new_tril[tril_index(ki,kj),ka] -= tmp
                                t2new[ki,kj,ka] -= tmp
        loader.slave_finished()
    cc.comm.Barrier()
    cput2 = log.timer_debug1('transforming Wovov (bi)', *cput2)

    cput2 = time.clock(), time.time()
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,1,nkpts_blksize,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    #######################################################
    # Making last of the Wovov terms... (part 2/2)
    #######################################################

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        if min(ranges0) >= max(ranges1): #continue if ki >= kj
            loader.slave_finished()
            continue

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in ranges0,ranges1,ranges2]

        eris_ovovRev_Xaj = _cp(eris.ovovRev[s1,s2,:])
        eris_ooovRev_Xaj = _cp(eris.ooovRev[s1,s2,:])
        eris_ovvvRev_Xaj = _cp(eris.ovvvRev[s1,s2,:])

        eris_vovvL1_ija = _cp(eris.vovvL1[s1,s2,:])

        for iterki,ki in enumerate(ranges0):
            for iterkj,kj in enumerate(ranges1):
                if ki < kj:
                    for iterka,ka in enumerate(ranges2):
                        kb = kconserv[ki,ka,kj]
                        ###################################
                        # t1 with ovvv terms ... (part 2/2)
                        ###################################
                        tmp2 = eris_vovvL1_ija[iterkj,iterka,iterki].transpose(3,2,1,0).conj() - \
                                einsum('kajc,kb->bajc',eris_ovovRev_Xaj[iterkj,iterka,kb],t1[kb]) #ovvv[kj,ki,kb,ka]  ovov[kb,ka,kj,ki]
                        tmp  = einsum('bajc,ic->ijab',tmp2,t1[ki])
                        #t2new_tril[tril_index(ki,kj),ka] += tmp
                        t2new[ki,kj,ka] += tmp

        for kblock in BLKSIZE2_ranges:
            kk_block_size = kblock[1]-kblock[0]
            kk_slice = slice(kblock[0],kblock[1])
            kk_range = range(kblock[0],kblock[1])

            oOVv   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)

            for iterkj,kj in enumerate(ranges1):
                for iterka,ka in enumerate(ranges2):
                    kc_list = kconserv[kk_slice,kj,ka]
                    ###################################
                    # Wovov term (kk,ka,kj,kc)
                    ###################################
                    _WOvoV = _cp(eris_ovovRev_Xaj[iterkj,iterka,kk_slice]).transpose(2,3,0,1,4).reshape(nvir,nocc,-1)                          #ovov[*,ka,kj,*]
                    _WOvoV -= einsum('ljc,la->ajc',eris_ooovRev_Xaj[iterkj,iterka,kk_slice].transpose(2,3,0,1,4).reshape(nocc,nocc,-1),t1[ka]) #ooov[*,ka,kj,*]
                    _WOvoV += einsum('adc,jd->ajc',eris_ovvvRev_Xaj[iterkj,iterka,kk_slice].transpose(2,3,0,1,4).reshape(nvir,nvir,-1),t1[kj]) #ovvv[*,ka,kj,*]
                    #
                    # Filling in the oovv terms...
                    #
                    for iterkk,kk in enumerate(kk_range):
                        oOVv[:,iterkk] = _cp(eris.oovv[:,kk,kc_list[iterkk]])
                    oOVv_f = oOVv.transpose(0,2,5,1,3,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)

                    tau2_OovV  = t2[:,kj,ka].copy()
                    tau2_OovV[ka] += 2*einsum('jd,la->ljad',t1[kj],t1[ka])
                    _WOvoV -= 0.5*einsum('dc,jad->ajc',oOVv_f,tau2_OovV.transpose(2,3,0,1,4).reshape(nocc,nvir,-1))

                    for iterki,ki in enumerate(ranges0):
                        if ki < kj:
                            kb = kconserv[ki,ka,kj]
                            tmp = einsum('ajc,ibc->ijab',_WOvoV,t2[kk_slice,ki,kb].transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            #t2new_tril[tril_index(ki,kj),ka] -= tmp
                            t2new[ki,kj,ka] -= tmp
        loader.slave_finished()
    cc.comm.Barrier()
    cput2 = log.timer_debug1('transforming Wovov (aj)', *cput2)

    #for ki in range(nkpts):
    #    for kj in range(nkpts):
    #        if ki <= kj:
    #            for ka in range(nkpts):
    #                t2new[ki,kj,ka] += t2new_tril[tril_index(ki,kj),ka]

    cc.comm.Barrier()
    cc.comm.Allreduce(MPI.IN_PLACE, t2new, op=MPI.SUM)
    #cc.comm.Allreduce(MPI.IN_PLACE, t2new_tril, op=MPI.SUM)

    for kj in range(nkpts):
        for ki in range(kj):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                t2new[kj,ki,kb] += t2new[ki,kj,ka].transpose(1,0,3,2)

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
            #if ki <= kj:
            #    t2new_tril[tril_index(ki,kj),ka] /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)
#    sys.exit("exiting for testing...")

    cc.comm.Barrier()

    #tril_energy = energy_tril(cc,t1new,t2new_tril,eris)

    return t1new, t2new

def energy_tril(cc, t1, t2, eris):
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
            if ki <= kj:
                #kb = kj
                t1t1[tril_index(ki,kj),ka] = einsum('ia,jb->ijab',t1[ki],t1[kj])
    tau = t2 + t1t1
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                if ki <= kj:
                   kb = kconserv[ki,ka,kj]
                   e += einsum('ijab,ijab', tau[tril_index(ki,kj),ka], (2.*eris.oovv[ki,kj,ka]-eris.oovv[ki,kj,kb].transpose(0,1,3,2)))
                if kj < ki:
                   kb = kconserv[ki,ka,kj]
                   e += einsum('ijab,ijab', tau[tril_index(kj,ki),kb].transpose(1,0,3,2), (2.*eris.oovv[ki,kj,ka]-eris.oovv[ki,kj,kb].transpose(0,1,3,2)))
    comm.Barrier()
    e /= nkpts
    print "tril energy = ", e.real
    return e.real

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
                if ki <= kj:
                    e += einsum('ijab,ijab', tau[ki,kj,ka], (2.*eris.oovv[ki,kj,ka]-eris.oovv[ki,kj,kb].transpose(0,1,3,2)))
                    norm = numpy.linalg.norm(tau[ki,kj,ka] - tau[kj,ki,kb].transpose(1,0,3,2))
                if ki > kj:
                    e += einsum('ijab,ijab', tau[kj,ki,kb].transpose(1,0,3,2), (2.*eris.oovv[ki,kj,ka]-eris.oovv[ki,kj,kb].transpose(0,1,3,2)))
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
        self.made_ip_imds = False
        self.made_ea_imds = False
        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)

    def dump_flags(self):
        pyscf.cc.ccsd.CCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** EOM CC flags ********')

    def init_amps_tril(self, eris):
        time0 = time.clock(), time.time()
        rank = self.rank
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = len(self._kpts)
        t1 = numpy.zeros((nkpts,nocc,nvir), dtype=numpy.complex128)
        tril_shape = ((nkpts)*(nkpts+1))/2
        t2_tril = numpy.zeros((tril_shape,nkpts,nocc,nocc,nvir,nvir),dtype=numpy.complex128)
        local_mp2 = numpy.array(0.0,dtype=numpy.complex128)
        #woovv = numpy.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=numpy.complex128)
        self.emp2 = 0
        foo = eris.fock[:,:nocc,:nocc].copy()
        fvv = eris.fock[:,nocc:,nocc:].copy()
        #eris_oovv = numpy.asarray(eris.oovv).copy()
        eia = numpy.zeros((nocc,nvir))
        eijab = numpy.zeros((nocc,nocc,nvir,nvir))

        kconserv = self.kconserv
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(1,1,nkpts,))
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
                    if ki <= kj:
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
                            t2_tril[tril_index(ki,kj),ka] = numpy.conj(oovv_ijab / eijab)
                            local_mp2 += numpy.dot(t2_tril[tril_index(ki,kj),ka].flatten(),woovv.flatten())
                    if kj < ki:
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
                            t2_tril[tril_index(kj,ki),kb] = numpy.conj(oovv_ijab / eijab)
                            local_mp2 += numpy.dot(t2_tril[tril_index(kj,ki),kb].flatten(),woovv.flatten())
            loader.slave_finished()

        self.comm.Allreduce(MPI.IN_PLACE, local_mp2, op=MPI.SUM)
        self.comm.Allreduce(MPI.IN_PLACE, t2_tril, op=MPI.SUM)
        self.emp2 = local_mp2.real
        self.emp2 /= nkpts

        if rank == 0:
            logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
            logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2_tril

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
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(1,1,nkpts,))
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

    def ipccsd(self, nroots=2*4, klist=None):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size =  nocc + nkpts*nkpts*nocc*nocc*nvir
        if klist is None:
            klist = range(nkpts)
        evals = numpy.zeros((len(klist),nroots),numpy.complex)
        evecs = numpy.zeros((len(klist),size,nroots),numpy.complex)
        for ikshift,kshift in enumerate(klist):
            self.kshift = kshift
            print "ip-ccsd eigenvalues at k= ", kshift
            evals[ikshift], evecs[ikshift] = eigs(self.ipccsd_matvec, size, nroots=nroots)
        return evals.real, evecs

    @profile
    def ipccsd_matvec(self, vector):
    ########################################################
    # FOLLOWING:                                           #
    # Z. Tu, F. Wang, and X. Li                            #
    # J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)        #
    ########################################################
        r1,r2 = self.vector_to_amplitudes_ip(vector)
        r1 = self.comm.bcast(r1, root=0)
        r2 = self.comm.bcast(r2, root=0)

        nproc = self.comm.Get_size()
        t1,t2 = self.t1, self.t2
        nkpts,nocc,nvir = self.t1.shape
        nkpts = self.nkpts
        kshift = self.kshift
        kconserv = self.kconserv

        if not self.made_ip_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ip(self)
            self.made_ip_imds = True

        imds = self.imds

        #Lvv = imdk.Lvv(self,t1,t2,eris)
        #Loo = imdk.Loo(self,t1,t2,eris)
        #Fov = imdk.cc_Fov(self,t1,t2,eris)
        #Wooov = imdk.Wooov(self,t1,t2,eris)
        #Wovvo = imdk.Wovvo(self,t1,t2,eris)
        #Wovoo = imdk.Wovoo(self,t1,t2,eris)
        #Woooo = imdk.Woooo(self,t1,t2,eris)
        #Wovov = imdk.Wovov(self,t1,t2,eris)
        #Woovv = eris.oovv
        cput2 = time.clock(), time.time()
        Hr1 = numpy.zeros(r1.shape,dtype=t1.dtype)
        #loader = mpi_load_balancer.load_balancer(BLKSIZE=(max(int(numpy.floor(nkpts/self.comm.Get_size())),1),))
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,))
        loader.set_ranges((range(nkpts),))

        good2go = True
        while(good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0 = loader.get_blocks_from_data(data)

            s0 = slice(min(ranges0),max(ranges0)+1)

            Wooov_Xls = _cp(imds.Wooov[:,s0,kshift])
            Wooov_lXs = _cp(imds.Wooov[s0,:,kshift])

            for iterkl,kl in enumerate(ranges0):
                Hr1 += einsum('ld,ild->i',imds.Fov[kl],2.*r2[kshift,kl]-r2[kl,kshift].transpose(1,0,2))
                Hr1 += einsum('xklid,xkld->i',-2.*Wooov_Xls[:,iterkl]+Wooov_lXs[iterkl,:].transpose(0,2,1,3,4),r2[:,kl])
            loader.slave_finished()
        self.comm.Allreduce(MPI.IN_PLACE, Hr1, op=MPI.SUM)
        Hr1 -= einsum('ki,k->i',imds.Loo[kshift],r1)

        Hr2 = numpy.zeros(r2.shape,dtype=t1.dtype)
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,1,))
        loader.set_ranges((range(nkpts),range(nkpts),))

        good2go = True
        while(good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0, ranges1 = loader.get_blocks_from_data(data)

            s0,s1 = [slice(min(x),max(x)+1) for x in ranges0,ranges1]
            Wovoo_sXi  = _cp(imds.Wovoo[kshift,:,s0])
            WooooS_Xij = _cp(imds.WooooS[:,s0,s1])

            tmp = numpy.zeros(nvir,dtype=t2.dtype)
            for kl in range(nkpts):
                kk_list = range(nkpts)
                kd_list = kconserv[kl,kshift,kk_list]
                tmp += einsum('lc,l->c',(2.*imds.Woovv[kl,kk_list,kd_list].transpose(1,0,2,3,4) \
                                           -imds.Woovv[kk_list,kl,kd_list].transpose(2,0,1,3,4)).reshape(-1,nvir),
                                         r2[kk_list,kl].transpose(2,0,1,3).reshape(-1))
                #for kk in range(nkpts):
                #    kd = kconserv[kl,kshift,kk]
                #    if numpy.linalg.norm(noob[kk] - imds.Woovv[kl,kk,kd]) > 1e-8:
                #        print "DIFFERENT"
                #        sys.exit()
                #    tmp += einsum('lkdc,kld->c',2.*imds.Woovv[kl,kk,kd]-imds.Woovv[kk,kl,kd].transpose(1,0,2,3),r2[kk,kl])
                #    #tmp += einsum('lkdc,kld->c',-imds.Woovv[kk,kl,kd].transpose(1,0,2,3),r2[kk,kl])
            for iterki, ki in enumerate(ranges0):
                for iterkj, kj in enumerate(ranges1):
                    Hr2[ki,kj] += -einsum('c,ijcb->ijb',tmp,t2[ki,kj,kshift])

            for iterki, ki in enumerate(ranges0):
                for iterkj, kj in enumerate(ranges1):
                    kb = kconserv[ki,kshift,kj]
                    Hr2[ki,kj] += einsum('bd,ijd->ijb',imds.Lvv[kb],r2[ki,kj])
                    Hr2[ki,kj] -= einsum('li,ljb->ijb',imds.Loo[ki],r2[ki,kj])
                    Hr2[ki,kj] -= einsum('lj,ilb->ijb',imds.Loo[kj],r2[ki,kj])
                    #Hr2[ki,kj] -= einsum('kbij,k->ijb',imds.Wovoo[kshift,kb,ki],r1)
                    Hr2[ki,kj] -= einsum('kbij,k->ijb',Wovoo_sXi[kb,iterki],r1)

                    kl_list = range(nkpts)
                    kk_list = kconserv[ki,kl_list,kj]
                    Hr2[ki,kj] += einsum('klij,klb->ijb',WooooS_Xij[kl_list,iterki,iterkj].transpose(1,0,2,3,4).reshape(nocc,nocc*nkpts,nocc,nocc),
                                                         r2[kk_list,kl_list].transpose(1,0,2,3).reshape(nocc,nocc*nkpts,nvir))
                    #Hr2[ki,kj] += einsum('bljd,ild->ijb',imds.Wvoov[kb,:,kj].transpose(1,0,2,3,4).reshape(nvir,nocc*nkpts,nocc,nvir),
                    #                                     (2.*r2[ki,:].transpose(1,0,2,3)-r2[:,ki].transpose(2,0,1,3)).reshape(nocc,nocc*nkpts,nvir))
                    #Hr2[ki,kj] += -einsum('lbjd,ild->ijb',imds.Wovov[:,kb,kj].reshape(nocc*nkpts,nvir,nocc,nvir),
                    #                                      r2[ki,:].transpose(1,0,2,3).reshape(nocc,nocc*nkpts,nvir)) #typo in nooijen's paper
                    #Hr2[ki,kj] += -einsum('lbid,ljd->ijb',imds.Wovov[:,kb,ki].reshape(nocc*nkpts,nvir,nocc,nvir),
                    #                                      r2[:,kj].reshape(nocc*nkpts,nocc,nvir))

            Wovov_Xbi = _cp(imds.Wovov[:,s1,s0])

            for iterki,ki in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges1):
                    kj = kconserv[kshift,ki,kb]
                    Hr2[ki,kj] += -einsum('lbid,ljd->ijb',Wovov_Xbi[:,iterkb,iterki].reshape(nocc*nkpts,nvir,nocc,nvir),
                                                          r2[:,kj].reshape(nocc*nkpts,nocc,nvir))
            Wvoov_bXj = _cp(imds.Wvoov[s1,:,s0])
            Wovov_Xbj = _cp(imds.Wovov[:,s1,s0])

            for iterkj,kj in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges1):
                    ki = kconserv[kshift,kj,kb]
                    Hr2[ki,kj] += einsum('bljd,ild->ijb',Wvoov_bXj[iterkb,:,iterkj].transpose(1,0,2,3,4).reshape(nvir,nocc*nkpts,nocc,nvir),
                                                         (2.*r2[ki,:].transpose(1,0,2,3)-r2[:,ki].transpose(2,0,1,3)).reshape(nocc,nocc*nkpts,nvir))
                    Hr2[ki,kj] += -einsum('lbjd,ild->ijb',Wovov_Xbj[:,iterkb,iterkj].reshape(nocc*nkpts,nvir,nocc,nvir),
                                                          r2[ki,:].transpose(1,0,2,3).reshape(nocc,nocc*nkpts,nvir)) #typo in nooijen's paper
            loader.slave_finished()
        self.comm.Allreduce(MPI.IN_PLACE, Hr2, op=MPI.SUM)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ip(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts

        r2 = vector[nocc:].copy().reshape(nkpts,nkpts,nocc,nocc,nvir)
        r1 = vector[:nocc].copy()
        return [r1,r2]

    def amplitudes_to_vector_ip(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size = nocc + nkpts*nkpts*nocc*nocc*nvir

        vector = numpy.zeros((size), r1.dtype)
        vector[:nocc] = r1.copy()
        vector[nocc:] = r2.copy().reshape(nkpts*nkpts*nocc*nocc*nvir)
        return vector

    def eaccsd(self, nroots=2*4, klist=None):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size =  nvir + nkpts*nkpts*nocc*nvir*nvir
        if klist is None:
            klist = range(nkpts)
        evals = numpy.zeros((len(klist),nroots),numpy.complex)
        evecs = numpy.zeros((len(klist),size,nroots),numpy.complex)
        for ikshift,kshift in enumerate(klist):
            self.kshift = kshift
            print "ea-ccsd eigenvalues at k= ", kshift
            evals[ikshift], evecs[ikshift] = eigs(self.eaccsd_matvec, size, nroots=nroots)
        return evals.real, evecs

    @profile
    def eaccsd_matvec(self, vector):
    ########################################################
    # FOLLOWING:                                           #
    # M. Nooijen and R. J. Bartlett,                       #
    # J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)        #
    ########################################################
        r1,r2 = self.vector_to_amplitudes_ea(vector)

        t1,t2 = self.t1, self.t2
        nkpts,nocc,nvir = self.t1.shape
        nkpts = self.nkpts
        kshift = self.kshift
        kconserv = self.kconserv

        if not self.made_ea_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ea(self)
            self.made_ea_imds = True

        imds = self.imds

        #Lvv = imdk.Lvv(self,t1,t2,eris)
        #Loo = imdk.Loo(self,t1,t2,eris)
        #Fov = imdk.cc_Fov(self,t1,t2,eris)
        #Wvovv = imdk.Wvovv(self,t1,t2,eris)
        #Wvvvo = imdk.Wvvvo(self,t1,t2,eris)
        #Wovvo = imdk.Wovvo(self,t1,t2,eris)
        #Wvvvv = imdk.Wvvvv(self,t1,t2,eris)
        #Woovv = eris.oovv
        #Wovov = imdk.Wovov(self,t1,t2,eris)

        Hr1 = numpy.zeros(r1.shape,dtype=t1.dtype)
        mem = 0.5e9
        pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
        nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)
        #loader = mpi_load_balancer.load_balancer(BLKSIZE=(max(int(numpy.floor(nkpts/self.comm.Get_size())),1),))
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts_blksize,))
        loader.set_ranges((range(nkpts),))

        good2go = True
        while(good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0 = loader.get_blocks_from_data(data)

            s0 = slice(min(ranges0),max(ranges0)+1)
            s1 = slice(kshift,kshift+1)

            Wvovv_slX = _cp(imds.Wvovv[s1,s0,:])

            for iterkl,kl in enumerate(ranges0):
                Hr1 += 2.*einsum('ld,lad->a',imds.Fov[kl],r2[kl,kshift])
                Hr1 +=   -einsum('ld,lda->a',imds.Fov[kl],r2[kl,kl])
                kd_list = numpy.array(kconserv[kshift,range(nkpts),kl])
                Hr1 += einsum('alxcd,lxcd->a',2.*Wvovv_slX[kshift,iterkl,:].transpose(1,2,0,3,4)-Wvovv_slX[kshift,iterkl,kd_list].transpose(1,2,0,4,3),
                               r2[kl,:].transpose(1,0,2,3))
                #for kc in range(nkpts):
                #    kd = kconserv[kshift,kc,kl]
                #    #Hr1 +=  einsum('alcd,lcd->a',2.*Wvovv_slX[iterkl,kc]-Wvovv_slX[iterkl,kd].transpose(0,1,3,2),r2[kl,kc])
                #    #Hr1 +=  einsum('alcd,lcd->a',2.*imds.Wvovv[kshift,kl,kc]-imds.Wvovv[kshift,kl,kd].transpose(0,1,3,2),r2[kl,kc])
                #    #Hr1 +=  einsum('alcd,lcd->a',2.*imds.Wvovv[kshift,kl,kc],r2[kl,kc])
                #    #Hr1 += -einsum('aldc,lcd->a',imds.Wvovv[kshift,kl,kd],r2[kl,kc])
            loader.slave_finished()
        self.comm.Allreduce(MPI.IN_PLACE, Hr1, op=MPI.SUM)
        Hr1 += einsum('ac,c->a',imds.Lvv[kshift],r1)

        Hr2 = numpy.zeros(r2.shape,dtype=t1.dtype)
        cput2 = time.clock(), time.time()
        mem = 0.5e9
        pre = 1.*nvir*nvir*nvir*nvir*nkpts*16
        nkpts_blksize  = min(max(int(numpy.floor(mem/pre)),1),nkpts)
        nkpts_blksize2 = min(max(int(numpy.floor(mem/(nkpts_blksize*pre))),1),nkpts)
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts_blksize2,nkpts_blksize,))
        loader.set_ranges((range(nkpts),range(nkpts),))

        good2go = True
        while(good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0, ranges1 = loader.get_blocks_from_data(data)

            s0,s1 = [slice(min(x),max(x)+1) for x in ranges0,ranges1]

            for iterkj,kj in enumerate(ranges0):
                for iterka,ka in enumerate(ranges1):
                    kb = kconserv[kshift,ka,kj]
                    Hr2[kj,ka] -= einsum('lj,lab->jab',imds.Loo[kj],r2[kj,ka])
                    Hr2[kj,ka] += einsum('ac,jcb->jab',imds.Lvv[ka],r2[kj,ka])
                    Hr2[kj,ka] += einsum('bd,jad->jab',imds.Lvv[kb],r2[kj,ka])

            #Wvvvo_abX = _cp(imds.Wvvvo[s0,s1,kshift])
            WvvvoR1_abX = _cp(imds.WvvvoR1[kshift,s0,s1])

            for iterka,ka in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges1):
                    kj = kconserv[ka,kshift,kb]
                    Hr2[kj,ka] += einsum('abcj,c->jab',WvvvoR1_abX[iterka,iterkb],r1)

            WovovRev_Xaj = _cp(imds.WovovRev[s0,s1,:])

            for iterkj,kj in enumerate(ranges0):
                for iterka,ka in enumerate(ranges1):
                    kb = kconserv[kshift,ka,kj]
                    kl_range = range(nkpts)
                    kd_range = kconserv[ka,kj,kl_range]
                    #Hr2[kj,ka] += -einsum('aldj,ldb->jab',
                    #                      imds.Wovov[:,ka,kj].transpose(2,0,1,4,3).reshape(nvir,nocc*nkpts,nvir,nocc),
                    #                      r2[kl_range,kd_range].reshape(nkpts*nocc,nvir,nvir))
                    #Hr2[kj,ka] += -einsum('axj,xb->jab',
                    #                      imds.Wovov[:,ka,kj].transpose(2,0,1,4,3).reshape(nvir,nocc*nvir*nkpts,nocc),
                    #                      r2[kl_range,kd_range].reshape(nkpts*nocc*nvir,nvir))
                    Hr2[kj,ka] += -einsum('axj,xb->jab',
                                          WovovRev_Xaj[iterkj,iterka,:].transpose(2,0,1,4,3).reshape(nvir,nocc*nvir*nkpts,nocc),
                                          r2[kl_range,kd_range].reshape(nkpts*nocc*nvir,nvir))

            tmp = numpy.zeros(nocc,dtype=t2.dtype)
            for kl in range(nkpts):
                kd_range = _cp(range(nkpts))
                kc_range = _cp(kconserv[kshift,kd_range,kl])
                tmp += einsum('kl,l->k',(2.*imds.Woovv[kshift,kl,kc_range].transpose(1,2,0,3,4)-
                                            imds.Woovv[kshift,kl,kd_range].transpose(1,2,0,4,3)).reshape(nocc,-1),
                                        r2[kl,kc_range].transpose(1,0,2,3).reshape(-1))
            #for kd in range(nkpts):
            #    for kc in range(nkpts):
            #        kk = kshift
            #        kl = kconserv[kc,kk,kd]
            #        #tmp = ( 2.*einsum('klcd,lcd->k',imds.Woovv[kk,kl,kc],r2[kl,kc])
            #        #          -einsum('kldc,lcd->k',imds.Woovv[kk,kl,kd],r2[kl,kc]) )
            #        tmp += einsum('klcd,lcd->k',2.*imds.Woovv[kk,kl,kc]-imds.Woovv[kk,kl,kd].transpose(0,1,3,2),r2[kl,kc])
            for iterkj,kj in enumerate(ranges0):
                for iterka,ka in enumerate(ranges1):
                    kb = kconserv[kshift,ka,kj]
                    Hr2[kj,ka] += -einsum('k,kjab->jab',tmp,t2[kshift,kj,ka])

            #Wovov_Xbj = _cp(imds.Wovov[:,s1,s0])
            #Wvoov_bXj = _cp(imds.Wvoov[s1,:,s0])
            WovovRev_Xbj = _cp(imds.WovovRev[s0,s1,:])
            WvoovR1_bXj = _cp(imds.WvoovR1[s0,s1,:])
            # shift + kj - kb = ka
            for iterkj,kj in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges1):
                    ka = kconserv[kshift,kb,kj]
                    Hr2[kj,ka] += -einsum('bldj,lad->jab',WovovRev_Xbj[iterkj,iterkb,:].transpose(2,0,1,4,3).reshape(nvir,nkpts*nocc,nvir,nocc),
                                          r2[:,ka].reshape(nkpts*nocc,nvir,nvir))
                    #Hr2[kj,ka] += -einsum('bldj,lad->jab',Wovov_Xbj[:,iterkb,iterkj].transpose(2,0,1,4,3).reshape(nvir,nkpts*nocc,nvir,nocc),
                    #                      r2[:,ka].reshape(nkpts*nocc,nvir,nvir))
                    kl_range = _cp(range(nkpts))
                    kd_range = _cp(kconserv[kb,kj,kl_range])
                    kl_slice = slice(0,nkpts)
                    #Hr2[kj,ka] += einsum('bljd,lad->jab',Wvoov_bXj[iterkb,:,iterkj].transpose(1,0,2,3,4).reshape(nvir,nocc*nkpts,nocc,nvir),
                    #                        (2.*r2[:,ka]-r2[kl_range,kd_range].transpose(0,1,3,2)).reshape(nocc*nkpts,nvir,nvir))
                    Hr2[kj,ka] += einsum('bljd,lad->jab',WvoovR1_bXj[iterkj,iterkb,:].transpose(1,0,2,3,4).reshape(nvir,nocc*nkpts,nocc,nvir),
                                            (2.*r2[:,ka]-r2[kl_slice,kd_range].transpose(0,1,3,2)).reshape(nocc*nkpts,nvir,nvir))

            Wvvvv_abX = _cp(imds.Wvvvv[s0,s1])
            # kj = ka - kshift + kb
            for iterka,ka in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges1):
                    kj = kconserv[ka,kshift,kb]
                    #Hr2[kj,ka] += einsum('abcd,jcd->jab',Wvvvv_abX[iterka,iterkb,:].transpose(1,2,0,3,4).reshape(nvir,nvir,nvir*nkpts,nvir),
                    #                     r2[kj,:].transpose(1,0,2,3).reshape(nocc,nvir*nkpts,nvir))
                    Hr2[kj,ka] += einsum('abx,jx->jab',Wvvvv_abX[iterka,iterkb,:].transpose(1,2,0,3,4).reshape(nvir,nvir,nvir*nkpts*nvir),
                                         r2[kj,:].transpose(1,0,2,3).reshape(nocc,nvir*nkpts*nvir))
            loader.slave_finished()
        self.comm.Allreduce(MPI.IN_PLACE, Hr2, op=MPI.SUM)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ea(self,vector):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts

        r1 = vector[:nvir].copy()
        r2 = vector[nvir:].copy().reshape(nkpts,nkpts,nocc,nvir,nvir)
        return [r1,r2]

    def amplitudes_to_vector_ea(self,r1,r2):
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = self.nkpts
        size = nvir + nkpts*nkpts*nocc*nvir*nvir

        vector = numpy.zeros((size), r1.dtype)
        vector[:nvir] = r1.copy()
        vector[nvir:] = r2.copy().reshape(nkpts*nkpts*nocc*nvir*nvir)
        return vector

class _ERIS:
    ##@profile
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
            or (cc.mol.incore_anyway and False)):
            kconserv = cc.kconserv
            khelper = cc.khelper #kpoint_helper.unique_pqr_list(cc._scf.cell,cc._kpts)

            unique_klist = khelper.get_uniqueList()
            nUnique_klist = khelper.nUnique
            loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,))
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
            if cc.rank == 0:
                _tmpfile1_name = "eris1.hdf5"
            _tmpfile1_name = cc.comm.bcast(_tmpfile1_name, root=0)
            print _tmpfile1_name
######
            read_feri=False
            if cc.rank == 0:
                if os.path.isfile(_tmpfile1_name):
                    read_feri=True
            read_feri = cc.comm.bcast(read_feri,root=0)

            if read_feri is True:
                self.feri1 = h5py.File(_tmpfile1_name, 'r', driver='mpio', comm=MPI.COMM_WORLD)
                self.oooo  = self.feri1['oooo']
                self.ooov  = self.feri1['ooov']
                self.ovoo  = self.feri1['ovoo']
                self.oovv  = self.feri1['oovv']
                self.ovov  = self.feri1['ovov']
                self.ovvo  = self.feri1['ovvo']
                self.voov  = self.feri1['voov']
                self.ovvv  = self.feri1['ovvv']
                self.vovv  = self.feri1['vovv']
                self.vvvv  = self.feri1['vvvv']

                self.ovovL1  = self.feri1['ovovL1']
                self.ooovL1  = self.feri1['ooovL1']
                #self.ovvvL1  = self.feri1['ovvvL1']
                self.voovR1  = self.feri1['voovR1']
                self.ooovR1  = self.feri1['ooovR1']
                self.vovvR1  = self.feri1['vovvR1']
                self.vovvL1  = self.feri1['vovvL1']
                self.ovovRev  = self.feri1['ovovRev']
                self.ooovRev  = self.feri1['ooovRev']
                self.ovvvRev  = self.feri1['ovvvRev']

                print "........WARNING : using oovv in memory......."
                new_oovv = numpy.empty( (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff.dtype)
                for kp in range(nkpts):
                    for kq in range(nkpts):
                        for kr in range(nkpts):
                            new_oovv[kp,kq,kr] = self.oovv[kp,kq,kr].copy()
                self.oovv = new_oovv

                print "lower triangular oovv"
                #self.triu_oovv = numpy.empty( ((nkpts*(nkpts+1))/2,nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff.dtype)
                #triu_indices = [list(x) for x in numpy.triu_indices(nkpts)]
                #self.triu_oovv = self.oovv[triu_indices]
                return
            cc.comm.Barrier()
######
            self.feri1 = h5py.File(_tmpfile1_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)

            ds_type = mo_coeff.dtype

            self.oooo  = self.feri1.create_dataset('oooo',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=ds_type)
            self.ooov  = self.feri1.create_dataset('ooov',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
            self.ovoo  = self.feri1.create_dataset('ovoo',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=ds_type)
            self.oovv  = self.feri1.create_dataset('oovv',  (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=ds_type)

            self.ovov  = self.feri1.create_dataset('ovov',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
            self.ovvo  = self.feri1.create_dataset('ovvo',  (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
            self.voov  = self.feri1.create_dataset('voov',  (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
            self.ovvv  = self.feri1.create_dataset('ovvv',  (nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=ds_type)
            self.vovv  = self.feri1.create_dataset('vovv',  (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=ds_type)
            self.vvvv  = self.feri1.create_dataset('vvvv',  (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=ds_type)

            self.ovovL1  = self.feri1.create_dataset('ovovL1',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
            self.ooovL1  = self.feri1.create_dataset('ooovL1',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
            #self.ovvvL1  = self.feri1.create_dataset('ovvvL1',  (nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=ds_type)
            self.ovovRev  = self.feri1.create_dataset('ovovRev',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
            self.ooovRev  = self.feri1.create_dataset('ooovRev',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
            self.ovvvRev  = self.feri1.create_dataset('ovvvRev',  (nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=ds_type)

            self.voovR1  = self.feri1.create_dataset('voovR1',  (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
            self.ooovR1  = self.feri1.create_dataset('ooovR1',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
            self.vovvR1  = self.feri1.create_dataset('vovvR1',  (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=ds_type)
            self.vovvL1  = self.feri1.create_dataset('vovvL1',  (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=ds_type)

            #######################################################
            ## Setting up permutational symmetry and MPI stuff    #
            #######################################################
            kconserv = cc.kconserv
            khelper = cc.khelper #kpoint_helper.unique_pqr_list(cc._scf.cell,cc._kpts)
            unique_klist = khelper.get_uniqueList()
            nUnique_klist = khelper.nUnique
            print "ints = ", nUnique_klist

####
            mem = 0.5e9
            pre = 1.*nocc*nocc*nmo*nmo*nkpts*16
            nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)
####
            BLKSIZE = (1,nkpts_blksize,nkpts,)
            if cc.rank == 0:
                print "ERI oopq blksize = (%3d %3d %3d)" % BLKSIZE
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
                ############################################################################
                self.oooo    [min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,:nocc]
                self.ooov    [min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:]
                self.ooovL1  [min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1,min(ranges0):max(ranges0)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:].transpose(1,2,0,3,4,5,6)
                self.ooovR1  [min(ranges2):max(ranges2)+1,min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:].transpose(2,0,1,3,4,5,6)
                self.ooovRev [min(ranges2):max(ranges2)+1,min(ranges1):max(ranges1)+1,min(ranges0):max(ranges0)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:].transpose(2,1,0,3,4,5,6)
                self.oovv    [min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:]

                loader.slave_finished()

            cc.comm.Barrier()
            cput1 = log.timer_debug1('transforming oopq', *cput1)

####
            mem = 0.5e9
            pre = 1.*nocc*nvir*nmo*nmo*nkpts*16
            nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)
####
            BLKSIZE = (1,nkpts_blksize,nkpts,)
            if cc.rank == 0:
                print "ERI ovpq blksize = (%3d %3d %3d)" % BLKSIZE
            loader1 = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
            loader1.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

            tmp_block_shape = BLKSIZE + (nocc,nvir,nmo,nmo)
            tmp_block  = numpy.empty(shape=tmp_block_shape,dtype=ds_type)

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
                            self.voov[kr,kp,ks] = eri_kpt.transpose(1,0,3,2)[:,:,:nocc,nocc:]
                            self.voovR1[ks,kr,kp] = eri_kpt.transpose(1,0,3,2)[:,:,:nocc,nocc:]
                            self.vovv[kr,kp,ks] = eri_kpt.transpose(1,0,3,2)[:,:,nocc:,nocc:]
                            self.vovvR1[ks,kr,kp] = eri_kpt.transpose(1,0,3,2)[:,:,nocc:,nocc:]
                            self.vovvL1[kp,ks,kr] = eri_kpt.transpose(1,0,3,2)[:,:,nocc:,nocc:]
                ############################################################################
                self.ovoo[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                                                            tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,:nocc]
                self.ovov[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                                                            tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:]
                self.ovovL1[min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1,min(ranges0):max(ranges0)+1] = \
                                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:].transpose(1,2,0,3,4,5,6)
                self.ovovRev[min(ranges2):max(ranges2)+1,min(ranges1):max(ranges1)+1,min(ranges0):max(ranges0)+1] = \
                                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:].transpose(2,1,0,3,4,5,6)
                self.ovvo[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                                                            tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,:nocc]
                self.ovvv[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                                                            tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:]
                #self.ovvvL1[min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1,min(ranges0):max(ranges0)+1] = \
                #                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:].transpose(1,2,0,3,4,5,6)
                self.ovvvRev[min(ranges2):max(ranges2)+1,min(ranges1):max(ranges1)+1,min(ranges0):max(ranges0)+1] = \
                                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:].transpose(2,1,0,3,4,5,6)
                loader1.slave_finished()

            cc.comm.Barrier()
            cput1 = log.timer_debug1('transforming ovpq', *cput1)

            #######################################################
            # Here we can exploit the full 4-permutational symm.  #
            # for 'vvvv' unlike in the cases above.               #
            #######################################################
####
            mem = 0.5e9
            pre = 1.*nvir*nvir*nvir*nvir*16
            nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nUnique_klist)
####
            BLKSIZE = (nkpts_blksize,)
            if cc.rank == 0:
                print "ERI vvvv blksize = %3d" % nkpts_blksize
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

            cc.comm.Barrier()
            cput1 = log.timer_debug1('transforming vvvv', *cput1)

            self.feri1.close()
            self.feri1 = h5py.File(_tmpfile1_name, 'r', driver='mpio', comm=MPI.COMM_WORLD)
            self.oooo  = self.feri1['oooo']
            self.ooov  = self.feri1['ooov']
            self.ovoo  = self.feri1['ovoo']
            self.oovv  = self.feri1['oovv']
            self.ovov  = self.feri1['ovov']
            self.ovvo  = self.feri1['ovvo']
            self.voov  = self.feri1['voov']
            self.ovvv  = self.feri1['ovvv']
            self.vovv  = self.feri1['vovv']
            self.vvvv  = self.feri1['vvvv']

            self.ovovL1  = self.feri1['ovovL1']
            self.ooovL1  = self.feri1['ooovL1']
            #self.ovvvL1  = self.feri1['ovvvL1']
            self.voovR1  = self.feri1['voovR1']
            self.ooovR1  = self.feri1['ooovR1']
            self.vovvR1  = self.feri1['vovvR1']
            self.vovvL1  = self.feri1['vovvL1']
            self.ovovRev  = self.feri1['ovovRev']
            self.ooovRev  = self.feri1['ooovRev']
            self.ovvvRev  = self.feri1['ovvvRev']

            print "........WARNING : using oovv in memory......."
            new_oovv = numpy.empty( (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff.dtype)
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        new_oovv[kp,kq,kr] = self.oovv[kp,kq,kr].copy()
            self.oovv = new_oovv

        log.timer('CCSD integral transformation', *cput0)

    def __del__(self):
        if hasattr(self, 'feri1'):
            self.feri1.close()

class _IMDS:
    def __init__(self, cc):
        return

    @profile
    def make_ip(self,cc):
        #cc = self.cc
        t1,t2,eris = cc.t1, cc.t2, cc.eris
        nkpts,nocc,nvir = t1.shape

        if not hasattr(self, 'feri1'):
            self.fint1 = None

        print "*** Using HDF5 ERI storage ***"
        tmpfile1_name = "eom_intermediates.hdf5"
        self.fint1 = h5py.File(tmpfile1_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)

        ds_type = t2.dtype

        self.Wooov  = self.fint1.create_dataset('Wooov',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
        self.Woooo  = self.fint1.create_dataset('Woooo',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=ds_type)
        self.WooooS = self.fint1.create_dataset('WooooS',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=ds_type)
        self.W1voov = self.fint1.create_dataset('W1voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        self.W2voov = self.fint1.create_dataset('W2voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        self.Wvoov  = self.fint1.create_dataset('Wvoov',  (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        #self.W1ovvo = self.fint1.create_dataset('W1ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
        #self.W2ovvo = self.fint1.create_dataset('W2ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
        #self.Wovvo  = self.fint1.create_dataset('Wovvo',  (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
        self.W1ovov = self.fint1.create_dataset('W1ovov',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        self.W2ovov = self.fint1.create_dataset('W2ovov',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        self.Wovov  = self.fint1.create_dataset('Wovov',   (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        self.Wovoo  = self.fint1.create_dataset('Wovoo',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=ds_type)

        self.Lvv = imdk.Lvv(cc,t1,t2,eris)
        self.Loo = imdk.Loo(cc,t1,t2,eris)
        self.Fov = imdk.cc_Fov(cc,t1,t2,eris)

        #
        # Order matters here for array creation
        self.Wooov = imdk.Wooov(cc,t1,t2,eris,self.fint1)

        self.W1voov = imdk.W1voov(cc,t1,t2,eris,self.fint1)
        self.W2voov = imdk.W2voov(cc,t1,t2,eris,self.fint1)
        self.Wvoov = imdk.Wvoov(cc,t1,t2,eris,self.fint1)

        #self.W1ovvo = imdk.W1ovvo(cc,t1,t2,eris,self.fint1)
        #self.W2ovvo = imdk.W2ovvo(cc,t1,t2,eris,self.fint1)
        #self.Wovvo = imdk.Wovvo(cc,t1,t2,eris,self.fint1)

        self.Woooo = imdk.Woooo(cc,t1,t2,eris,self.fint1)
        self.WooooS = imdk.WooooS(cc,t1,t2,eris,self.fint1)

        self.W1ovov = imdk.W1ovov(cc,t1,t2,eris,self.fint1)
        self.W2ovov = imdk.W2ovov(cc,t1,t2,eris,self.fint1)
        self.Wovov  = imdk.Wovov(cc,t1,t2,eris,self.fint1)

        self.Woovv = eris.oovv

        self.Wovoo = imdk.Wovoo(cc,t1,t2,eris,self.fint1)

        self.fint1.close()
        self.fint1 = h5py.File(tmpfile1_name, 'r', driver='mpio', comm=MPI.COMM_WORLD)

        self.Wooov  = self.fint1['Wooov' ]
        self.Woooo  = self.fint1['Woooo' ]
        self.WooooS = self.fint1['WooooS' ]
        #self.W1ovvo = self.fint1['W1ovvo']
        #self.W2ovvo = self.fint1['W2ovvo']
        #self.Wovvo  = self.fint1['Wovvo' ]
        self.W1voov = self.fint1['W1voov']
        self.W2voov = self.fint1['W2voov']
        self.Wvoov  = self.fint1['Wvoov' ]
        self.W1ovov = self.fint1['W1ovov']
        self.W2ovov = self.fint1['W2ovov']
        self.Wovov  = self.fint1['Wovov' ]
        self.Wovoo  = self.fint1['Wovoo' ]

    @profile
    def make_ea(self,cc):
        t1,t2,eris = cc.t1, cc.t2, cc.eris
        nkpts,nocc,nvir = t1.shape

        if not hasattr(self, 'feri1'):
            self.fint1 = None

        print "*** Using HDF5 ERI storage ***"
        tmpfile1_name = "eom_intermediates.hdf5"
        self.fint1 = h5py.File(tmpfile1_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)

        ds_type = t2.dtype

        self.Wooov  = self.fint1.create_dataset('Wooov',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
        self.Wvovv  = self.fint1.create_dataset('Wvovv',  (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=ds_type)
        #self.W1ovvo = self.fint1.create_dataset('W1ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
        #self.W2ovvo = self.fint1.create_dataset('W2ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
        #self.Wovvo  = self.fint1.create_dataset('Wovvo',  (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
        self.W1voov = self.fint1.create_dataset('W1voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        self.W2voov = self.fint1.create_dataset('W2voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        #self.Wvoov  = self.fint1.create_dataset('Wvoov',  (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        self.WvoovR1 = self.fint1.create_dataset('WvoovR1',  (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        self.Wvvvv  = self.fint1.create_dataset('Wvvvv',  (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=ds_type)
        self.W1ovov = self.fint1.create_dataset('W1ovov',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        self.W2ovov = self.fint1.create_dataset('W2ovov',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        #self.Wovov  = self.fint1.create_dataset('Wovov',   (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        self.WovovRev  = self.fint1.create_dataset('WovovRev',   (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        #self.Wvvvo  = self.fint1.create_dataset('Wvvvo',  (nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), dtype=ds_type)
        self.WvvvoR1  = self.fint1.create_dataset('WvvvoR1',  (nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), dtype=ds_type)

        self.Lvv = imdk.Lvv(cc,t1,t2,eris)
        self.Loo = imdk.Loo(cc,t1,t2,eris)
        self.Fov = imdk.cc_Fov(cc,t1,t2,eris)

        #
        # Order matters here for array creation
        self.Wooov = imdk.Wooov(cc,t1,t2,eris,self.fint1)

        self.Wvovv = imdk.Wvovv(cc,t1,t2,eris,self.fint1)

        self.W1voov = imdk.W1voov(cc,t1,t2,eris,self.fint1)
        self.W2voov = imdk.W2voov(cc,t1,t2,eris,self.fint1)
        #self.Wvoov = imdk.Wvoov(cc,t1,t2,eris,self.fint1)
        self.WvoovR1 = imdk.WvoovR1(cc,t1,t2,eris,self.fint1)

        #self.W1ovvo = imdk.W1ovvo(cc,t1,t2,eris,self.fint1)
        #self.W2ovvo = imdk.W2ovvo(cc,t1,t2,eris,self.fint1)
        #self.Wovvo = imdk.Wovvo(cc,t1,t2,eris,self.fint1)

        self.Wvvvv = imdk.Wvvvv(cc,t1,t2,eris,self.fint1)

        self.Woovv = eris.oovv

        self.W1ovov = imdk.W1ovov(cc,t1,t2,eris,self.fint1)
        self.W2ovov = imdk.W2ovov(cc,t1,t2,eris,self.fint1)
        #self.Wovov  = imdk.Wovov(cc,t1,t2,eris,self.fint1)
        self.WovovRev  = imdk.WovovRev(cc,t1,t2,eris,self.fint1)

        #self.Wvvvo = imdk.Wvvvo(cc,t1,t2,eris,self.fint1)
        self.WvvvoR1 = imdk.WvvvoR1(cc,t1,t2,eris,self.fint1)

        self.fint1.close()
        self.fint1 = h5py.File(tmpfile1_name, 'r', driver='mpio', comm=MPI.COMM_WORLD)

        self.Wooov  = self.fint1['Wooov' ]
        self.Wvovv  = self.fint1['Wvovv' ]
        #self.W1ovvo = self.fint1['W1ovvo']
        #self.W2ovvo = self.fint1['W2ovvo']
        #self.Wovvo  = self.fint1['Wovvo' ]
        self.W1voov = self.fint1['W1voov']
        self.W2voov = self.fint1['W2voov']
        #self.Wvoov  = self.fint1['Wvoov' ]
        self.WvoovR1  = self.fint1['WvoovR1' ]
        self.Wvvvv  = self.fint1['Wvvvv' ]
        self.W1ovov = self.fint1['W1ovov']
        self.W2ovov = self.fint1['W2ovov']
        #self.Wovov  = self.fint1['Wovov' ]
        self.WovovRev  = self.fint1['WovovRev' ]
        #self.Wvvvo  = self.fint1['Wvvvo' ]
        self.WvvvoR1  = self.fint1['WvvvoR1' ]

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
