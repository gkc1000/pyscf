import numpy as np
import time
import pyscf.pbc.tools as tools
from pyscf.lib import logger
import mpi_load_balancer
from pyscf import lib
from pyscf.pbc import lib as pbclib

#einsum = np.einsum
einsum = pbclib.einsum
dot = np.dot

#################################################
# FOLLOWING:                                    #
# S. Hirata, ..., R. J. Bartlett                #
# J. Chem. Phys. 120, 2581 (2004)               #
#################################################

### Eqs. (37)-(39) "kappa"

@profile
def cc_tau1(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    BLKSIZE = (1,1,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    tmp_oovv_shape = BLKSIZE + (nocc,nocc,nvir,nvir)
    tmp_oovv  = np.empty(tmp_oovv_shape,dtype=t1.dtype)
    tmp2_oovv = np.zeros_like(tmp_oovv)

    tau1_ooVv = feri2['tau1_ooVv']
    tau1_oOvv = feri2['tau1_oOvv']
    tau1_oovv_rev = feri2['tau1_oovv_rev']
    tau2_Oovv = feri2['tau2_Oovv']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        for iterki, ki in enumerate(ranges0):
            for iterkj, kj in enumerate(ranges1):
                for iterka, ka in enumerate(ranges2):
                    kb = kconserv[ki,ka,kj]
                    tmp_oovv[iterki,iterkj,iterka] = t2[ki,kj,ka].copy()
                    tmp2_oovv[iterki,iterkj,iterka] *= 0.0
                    if ki == ka and kj == kb:
                        tmp2_oovv[iterki,iterkj,iterka] = einsum('ia,jb->ijab',t1[ki],t1[kj])

                    tau1_oovv_rev[kj,ka,kb] = (tmp_oovv[iterki,iterkj,iterka] + tmp2_oovv[iterki,iterkj,iterka])

        tau1_ooVv[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,nvir*min(ranges2):nvir*(max(ranges2)+1)] = \
                ( tmp_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] +
                        tmp2_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] ).transpose(0,1,2,5,3,4,6).reshape(len(ranges0),len(ranges1),len(ranges2)*nvir,nocc,nocc,nvir)
        tau1_oOvv[min(ranges0):max(ranges0)+1,min(ranges2):max(ranges2)+1,nocc*min(ranges1):nocc*(max(ranges1)+1)] = \
                ( tmp_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] +
                        tmp2_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] ).transpose(0,2,1,4,3,5,6).reshape(len(ranges0),len(ranges2),len(ranges1)*nocc,nocc,nvir,nvir)
        tau2_Oovv[min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1,nocc*min(ranges0):nocc*(max(ranges0)+1)] = \
                ( tmp_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] +
                        2*tmp2_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] ).transpose(1,2,0,3,4,5,6).reshape(len(ranges1),len(ranges2),len(ranges0)*nocc,nocc,nvir,nvir)
        loader.slave_finished()
    cc.comm.Barrier()
    return

@profile
def cc_Foo(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    Fki = np.empty((nkpts,nocc,nocc),dtype=t2.dtype)

    tau1_oOvv = feri2['tau1_oOvv']
    for ki in range(nkpts):
        kk = ki
        Fki[ki] = eris.fock[ki,:nocc,:nocc].copy()
        for kc in range(nkpts):
            Fki[ki] += einsum('lkcd,licd->ki',eris.SoOvv[kk,kc],tau1_oOvv[ki,kc])
            #for kl in range(nkpts):
            #    kd = kconserv[kk,kc,kl]
            #    Soovv = 2*eris.oovv[kk,kl,kc] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
            #    Fki[ki] += einsum('klcd,ilcd->ki',Soovv,t2[ki,kl,kc])
            ##if ki == kc:
            #kd = kconserv[kk,ki,kl]
            #Soovv = 2*eris.oovv[kk,kl,ki] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
            #Fki[ki] += einsum('klcd,ic,ld->ki',Soovv,t1[ki],t1[kl])
    return Fki

@profile
def cc_Fvv(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    Fac = np.empty((nkpts,nvir,nvir),dtype=t2.dtype)

    tau1_oOvv = feri2['tau1_oOvv']
    for ka in range(nkpts):
        kc = ka
        Fac[ka] = eris.fock[ka,nocc:,nocc:].copy()
        for kk in range(nkpts):
            Fac[ka] += -einsum('lkcd,lkad->ac',eris.SoOvv[kk,kc],tau1_oOvv[kk,ka])

    return Fac

@profile
def cc_Fov(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    Fkc = np.empty((nkpts,nocc,nvir),dtype=t2.dtype)
    Fkc[:] = eris.fock[:,:nocc,nocc:].copy()
    for kk in range(nkpts):
        Fkc[kk] += einsum('lkcd,ld->kc',eris.SoOvv[kk,kk],t1.reshape(nkpts*nocc,nvir))
    return Fkc

### Eqs. (40)-(41) "lambda"

@profile
def Loo(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lki = cc_Foo(cc,t1,t2,eris,feri2)
    for ki in range(nkpts):
        Lki[ki] += einsum('kc,ic->ki',fov[ki],t1[ki])
        SoOov = (2*eris.ooov[ki,:,ki] - eris.ooov[:,ki,ki].transpose(0,2,1,3,4)).transpose(0,2,1,3,4).reshape(nkpts*nocc,nocc,nocc,nvir)
        Lki[ki] += einsum('lkic,lc->ki',SoOov,t1.reshape(nkpts*nocc,nvir))
    return Lki

@profile
def Lvv(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lac = cc_Fvv(cc,t1,t2,eris,feri2)
    for ka in range(nkpts):
        Lac[ka] += -einsum('kc,ka->ac',fov[ka],t1[ka])
        for kk in range(nkpts):
            Svovv = 2*eris.ovvv[kk,ka,kk].transpose(1,0,3,2) - eris.ovvv[kk,ka,ka].transpose(1,0,2,3)
            Lac[ka] += einsum('akcd,kd->ac',Svovv,t1[kk])
    return Lac

### Eqs. (42)-(45) "chi"

@profile
def cc_Woooo(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    khelper = cc.khelper

    #Wklij = np.array(eris.oooo, copy=True)
    #for pqr in range(nUnique_klist):
    #    kk, kl, ki = unique_klist[pqr]
    BLKSIZE = (1,1,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    oooo_tmp_shape = BLKSIZE + (nocc,nocc,nocc,nocc)
    oooo_tmp = np.empty(shape=oooo_tmp_shape,dtype=t1.dtype)

    tau1_ooVv = feri2['tau1_ooVv']
    #Woooo     = feri2['Woooo']
    Woooo_rev = feri2['Woooo_rev']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        for iterkk, kk in enumerate(ranges0):
            for iterkl, kl in enumerate(ranges1):
                for iterki, ki in enumerate(ranges2):
                    kj = kconserv[kk,ki,kl]
                    oooo_tmp[iterkk,iterkl,iterki] = np.array(eris.oooo[kk,kl,ki],copy=True)
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('klic,jc->klij',eris.ooov[kk,kl,ki],t1[kj])
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('klcj,ic->klij',eris.ooov[kl,kk,kj].transpose(1,0,3,2),t1[ki])

                    ###################################################################################
                    # Note the indices and way the tau1 is stored : instead of a loop over kpt='kc' and
                    # loop over mo='c', the (kc,k,l,c,d) index is changed instead to (nkpts*nvir,k,l,d)
                    # so that we only have to loop over the first index, saving read operations.
                    ###################################################################################
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('ckld,cijd->klij',eris.ooVv[kk,kl],tau1_ooVv[ki,kj])

                    #for kc in range(nkpts):
                    #    oooo_tmp[iterkk,iterkl,iterki] += einsum('klcd,ijcd->klij',eris.oovv[kk,kl,kc],t2[ki,kj,kc])
                    #oooo_tmp[iterkk,iterkl,iterki] += einsum('klcd,ic,jd->klij',eris.oovv[kk,kl,ki],t1[ki],t1[kj])
                    #Woooo[kk,kl,ki] = oooo_tmp[iterkk,iterkl,iterki]
                    #Woooo[kl,kk,kj] = oooo_tmp[iterkk,iterkl,iterki].transpose(1,0,3,2)
                    Woooo_rev[kl,ki,kj] = oooo_tmp[iterkk,iterkl,iterki]

        #Woooo[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
        #                oooo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]

        #
        # for if you want to take into account symmetry of Woooo integral
        #
        #feri2.Woooo[min(ranges1):max(ranges1)+1,min(ranges0):max(ranges0)+1,min(ranges2):max(ranges2)+1] = \
        #                oooo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)].transpose(0,1,2,4,3,6,5)
        loader.slave_finished()
    cc.comm.Barrier()
    return

@profile
def cc_Wvvvv(cc,t1,t2,eris,feri2=None):
    ## Slow:
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    vvvv_tmp = np.empty((nvir,nvir,nvir,nvir),dtype=t1.dtype)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(1,nkpts,))
    loader.set_ranges((range(nkpts),range(nkpts),))

    Wvvvv = feri2['Wvvvv']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1 = loader.get_blocks_from_data(data)
        for ka in ranges0:
            for kc in ranges1:
                for kb in range(ka+1):
                    kd = kconserv[ka,kc,kb]
                    vvvv_tmp = np.array(eris.vvvv[ka,kb,kc],copy=True)
                    vvvv_tmp += einsum('akcd,kb->abcd',eris.ovvv[kb,ka,kd].transpose(1,0,3,2),-t1[kb])
                    vvvv_tmp += einsum('kbcd,ka->abcd',eris.ovvv[ka,kb,kc],-t1[ka])
                    Wvvvv[ka,kb,kc] = vvvv_tmp
                    Wvvvv[kb,ka,kd] = vvvv_tmp.transpose(1,0,3,2)
        loader.slave_finished()

    ## Fast
    #nocc,nvir = t1.shape
    #Wabcd = np.empty((nvir,)*4)
    #for a in range(nvir):
    #    Wabcd[a,:] = einsum('kcd,kb->bcd',eris.vovv[a],-t1)
    ##Wabcd += einsum('kbcd,ka->abcd',eris.ovvv,-t1)
    #Wabcd += lib.dot(-t1.T,eris.ovvv.reshape(nocc,-1)).reshape((nvir,)*4)
    #Wabcd += np.asarray(eris.vvvv)

    cc.comm.Barrier()
    return

@profile
def cc_Wvoov(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    #Wakic = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir),dtype=t1.dtype)

    BLKSIZE = (1,1,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    voov_tmp_shape = BLKSIZE + (nvir,nocc,nocc,nvir)
    voov_tmp = np.empty(voov_tmp_shape,dtype=t1.dtype)

    tau2_Oovv = feri2['tau2_Oovv']
    #Wvoov     = feri2['Wvoov']
    WvOov     = feri2['WvOov']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        ix = sum([[min(x),max(x)+1] for x in ranges0,ranges1,ranges2], [])

        #eris_ooov = eris.ooov[ix[0]:ix[1], ix[2]:ix[3], ix[4]:ix[5]]
        for iterka, ka in enumerate(ranges0):
            for iterkk, kk in enumerate(ranges1):
                for iterki, ki in enumerate(ranges2):
                    kc = kconserv[ka,ki,kk]
                    voov_tmp[iterka,iterkk,iterki] = np.array(eris.ovvo[kk,ka,kc]).transpose(1,0,3,2)
                    voov_tmp[iterka,iterkk,iterki] -= einsum('lkic,la->akic',eris.ooov[ka,kk,ki],t1[ka])
                    voov_tmp[iterka,iterkk,iterki] += einsum('akdc,id->akic',eris.ovvv[kk,ka,kc].transpose(1,0,3,2),t1[ki])
                    # ==== Beginning of change ====
                    #
                    #for kl in range(nkpts):
                    #    # kl - kd + kk = kc
                    #    # => kd = kl - kc + kk
                    #    kd = kconserv[kl,kc,kk]
                    #    Soovv = 2*np.array(eris.oovv[kl,kk,kd]) - np.array(eris.oovv[kl,kk,kc]).transpose(0,1,3,2)
                    #    voov_tmp[iterka,iterkk,iterki] += 0.5*einsum('lkdc,ilad->akic',Soovv,t2[ki,kl,ka])
                    #    voov_tmp[iterka,iterkk,iterki] -= 0.5*einsum('lkdc,ilda->akic',eris.oovv[kl,kk,kd],t2[ki,kl,kd])
                    #voov_tmp[iterka,iterkk,iterki] -= einsum('lkdc,id,la->akic',eris.oovv[ka,kk,ki],t1[ki],t1[ka])
                    #Wvoov[ka,kk,ki] = voov_tmp[iterka,iterkk,iterki]

                    #
                    # Making various intermediates...
                    #
                    t2_oOvv = t2[ki,:,ka].transpose(0,2,1,3,4).reshape(nkpts*nocc,nocc,nvir,nvir)
                    #eris_oOvv = eris.oovv[kk,:,kc].transpose(0,2,1,3,4).reshape(nkpts*nocc,nocc,nvir,nvir)

                    voov_tmp[iterka,iterkk,iterki] += 0.5*einsum('lkcd,liad->akic',eris.SoOvv[kk,kc],t2_oOvv)
                    voov_tmp[iterka,iterkk,iterki] -= 0.5*einsum('lkcd,liad->akic',eris.oOvv[kk,kc],tau2_Oovv[ki,ka])

                    # =====   End of change  =====
        #Wvoov[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
        #        voov_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        WvOov[min(ranges0):max(ranges0)+1,min(ranges2):max(ranges2)+1,nocc*min(ranges1):nocc*(max(ranges1)+1)] = \
                voov_tmp[:len(ranges0),:len(ranges1),:len(ranges2)].transpose(0,2,1,4,3,5,6).reshape(len(ranges1),len(ranges2),len(ranges0)*nocc,nvir,nocc,nvir)
        loader.slave_finished()
    cc.comm.Barrier()
    return

@profile
def cc_Wvovo(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    BLKSIZE = (1,1,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    vovo_tmp_shape = BLKSIZE + (nvir,nocc,nvir,nocc)
    vovo_tmp = np.empty(shape=vovo_tmp_shape,dtype=t1.dtype)

    Wvovo = feri2['Wvovo']
    WvOVo = feri2['WvOVo']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        for iterka, ka in enumerate(ranges0):
            for iterkk, kk in enumerate(ranges1):
                for iterkc, kc in enumerate(ranges2):
                    ki = kconserv[ka,kc,kk]
                    vovo_tmp[iterka,iterkk,iterkc] = np.array(eris.ovov[kk,ka,ki]).transpose(1,0,3,2)
                    vovo_tmp[iterka,iterkk,iterkc] -= einsum('lkci,la->akci',eris.ooov[kk,ka,ki].transpose(1,0,3,2),t1[ka])
                    vovo_tmp[iterka,iterkk,iterkc] += einsum('akcd,id->akci',eris.ovvv[kk,ka,ki].transpose(1,0,3,2),t1[ki])
                    # ==== Beginning of change ====
                    #
                    #for kl in range(nkpts):
                    #    kd = kconserv[kl,kc,kk]
                    #    vovo_tmp[iterka,iterkk,iterkc] -= 0.5*einsum('lkcd,ilda->akci',eris.oovv[kl,kk,kc],t2[ki,kl,kd])
                    #vovo_tmp[iterka,iterkk,iterkc] -= einsum('lkcd,id,la->akci',eris.oovv[ka,kk,kc],t1[ki],t1[ka])
                    #Wvovo[ka,kk,kc] = vovo_tmp[iterka,iterkk,iterkc]

                    oovvf = eris.oovv[:,kk,kc].reshape(nkpts*nocc,nocc,nvir,nvir)
                    t2f   = t2[:,ki,ka].copy() #This is a tau like term
                    #for kl in range(nkpts):
                    #    kd = kconserv[kl,kc,kk]
                    #    if ki == kd and kl == ka:
                    #        t2f[kl] += 2*einsum('id,la->liad',t1[ki],t1[ka])
                    kd = kconserv[ka,kc,kk]
                    t2f[ka] += 2*einsum('id,la->liad',t1[kd],t1[ka])
                    t2f = t2f.reshape(nkpts*nocc,nocc,nvir,nvir)
                    vovo_tmp[iterka,iterkk,iterkc] -= 0.5*einsum('lkcd,liad->akci',oovvf,t2f)

        Wvovo[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                vovo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        WvOVo[min(ranges0):max(ranges0)+1,nocc*min(ranges1):nocc*(max(ranges1)+1),nvir*min(ranges2):nvir*(max(ranges2)+1)] = \
                vovo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)].transpose(0,1,4,2,5,3,6).reshape(len(ranges0),len(ranges1)*nocc,len(ranges2)*nvir,nvir,nocc)
                    # =====   End of change  = ====
        loader.slave_finished()
    cc.comm.Barrier()
    return

@profile
def cc_Wovov(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    BLKSIZE = (1,1,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    ovov_tmp_shape = BLKSIZE + (nocc,nvir,nocc,nvir)
    ovov_tmp = np.empty(shape=ovov_tmp_shape,dtype=t1.dtype)

    #Wovov = feri2['Wovov']
    WOvov = feri2['WOvov']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        for iterkk, kk in enumerate(ranges0):
            for iterka, ka in enumerate(ranges1):
                for iterki, ki in enumerate(ranges2):
                    kc = kconserv[kk,ki,ka]
                    ovov_tmp[iterkk,iterka,iterki] = np.array(eris.ovov[kk,ka,ki],copy=True)
                    ovov_tmp[iterkk,iterka,iterki] -= einsum('lkci,la->kaic',eris.ooov[kk,ka,ki].transpose(1,0,3,2),t1[ka])
                    ovov_tmp[iterkk,iterka,iterki] += einsum('akcd,id->kaic',eris.ovvv[kk,ka,ki].transpose(1,0,3,2),t1[ki])
                    # ==== Beginning of change ====
                    #
                    #for kl in range(nkpts):
                    #    kd = kconserv[kl,kc,kk]
                    #    ovov_tmp[iterka,iterkk,iterkc] -= 0.5*einsum('lkcd,ilda->akci',eris.oovv[kl,kk,kc],t2[ki,kl,kd])
                    #ovov_tmp[iterka,iterkk,iterkc] -= einsum('lkcd,id,la->akci',eris.oovv[ka,kk,kc],t1[ki],t1[ka])
                    #Wvovo[ka,kk,kc] = ovov_tmp[iterka,iterkk,iterkc]

                    oovvf = eris.oovv[:,kk,kc].reshape(nkpts*nocc,nocc,nvir,nvir)
                    t2f   = t2[:,ki,ka].copy() #This is a tau like term
                    #for kl in range(nkpts):
                    #    kd = kconserv[kl,kc,kk]
                    #    if ki == kd and kl == ka:
                    #        t2f[kl] += 2*einsum('id,la->liad',t1[ki],t1[ka])
                    kd = kconserv[ka,kc,kk]
                    t2f[ka] += 2*einsum('id,la->liad',t1[kd],t1[ka])
                    t2f = t2f.reshape(nkpts*nocc,nocc,nvir,nvir)
                    ovov_tmp[iterkk,iterka,iterki] -= 0.5*einsum('lkcd,liad->kaic',oovvf,t2f)

        #Wovov[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
        #        ovov_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        WOvov[min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1,nocc*min(ranges0):nocc*(max(ranges0)+1)] = \
                ovov_tmp[:len(ranges0),:len(ranges1),:len(ranges2)].transpose(1,2,0,3,4,5,6).reshape(len(ranges1),len(ranges2),len(ranges0)*nocc,nvir,nocc,nvir)
                    # =====   End of change  = ====
        loader.slave_finished()
    cc.comm.Barrier()
    return

########################################################
#        EOM Intermediates w/ k-points                 #
########################################################

# Indices in the following can be safely permuted.

def Wooov(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    Wklid = np.array(eris.ooov, copy=True)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kl]
                Wklid[kk,kl,ki] += einsum('ic,klcd->klid',t1[ki],eris.oovv[kk,kl,ki])
    return Wklid

def Wvovv(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    Walcd = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir),dtype=t1.dtype)
    for ka in range(nkpts):
        for kl in range(nkpts):
            for kc in range(nkpts):
                kd = kconserv[ka,kc,kl]
                # vovv[ka,kl,kc,kd] <= ovvv[kl,ka,kd,kc].transpose(1,0,3,2)
                Walcd[ka,kl,kc] = np.array(eris.ovvv[kl,ka,kd]).transpose(1,0,3,2)
                Walcd[ka,kl,kc] += -einsum('ka,klcd->alcd',t1[ka],eris.oovv[ka,kl,kc])
    return Walcd

def W1ovvo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    Wkaci = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc),dtype=t1.dtype)
    for kk in range(nkpts):
        for ka in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kk,kc,ka]
                # ovvo[kk,ka,kc,ki] => voov[ka,kk,ki,kc]
                Wkaci[kk,ka,kc] = np.array(eris.voov[ka,kk,ki]).transpose(1,0,3,2)
                for kl in range(nkpts):
                    kd = kconserv[ki,ka,kl]
                    St2 = 2.*t2[ki,kl,ka] - t2[kl,ki,ka].transpose(1,0,2,3)
                    Wkaci[kk,ka,kc] +=  einsum('klcd,ilad->kaci',eris.oovv[kk,kl,kc],St2)
                    Wkaci[kk,ka,kc] += -einsum('kldc,ilad->kaci',eris.oovv[kk,kl,kd],t2[ki,kl,ka])
    return Wkaci

def W2ovvo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    Wkaci = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc),dtype=t1.dtype)
    WWooov = Wooov(cc,t1,t2,eris)
    for kk in range(nkpts):
        for ka in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kk,kc,ka]
                Wkaci[kk,ka,kc] =  einsum('la,lkic->kaci',-t1[ka],WWooov[ka,kk,ki])
                Wkaci[kk,ka,kc] += einsum('akdc,id->kaci',eris.vovv[ka,kk,ki],t1[ki])
    return Wkaci

def Wovvo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    return W1ovvo(cc,t1,t2,eris) + W2ovvo(cc,t1,t2,eris)

def W1ovov(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    Wkbid = np.array(eris.ovov, copy=True)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kb]
                #   kk + kl - kc - kd = 0
                # => kc = kk - kd + kl
                for kl in range(nkpts):
                    kc = kconserv[kk,kd,kl]
                    Wkbid[kk,kb,ki] += -einsum('klcd,ilcb->kbid',eris.oovv[kk,kl,kc],t2[ki,kl,kc])
    return Wkbid

def W2ovov(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    Wkbid = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir),dtype=t1.dtype)
    WWooov = Wooov(cc,t1,t2,eris)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kb]
                Wkbid[kk,kb,ki] = einsum('klid,lb->kbid',WWooov[kk,kb,ki],-t1[kb])
                Wkbid[kk,kb,ki] += einsum('bkdc,ic->kbid',eris.vovv[kb,kk,kd],t1[ki])
    return Wkbid

def Wovov(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    return W1ovov(cc,t1,t2,eris) + W2ovov(cc,t1,t2,eris)

def Woooo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    Wklij = np.array(eris.oooo, copy=True)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kl]
                for kc in range(nkpts):
                    kd = kconserv[kk,kc,kl]
                    Wklij[kk,kl,ki] += einsum('klcd,ijcd->klij',eris.oovv[kk,kl,kc],t2[ki,kj,kc])
                Wklij[kk,kl,ki] += einsum('klcd,ic,jd->klij',eris.oovv[kk,kl,ki],t1[ki],t1[kd])
                Wklij[kk,kl,ki] += einsum('klid,jd->klij',eris.ooov[kk,kl,ki],t1[kj])
                Wklij[kk,kl,ki] += einsum('lkjc,ic->klij',eris.ooov[kl,kk,kj],t1[ki])
    return Wklij

def Wvvvv(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    Wabcd = np.array(eris.vvvv, copy=True)
    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                kd = kconserv[ka,kc,kb]
                for kk in range(nkpts):
                    # kk + kl - kc - kd = 0
                    # => kl = kc - kk + kd
                    kl = kconserv[kc,kk,kd]
                    Wabcd[ka,kb,kc] += einsum('klcd,klab->abcd',eris.oovv[kk,kl,kc],t2[kk,kl,ka])
                Wabcd[ka,kb,kc] += einsum('klcd,ka,lb->abcd',eris.oovv[ka,kb,kc],t1[ka],t1[kb])
                Wabcd[ka,kb,kc] += einsum('alcd,lb->abcd',eris.vovv[ka,kb,kc],-t1[kb])
                Wabcd[ka,kb,kc] += einsum('bkdc,ka->abcd',eris.vovv[kb,ka,kd],-t1[ka])
    return Wabcd

def Wvvvo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    Wabcj = np.empty((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc),dtype=t1.dtype)
    WWvvvv = Wvvvv(cc,t1,t2,eris)
    WW1ovov = W1ovov(cc,t1,t2,eris)
    WW1ovvo = W1ovvo(cc,t1,t2,eris)
    FFov = cc_Fov(cc,t1,t2,eris)
    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                kj = kconserv[ka,kc,kb]
                # vvvo[ka,kb,kc,kj] <= vovv[kc,kj,ka,kb].transpose(2,3,0,1).conj()
                Wabcj[ka,kb,kc] = np.array(eris.vovv[kc,kj,ka]).transpose(2,3,0,1).conj()
                Wabcj[ka,kb,kc] += einsum('abcd,jd->abcj',WWvvvv[ka,kb,kc],t1[kj])
                # Wvovo[ka,kl,kc,kj] <= Wovov[kl,ka,kj,kc].transpose(1,0,3,2)
                Wabcj[ka,kb,kc] += einsum('alcj,lb->abcj',WW1ovov[kb,ka,kj].transpose(1,0,3,2),-t1[kb])
                Wabcj[ka,kb,kc] += einsum('kbcj,ka->abcj',WW1ovvo[ka,kb,kc],-t1[ka])

                for kl in range(nkpts):
                    # ka + kl - kc - kd = 0
                    # => kd = ka - kc + kl
                    kd = kconserv[ka,kc,kl]
                    St2 = 2.*t2[kl,kj,kd] - t2[kl,kj,kb].transpose(0,1,3,2)
                    Wabcj[ka,kb,kc] += einsum('alcd,ljdb->abcj',eris.vovv[ka,kl,kc], St2)
                    Wabcj[ka,kb,kc] += einsum('aldc,ljdb->abcj',eris.vovv[ka,kl,kd], -t2[kl,kj,kd])
                    # kb - kc + kl = kd
                    kd = kconserv[kb,kc,kl]
                    Wabcj[ka,kb,kc] += einsum('bldc,jlda->abcj',eris.vovv[kb,kl,kd], -t2[kj,kl,kd])

                    # kl + kk - kb - ka = 0
                    # => kk = kb + ka - kl
                    kk = kconserv[kb,kl,ka]
                    Wabcj[ka,kb,kc] += einsum('lkjc,lkba->abcj',eris.ooov[kl,kk,kj],t2[kl,kk,kb])
                Wabcj[ka,kb,kc] += einsum('lkjc,lb,ka->abcj',eris.ooov[kb,ka,kj],t1[kb],t1[ka])
                Wabcj[ka,kb,kc] += einsum('lc,ljab->abcj',-FFov[kc],t2[kc,kj,ka])
    return Wabcj

def Wovoo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    WW1ovov = W1ovov(cc,t1,t2,eris)
    WWoooo = Woooo(cc,t1,t2,eris)
    WW1ovvo = W1ovvo(cc,t1,t2,eris)
    FFov = cc_Fov(cc,t1,t2,eris)

    Wkbij = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc),dtype=t1.dtype)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kb]
                # ovoo[kk,kb,ki,kj] <= oovo[kj,ki,kb,kk].transpose(3,2,1,0).conj()
                Wkbij[kk,kb,ki] = np.array(eris.oovo[kj,ki,kb]).transpose(3,2,1,0).conj()
                Wkbij[kk,kb,ki] += einsum('kbid,jd->kbij',WW1ovov[kk,kb,ki], t1[kj])
                Wkbij[kk,kb,ki] += einsum('klij,lb->kbij',WWoooo[kk,kb,ki],-t1[kb])
                Wkbij[kk,kb,ki] += einsum('kbcj,ic->kbij',WW1ovvo[kk,kb,ki],t1[ki])

                for kd in range(nkpts):
                    # kk + kl - ki - kd = 0
                    # => kl = ki - kk + kd
                    kl = kconserv[ki,kk,kd]
                    St2 = 2.*t2[kl,kj,kd] - t2[kj,kl,kd].transpose(1,0,2,3)
                    Wkbij[kk,kb,ki] += einsum('klid,ljdb->kbij',  eris.ooov[kk,kl,ki], St2)
                    Wkbij[kk,kb,ki] += einsum('lkid,ljdb->kbij', -eris.ooov[kl,kk,ki],t2[kl,kj,kd])
                    kl = kconserv[kb,ki,kd]
                    Wkbij[kk,kb,ki] += einsum('lkjd,libd->kbij', -eris.ooov[kl,kk,kj],t2[kl,ki,kb])

                    # kb + kk - kd = kc
                    kc = kconserv[kb,kd,kk]
                    Wkbij[kk,kb,ki] += einsum('bkdc,jidc->kbij',eris.vovv[kb,kk,kd],t2[kj,ki,kd])
                Wkbij[kk,kb,ki] += einsum('bkdc,jd,ic->kbij',eris.vovv[kb,kk,kj],t1[kj],t1[ki])
                Wkbij[kk,kb,ki] += einsum('kc,ijcb->kbij',FFov[kk],t2[ki,kj,kk])
    return Wkbij
