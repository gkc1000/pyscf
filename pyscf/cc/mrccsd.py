
#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Timothy Berkelbach <tim.berkelbach@@gmail.com>
#

'''
MRCCSD
'''

import time, math, scipy
import numpy as np

from pyscf import lib, ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd, uccsd
from pyscf import __config__
from pyscf.fci import cistring, direct_uhf

np.set_printoptions(precision=8,suppress=True)
einsum = lib.einsum
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

make_tau = uccsd.make_tau
make_tau_aa = uccsd.make_tau_aa
make_tau_ab = uccsd.make_tau_ab

_gen_occslst = cistring._gen_occslst
num_strings = cistring.num_strings
addr2str = cistring.addr2str
str2addr = cistring.str2addr
cre_des_sign = cistring.cre_des_sign

def get_frozen(mf, frozen):
    nea, neb = mf.mol.nelec
    if type(frozen) is int:
        nfc = frozen, frozen
        nfv = 0,0
    if type(frozen) is list:
        nfc = len(frozen[frozen<nea]),len(frozen[frozen<neb])
        nfv = len(frozen[frozen>nea]),len(frozen[frozen>neb])
    if type(frozen) is tuple:
        nfc = len(frozen[0][frozen[0]<nea]),len(frozen[1][frozen[1]<neb])
        nfv = len(frozen[0][frozen[0]>nea]),len(frozen[1][frozen[1]>neb])
    return nfc, nfv

def get_occ(mf, ncas, nelecas, occslst, ci, ndet, cutoff):
    if occslst is None:
        if ci is None: return cas2occ(mf, ncas, nelecas)
        else: return ci2occ(mf, ci, ndet, cutoff)
    else: return occslst
        
def ci2occ(mf, ci, ndet=None, cutoff=None, ne=None, norb=None):
    nea, neb = mf.mol.nelec if ne is None else ne
    norba = mf.mo_coeff[0].shape[1] if norb is None else norb[0]
    norbb = mf.mo_coeff[1].shape[1] if norb is None else norb[1]
    nca, ncb = mf.mol.nelec[0]-nea, mf.mol.nelec[1]-neb
    occslsta = _gen_occslst(range(nca,nca+norba),nea)
    occslstb = _gen_occslst(range(ncb,ncb+norbb),neb)
    Na, Nb = occslsta.shape[0], occslstb.shape[0]
    assert ci.shape == (Na,Nb)
    ci = np.ravel(ci)
    if ndet is None:
        ndet = ci[abs(ci)>cutoff].size
    idx = np.argsort(abs(ci))[-ndet:]
    idx = np.flip(idx,axis=0)
    occ = np.zeros((ndet,nea+neb),dtype=int)
    for I in range(ndet):
        Ia, Ib = np.divmod(idx[I],Na)
        occ[I,:nea] = occslsta[Ia,:]
        occ[I,nea:] = occslstb[Ib,:]
    print('important determinants:\n{}'.format(occ))
    print('coeffs for important determinants:\n{}'.format(ci[idx]))
    print('ci vector size: {}'.format(Na*Nb))
    return occ

def cas2occ(mf, ncas, nelecas):
    nea, neb = mf.mol.nelec 
    nca = nea-nelecas[0]
    ncb = neb-nelecas[1]
    occslsta = _gen_occslst(range(nca,nca+ncas[0]),nelecas[0])
    occslstb = _gen_occslst(range(ncb,ncb+ncas[1]),nelecas[1])
    ca = np.empty((occslsta.shape[0],nca),dtype=int)
    cb = np.empty((occslstb.shape[0],ncb),dtype=int)
    for i in range(nca): 
        ca[:, i] = i
    for i in range(ncb): 
        cb[:, i] = i
    occslsta = np.hstack((ca, occslsta))
    occslstb = np.hstack((cb, occslstb))
    Na = occslsta.shape[0]
    Nb = occslstb.shape[0]
    ndet = Na*Nb
    occ = np.empty((ndet,nea+neb),dtype=int)
    for Ia in range(Na):
        for Ib in range(Nb):
            I = Ia*Na + Ib
            occ[I,:] = np.hstack((occslsta[Ia,:],occslstb[Ib,:]))
    print('number of active determinant: {}'.format(ndet))
    return occ 

def perm_mo(mf, mo0, occ):
    nea, neb = mf.mol.nelec
    moa, mob = mo0
    nmoa, nmob = moa.shape[1], mob.shape[1]
    occa, occb = occ[:nea], occ[nea:]
    vira = list(set(range(nmoa))-set(occa)) 
    virb = list(set(range(nmoa))-set(occb))
    moa_occ, mob_occ = moa[:,list(occa)], mob[:,list(occb)]
    moa_vir, mob_vir = moa[:,vira], mob[:,virb]
    vira, virb = np.array(vira), np.array(virb)
    moa, mob = np.hstack((moa_occ,moa_vir)), np.hstack((mob_occ,mob_vir))
    occ = np.hstack((occa,occb))
    vir = np.hstack((vira,virb))
    return (moa, mob), occ, vir

def ref_energy(mf):
    moa, mob = mf.mo_coeff
    nmoa, nmob = moa.shape[1], mob.shape[1]
    nea, neb = mf.mol.nelec
    hcore = mf.get_hcore()
    ha = np.linalg.multi_dot([moa.T,hcore,moa])
    hb = np.linalg.multi_dot([mob.T,hcore,mob])
    E0 = sum(ha.diagonal()[:nea]) + sum(hb.diagonal()[:neb])

    eriao = mf._eri
    eri_aa = ao2mo.restore(1, ao2mo.full(eriao, moa), nmoa)
    eri_bb = ao2mo.restore(1, ao2mo.full(eriao, mob), nmob)
    eri_ab = ao2mo.general(eriao, (moa,moa,mob,mob), compact=False)
    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    oooo = eri_aa[:nea,:nea,:nea,:nea].copy()
    OOOO = eri_bb[:neb,:neb,:neb,:neb].copy()
    ooOO = eri_ab[:nea,:nea,:neb,:neb].copy()

    temp = einsum('iijj',OOOO)-einsum('ijji',OOOO)
    temp += einsum('iijj',oooo)-einsum('ijji',oooo) 
    temp += 2*einsum('iijj',ooOO)
    E0 += 0.5*temp
    return E0 

def gen_denom(mf, nfc, nfv, homo):
    nea, neb = mf.mol.nelec
    nfca, nfcb = nfc
    nfva, nfvb = nfv
    moa, mob = mf.mo_coeff
    nmoa, nmob = moa.shape[1], mob.shape[1]
    dma, dmb = np.dot(moa[:,:nea],moa[:,:nea].T), np.dot(mob[:,:neb],mob[:,:neb].T)
    fockao = mf.get_fock(dm=(dma,dmb))
    moa_occ, mob_occ = moa[:,nfca:nea], mob[:,nfcb:neb]
    moa_vir, mob_vir = moa[:,nea:nmoa-nfva], mob[:,neb:nmob-nfvb]
    focka_occ = np.linalg.multi_dot([moa_occ.T, fockao[0], moa_occ])
    fockb_occ = np.linalg.multi_dot([mob_occ.T, fockao[1], mob_occ])
    focka_vir = np.linalg.multi_dot([moa_vir.T, fockao[0], moa_vir])
    fockb_vir = np.linalg.multi_dot([mob_vir.T, fockao[1], mob_vir])
    eoa, eob = focka_occ.diagonal().copy(), fockb_occ.diagonal().copy()
    eva, evb = focka_vir.diagonal().copy(), fockb_vir.diagonal().copy()
    # setting newly occupied orbital energies to HOMO
#    occa0, occb0 = set(range(nfca,nea)), set(range(nfcb,neb))
#    occa, occb = list(occ[nfca:nea]), list(occ[nea+nfcb:])
#    idxa = [occa.index(x) for x in set(occa) - occa0]
#    idxb = [occb.index(x) for x in set(occb) - occb0]
#    eoa[idxa] = homo[0]
#    eob[idxb] = homo[1]
    eoa[eoa>homo[0]] = homo[0]
    eob[eob>homo[1]] = homo[1]
    # shifting virtual orbitals
    thresh = 1.0
    for i in range(len(eva)):
        delta = eva[i] - homo[0]
        eva[i] = eva[i] + abs(delta) + thresh if delta < thresh else eva[i]
    for i in range(len(evb)):
        delta = evb[i] - homo[1]
        evb[i] = evb[i] + abs(delta) + thresh if delta < thresh else evb[i]
    return (eoa, eob, eva, evb)

def get_init_guess(cc, eris=None, denom=None):
    return init_amps(cc, eris, denom)[1:]

def init_amps(cc, eris=None, denom=None):
    time0 = time.clock(), time.time()
    if eris is None:
        eris = cc.ao2mo(cc.mo_coeff)
    nocca, noccb = cc.nocc

    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    if denom is None:
        mo_ea_o = eris.mo_energy[0][:nocca]
        mo_eb_o = eris.mo_energy[1][:noccb]
        mo_ea_v = eris.mo_energy[0][nocca:]
        mo_eb_v = eris.mo_energy[1][noccb:]
    else:
        mo_ea_o, mo_eb_o, mo_ea_v, mo_eb_v = denom

    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)
    eijab_aa = lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    eijab_ab = lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    eijab_bb = lib.direct_sum('ia+jb->ijab', eia_b, eia_b)

    t1a = fova.conj() / eia_a
    t1b = fovb.conj() / eia_b

    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    t2aa = eris_ovov.transpose(0,2,1,3) / eijab_aa
    t2ab = eris_ovOV.transpose(0,2,1,3) / eijab_ab
    t2bb = eris_OVOV.transpose(0,2,1,3) / eijab_bb
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    cc.emp2 = cc.energy((t1a,t1b), (t2aa,t2ab,t2bb), eris)
    logger.info(cc, 'Init t2, MP2 energy = %.15g', cc.emp2)
    logger.timer(cc, 'init mp2', *time0)
    return cc.emp2, (t1a,t1b), (t2aa,t2ab,t2bb)

def update_amps(cc, t1, t2, eris, denom=None):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    if denom is None:
        mo_ea_o = eris.mo_energy[0][:nocca]
        mo_eb_o = eris.mo_energy[1][:noccb]
        mo_ea_v = eris.mo_energy[0][nocca:]
        mo_eb_v = eris.mo_energy[1][noccb:]
    else:
        mo_ea_o, mo_eb_o, mo_ea_v, mo_eb_v = denom

    u1a = np.zeros_like(t1a)
    u1b = np.zeros_like(t1b)
    #:eris_vvvv = aself.nmo[0]o2mo.restore(1, np.asarray(eris.vvvv), nvirb)
    #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
    #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
    #:u2aa += lib.einsum('ijef,aebf->ijab', tauaa, eris_vvvv) * .5
    #:u2bb += lib.einsum('ijef,aebf->ijab', taubb, eris_VVVV) * .5
    #:u2ab += lib.einsum('iJeF,aeBF->iJaB', tauab, eris_vvVV)
    tauaa, tauab, taubb = make_tau(t2, t1, t1)
    u2aa, u2ab, u2bb = cc._add_vvvv(None, (tauaa,tauab,taubb), eris)
    u2aa *= .5
    u2bb *= .5

    Fooa =  .5 * lib.einsum('me,ie->mi', fova, t1a)
    Foob =  .5 * lib.einsum('me,ie->mi', fovb, t1b)
    Fvva = -.5 * lib.einsum('me,ma->ae', fova, t1a)
    Fvvb = -.5 * lib.einsum('me,ma->ae', fovb, t1b)
    Fooa += eris.focka[:nocca,:nocca] - np.diag(mo_ea_o)
    Foob += eris.fockb[:noccb,:noccb] - np.diag(mo_eb_o)
    Fvva += eris.focka[nocca:,nocca:] - np.diag(mo_ea_v)
    Fvvb += eris.fockb[noccb:,noccb:] - np.diag(mo_eb_v)
    dtype = u2aa.dtype
    wovvo = np.zeros((nocca,nvira,nvira,nocca), dtype=dtype)
    wOVVO = np.zeros((noccb,nvirb,nvirb,noccb), dtype=dtype)
    woVvO = np.zeros((nocca,nvirb,nvira,noccb), dtype=dtype)
    woVVo = np.zeros((nocca,nvirb,nvirb,nocca), dtype=dtype)
    wOvVo = np.zeros((noccb,nvira,nvirb,nocca), dtype=dtype)
    wOvvO = np.zeros((noccb,nvira,nvira,noccb), dtype=dtype)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now)
    if nvira > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            Fvva += np.einsum('mf,mfae->ae', t1a[p0:p1], ovvv)
            wovvo[p0:p1] += lib.einsum('jf,mebf->mbej', t1a, ovvv)
            u1a += 0.5*lib.einsum('mief,meaf->ia', t2aa[p0:p1], ovvv)
            u2aa[:,p0:p1] += lib.einsum('ie,mbea->imab', t1a, ovvv.conj())
            tmp1aa = lib.einsum('ijef,mebf->ijmb', tauaa, ovvv)
            u2aa -= lib.einsum('ijmb,ma->ijab', tmp1aa, t1a[p0:p1]*.5)
            ovvv = tmp1aa = None

    if nvirb > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            Fvvb += np.einsum('mf,mfae->ae', t1b[p0:p1], OVVV)
            wOVVO[p0:p1] = lib.einsum('jf,mebf->mbej', t1b, OVVV)
            u1b += 0.5*lib.einsum('MIEF,MEAF->IA', t2bb[p0:p1], OVVV)
            u2bb[:,p0:p1] += lib.einsum('ie,mbea->imab', t1b, OVVV.conj())
            tmp1bb = lib.einsum('ijef,mebf->ijmb', taubb, OVVV)
            u2bb -= lib.einsum('ijmb,ma->ijab', tmp1bb, t1b[p0:p1]*.5)
            OVVV = tmp1bb = None

    if nvirb > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            Fvvb += np.einsum('mf,mfAE->AE', t1a[p0:p1], ovVV)
            woVvO[p0:p1] = lib.einsum('JF,meBF->mBeJ', t1b, ovVV)
            woVVo[p0:p1] = lib.einsum('jf,mfBE->mBEj',-t1a, ovVV)
            u1b += lib.einsum('mIeF,meAF->IA', t2ab[p0:p1], ovVV)
            u2ab[p0:p1] += lib.einsum('IE,maEB->mIaB', t1b, ovVV.conj())
            tmp1ab = lib.einsum('iJeF,meBF->iJmB', tauab, ovVV)
            u2ab -= lib.einsum('iJmB,ma->iJaB', tmp1ab, t1a[p0:p1])
            ovVV = tmp1ab = None

    if nvira > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            Fvva += np.einsum('MF,MFae->ae', t1b[p0:p1], OVvv)
            wOvVo[p0:p1] = lib.einsum('jf,MEbf->MbEj', t1a, OVvv)
            wOvvO[p0:p1] = lib.einsum('JF,MFbe->MbeJ',-t1b, OVvv)
            u1a += lib.einsum('iMfE,MEaf->ia', t2ab[:,p0:p1], OVvv)
            u2ab[:,p0:p1] += lib.einsum('ie,MBea->iMaB', t1a, OVvv.conj())
            tmp1abba = lib.einsum('iJeF,MFbe->iJbM', tauab, OVvv)
            u2ab -= lib.einsum('iJbM,MA->iJbA', tmp1abba, t1b[p0:p1])
            OVvv = tmp1abba = None

    eris_ovov = np.asarray(eris.ovov)
    eris_ovoo = np.asarray(eris.ovoo)
    Woooo = lib.einsum('je,nemi->mnij', t1a, eris_ovoo)
    Woooo = Woooo - Woooo.transpose(0,1,3,2)
    Woooo += np.asarray(eris.oooo).transpose(0,2,1,3)
    Woooo += lib.einsum('ijef,menf->mnij', tauaa, eris_ovov) * .5
    u2aa += lib.einsum('mnab,mnij->ijab', tauaa, Woooo*.5)
    Woooo = tauaa = None
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    Fooa += np.einsum('ne,nemi->mi', t1a, ovoo)
    u1a += 0.5*lib.einsum('mnae,meni->ia', t2aa, ovoo)
    wovvo += lib.einsum('nb,nemj->mbej', t1a, ovoo)
    ovoo = eris_ovoo = None

    tilaa = make_tau_aa(t2[0], t1a, t1a, fac=0.5)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    Fvva -= .5 * lib.einsum('mnaf,menf->ae', tilaa, ovov)
    Fooa += .5 * lib.einsum('inef,menf->mi', tilaa, ovov)
    Fova = np.einsum('nf,menf->me',t1a, ovov)
    u2aa += ovov.conj().transpose(0,2,1,3) * .5
    wovvo -= 0.5*lib.einsum('jnfb,menf->mbej', t2aa, ovov)
    woVvO += 0.5*lib.einsum('nJfB,menf->mBeJ', t2ab, ovov)
    tmpaa = lib.einsum('jf,menf->mnej', t1a, ovov)
    wovvo -= lib.einsum('nb,mnej->mbej', t1a, tmpaa)
    eirs_ovov = ovov = tmpaa = tilaa = None

    eris_OVOV = np.asarray(eris.OVOV)
    eris_OVOO = np.asarray(eris.OVOO)
    WOOOO = lib.einsum('je,nemi->mnij', t1b, eris_OVOO)
    WOOOO = WOOOO - WOOOO.transpose(0,1,3,2)
    WOOOO += np.asarray(eris.OOOO).transpose(0,2,1,3)
    WOOOO += lib.einsum('ijef,menf->mnij', taubb, eris_OVOV) * .5
    u2bb += lib.einsum('mnab,mnij->ijab', taubb, WOOOO*.5)
    WOOOO = taubb = None
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    Foob += np.einsum('ne,nemi->mi', t1b, OVOO)
    u1b += 0.5*lib.einsum('mnae,meni->ia', t2bb, OVOO)
    wOVVO += lib.einsum('nb,nemj->mbej', t1b, OVOO)
    OVOO = eris_OVOO = None

    tilbb = make_tau_aa(t2[2], t1b, t1b, fac=0.5)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    Fvvb -= .5 * lib.einsum('MNAF,MENF->AE', tilbb, OVOV)
    Foob += .5 * lib.einsum('inef,menf->mi', tilbb, OVOV)
    Fovb = np.einsum('nf,menf->me',t1b, OVOV)
    u2bb += OVOV.conj().transpose(0,2,1,3) * .5
    wOVVO -= 0.5*lib.einsum('jnfb,menf->mbej', t2bb, OVOV)
    wOvVo += 0.5*lib.einsum('jNbF,MENF->MbEj', t2ab, OVOV)
    tmpbb = lib.einsum('jf,menf->mnej', t1b, OVOV)
    wOVVO -= lib.einsum('nb,mnej->mbej', t1b, tmpbb)
    eris_OVOV = OVOV = tmpbb = tilbb = None

    eris_OVoo = np.asarray(eris.OVoo)
    eris_ovOO = np.asarray(eris.ovOO)
    Fooa += np.einsum('NE,NEmi->mi', t1b, eris_OVoo)
    u1a -= lib.einsum('nMaE,MEni->ia', t2ab, eris_OVoo)
    wOvVo -= lib.einsum('nb,MEnj->MbEj', t1a, eris_OVoo)
    woVVo += lib.einsum('NB,NEmj->mBEj', t1b, eris_OVoo)
    Foob += np.einsum('ne,neMI->MI', t1a, eris_ovOO)
    u1b -= lib.einsum('mNeA,meNI->IA', t2ab, eris_ovOO)
    woVvO -= lib.einsum('NB,meNJ->mBeJ', t1b, eris_ovOO)
    wOvvO += lib.einsum('nb,neMJ->MbeJ', t1a, eris_ovOO)
    WoOoO = lib.einsum('JE,NEmi->mNiJ', t1b, eris_OVoo)
    WoOoO+= lib.einsum('je,neMI->nMjI', t1a, eris_ovOO)
    WoOoO += np.asarray(eris.ooOO).transpose(0,2,1,3)
    eris_OVoo = eris_ovOO = None

    eris_ovOV = np.asarray(eris.ovOV)
    WoOoO += lib.einsum('iJeF,meNF->mNiJ', tauab, eris_ovOV)
    u2ab += lib.einsum('mNaB,mNiJ->iJaB', tauab, WoOoO)
    WoOoO = None

    tilab = make_tau_ab(t2[1], t1 , t1 , fac=0.5)
    Fvva -= lib.einsum('mNaF,meNF->ae', tilab, eris_ovOV)
    Fvvb -= lib.einsum('nMfA,nfME->AE', tilab, eris_ovOV)
    Fooa += lib.einsum('iNeF,meNF->mi', tilab, eris_ovOV)
    Foob += lib.einsum('nIfE,nfME->MI', tilab, eris_ovOV)
    Fova += np.einsum('NF,meNF->me',t1b, eris_ovOV)
    Fovb += np.einsum('nf,nfME->ME',t1a, eris_ovOV)
    u2ab += eris_ovOV.conj().transpose(0,2,1,3)
    wovvo += 0.5*lib.einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
    wOVVO += 0.5*lib.einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
    wOvVo -= 0.5*lib.einsum('jnfb,nfME->MbEj', t2aa, eris_ovOV)
    woVvO -= 0.5*lib.einsum('JNFB,meNF->mBeJ', t2bb, eris_ovOV)
    woVVo += 0.5*lib.einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
    wOvvO += 0.5*lib.einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)
    tmpabab = lib.einsum('JF,meNF->mNeJ', t1b, eris_ovOV)
    tmpbaba = lib.einsum('jf,nfME->MnEj', t1a, eris_ovOV)
    woVvO -= lib.einsum('NB,mNeJ->mBeJ', t1b, tmpabab)
    wOvVo -= lib.einsum('nb,MnEj->MbEj', t1a, tmpbaba)
    woVVo += lib.einsum('NB,NmEj->mBEj', t1b, tmpbaba)
    wOvvO += lib.einsum('nb,nMeJ->MbeJ', t1a, tmpabab)
    tmpabab = tmpbaba = tilab = None

    Fova += fova
    Fovb += fovb
    u1a += fova.conj()
    u1a += np.einsum('ie,ae->ia', t1a, Fvva)
    u1a -= np.einsum('ma,mi->ia', t1a, Fooa)
    u1a -= np.einsum('imea,me->ia', t2aa, Fova)
    u1a += np.einsum('iMaE,ME->ia', t2ab, Fovb)
    u1b += fovb.conj()
    u1b += np.einsum('ie,ae->ia',t1b,Fvvb)
    u1b -= np.einsum('ma,mi->ia',t1b,Foob)
    u1b -= np.einsum('imea,me->ia', t2bb, Fovb)
    u1b += np.einsum('mIeA,me->IA', t2ab, Fova)

    eris_oovv = np.asarray(eris.oovv)
    eris_ovvo = np.asarray(eris.ovvo)
    wovvo -= eris_oovv.transpose(0,2,3,1)
    wovvo += eris_ovvo.transpose(0,2,1,3)
    oovv = eris_oovv - eris_ovvo.transpose(0,3,2,1)
    u1a-= np.einsum('nf,niaf->ia', t1a,      oovv)
    tmp1aa = lib.einsum('ie,mjbe->mbij', t1a,      oovv)
    u2aa += 2*lib.einsum('ma,mbij->ijab', t1a, tmp1aa)
    eris_ovvo = eris_oovv = oovv = tmp1aa = None

    eris_OOVV = np.asarray(eris.OOVV)
    eris_OVVO = np.asarray(eris.OVVO)
    wOVVO -= eris_OOVV.transpose(0,2,3,1)
    wOVVO += eris_OVVO.transpose(0,2,1,3)
    OOVV = eris_OOVV - eris_OVVO.transpose(0,3,2,1)
    u1b-= np.einsum('nf,niaf->ia', t1b,      OOVV)
    tmp1bb = lib.einsum('ie,mjbe->mbij', t1b,      OOVV)
    u2bb += 2*lib.einsum('ma,mbij->ijab', t1b, tmp1bb)
    eris_OVVO = eris_OOVV = OOVV = None

    eris_ooVV = np.asarray(eris.ooVV)
    eris_ovVO = np.asarray(eris.ovVO)
    woVVo -= eris_ooVV.transpose(0,2,3,1)
    woVvO += eris_ovVO.transpose(0,2,1,3)
    u1b+= np.einsum('nf,nfAI->IA', t1a, eris_ovVO)
    tmp1ab = lib.einsum('ie,meBJ->mBiJ', t1a, eris_ovVO)
    tmp1ab+= lib.einsum('IE,mjBE->mBjI', t1b, eris_ooVV)
    u2ab -= lib.einsum('ma,mBiJ->iJaB', t1a, tmp1ab)
    eris_ooVV = eris_ovVo = tmp1ab = None

    eris_OOvv = np.asarray(eris.OOvv)
    eris_OVvo = np.asarray(eris.OVvo)
    wOvvO -= eris_OOvv.transpose(0,2,3,1)
    wOvVo += eris_OVvo.transpose(0,2,1,3)
    u1a+= np.einsum('NF,NFai->ia', t1b, eris_OVvo)
    tmp1ba = lib.einsum('IE,MEbj->MbIj', t1b, eris_OVvo)
    tmp1ba+= lib.einsum('ie,MJbe->MbJi', t1a, eris_OOvv)
    u2ab -= lib.einsum('MA,MbIj->jIbA', t1b, tmp1ba)
    eris_OOvv = eris_OVvO = tmp1ba = None

    u2aa += 2*lib.einsum('imae,mbej->ijab', t2aa, wovvo)
    u2aa += 2*lib.einsum('iMaE,MbEj->ijab', t2ab, wOvVo)
    u2bb += 2*lib.einsum('imae,mbej->ijab', t2bb, wOVVO)
    u2bb += 2*lib.einsum('mIeA,mBeJ->IJAB', t2ab, woVvO)
    u2ab += lib.einsum('imae,mBeJ->iJaB', t2aa, woVvO)
    u2ab += lib.einsum('iMaE,MBEJ->iJaB', t2ab, wOVVO)
    u2ab += lib.einsum('iMeA,MbeJ->iJbA', t2ab, wOvvO)
    u2ab += lib.einsum('IMAE,MbEj->jIbA', t2bb, wOvVo)
    u2ab += lib.einsum('mIeA,mbej->jIbA', t2ab, wovvo)
    u2ab += lib.einsum('mIaE,mBEj->jIaB', t2ab, woVVo)
    wovvo = wOVVO = woVvO = wOvVo = woVVo = wOvvO = None

    Ftmpa = Fvva - .5*lib.einsum('mb,me->be', t1a, Fova)
    Ftmpb = Fvvb - .5*lib.einsum('mb,me->be', t1b, Fovb)
    u2aa += lib.einsum('ijae,be->ijab', t2aa, Ftmpa)
    u2bb += lib.einsum('ijae,be->ijab', t2bb, Ftmpb)
    u2ab += lib.einsum('iJaE,BE->iJaB', t2ab, Ftmpb)
    u2ab += lib.einsum('iJeA,be->iJbA', t2ab, Ftmpa)
    Ftmpa = Fooa + 0.5*lib.einsum('je,me->mj', t1a, Fova)
    Ftmpb = Foob + 0.5*lib.einsum('je,me->mj', t1b, Fovb)
    u2aa -= lib.einsum('imab,mj->ijab', t2aa, Ftmpa)
    u2bb -= lib.einsum('imab,mj->ijab', t2bb, Ftmpb)
    u2ab -= lib.einsum('iMaB,MJ->iJaB', t2ab, Ftmpb)
    u2ab -= lib.einsum('mIaB,mj->jIaB', t2ab, Ftmpa)

    eris_ovoo = np.asarray(eris.ovoo).conj()
    eris_OVOO = np.asarray(eris.OVOO).conj()
    eris_OVoo = np.asarray(eris.OVoo).conj()
    eris_ovOO = np.asarray(eris.ovOO).conj()
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    u2aa -= lib.einsum('ma,jbim->ijab', t1a, ovoo)
    u2bb -= lib.einsum('ma,jbim->ijab', t1b, OVOO)
    u2ab -= lib.einsum('ma,JBim->iJaB', t1a, eris_OVoo)
    u2ab -= lib.einsum('MA,ibJM->iJbA', t1b, eris_ovOO)
    eris_ovoo = eris_OVoo = eris_OVOO = eris_ovOO = None

    u2aa *= .5
    u2bb *= .5
    u2aa = u2aa - u2aa.transpose(0,1,3,2)
    u2aa = u2aa - u2aa.transpose(1,0,2,3)
    u2bb = u2bb - u2bb.transpose(0,1,3,2)
    u2bb = u2bb - u2bb.transpose(1,0,2,3)
    time0 = log.timer_debug1('update t1 t2', *time0)
    return (u1a, u1b), (u2aa, u2ab, u2bb)

def kernel(cc, eris=None, t1=None, t2=None, denom=None, mbpt2=False, beta=-1.0, dt=None):
    max_cycle = cc.max_cycle
    tol = cc.conv_tol
    tolnormt = cc.conv_tol_normt

    assert(cc.mo_coeff is not None)
    assert(cc.mo_occ is not None)
    if cc.verbose >= logger.WARN:
        ccsd.CCSD.check_sanity(cc)
    ccsd.CCSD.dump_flags(cc)
    log = logger.new_logger(cc, cc.verbose)

    eris = cc.ao2mo(cc.mo_coeff) if eris is None else eris
    # MBPT2
    if mbpt2:
        t1, t2 = get_init_guess(cc, eris, denom=None)
        eccsd = cc.energy(t1, t2, eris)
        return True, eccsd, t1, t2
     
    if t1 is None and t2 is None:
        t1, t2 = get_init_guess(cc, eris, denom)
    elif t2 is None:
        t2 = get_init_guess(cc, eris, denom)[1]

    # generate denominator
    if denom is None:
        mo_ea_o = eris.mo_energy[0][:nocca]
        mo_eb_o = eris.mo_energy[1][:noccb]
        mo_ea_v = eris.mo_energy[0][nocca:]
        mo_eb_v = eris.mo_energy[1][noccb:]
    else:
        mo_ea_o, mo_eb_o, mo_ea_v, mo_eb_v = denom
    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)
    eijab_aa = lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    eijab_ab = lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    eijab_bb = lib.direct_sum('ia+jb->ijab', eia_b, eia_b)

    cput1 = cput0 = (time.clock(), time.time())
    eold = 0
    conv = False
    eccsd = cc.energy(t1, t2, eris)
    log.info('Init E(CCSD) = %.15g', eccsd)

    if beta < 0.0:
        if isinstance(cc.diis, lib.diis.DIIS):
            adiis = cc.diis
        elif cc.diis:
            adiis = lib.diis.DIIS(cc, cc.diis_file, incore=cc.incore_complete)
            adiis.space = cc.diis_space
        else:
            adiis = None
    
        for istep in range(max_cycle):
            u1, u2 = update_amps(cc, t1, t2, eris, denom) # residue
            u1a, u1b = u1
            u2aa, u2ab, u2bb = u2
            t1a = u1a / eia_a
            t1b = u1b / eia_b
            t2aa = u2aa / eijab_aa
            t2ab = u2ab / eijab_ab
            t2bb = u2bb / eijab_bb
            t1new = t1a, t1b
            t2new = t2aa, t2ab, t2bb            

            normt = np.linalg.norm(cc.amplitudes_to_vector(t1new, t2new) -
                                   cc.amplitudes_to_vector(t1, t2))
            if cc.iterative_damping < 1.0:
                alpha = cc.iterative_damping
                t1new = (1-alpha) * t1 + alpha * t1new
                t2new *= alpha
                t2new += (1-alpha) * t2
            t1, t2 = t1new, t2new
            t1new = t2new = None
            t1, t2 = cc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
            eold, eccsd = eccsd, cc.energy(t1, t2, eris)
            log.info('cycle = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                     istep+1, eccsd, eccsd - eold, normt)
            cput1 = log.timer('CCSD iter', *cput1)
            if abs(eccsd) > 10.0:
                print('Divergent energy!')
                break
            if abs(eccsd-eold) < tol and normt < tolnormt:
                conv = True
                break
    else:
        for tau in np.arange(0.0, beta, dt):
            u1, u2 = update_amps(cc, t1, t2, eris, denom) # residue
            u1a, u1b = u1
            u2aa, u2ab, u2bb = u2
            t1a, t1b = t1
            t2aa, t2ab, t2bb = t2
            t1a += dt*(np.multiply(eia_a,t1a) - u1a)
            t1b += dt*(np.multiply(eia_b,t1b) - u1b)
            t2aa += dt*(np.multiply(eijab_aa,t2aa) - u2aa)
            t2ab += dt*(np.multiply(eijab_ab,t2ab) - u2ab)
            t2bb += dt*(np.multiply(eijab_bb,t2bb) - u2bb)
            t1new = t1a, t1b
            t2new = t2aa, t2ab, t2bb            

            normt = np.linalg.norm(cc.amplitudes_to_vector(t1new, t2new) -
                                   cc.amplitudes_to_vector(t1, t2))/dt
            t1, t2 = t1new, t2new
            t1new = t2new = None
            eold, eccsd = eccsd, cc.energy(t1, t2, eris)
            log.info('tau = %.4g  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                     tau, eccsd, eccsd - eold, normt)
            cput1 = log.timer('CCSD iter', *cput1)
            if abs(eccsd) > 10.0:
                print('Divergent energy!')
                break
            if abs(eccsd-eold) < tol and normt < tolnormt:
                conv = True
                beta = tau
                break
    log.timer('CCSD', *cput0)

    cc.converged = conv
    cc.e_corr = eccsd
    ccsd.CCSD._finalize(cc)
    return conv, eccsd, t1, t2, beta

def occ2str(occ):
    string = 0
    for i in occ: 
        string ^= 1 << i
    return string

def ci_imt(N,nmo,ne,no,nv,nfc,occ,vir,t1,t2):
    S = np.zeros((N, N, no, nv))
    T = np.zeros((N, N))
    for I in range(N): 
        strI = addr2str(nmo,ne,I) 
        for i in range(no):
            des1 = occ[i+nfc]
            h1 = 1 << des1
            if strI & h1 != 0:
                for a in range(nv):
                    cre1 = vir[a]
                    p1 = 1 << cre1
                    if strI & p1 == 0:
                        str1 = strI ^ h1 | p1
                        K = str2addr(nmo,ne,str1)
                        sgn1 = cre_des_sign(cre1,des1,strI)
                        S[K,I,i,a] += sgn1
                        T[K,I] += t1[i,a]*sgn1
                        for j in range(i):
                            des2 = occ[j+nfc]
                            h2 = 1 << des2
                            if strI & h2 != 0:
                                for b in range(a):
                                    cre2 = vir[b]
                                    p2 = 1 << cre2
                                    if strI & p2 == 0:
                                        str2 = str1 ^ h2 | p2
                                        K = str2addr(nmo,ne,str2)
                                        sgn2 = cre_des_sign(cre2,des2,str1)
                                        T[K,I] += t2[i,j,a,b]*sgn1*sgn2
    return S, T

def det2ind(nmo,tar,occ,vir,nfc):
    occ = list(occ)
    vir = list(vir)
    h_abs = set(occ) - set(tar)
    p_abs = set(tar) - set(occ)
    h_rel = [occ.index(x) - nfc for x in h_abs]
    p_rel = [vir.index(x) for x in p_abs]
    return p_rel, h_rel

class MRCCSD(ccsd.CCSD):

    conv_tol = getattr(__config__, 'cc_uccsd_UCCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_uccsd_UCCSD_conv_tol_normt', 1e-6)

# Attribute frozen can be
# * An integer : The same number of inner-most alpha and beta orbitals are frozen
# * One list : Same alpha and beta orbital indices to be frozen
# * A pair of list : First list is the orbital indices to be frozen for alpha
#       orbitals, second list is for beta orbitals
    def __init__(self, mf, 
                 ncas=None, nelecas=None, 
                 occslst=None, 
                 ci=None, ndet=None, cutoff=None, 
                 frozen=0, beta=-1.0, dt=None):

        nea, neb = mf.mol.nelec
        moa, mob = mf.mo_coeff
        self.mo0 = moa.copy(), mob.copy()
        nmoa, nmob = moa.shape[1], mob.shape[1]
        self.nfc, self.nfv = get_frozen(mf, frozen)
        self.pbra = 0
        self.pket = nea + neb - sum(self.nfc)
        self.mbpt2 = False
        self.homo = mf.mo_energy[0][nea-1], mf.mo_energy[1][neb-1]
        print('nmo: {}, nele: {}'.format((nmoa, nmob), (nea, neb)))
        print('frozen core: {}, virtual: {}'.format(self.nfc, self.nfv))

        self.occ = get_occ(mf, ncas, nelecas, occslst, ci, ndet, cutoff)
        ndet = self.occ.shape[0]
        self.vir = np.empty((ndet,nmoa+nmob-nea-neb),dtype=int)
        self.beta = np.ones(ndet) * beta
        self.dt = dt

        noa, nob = nea - self.nfc[0], neb - self.nfc[1]
        nva, nvb = nmoa - nea - self.nfv[0], nmob - neb - self.nfv[1] 
        self.t1a = np.empty((ndet, noa, nva))
        self.t1b = np.empty((ndet, nob, nvb))
        self.t2aa = np.empty((ndet, noa, noa, nva, nva))
        self.t2ab = np.empty((ndet, noa, nob, nva, nvb))
        self.t2bb = np.empty((ndet, nob, nob, nvb, nvb))

        self.cc = uccsd.UCCSD(mf, frozen)
#        self.cc.diis_space = 10
#        self.cc.iterative_damping = 0.3

    def kernel(self, H=None, S=None, H_=None, S_=None):
        enuc = self.cc._scf.mol.energy_nuc()
        if H is None:
            eccsd = self.ccsd()
            H, S = self.fci()
        if self.beta[0] < 0.0:
            if not self.mbpt2:
                if H_ is None: 
                    H_, S_ = self.contract(eccsd)
                    print('contraction:\n{}\n{}'.format(H_,S_))
                    print('fci:\n{}\n{}'.format(H,S))
        np.set_printoptions(precision=20,suppress=True)
        print('diagonal: {}'.format(
              np.divide(np.diagonal(H),np.diagonal(S)) + enuc))
        np.set_printoptions(precision=8,suppress=True)

        w, v = scipy.linalg.eig(H, S)
        idx = np.argsort(w.real) 
        w, v = w[idx], v[:, idx]

        self.etot = w.real[0] + enuc()
        self.vact = v.real[:,0]
        np.set_printoptions(precision=20,suppress=True)
        print('MRCCSD ground state energy: {}'.format(self.etot))
        np.set_printoptions(precision=8,suppress=True)
        print('Imaginary parts: {}'.format(w.imag[0]))
        return self.etot, self.vact 

    def ccsd(self):
        ndet = self.occ.shape[0]
        eccsd = np.empty(ndet)
        for I in range(ndet):
            print('\nreference {} occupation: {}'.format(I, self.occ[I,:]))
            mo_coeff, self.occ[I,:], self.vir[I,:] = \
                perm_mo(self.cc._scf, self.mo0, self.occ[I,:])
            self.cc.mo_coeff = self.cc._scf.mo_coeff = mo_coeff
            e_ref = ref_energy(self.cc._scf)
            self.cc._scf.e_tot = e_ref + self.cc._scf.mol.energy_nuc()
            print('reference energy: {}'.format(self.cc._scf.e_tot))
            denom = gen_denom(self.cc._scf,self.nfc,self.nfv,self.homo)
            conv, Ecorr, t1, t2, self.beta[I] = kernel(self.cc,denom=denom,
                mbpt2=self.mbpt2,beta=self.beta[I],dt=self.dt)
            t1a, t1b = t1
            t2aa, t2ab, t2bb = t2
            self.t1a[I,:,:] = t1a.copy()
            self.t1b[I,:,:] = t1b.copy()
            self.t2aa[I,:,:,:,:] = t2aa.copy()
            self.t2ab[I,:,:,:,:] = t2ab.copy()
            self.t2bb[I,:,:,:,:] = t2bb.copy()
            eccsd[I] = e_ref + Ecorr
            print('max element: t1a, t1b\n{}, {}'.format(
                  np.amax(abs(t1a)),np.amax(abs(t1b))))
            print('max element: t2aa, t2ab, t2bb\n{}, {}, {}'.format(
                  np.amax(abs(t2aa)),np.amax(abs(t2ab)),np.amax(abs(t2bb))))
        return eccsd

    def fci(self, ket=None, bra=None):
        if ket is None:
            ndet = self.occ.shape[0]
            nmoa, nmob = self.mo0[0].shape[1], self.mo0[1].shape[1]
            noa, nob, _, _ = self.t2ab.shape[1:]
            nea, neb = noa + self.nfc[0], nob + self.nfc[1]
            Na = num_strings(nmoa, nea)
            Nb = num_strings(nmob, neb)
            ket = np.empty((ndet,Na,Nb))
            bra = np.empty((ndet,Na,Nb))
            for I in range(ndet):
                t1a = self.t1a[I,:,:]
                t1b = self.t1b[I,:,:]
                t2aa = self.t2aa[I,:,:,:,:]
                t2ab = self.t2ab[I,:,:,:,:]
                t2bb = self.t2bb[I,:,:,:,:]
                t = time.time()
                bra[I,:,:], ket[I,:,:] = self.get_ci(t1a,t1b,t2aa,t2ab,t2bb,
                                                     self.occ[I,:],self.vir[I,:])
                print('expT mapping time for reference {}: {}'.format(
                      I, time.time()-t))

        ndet = ket.shape[0]
        hop = self._hop()
        H = np.empty((ndet,ndet)) 
        S = np.empty((ndet,ndet)) 
        for I in range(ndet):
            ci = ket[I,:,:]
            hci = hop(ci)
            for J in range(ndet):
                H[J,I] = einsum('JI,JI',bra[J,:,:],hci)
                S[J,I] = einsum('JI,JI',bra[J,:,:],ci)
        return H, S

    def get_ci(self, t1a, t1b, t2aa, t2ab, t2bb, occ, vir):
        noa, nob, nva, nvb = t2ab.shape
        nfca, nfcb = self.nfc
        nfva, nfvb = self.nfv
        nea, neb = noa + nfca, nob + nfcb
        nmoa, nmob = nea + nva + nfvb, neb + nvb + nfvb
        occa, occb = occ[:nea], occ[nea:] 
        vira, virb = vir[:nmoa-nea], vir[nmoa-nea:]
        Na = num_strings(nmoa,nea)
        Nb = num_strings(nmob,neb)
        stra = occ2str(occa)
        strb = occ2str(occb)
        addra = str2addr(nmoa, nea, stra)
        addrb = str2addr(nmob, neb, strb)
        ci = np.zeros((Na,Nb))
        ci[addra, addrb] = 1.0
    
        t = time.time()
        Sa, Taa = ci_imt(Na,nmoa,nea,noa,nva,nfca,occa,vira,t1a,t2aa)
        Sb, Tbb = ci_imt(Nb,nmob,neb,nob,nvb,nfcb,occb,virb,t1b,t2bb)
        temp = einsum('KIia,ijab->KIjb',Sa,t2ab) 
        print('tensor construction: {}'.format(time.time()-t))
        def T(c):
            c_  = np.dot(Taa,c)
            c_ += np.dot(Tbb,c.T).T
            c_ += einsum('KIjb,LJjb,IJ->KL',temp,Sb,c)
            return c_
        bra = ci.copy() 
        for n in range(1, self.pbra+1):
            ci = T(ci)
            bra += ci/math.factorial(n)
        ket = bra.copy()
        for n in range(self.pbra+1, self.pket+1):
            ci = T(ci)
            ket += ci/math.factorial(n)
        return bra, ket

    def _hop(self):
        moa, mob = self.mo0
        hcore = self.cc._scf.get_hcore()
        eri = self.cc._scf._eri
        nelec = self.cc._scf.mol.nelec
        nmoa = moa.shape[1]
        nmob = mob.shape[1]
    
        h1e_a = np.linalg.multi_dot([moa.T, hcore, moa])
        h1e_b = np.linalg.multi_dot([mob.T, hcore, mob])
        g2e_aa = ao2mo.incore.general(eri, (moa,)*4, compact=False)
        g2e_aa = g2e_aa.reshape(nmoa,nmoa,nmoa,nmoa)
        g2e_ab = ao2mo.incore.general(eri, (moa,moa,mob,mob), compact=False)
        g2e_ab = g2e_ab.reshape(nmoa,nmoa,nmob,nmob)
        g2e_bb = ao2mo.incore.general(eri, (mob,)*4, compact=False)
        g2e_bb = g2e_bb.reshape(nmob,nmob,nmob,nmob)
        h1e = (h1e_a, h1e_b)
        eri = (g2e_aa, g2e_ab, g2e_bb)
    
        h2e = direct_uhf.absorb_h1e(h1e, eri, nmoa, nelec, .5) 
        def hop(c): 
            return direct_uhf.contract_2e(h2e, c, nmoa, nelec)
        return hop

    def contract(self, eccsd, t1a=None, t1b=None, t2aa=None, t2ab=None, t2bb=None, 
                 occ=None, vir=None):
        t1as = self.t1a if t1a is None else t1a
        t1bs = self.t1b if t1b is None else t1b
        t2aas = self.t2aa if t2aa is None else t2aa
        t2abs = self.t2ab if t2ab is None else t2ab
        t2bbs = self.t2bb if t2bb is None else t2bb
        occ = self.occ if occ is None else occ
        vir = self.vir if vir is None else vir
        ndet, noa, nob, nva, nvb = t2ab.shape
        nea, neb = noa + self.nfc[0], nob + self.nfc[1]
        H = np.zeros((ndet, ndet)) 
        S = np.zeros((ndet, ndet)) 
        nmoa, nmob = self.mo0[0].shape[1], self.mo0[1].shape[1]
        nva, nvb = nmoa - nea, nmob - neb
        for I in range(ndet):
            occa, occb = occ[I,:][:nea], occ[I,:][nea:]
            vira, virb = vir[I,:][:nva], vir[I,:][nva:]
            t1a = t1as[I,:,:]
            t1b = t1bs[I,:,:]
            t2aa = t2aas[I,:,:,:,:]
            t2ab = t2abs[I,:,:,:,:]
            t2bb = t2bbs[I,:,:,:,:]
            for J in range(ndet):
                tara, tarb = occ[J,:][:nea], occ[J,:][nea:]
                pa, ha = det2ind(nmoa, tara, occa, vira, self.nfc[0]) 
                pb, hb = det2ind(nmob, tarb, occb, virb, self.nfc[1]) 
                exc = len(pa) + len(pb)
                if exc == 0: 
                    S[J,I] += 1.0
                if exc == 1: 
                    a,i = pa + pb + ha + hb
                    if len(pa) == 1:
                        S[J,I] += t1a[i,a]
                    if len(pa) == 0:
                        S[J,I] += t1b[i,a]
                if exc == 2: 
                    a,b,i,j = pa + pb + ha + hb
                    if len(pa) == 2:
                        S[J,I] += t2aa[i,j,a,b]+t1a[i,a]*t1a[j,b]-t1a[i,b]*t1a[j,a]
                    if len(pa) == 1:
                        S[J,I] += t2ab[i,j,a,b]+t1a[i,a]*t1b[j,b]
                    if len(pa) == 0:
                        S[J,I] += t2bb[i,j,a,b]+t1b[i,a]*t1b[j,b]-t1b[i,b]*t1b[j,a]
                H[J,I] += S[J,I]*eccsd[I]
        return H, S

    def get_coeff(self,state=0,t1a=None,t1b=None,t2aa=None,t2ab=None,t2bb=None,
                  vact=None,occ=None,vir=None):
        t1a = self.t1a if t1a is None else t1a
        t1b = self.t1b if t1b is None else t1b
        t2aa = self.t2aa if t2aa is None else t2aa
        t2ab = self.t2ab if t2ab is None else t2ab
        t2bb = self.t2bb if t2bb is None else t2bb
        v = self.vact[:,state] if vact is None else vact[:,state]
        occ = self.occ if occ is None else occ
        vir = self.vir if vir is None else vir
        nea, neb = self.cc._scf.mol.nelec
        nmoa, nmob = self.mo0[0].shape[1], self.mo0[1].shape[1]
        Na, Nb = num_strings(nmoa,nea), num_strings(nmob,neb)

        ndet = occ.shape[0]
        idx = np.zeros(ndet,dtype=int)
        ci = np.zeros(Na*Nb)
        for I in range(ndet):
            Ia, Ib = occ[I,:nea], occ[I,nea:]
            Ia, Ib = occ2str(Ia), occ2str(Ib)
            Ia, Ib = str2addr(nmoa,nea,Ia), str2addr(nmoa,nea,Ib)
            idx[I] = Ia*Na+Ib
        for I in range(ndet):
            _, ket = self.get_ci(t1a[I,:,:],t1b[I,:,:],
                                 t2aa[I,:,:,:,:],t2ab[I,:,:,:,:],t2bb[I,:,:,:,:],
                                 occ[I,:],vir[I,:])
            ci += v[I]*np.ravel(ket)
        ci /= np.linalg.norm(ci)
        print('MRCC state {} coeffs for important determinants:\n{}'.format(
              state,ci[idx]))
        return

#if __name__ == '__main__':
#    import copy
#    from pyscf import scf
#    from pyscf import gto
#
#    mol = gto.Mole()
#    mol.atom = [['O', (0.,   0., 0.)],
#                ['O', (1.21, 0., 0.)]]
#    mol.basis = 'cc-pvdz'
#    mol.spin = 2
#    mol.build()
#    mf = scf.UHF(mol).run()
#    # Freeze 1s electrons
#    # also acceptable
#    #frozen = 4 or [2,2]
#    frozen = [[0,1], [0,1]]
#    ucc = UCCSD(mf, frozen=frozen)
#    eris = ucc.ao2mo()
#    ecc, t1, t2 = ucc.kernel(eris=eris)
#    print(ecc - -0.3486987472235819)
#
#    mol = gto.Mole()
#    mol.atom = [
#        [8 , (0. , 0.     , 0.)],
#        [1 , (0. , -0.757 , 0.587)],
#        [1 , (0. , 0.757  , 0.587)]]
#    mol.basis = 'cc-pvdz'
#    mol.spin = 0
#    mol.build()
#    mf = scf.UHF(mol).run()
#
#    mycc = UCCSD(mf)
#    mycc.direct = True
#    ecc, t1, t2 = mycc.kernel()
#    print(ecc - -0.2133432712431435)
#    print(mycc.ccsd_t() - -0.003060021865720902)
#
#    e,v = mycc.ipccsd(nroots=8)
#    print(e[0] - 0.4335604332073799)
#    print(e[2] - 0.5187659896045407)
#    print(e[4] - 0.6782876002229172)
#
#    e,v = mycc.eaccsd(nroots=8)
#    print(e[0] - 0.16737886338859731)
#    print(e[2] - 0.24027613852009164)
#    print(e[4] - 0.51006797826488071)
#
#    e,v = mycc.eeccsd(nroots=4)
#    print(e[0] - 0.2757159395886167)
#    print(e[1] - 0.2757159395886167)
#    print(e[2] - 0.2757159395886167)
#    print(e[3] - 0.3005716731825082)
#
