#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

'''Non-relativistic and relativistic Hartree-Fock
   for periodic systems at a *single* k-point.

'''

from pyscf.pbc.scf_test import hf
from pyscf.pbc.scf_test import hf as rhf

def RHF(mol, *args, **kwargs):
    '''This is a wrap function to mimic pyscf
    '''
    return rhf.RHF(mol, *args, **kwargs)

def KRHF(mol, *args, **kwargs):
    '''This is a wrap function to mimic pyscf
    '''
    from pyscf.pbc.scf_test import khf
    from pyscf.pbc.scf_test import khf as krhf
    return krhf.KRHF(mol, *args, **kwargs)
