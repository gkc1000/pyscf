from pyscf import gto, scf, cc, fci
from pyscf.cc import mrccsd 
from math import radians, sin, cos

def test():
    re = 0.9929
    theta = radians(109.57)
    frozen = 1

    r = 1.*re
    mol = gto.Mole()
    mol.atom = [['O', (0.0, 0.0, 0.0)],
                ['H', (  r, 0.0, 0.0)], 
                ['H', (r*cos(theta),r*sin(theta),0.0)]] 
    mol.verbose = 4
    mol.symmetry = True
    mol.basis = '6-31g'
    mol.unit = 'Angstrom'
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 100
    mf.kernel()
    mf.analyze()
    
    cisolver = fci.FCI(mf)
    cisolver.nroots = 1
    cisolver.kernel()

    # mf = mf.to_uhf()
    # print mf.mo_occ

    # mo_occ = mf.mo_occ
    # mo_occ[0][4] = 0.
    # mo_occ[1][4] = 0.
    # mo_occ[0][9] = 1.
    # mo_occ[1][9] = 1.
    # mycc = cc.UCCSD(mf, mo_coeff=mf.mo_coeff, mo_occ=mo_occ)
    # mycc.kernel()

    # mo_occ = mf.mo_occ
    # mo_occ[0][2] = 0.
    # mo_occ[1][2] = 0.
    # mo_occ[0][6] = 1.
    # mo_occ[1][6] = 1.
    # mycc = cc.CCSD(mf, mo_coeff=mf.mo_coeff, mo_occ=mo_occ)
    # mycc.kernel()

    # ddd
    # top 16
    # dets = [[ 0  1  2  3  4  0  1  2  3  4]
    #         [ 0  1  2  3  9  0  1  2  3  9]
    #         [ 0  1  3  4  6  0  1  3  4  6]
    #         [ 0  1  2  4 10  0  1  2  4 10]
    #         [ 0  1  2  4 10  0  1  2  3  9]
    #         [ 0  1  2  3  9  0  1  2  4 10]
    #         [ 0  1  3  4  7  0  1  3  4  7]
    #         [ 0  1  3  4  8  0  1  3  4  8]
    #         [ 0  1  2  4  5  0  1  2  4  5]
    #         [ 0  1  3  4 11  0  1  2  3  9]
    #         [ 0  1  2  3  9  0  1  3  4 11]
    #         [ 0  1  3  4  6  0  1  2  3  9]
    #         [ 0  1  2  3  9  0  1  3  4  6]
    #         [ 0  1  3  4 11  0  1  3  4 11]
    #         [ 0  1  3  4  6  0  1  2  4  5]
    #         [ 0  1  2  4  5  0  1  3  4  6]]

    #mf = mf.to_uhf()
    mo_occ = mf.mo_occ
    mycc = cc.CCSD(mf, mo_coeff=mf.mo_coeff, mo_occ=mo_occ)
    mycc.level_shift = 0.3
    mycc.kernel()

    
    mo_occ = mf.mo_occ
    mo_occ[4] = 0.
    mo_occ[9] = 2.
    mycc = cc.CCSD(mf, mo_coeff=mf.mo_coeff, mo_occ=mo_occ)
    mycc.level_shift = 0.3
    mycc.kernel()

    mo_occ = mf.mo_occ
    mo_occ[2] = 0.
    mo_occ[6] = 2.
    mycc = cc.CCSD(mf, mo_coeff=mf.mo_coeff, mo_occ=mo_occ)
    mycc.max_cycle = 100
    mycc.level_shift = 1.
    mycc.kernel()

    # mo_occ = mf.mo_occ
    # mo_occ[3] = 0.
    # mo_occ[10] = 2.
    # mycc = cc.CCSD(mf, mo_coeff=mf.mo_coeff, mo_occ=mo_occ)
    # mycc.kernel()

    # ddd

    # mf = mf.to_uhf()
    # mr = mrccsd.MRCCSD(mf, ci=cisolver.ci, ndet=16, frozen=frozen)
    # mr.cc.conv_tol = 1e-8
    # mr.cc.conv_tol_normt = 1e-5
    # mr.cc.max_cycle = 100
    # mr.mbpt2 = False
    # eccsd = mr.ccsd()
