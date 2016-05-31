import numpy as np
import shutil
import os.path
import os
from pyscf.pbc import cc as pbccc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_mpi_cc(cell, nmp):
    from scf import run_khf
    from cc import run_krccsd

    kpt = None
    mf = run_khf(cell, nmp=nmp, scaled_kshift=kpt, exxdiv=None)
    # Setting up everything for MPI after mean field
    comm.Barrier()
    mo_coeff  = comm.bcast(mf.mo_coeff,root=0)
    mo_energy = comm.bcast(mf.mo_energy,root=0)
    mo_occ    = comm.bcast(mf.mo_occ,root=0)
    kpts      = comm.bcast(mf.kpts,root=0)
    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy
    mf.mo_occ = mo_occ
    mf.kpts   = kpts
    comm.Barrier()
    # Done with setting up

    # Running ccsd
    cc = run_krccsd(mf)

def main():
    import sys
    sys.path.append('/home/jmcclain/pyscf/pyscf/pbc/examples/')
    from helpers import get_ase_atom, build_cell, get_bandpath_fcc

    args = sys.argv[1:]
    if len(args) != 5:
        print 'usage: formula basis nkx nky nkz'
        sys.exit(1)
    formula = args[0]
    bas = args[1]
    nmp = np.array([int(nk) for nk in args[2:5]])

    ase_atom = get_ase_atom(formula)
    cell = build_cell(ase_atom, ke=40.0, basis=bas, incore_anyway=True)
    run_mpi_cc(cell, nmp)

if __name__ == '__main__':
    main()
