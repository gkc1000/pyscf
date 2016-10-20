import numpy as np
import shutil
import os.path
import sys
import os
from pyscf.pbc import cc as pbccc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_krccsd(mf):
    cc = pbccc.KRCCSD(mf, mf.kpts)
    cc.verbose = 7
    #cc.ccsd()
    cc.kernel()
    return cc

def run_ip_krccsd(cc, nroots=3, klist=None):
    e,c = cc.ipccsd(nroots, klist)
    comm.Barrier()
    return e,c

def run_ea_krccsd(cc, nroots=3, klist=None):
    e,c = cc.eaccsd(nroots, klist)
    comm.Barrier()
    return e,c

def run_eom_krccsd_bands(cell, nmp, kpts_red):
    from scf import run_khf
    e_kn = []
    qp_kn = []
    vbmax = -99
    cbmin = 99
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    for kpt in kpts_red:
        mf = run_khf(cell, nmp=nmp, scaled_kshift=kpt, gamma=True, exxdiv=None)
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
        nocc = cc.nocc()
        nvir = cc.nmo() - nocc

        eip,cip = run_ip_krccsd(cc, klist=[0])
        eip, cip = eip[0], cip[0]
        qpip = np.linalg.norm(cip[:nocc],axis=0)**2

        #if os.path.isfile("eom_intermediates_IP.hdf5") is True:
        #    os.remove("eom_intermediates_IP.hdf5")

        eea,cea = run_ea_krccsd(cc, klist=[0])
        eea, cea = eea[0], cea[0]
        qpea = np.linalg.norm(cea[:nvir],axis=0)**2

        e_kn.append( np.append(-eip[::-1], eea) )
        qp_kn.append( np.append(qpip[::-1], qpea) )
        if rank == 0:
            filename = "kpt_%.4f_%.4f_%.4f-band.dat"%(kpt[0], kpt[1], kpt[2])
            f = open(filename,'w')
            f.write("# IP\n")
            for ekn, qpkn in zip(eip,qpip):
                f.write("%0.6f %0.6f \n"%(ekn, qpkn))
            f.write("# EA \n")
            for ekn, qpkn in zip(eea,qpea):
                f.write("%0.6f %0.6f \n"%(ekn, qpkn))
            f.write("\n")
            f.close()
        if np.max(-eip) > vbmax:
            vbmax = np.max(-eip)
        if np.min(eea) < cbmin:
            cbmin = np.min(eea)

        cc = None

        if rank == 0:
            if os.path.isfile("eris1.hdf5") is True:
                os.remove("eris1.hdf5")
            if os.path.isfile("__ip_dvdson__.hdf5") is True:
                os.remove("__ip_dvdson__.hdf5")
            if os.path.isfile("__ea_dvdson__.hdf5") is True:
                os.remove("__ea_dvdson__.hdf5")
                #shutil.rmtree('./tmp')
        comm.Barrier()

    for k, ek in enumerate(e_kn):
        e_kn[k] = ek-vbmax
    bandgap = cbmin - vbmax
    return e_kn, qp_kn, bandgap

def read_eom_krccsd_bands(cell, nmp, kpts_red):
    from scf import run_khf
    e_kn = []
    qp_kn = []
    vbmax = -99
    cbmin = 99

    for kpt in kpts_red:
        eip,cip = run_ip_krccsd(cc, klist=[0])
        eip, cip = eip[0], cip[0]
        qpip = np.linalg.norm(cip[:nocc],axis=0)**2

        eea,cea = run_ea_krccsd(cc, klist=[0])
        eea, cea = eea[0], cea[0]
        qpea = np.linalg.norm(cea[:nvir],axis=0)**2

        e_kn.append( np.append(-eip[::-1], eea) )
        qp_kn.append( np.append(qpip[::-1], qpea) )
        if rank == 0:
            filename = "kpt_%.4f_%.4f_%.4f-band.dat"%(kpt[0], kpt[1], kpt[2])
            f = open(filename,'r')
            lines = f.readlines()
            f.write("# IP\n")
            for ekn, qpkn in zip(eip,qpip):
                f.write("%0.6f %0.6f \n"%(ekn, qpkn))
            f.write("# EA \n")
            for ekn, qpkn in zip(eea,qpea):
                f.write("%0.6f %0.6f \n"%(ekn, qpkn))
            f.write("\n")
            f.close()
        if np.max(-eip) > vbmax:
            vbmax = np.max(-eip)
        if np.min(eea) < cbmin:
            cbmin = np.min(eea)

        cc = None

        if rank == 0:
            if os.path.isfile("eris1.hdf5") is True:
                os.remove("eris1.hdf5")
                #shutil.rmtree('./tmp')
        comm.Barrier()

    for k, ek in enumerate(e_kn):
        e_kn[k] = ek-vbmax
    bandgap = cbmin - vbmax
    return e_kn, qp_kn, bandgap

def main():
    import sys
    sys.path.append('/home/jmcclain/pyscf/pyscf/pbc/examples/')
    from helpers import get_ase_atom, build_cell, get_bandpath_fcc

    args = sys.argv[1:]
    if len(args) != 5 and len(args) != 7:
        print 'usage: formula basis nkx nky nkz [start_band end_band]'
        sys.exit(1)
    formula = args[0]
    bas = args[1]
    nmp = np.array([int(nk) for nk in args[2:5]])
    start_band = 0
    end_band = 30
    if len(args) == 7:
        start_band = int(args[5])
        end_band =   int(args[6])

    ase_atom = get_ase_atom(formula)
    cell = build_cell(ase_atom, ke=40.0, basis=bas, incore_anyway=True)

    kpts_red, kpts_cart, kpath, sp_points = get_bandpath_fcc(ase_atom,npoints=30)

    e_kn, qp_kn, bandgap = run_eom_krccsd_bands(cell, nmp, kpts_red[start_band:end_band,:])

    if rank == 0:
        filename = "%s_%s_%d%d%d-bands.dat"%(formula.lower(), bas[4:],
                                             nmp[0], nmp[1], nmp[2])
        f = open(filename,'w')
        f.write("# Bandgap = %0.6f au = %0.6f eV\n"%(bandgap, bandgap*27.2114))
        f.write("# Special points:\n")
        for point, label in zip(sp_points,['L', 'G', 'X', 'W', 'K', 'G']):
            f.write("# %0.6f %s\n"%(point,label))
        for kk, ek, qpk in zip(kpath, e_kn, qp_kn):
            f.write("%0.6f "%(kk))
            for ekn, qpkn in zip(ek,qpk):
                f.write("%0.6f %0.6f "%(ekn, qpkn))
            f.write("\n")
        f.close()

if __name__ == '__main__':
    main()
