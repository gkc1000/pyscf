import sys
import numpy as np
import scipy.linalg
from pyscf import lib

import pyfftw


def fft(f, gs):
    '''Perform the 3D FFT from real (R) to reciprocal (G) space.

    Re: MH (3.25), we assume Ns := ngs = 2*gs+1

    After FFT, (u, v, w) -> (j, k, l).
    (jkl) is in the index order of Gv.

    FFT normalization factor is 1., as in MH and in `numpy.fft`.

    Args:
        f : (nx*ny*nz,) ndarray
            The function to be FFT'd, flattened to a 1D array corresponding
            to the index order of :func:`cartesian_prod`.
        gs : (3,) ndarray of ints
            The number of *positive* G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The FFT 1D array in same index order as Gv (natural order of
            numpy.fft).

    '''
    ngs = 2*np.asarray(gs)+1
    f3d = np.reshape(f, ngs)
    #g3d = np.fft.fftn(f3d)
    pyfftw.interfaces.cache.enable()
    g3d = pyfftw.interfaces.numpy_fft.fftn(f3d,
            overwrite_input=True,
            planner_effort='FFTW_MEASURE',
            #threads=2,
            auto_align_input=False)
    return np.ravel(g3d)


def ifft(g, gs):
    '''Perform the 3D inverse FFT from reciprocal (G) space to real (R) space.

    Inverse FFT normalization factor is 1./N, same as in `numpy.fft` but
    **different** from MH (they use 1.).

    Args:
        g : (nx*ny*nz,) ndarray
            The function to be inverse FFT'd, flattened to a 1D array
            corresponding to the index order of `span3`.
        gs : (3,) ndarray of ints
            The number of *positive* G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The inverse FFT 1D array in same index order as Gv (natural order
            of numpy.fft).

    '''
    ngs = 2*np.asarray(gs)+1
    g3d = np.reshape(g, ngs)
    #f3d = np.fft.ifftn(g3d)
    pyfftw.interfaces.cache.enable()
    f3d = pyfftw.interfaces.numpy_fft.ifftn(g3d,
            overwrite_input=True,
            planner_effort='FFTW_MEASURE',
            #threads=2,
            auto_align_input=False)
    return np.ravel(f3d)


def fftk(f, gs, expmikr):
    '''Perform the 3D FFT of a real-space function which is (periodic*e^{ikr}).

    fk(k+G) = \sum_r fk(r) e^{-i(k+G)r} = \sum_r [f(k)e^{-ikr}] e^{-iGr}
    '''
    return fft(f*expmikr, gs)


def ifftk(g, gs, expikr):
    '''Perform the 3D inverse FFT of f(k+G) into a function which is (periodic*e^{ikr}).

    fk(r) = (1/Ng) \sum_G fk(k+G) e^{i(k+G)r} = (1/Ng) \sum_G [fk(k+G)e^{iGr}] e^{ikr}
    '''
    return ifft(g, gs) * expikr


def get_coulG(cell, k=np.zeros(3), exx=False, mf=None, G0eq0=True):
    '''Calculate the Coulomb kernel for all G-vectors, handling G=0 and exchange.

    Args:
        cell : instance of :class:`Cell`
        k : (3,) ndarray
        exx : bool
            Whether this is an exchange matrix element.
        mf : instance of :class:`SCF`

    Returns:
        coulG : (ngs,) ndarray
            The Coulomb kernel.

    '''
    kG = k + cell.Gv

    # Here we 'wrap around' the high frequency k+G vectors into their lower
    # frequency counterparts.  Important if you want the gamma point and k-point
    # answers to agree
    box_edge = np.dot(2.*np.pi*np.diag(cell.gs+0.5), np.linalg.inv(cell._h))
    reduced_coords = np.dot(kG, np.linalg.inv(box_edge))
    equal2boundary = np.where( abs(abs(reduced_coords) - 1.) < 1e-14 )[0]
    factor = np.trunc(reduced_coords)
    kG -= 2.*np.dot(np.sign(factor), box_edge)
    #kG[equal2boundary] = [0.0, 0.0, 0.0]
    # Done wrapping.

    absG2 = np.einsum('gi,gi->g', kG, kG)

    try:
        kpts = mf.kpts
    except AttributeError:
        kpts = k.reshape(1,3)
    Nk = len(kpts)

    if exx is False or mf.exxdiv is None:
        if cell.dimension == 1:
            if mf.exx_built == False:
                mf.precompute_exx1D()
            with np.errstate(divide='ignore',invalid='ignore'):
                coulG = 4*np.pi/absG2*(1.0 - np.exp(-absG2/(4*mf.exx_alpha**2))) + 0j
            if np.linalg.norm(k) < 1e-8:
                coulG[0] = np.pi / mf.exx_alpha**2

            # Index k+cell.Gv into the precomputed vq and add on
            gxyz = np.round(np.dot(kG, mf.exx_kcell.h)/(2*np.pi)).astype(int)
            ngs  = 2*mf.exx_kcell.gs+1
            gxyz = (gxyz + ngs)%(ngs)
            old_qidx = (gxyz[:,0]*ngs[1] + gxyz[:,1])*ngs[2] + gxyz[:,2]
            for igz,gz in enumerate(np.unique(kG[:,2].round(decimals=14))):
                # for a given gz, finds all kG with kGz = gz
                kGz_loc = [ abs(kG[:,2] - gz)<1e-10 ]
                # finds the index in our convolution list of the given gz
                exx_loc = np.where( abs(mf.exx_gz - abs(gz))<1e-10 )[0][0]
                qidx = old_qidx[ kGz_loc ]
                coulG[ kGz_loc ] += mf.exx_vq[ exx_loc, qidx ]

                #if exx is True:
                #    if np.linalg.norm(k) < 1e-8 and abs(gz) < 1e-8:
                #        coulG[0] = 0.0

                if G0eq0 is True:
                    if np.linalg.norm(k) < 1e-8 and abs(gz) < 1e-8:
                        coulG[0] = 0.0

            # The following is an implementation of Rozzi's cylindrical cut-off
            #
            # NOTE: This hasn't been implemented in the ewald treatment, so it isn't
            #       too useful besides to check the above Wigner-Seitz method...

            #Rin = min(cell._h[0,0],cell._h[1,1])/2.
            #Gz = kG[:,2]
            #Gp = np.array([np.linalg.norm(x) for x in kG[:,:2]])
            #Gz_N0 = abs(Gz) > 0
            #coulG = 0.0*absG2
            #with np.errstate(divide='ignore',invalid='ignore'):
            #    coulG[ Gz_N0 ] = 4*np.pi/absG2[ Gz_N0 ] * ( 1. + Gp[ Gz_N0 ]*Rin*scipy.special.jn(1,Gp[ Gz_N0 ]*Rin)*scipy.special.kn(0,abs(Gz[ Gz_N0 ])*Rin)
            #                                              - abs(Gz[ Gz_N0 ])*Rin*scipy.special.jn(0,Gp[ Gz_N0 ]*Rin)*scipy.special.kn(1,abs(Gz[ Gz_N0 ])*Rin))
            #for i in np.where(Gz_N0==False)[0]:
            #    if abs(Gz[i]) + Gp[i] > 1e-8:
            #        def func(x):
            #            return x*scipy.special.jn(0,Gp[i]*x)*np.log(x)
            #        coulG[ i ] = - 4.*np.pi*scipy.integrate.quad(func,1e-6,Rin)[0]

            #if np.linalg.norm(k) < 1e-8:
            #    #if G0eq0 is True:
            #    #    coulG[0] = 0.0
            #    #else:
            #    coulG[0] = -np.pi * Rin**2 * ( 2.*np.log(Rin) - 1 )
        elif cell.dimension == 2:
            L = cell._h[2,2]
            Gz = kG[:,2]
            Gp = np.array([np.linalg.norm(x) for x in kG[:,:2]])
            with np.errstate(divide='ignore',invalid='ignore'):
                coulG = 4*np.pi/absG2 * ( 1. - np.cos(Gz*L/2.)*np.exp(-Gp*L/2.) )
            if np.linalg.norm(k) < 1e-8:
                coulG[0] = -np.pi * L**2 / 2.
        else:
            with np.errstate(divide='ignore'):
                coulG = 4*np.pi/absG2
            if np.linalg.norm(k) < 1e-8:
                coulG[0] = 0.
    elif mf.exxdiv == 'vcut_sph':
        Rc = (3*Nk*cell.vol/(4*np.pi))**(1./3)
        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*np.pi/absG2*(1.0 - np.cos(np.sqrt(absG2)*Rc))
        if np.linalg.norm(k) < 1e-8:
            coulG[0] = 4*np.pi*0.5*Rc**2
    elif mf.exxdiv == 'ewald':
        if cell.dimension == 1:
            if mf.exx_built == False:
                mf.precompute_exx1D()
            with np.errstate(divide='ignore',invalid='ignore'):
                coulG = 4*np.pi/absG2*(1.0 - np.exp(-absG2/(4*mf.exx_alpha**2))) + 0j
            if np.linalg.norm(k) < 1e-8:
                coulG[0] = np.pi / mf.exx_alpha**2

            # Index k+cell.Gv into the precomputed vq and add on
            gxyz = np.round(np.dot(kG, mf.exx_kcell.h)/(2*np.pi)).astype(int)
            ngs  = 2*mf.exx_kcell.gs+1
            gxyz = (gxyz + ngs)%(ngs)
            old_qidx = (gxyz[:,0]*ngs[1] + gxyz[:,1])*ngs[2] + gxyz[:,2]
            for igz,gz in enumerate(np.unique(kG[:,2].round(decimals=14))):
                # for a given gz, finds all kG with kGz = gz
                kGz_loc = [ abs(kG[:,2] - gz)<1e-10 ]
                # finds the index in our convolution list of the given gz
                exx_loc = np.where( abs(mf.exx_gz - abs(gz))<1e-10 )[0][0]
                qidx = old_qidx[ kGz_loc ]
                coulG[ kGz_loc ] += mf.exx_vq[ exx_loc, qidx ]

                if np.linalg.norm(k) < 1e-8 and abs(gz) < 1e-8:
                    print coulG[0], madelung(cell,kpts, alpha = mf.exx_alpha), cell.vol, Nk
                    coulG[0] += Nk*cell.vol*madelung(cell, kpts, alpha = mf.exx_alpha)
        elif cell.dimension == 2:
            raise NotImplementedError
        else:
            with np.errstate(divide='ignore'):
                coulG = 4*np.pi/absG2
            if np.linalg.norm(k) < 1e-8:
                #print "madelung ", madelung(cell, kpts)
                #print "madelung ", madelung(cell, kpts)*Nk
                #print "madelung ", madelung(cell, kpts)*Nk*cell.vol
                coulG[0] = Nk*cell.vol*madelung(cell, kpts)
    elif mf.exxdiv == 'vcut_ws':
        if mf.exx_built == False:
            mf.precompute_exx()
        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*np.pi/absG2*(1.0 - np.exp(-absG2/(4*mf.exx_alpha**2))) + 0j
        if np.linalg.norm(k) < 1e-8:
            coulG[0] = np.pi / mf.exx_alpha**2
        # Index k+cell.Gv into the precomputed vq and add on
        gxyz = np.round(np.dot(kG, mf.exx_kcell.h)/(2*np.pi)).astype(int)
        ngs = 2*mf.exx_kcell.gs+1
        gxyz = (gxyz + ngs)%(ngs)
        qidx = (gxyz[:,0]*ngs[1] + gxyz[:,1])*ngs[2] + gxyz[:,2]
        #qidx = [np.linalg.norm(mf.exx_q-kGi,axis=1).argmin() for kGi in kG]
        maxqv = abs(mf.exx_q).max(axis=0)
        is_lt_maxqv = (abs(kG) <= maxqv).all(axis=1)
        coulG += mf.exx_vq[qidx] * is_lt_maxqv

    coulG[ coulG == np.inf ] = 0.0
    coulG[ equal2boundary ] = 0.0

    return coulG


def madelung(cell, kpts, alpha=None):
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc.scf.hf import ewald

    Nk = get_monkhorst_pack_size(cell, kpts)
    ecell = pbcgto.Cell()
    ecell.atom = 'H 0. 0. 0.'
    ecell.spin = 1
    ecell.gs = cell.gs
    ecell.precision = 1e-16
    ecell.unit = 'B'
    ecell.h = cell._h * Nk
    ecell.build(False,False)
    if alpha is None:
        alpha = ecell.ew_eta
    return -2*ewald(ecell, alpha, ecell.ew_cut)

def get_monkhorst_pack_size(cell, kpts):
    skpts = cell.get_scaled_kpts(kpts).round(decimals=6)
    Nk = np.array([len(np.unique(ki)) for ki in skpts.T])
    return Nk

def f_aux(cell, q):
    a = cell._h.T
    b = 2*np.pi*scipy.linalg.inv(cell._h)
    denom = 4 * np.dot(b*np.sin(a*q/2.), b*np.sin(a*q/2.)) \
          + 2 * np.dot(b*np.sin(a*q), np.roll(b,1,axis=0)*np.sin(np.roll(a,1,axis=0)*q ))
    return 1./(2*np.pi)**2 * 1./denom


def get_lattice_Ls(cell, nimgs):
    '''Get the (Cartesian, unitful) lattice translation vectors for nearby images.'''
    Ts = [[i,j,k] for i in range(-nimgs[0],nimgs[0]+1)
                  for j in range(-nimgs[1],nimgs[1]+1)
                  for k in range(-nimgs[2],nimgs[2]+1)
                  if i**2+j**2+k**2 <= 1./3*np.dot(nimgs,nimgs)]
    Ts = np.array(Ts)
    Ls = np.dot(cell._h, Ts.T).T
    return Ls


def super_cell(cell, ncopy):
    '''Create an ncopy[0] x ncopy[1] x ncopy[2] supercell of the input cell

    Args:
        cell : instance of :class:`Cell`
        ncopy : (3,) array

    Returns:
        supcell : instance of :class:`Cell`
    '''
    supcell = cell.copy()
    supcell.atom = []
    for Lx in range(ncopy[0]):
        for Ly in range(ncopy[1]):
            for Lz in range(ncopy[2]):
                # Using cell._atom guarantees coord is in Bohr
                for atom, coord in cell._atom:
                    L = np.dot(cell._h, [Lx, Ly, Lz])
                    supcell.atom.append([atom, coord + L])
    supcell.unit = 'B'
    supcell.h = np.dot(cell._h, np.diag(ncopy))
    supcell.gs = np.array([ncopy[0]*cell.gs[0] + (ncopy[0]-1)//2,
                           ncopy[1]*cell.gs[1] + (ncopy[1]-1)//2,
                           ncopy[2]*cell.gs[2] + (ncopy[2]-1)//2])
    supcell.build(False, False)
    return supcell

kconserver = None

@profile
def get_kconserv3(cell, kpts, kijkab):
    '''Get the momentum conservation array for a set of k-points.

    Given k-point indices (k, l, m) the array kconserv[k,l,m] returns
    the index n that satifies momentum conservation,

       k(k) - k(l) = - k(m) + k(n)

    This is used for symmetry e.g. integrals of the form
        [\phi*[k](1) \phi[l](1) | \phi*[m](2) \phi[n](2)]
    are zero unless n satisfies the above.
    '''
    nkpts = kpts.shape[0]
    KLMN = np.zeros([nkpts,nkpts,nkpts], np.int)
    kvecs = 2*np.pi*scipy.linalg.inv(cell._h)
    kijkab = np.array(kijkab)

    idx_sum = np.array([not(isinstance(x,int) or isinstance(x,np.int)) for x in kijkab])
    idx_range = kijkab[idx_sum]
    min_idx_range = np.zeros(5,dtype=int)
    min_idx_range = np.array([min(x) for x in idx_range])
    out_array_shape = tuple([len(x) for x in idx_range])
    out_array = np.zeros(shape=out_array_shape,dtype=int)
    #kijkab = kijkab[0:3].sum(axis=0) - kijkab[3:5].sum(axis=0)

    #print kijkab

    #for k in lib.cartesian_prod(klist):
    kpqrst_idx = np.zeros(5,dtype=int)

    # Order here matters! Speedup of around 50x when going from
    # [-2,-1,...,2] to [0,-1,1,-2,2]
    temp = [0,-1,1,-2,2]
    xyz = lib.cartesian_prod((temp,temp,temp))
    kshift = np.dot(xyz,kvecs)

    for L, kvL in enumerate(lib.cartesian_prod(idx_range)):
        kpqrst_idx[idx_sum], kpqrst_idx[~idx_sum] = kvL, kijkab[~idx_sum]
        idx = tuple(kpqrst_idx[idx_sum]-min_idx_range)

        kvec = kpts[kpqrst_idx]
        kvec = kvec[0:3].sum(axis=0) - kvec[3:5].sum(axis=0)

        found = 0
        kvNs = kvec + kshift
        for ishift in xrange(len(xyz)):
            #kvN = kvec + np.dot(xyz[ishift],kvecs)
            #kvN = kvec + kshift[ishift]
            kvN = kvNs[ishift]
            finder = np.where(np.logical_and(kpts < kvN + 1.e-12, kpts > kvN - 1.e-12).sum(axis=1)==3)
            # The k-point should be the same in all 3 indices as kvN
            if len(finder[0]) > 0:
                found = 1
                out_array[idx] = finder[0][0]
                break

        if found == 0:
            print "** ERROR: Problem in get_kconserv. Quitting."
            print kijkab
            sys.exit()
    return out_array

def get_kconserv(cell, kpts):
    '''Get the momentum conservation array for a set of k-points.

    Given k-point indices (k, l, m) the array kconserv[k,l,m] returns
    the index n that satifies momentum conservation,

       k(k) - k(l) = - k(m) + k(n)

    This is used for symmetry e.g. integrals of the form
        [\phi*[k](1) \phi[l](1) | \phi*[m](2) \phi[n](2)]
    are zero unless n satisfies the above.
    '''
    nkpts = kpts.shape[0]
    KLMN = np.zeros([nkpts,nkpts,nkpts], np.int)
    kvecs = 2*np.pi*scipy.linalg.inv(cell._h)

    for K, kvK in enumerate(kpts):
        for L, kvL in enumerate(kpts):
            for M, kvM in enumerate(kpts):
                # Here we find where kvN = kvM + kvL - kvK (mod K)
                temp = range(-1,2)
                xyz = lib.cartesian_prod((temp,temp,temp))
                found = 0
                kvMLK = kvK - kvL + kvM
                kvN = kvMLK
                for ishift in xrange(len(xyz)):
                    kvN = kvMLK + np.dot(xyz[ishift],kvecs)
                    finder = np.where(np.logical_and(kpts < kvN + 1.e-12,
                                                     kpts > kvN - 1.e-12).sum(axis=1)==3)
                    # The k-point should be the same in all 3 indices as kvN
                    if len(finder[0]) > 0:
                        KLMN[K, L, M] = finder[0][0]
                        found = 1
                        break

                if found == 0:
                    print "** ERROR: Problem in get_kconserv. Quitting."
                    print kvMLK
                    sys.exit()
    return KLMN


def cutoff_to_gs(h, cutoff):
    '''
    Convert KE cutoff to #grid points (gs variable)

        uses KE = k^2 / 2, where k_max ~ \pi / grid_spacing

    Args:
        h : (3,3) ndarray
            The unit cell lattice vectors, a "three-column" array [a1|a2|a3], in Bohr
        cutoff : float
            KE energy cutoff in a.u.

    Returns:
        gs : (3,) array
    '''
    grid_spacing = np.pi / np.sqrt(2 * cutoff)

    #print grid_spacing
    #print h

    h0 = np.linalg.norm(h[:,0])
    h1 = np.linalg.norm(h[:,1])
    h2 = np.linalg.norm(h[:,2])

    #print h0, h1, h2
    # number of grid points is 2gs+1 (~ 2 gs) along each direction
    gs = np.ceil([h0 / (2*grid_spacing),
                  h1 / (2*grid_spacing),
                  h2 / (2*grid_spacing)])
    return gs.astype(int)

