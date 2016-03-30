from mpi4py import MPI
import pyscf.pbc.tools.pbc as tools
import pyscf.pbc.ao2mo
import pyscf.lib
import numpy

DEBUG = 0

class unique_pqr_list:
    #####################################################################################
    # The following only computes the integrals not related by permutational symmetries.
    # Wasn't sure how to do this 'cleanly', but it's fairly straightforward
    #####################################################################################
    def __init__(self,cell,kpts):
        kconserv = tools.get_kconserv(cell,kpts)
        nkpts = len(kpts)
        temp = range(0,nkpts)
        klist = pyscf.lib.cartesian_prod((temp,temp,temp))
        completed = numpy.zeros((nkpts,nkpts,nkpts),dtype=int)

        self.operations = numpy.zeros((nkpts,nkpts,nkpts),dtype=int)
        self.equivalentList = numpy.zeros((nkpts,nkpts,nkpts,3),dtype=int)
        self.nUnique = 0
        self.uniqueList = numpy.array([])

        ivec = 0
        not_done = True
        while( not_done ):
            current_kvec = klist[ivec]
            # check to see if it's been done...
            kp = current_kvec[0]
            kq = current_kvec[1]
            kr = current_kvec[2]
            #print "computing ",kp,kq,kr
            if completed[kp,kq,kr] == 0:
                self.nUnique += 1
                self.uniqueList = numpy.append(self.uniqueList,current_kvec)
                ks = kconserv[kp,kq,kr]

                # Now find all equivalent kvectors by permuting it all possible ways...
                # and then storing how its related by symmetry
                completed[kp,kq,kr] = 1
                self.operations[kp,kq,kr] = 0
                self.equivalentList[kp,kq,kr] = current_kvec.copy()

                completed[kr,ks,kp] = 1
                self.operations[kr,ks,kp] = 1 #.transpose(2,3,0,1)
                self.equivalentList[kr,ks,kp] = current_kvec.copy()

                completed[kq,kp,ks] = 1
                self.operations[kq,kp,ks] = 2 #numpy.conj(.transpose(1,0,3,2))
                self.equivalentList[kq,kp,ks] = current_kvec.copy()

                completed[ks,kr,kq] = 1
                self.operations[ks,kr,kq] = 3 #numpy.conj(.transpose(3,2,1,0))
                self.equivalentList[ks,kr,kq] = current_kvec.copy()

            ivec += 1
            if ivec == len(klist):
                not_done = False

        if MPI.COMM_WORLD.Get_rank == 0:
            print "(4symm) original klist size : %10d   new size : %10d" % (nkpts*nkpts*nkpts,self.nUnique)
        self.uniqueList = self.uniqueList.reshape(self.nUnique,-1)
        if DEBUG == 1:
            print "::: kpoint helper :::"
            print "kvector list (in)"
            print "   shape = ", klist.shape
            print "kvector list (out)"
            print "   shape  = ", self.uniqueList.shape
            print "   unique list ="
            print self.uniqueList
            print "transformation ="
            for i in range(klist.shape[0]):
                pqr = klist[i]
                irr_pqr = self.equivalentList[pqr[0],pqr[1],pqr[2]]
                print "%3d %3d %3d   ->  %3d %3d %3d" % (pqr[0],pqr[1],pqr[2],
                                                         irr_pqr[0],irr_pqr[1],irr_pqr[2])

        #
        # Now doing everything for integrals with only 2 symmetries...
        # the identity symmetry and (1,0,3,2) symmetry, useful for coupled
        # cluster-type intermediates that don't have the four-fold symmetry
        #

        completed = numpy.zeros((nkpts,nkpts,nkpts),dtype=int)

        self.sym2_operations = numpy.zeros((nkpts,nkpts,nkpts),dtype=int)
        self.sym2_equivalentList = numpy.zeros((nkpts,nkpts,nkpts,3),dtype=int)
        self.sym2_nUnique = 0
        self.sym2_uniqueList = numpy.array([])

        ivec = 0
        not_done = True
        while( not_done ):
            current_kvec = klist[ivec]
            # check to see if it's been done...
            kp = current_kvec[0]
            kq = current_kvec[1]
            kr = current_kvec[2]
            #print "computing ",kp,kq,kr
            if completed[kp,kq,kr] == 0:
                self.sym2_nUnique += 1
                self.sym2_uniqueList = numpy.append(self.sym2_uniqueList,current_kvec)
                ks = kconserv[kp,kq,kr]

                # Now find all equivalent kvectors by permuting it all possible ways...
                # and then storing how its related by symmetry
                completed[kp,kq,kr] = 1
                self.sym2_operations[kp,kq,kr] = 0
                self.sym2_equivalentList[kp,kq,kr] = current_kvec.copy()

                completed[kq,kp,ks] = 1
                self.sym2_operations[kq,kp,ks] = 2 #numpy.conj(.transpose(1,0,3,2))
                self.sym2_equivalentList[kq,kp,ks] = current_kvec.copy()

            ivec += 1
            if ivec == len(klist):
                not_done = False

        #print "(2symm) original klist size : %10d   new size : %10d" % (nkpts*nkpts*nkpts,self.sym2_nUnique)
        self.sym2_uniqueList = self.sym2_uniqueList.reshape(self.sym2_nUnique,-1)
        if DEBUG == 1:
            print "::: kpoint helper :::"
            print "kvector list (in)"
            print "   shape = ", klist.shape
            print "kvector list (out)"
            print "   shape  = ", self.sym2_uniqueList.shape
            print "   unique list ="
            print self.sym2_uniqueList
            print "transformation ="
            for i in range(klist.shape[0]):
                pqr = klist[i]
                irr_pqr = self.sym2_equivalentList[pqr[0],pqr[1],pqr[2]]
                print "%3d %3d %3d   ->  %3d %3d %3d" % (pqr[0],pqr[1],pqr[2],
                                                         irr_pqr[0],irr_pqr[1],irr_pqr[2])

    def get_uniqueList(self):
        return self.uniqueList

    def get_irrVec(self,kp,kq,kr):
        return self.equivalentList[kp,kq,kr]

    def get_transformation(self,kp,kq,kr):
        return self.operations[kp,kq,kr]

    ######################################################
    # for the invec created out of our unique list from
    # the irreducible brillouin zone, we transform it to
    # arbitrary kp,kq,kr
    ######################################################
    def transform_irr2full(self,invec,kp,kq,kr):
        operation = self.get_transformation(kp,kq,kr)
        if operation == 0:
            return invec
        if operation == 1:
            return invec.transpose(2,3,0,1)
        if operation == 2:
            return numpy.conj(invec.transpose(1,0,3,2))
        if operation == 3:
            return numpy.conj(invec.transpose(3,2,1,0))

    def get_sym2_uniqueList(self):
        return self.sym2_uniqueList

    def get_sym2_irrVec(self,kp,kq,kr):
        return self.sym2_equivalentList[kp,kq,kr]

    def get_sym2_transformation(self,kp,kq,kr):
        return self.sym2_operations[kp,kq,kr]

    def sym2_transform_irr2full(self,invec,kp,kq,kr):
        operation = self.get_sym2_transformation(kp,kq,kr)
        if operation == 0:
            return invec
        if operation == 1:
            return invec.transpose(2,3,0,1)
        if operation == 2:
            return numpy.conj(invec.transpose(1,0,3,2))
        if operation == 3:
            return numpy.conj(invec.transpose(3,2,1,0))

    def get_sym2_irrVec_physics(self,kp,kq,kr):
        return self.sym2_equivalentList[kp,kr,kq]

    def sym2_transform_irr2full_physics(self,invec,kp,kq,kr):
        operation = self.get_sym2_transformation(kp,kr,kq)
        if operation == 0:
            return invec
        if operation == 2:
            return invec.transpose(1,0,3,2)
