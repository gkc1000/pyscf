import sys
import scipy.misc
import numpy as np
import scipy.special
import scipy.optimize
import scipy.integrate
import numpy.polynomial.laguerre
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
#import matplotlib.pyplot as plt

T = np.identity(3)
T[0,0] = 26.27579633
T[1,1] = 18.89725989
T[2,2] = 4.26

G = np.linalg.inv(T) * 2. * np.pi
exx_alpha = 1.7

nd128 = 9./128
od8   = 1./8

# alpha here is fixed along with the G_z vector
class conv1Dfunc:
    def __init__(self,alpha,Gvecs,rhoVals,tol=8,kpts=np.array([[0,0,0],]),smoothing=True,method="slow_quad"):
        self.alpha = alpha
        self.Gvecs  = Gvecs
        self.nGvecs = Gvecs.shape[0]
        self.method = method
        self.createdMatrix = False
        self.tol=10**(-tol)
        self.kpts = kpts.copy()
        self.smoothing = smoothing

        self.createFullCmatrix(rhoVals)

    # Originally used a fitting function for the tail of the Cmatrix, but
    # the scipy fit functions were pretty terrible, as was the fit.
    # TODO: work out the asymptotics for large rho of the integral (should be
    # somewhat tedious but straightforward)
    def createTailCmatrix(self,rhoList):
        raise NotImplementedError

    def createBodyCmatrix(self,rhoList):
        bodyRhoList = np.array(rhoList)
        self.bodyRhoList = bodyRhoList.copy()
        self.bodyCakFit = []
        for ig,gz in enumerate(self.gz):
            print "creating Cmatrix for gz = ", gz
            cakVals = numpy.zeros(bodyRhoList.shape[0])
            self.lastHeadValZero = False
            self.lastValZero = False
            # Special analytical case for |Gz| == 0
            if abs(gz) < 1e-14:
                cakVals = self.cak0(self.alpha,bodyRhoList)
            else:
                for ir,rho in enumerate(bodyRhoList):
                    if self.lastValZero is True:
                        cakVals[ir:] = 0.0
                        break
                    cakVals[ir] = self.evalRho(self.alpha,gz,rho)
                    if abs(cakVals[ir]) < 1e-14:
                        self.lastValZero = True
                    #print "rho, val = %12.8e %12.8e" % (rho,cakVals[ir])
            if self.smoothing is True:
                self.bodyCakFit.append(InterpolatedUnivariateSpline(bodyRhoList,cakVals))
            else:
                self.bodyCakFit.append(cakVals)
            #plt.plot(bodyRhoList,self.bodyCakFit[ig](bodyRhoList))
            #plt.show()
        self.createdMatrix = True
        return

    def createFullCmatrix(self,rhoList):
        self.gz = numpy.unique(self.Gvecs[:,2])
        self.gz = self.gz.reshape(-1,1)

        self.kz = numpy.unique(self.kpts[:,2]).reshape(-1,1)
        self.kz = self.kz.reshape(1,-1)
        self.kz = self.kz - self.kz.T
        self.kz = self.kz.reshape(1,-1)

        self.gz = numpy.unique(abs(self.kz+self.gz).round(decimals=14))
        print "unique |Gz| vectors = "
        print self.gz
        self.createBodyCmatrix(rhoList)
        return

    def getCmatrixAtGz(self,gz,rho):
        assert self.createdMatrix is True and "Must createFullCMatrix before calling getCmatrix"
        ig = np.where(abs(self.gz-abs(gz))<1e-10)[0]
        if ig.shape[0] == 0:
            print "Error: createFullCmatrix needs to be called initially with all Gvectors as input!"
            print "   initialized Gvectors = ", self.gz
            print "   input Gvector = ", gz
            sys.exit()
        if self.smoothing is True:
            return self.bodyCakFit[ig[0]](rho)
        else:
            assert np.linalg.norm(rho-self.bodyRhoList) < 1e-16 and "Rho must match input rholist"
            return self.bodyCakFit[ig[0]]

    #@profile
    def ck(self,gz,rho):
        if gz < 1e-14:
            ck = -2.*np.log(rho)
        else:
            ck = 2.*scipy.special.kn(0,rho*gz)
        return ck

    # Given in Arias paper (see comment below about expi)
    def cak0(self,alpha,rho):
        small_cutoff = 1e-14
        small_expansion = rho <  small_cutoff
        large_expansion = rho >= small_cutoff
        val = np.zeros_like(rho)
        val[ small_expansion ] = np.euler_gamma + 2.*np.log(alpha) + (alpha * rho[ small_expansion ])
        val[ large_expansion ] = -2.*np.log(rho[ large_expansion ]) + scipy.special.expi(-(alpha**2 * rho[ large_expansion ]**2))
        return val

    # Looks basically like a gaussian multiplied by a skewing function.
    # A higher rho skews it more (from the iv function).
    #
    @profile
    def cak(self,rhop,alpha,gz,rho):
        rhop = np.array(rhop)
        if rhop.shape == ():
            return self.cak_scalar(rhop,alpha,gz,rho)
        arg = 2.*rhop*rho*alpha**2
        argTrue  = (arg < 8000.0)
        argFalse = (arg >= 8000.0)
        val = np.zeros_like(rhop)
        #self.prefac = np.exp(-gz**2/(4.*alpha**2))
        val[argTrue] = \
              2. * alpha**2 * rhop[argTrue] * \
              np.exp(-alpha**2 * (rho**2 + rhop[argTrue]**2) + arg[argTrue] - gz*rhop[argTrue]) * \
              scipy.special.ive(0,arg[argTrue]) * \
              2.*scipy.special.kve(0,gz*rhop[argTrue])
              #self.ck(gz,rhop[argTrue])
        #val[argTrue] = \
        #      2. * alpha**2 * rhop[argTrue] * \
        #      np.exp(-alpha**2 * (rho**2 + rhop[argTrue]**2)) * \
        #      scipy.special.iv(0,arg[argTrue]) * \
        #      self.ck(gz,rhop[argTrue])
        val[argFalse] = \
              2. * alpha**2 * rhop[argFalse] * \
              np.exp(-alpha**2 * (rho**2 + rhop[argFalse]**2) + arg[argFalse]) * \
              1./np.sqrt(2.*np.pi*arg[argFalse])*(1. + od8/arg[argFalse] + nd128/arg[argFalse]**2) * \
              self.ck(gz,rhop[argFalse])
        return val

    @profile
    def cak_scalar(self,rhop,alpha,gz,rho):
        a_sq = alpha**2
        arg = 2.*rhop*rho*a_sq
        #self.prefac = np.exp(-gz**2/(4.*alpha**2))
        if arg < 8000.0:
            #val = \
            #      self.prefac * 2. * alpha**2 * rhop *\
            #      np.exp(-alpha**2 * (rho**2 + rhop**2)) * \
            #      scipy.special.iv(0,arg) * \
            #      self.ck(gz,rhop)
            val = \
              2. * alpha**2 * rhop * \
              np.exp(-alpha**2 * (rho**2 + rhop**2) + arg - gz*rhop) * \
              scipy.special.ive(0,arg) * \
              2.*scipy.special.kve(0,gz*rhop)
        else:
            val = \
                  2. * alpha**2 * rhop * \
                  np.exp(-alpha**2 * (rho**2 + rhop**2) + arg) * \
                  1./np.sqrt(2.*np.pi*arg)*(1. + od8/arg + nd128/arg**2) * \
                  self.ck(gz,rhop)
        return val

    def large_rho_analytical(self,x,alpha,gz,rho):
        analytical = \
                    1./np.sqrt(rho*gz*2) *\
                    np.sqrt(np.pi) * \
                    np.exp(0.25 * gz * ( gz/alpha**2 - 4.*rho) ) * scipy.special.erf(0.5*gz/alpha + alpha*x) * \
                    ( 1. - od8/gz/rho + 1./(16*alpha**2*rho**2))
        return analytical

    def my_integral_scheme(self,alpha,gz,rho):
        xstart = 0.05

        # Getting the correct limits of integration given the error
        # This is not great for small rho... as it was derived in the limits
        # of large rho.
        #
        a = alpha**2
        b = -gz
        c = (24 + gz * rho)
        lim1 = (-b + np.sqrt( max(0,b**2 - 4*a*c))) / (2.*a)
        xvals = np.linspace(0,2.*lim1,num=800)
        plt.plot(xvals,self.cak(xvals,alpha,gz,rho))
        lim2 = (self.tol - 2.*alpha**2*rho**2)/(2.*alpha**2*rho)
        lim2 = -(1./16/alpha**2/rho - 1/8./gz)/1e-3 - rho
        xvals = np.linspace(0,lim2,num=800)
        plt.plot(xvals,self.cak(xvals,alpha,gz,rho))
        plt.show()

        # Integrating the head of function. Spline doesn't work too well since sometimes
        # this function isn't very well-behaved at the origin.
        #
        if self.lastHeadValZero is True:
            quad_head, quad_err = 0.0, self.tol
        else:
            quad_head, quad_err = scipy.integrate.quad(self.cak,a=1e-8,b=xstart,args=(alpha,gz,rho,))
        head = quad_head
        #print "Quad head = ", quad_head

        # If we are increasing rho, then the function is more and more well-behaved at the origin.
        # So we ignore the integral at the head (which should be -> 0 with increasing rho)
        #
        if quad_head < self.tol:
            self.lastHeadValZero = True

        # Integrating the body of the function with splines.
        #
        nx = int((lim1 - xstart) / dx)
        x = np.linspace(xstart,xstart+nx*dx,num=nx)
        rhopVals = self.cak(x,alpha,gz,rho)

        # Fitting to the gaussian-like function with a spline and integrating
        rhoFit = InterpolatedUnivariateSpline(x, rhopVals)
        body = rhoFit.integral(min(x),max(x))
        quad_body, quad_err = scipy.integrate.quad(self.cak,a=xstart,b=lim1,args=(alpha,gz,rho,))
        print "Old  integral    = %.15g" % body
        print "Quad integral    = %.15g" % quad_body

        # Integrating the tail of the gaussian
        #
        quad_tail, quad_err = scipy.integrate.quad(self.cak,a=lim1,b=np.inf,args=(alpha,gz,rho,))
        tail = quad_tail
        #print "Quad tail = ", quad_tail

        full_integral = (head+tail+body)
        print "Total integral = %.15g" % full_integral
        return full_integral

    def integrate(self,alpha,gz,rho):
        if self.method == "slow_quad":
            integral, error = scipy.integrate.quad(self.cak,a=1e-8,b=np.inf,args=(alpha,gz,rho,))
            #integral1, error = scipy.integrate.quad(self.cak,a=1e-8,b=2.,args=(alpha,gz,rho,))
            #integral2, error = scipy.integrate.quad(self.cak,a=2.,b=np.inf,args=(alpha,gz,rho,))
            #integral = integral1 + integral2
        else:
            raise NotImplementedError("Method %s not implemented!" % self.method)
        return integral

    @profile
    def evalRho(self,alpha,gz,rho):
        # Parameters:
        #
        #   dx     - determines the spline spacing
        #   xstart - where the integration changes from quad to spline
        #
        dx = 0.01

        self.prefac = np.exp(-gz**2/(4.*alpha**2))

        # Return 0.0 if exponential argument is too small
        #
        if self.prefac < self.tol:
            return 0.0

        # Otherwise, call the appropriate integration scheme
        #
        integral = self.prefac * self.integrate(alpha,gz,rho)

        return integral

        #def func3(x,alpha,gz,rho):
        #    val = np.exp(-gz**2/(4.*alpha**2))*2.*alpha**2*(rho+x)*np.exp(-alpha**2*x**2)*\
        #            scipy.special.iv(0,2.*alpha**2*rho*(rho+x))*\
        #            self.ck(gz,(rho+x))*np.exp(-2*alpha**2*rho*(rho+x))
        #    #val = np.exp(-gz**2/(4.*alpha**2))*2.*alpha**2*(rho+x)*np.exp(-alpha**2*x**2)*\
        #    #        1./np.sqrt(2.*np.pi*(2*alpha**2*rho*(rho+x)))*(1+od8/(2*alpha**2*rho*(rho+x))+nd128/(2*alpha**2*rho*(rho+x))**2)*\
        #    #        2.*(np.sqrt(0.5*np.pi/gz/(rho+x))*np.exp(-gz*(rho+x)))*(1-od8/gz/(rho+x)+nd128/gz**2/(rho+x)**2)
        #    return val

        ###if alpha**2*rho**2 > 30:
        ##quad_full, quad_err = scipy.integrate.quad(func3,a=-rho,b=rho,args=(alpha,gz,rho,))
        ##print np.exp( -(gz*rho + alpha**2 * ( rho + 0.5*gz/alpha**2 )**2))
        ##print func3(rho,alpha,gz,rho)
        ##xvals = np.linspace(-rho,10,num=800)
        ##plt.plot(xvals,func3(xvals,alpha,gz,rho))
        ##plt.show()
        ##print abs((2.*rho*alpha**2 - gz)/(16.*alpha**2*rho**3*gz))
        #if abs((2.*rho*alpha**2 - gz)/(16.*alpha**2*rho**3*gz)) < 0.001:
        #    #return 2.*self.large_rho_analytical(rho,alpha,gz,rho)
        #    approx_int = 2.*self.large_rho_analytical(rho,alpha,gz,rho)
        #    print "Diff = %.15g" % (full_integral-approx_int)
        ##sys.exit()
        ##    #val = np.exp(-gz**2/(4.*alpha**2))*2.*alpha**2*(rho+x)*np.exp(-alpha**2*x**2)*\
        ##    #        1./np.sqrt(2.*np.pi*(2*alpha**2*rho*(rho+x)))*(1+od8/(2*alpha**2*rho*(rho+x))+nd128/(2*alpha**2*rho*(rho+x))**2)*\
        ##    #        self.ck(gz,(rho+x))
        ##    #val = np.exp(-gz**2/(4.*alpha**2))*2.*alpha**2*(rho+x)*np.exp(-alpha**2*x**2)*\
        ##    #        1./np.sqrt(2.*np.pi*(2*alpha**2*rho*(rho+x)))*(1+od8/(2*alpha**2*rho*(rho+x))+nd128/(2*alpha**2*rho*(rho+x))**2)*\
        ##    #        2.*(np.sqrt(0.5*np.pi/gz/(rho+x))*np.exp(-gz*(rho+x)))*(1-od8/gz/(rho+x)+nd128/gz**2/(rho+x)**2)
        ##    #val = np.exp(-gz**2/(4.*alpha**2))*2.*alpha**2*(rho+x)*np.exp(-alpha**2*x**2)*\
        ##    #        1./np.sqrt(2.*np.pi*(2*alpha**2*rho*(rho+x)))*(1)*\
        ##    #        self.ck(gz,(rho+x))
        ##    #val = np.exp(-gz**2/(4.*alpha**2))*2.*alpha**2*(rho+x)*np.exp(-alpha**2*x**2)*\
        ##    #        1./np.sqrt(2.*np.pi*(2*alpha**2*rho*(rho+x)))*(1)*\
        ##    #        2.*(np.sqrt(0.5*np.pi/gz/(rho+x))*np.exp(-gz*(rho+x)))*(1-od8/gz/(rho+x)+nd128/gz**2/(rho+x)**2)
        ##    val = np.exp(-gz**2/(4.*alpha**2))*2.*alpha**2*np.exp(-alpha**2*x**2)*\
        ##            1./np.sqrt(2.*np.pi*(2*alpha**2*rho))*(1)*\
        ##            2.*(np.sqrt(0.5*np.pi/gz)*np.exp(-gz*(rho+x)))*(1)
        ##    return val

        ##def analytic1(x,alpha,gz,rho):
        ##    analytical = np.exp(-gz**2/(4.*alpha**2))* \
        ##                1./np.sqrt(rho*gz*2) *\
        ##                np.sqrt(np.pi) * np.exp(0.25 * gz * ( gz/alpha**2 - 4.*rho) ) * scipy.special.erf(0.5*gz/alpha + alpha*x)
        ##    return analytical
        ##def analytic2(x,alpha,gz,rho):
        ##    analytical = np.exp(-gz**2/(4.*alpha**2))* \
        ##                1./np.sqrt(rho*gz*2) *\
        ##                np.sqrt(np.pi) * np.exp(0.25 * gz * ( gz/alpha**2 - 4.*rho) ) * scipy.special.erf(0.5*gz/alpha + alpha*x) * ( 1. - od8/gz/rho)
        ##    return analytical
        ##def analytic3(x,alpha,gz,rho):
        ##    analytical = np.exp(-gz**2/(4.*alpha**2))* \
        ##                1./np.sqrt(rho*gz*2) *\
        ##                np.sqrt(np.pi) * \
        ##                np.exp(0.25 * gz * ( gz/alpha**2 - 4.*rho) ) * scipy.special.erf(0.5*gz/alpha + alpha*x) * \
        ##                ( 1. - od8/gz/rho + 1./(16*alpha**2*rho**2))
        ##    return analytical
        ##def analytic4(x,alpha,gz,rho):
        ##    analytical = np.exp(-gz**2/(4.*alpha**2))* \
        ##                1./np.sqrt(rho*gz*2) *\
        ##                np.sqrt(np.pi) * \
        ##                np.exp(0.25 * gz * ( gz/alpha**2 - 4.*rho) ) * scipy.special.erf(0.5*gz/alpha + alpha*x) * \
        ##                ( 1. - od8/gz/rho + 17./(128*alpha**2*rho*rho) + 9./(128*gz**2*rho**2) - 17./(1024*alpha**2*rho*gz*rho**2))
        ##    return analytical
        ##quad_new, err = scipy.integrate.quad(func3,a=-rho+0.001,b=rho-0.001,args=(alpha,gz,rho))
        ##print "new integral, diff = %12.8e %12.8e" % (quad_new,abs((head+tail+intgrl)-quad_new))
        ##analytical1 = analytic1(rho-0.001,alpha,gz,rho) - analytic1(-rho+0.001,alpha,gz,rho)
        ##print "an1 integral, diff = %12.8e %12.8e" % (analytical1,abs((head+tail+intgrl)-analytical1))
        ##analytical2 = analytic2(rho-0.001,alpha,gz,rho) - analytic2(-rho+0.001,alpha,gz,rho)
        ##print "an2 integral, diff = %12.8e %12.8e" % (analytical2,abs((head+tail+intgrl)-analytical2))
        ##analytical3 = analytic3(rho-0.001,alpha,gz,rho) - analytic3(-rho+0.001,alpha,gz,rho)
        ##print "an3 integral, diff = %12.8e %12.8e" % (analytical3,abs((head+tail+intgrl)-analytical3))
        ##analytical4 = analytic4(rho-0.001,alpha,gz,rho) - analytic4(-rho+0.001,alpha,gz,rho)
        ##print "an4 integral, diff = %12.8e %12.8e" % (analytical4,abs((head+tail+intgrl)-analytical4))


if __name__ == "__main__":
    rhoMin = 0.01
    rhoMax = 20.879696407
    dx = 0.05
    nrhoVals = int((rhoMax - rhoMin)/dx)
    exx_alpha = 0.01
    print "alpha = ", exx_alpha
    c1f = conv1Dfunc(exx_alpha, np.array([[0,0,0.0001]]), np.linspace(rhoMin,rhoMax,nrhoVals))
