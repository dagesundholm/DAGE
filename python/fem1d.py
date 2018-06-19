#!/usr/bin/env python
# -*- coding: latin-1 -*-
"""---------------------------------------------------------------------------------*
 *    Copyright (c) 2010-2018 Pauli Parkkinen, Eelis Solala, Wen-Hua Xu,            *
 *                            Sergio Losilla, Elias Toivanen, Jonas Juselius        *
 *                                                                                  *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy  *
 *    of this software and associated documentation files (the "Software"), to deal *
 *    in the Software without restriction, including without limitation the rights  *
 *    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     *
 *    copies of the Software, and to permit persons to whom the Software is         *
 *    furnished to do so, subject to the following conditions:                      *
 *                                                                                  *
 *    The above copyright notice and this permission notice shall be included in all*
 *    copies or substantial portions of the Software.                               *
 *                                                                                  *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   *
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, *
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE *
 *    SOFTWARE.                                                                     *
 *----------------------------------------------------------------------------------"""

import numpy as np

class LocalPoly(np.poly1d):
    """Instead of P(x) we store P(x'), where x'=(x-x_m)/h. Therefore
    we work on a polynomial defined around 0, but it is actually defined
    somewhere else.
    """
    def __init__(self,coeffs,middle,step):
        self.__dict__['middle']=middle
        self.__dict__['step']=step
        np.poly1d.__init__(self,coeffs)

    def __call__(self,x):
        return np.poly1d.__call__(self,(x-self.middle)/self.step)

    def integ(self, m=1, k=0):
        return LocalPoly(np.polyint(self.coeffs, m=m, k=k)*self.step**m,\
                self.middle,self.step)

    def deriv(self, m=1):
        return LocalPoly(np.polyder(self.coeffs, m=m)*self.step**-m,\
                self.middle,self.step)

    def normal(self):
        pout=np.poly1d(0.)
        p0=np.poly1d([1.,-self.middle])/self.step
        for cf in self.coeffs[:-1]:
            pout+=cf
            pout*=p0
        return pout+self.coeffs[-1]

    def translate(self, middle, step):
        """ Move the polynomial to the interval
        centered at middle with step step, scaling it to preserve area """
        m=step/self.step
#        n=middle-m*self.middle
        return LocalPoly(self.coeffs/m, middle, step)

def gen_LIP(valpairs):
    """ Convenience function to generate a Lagrange Interpolation Polynomial
    from a set of value pairs ((x_0,f(x_0)),(x_1,f(x_1)),...(x_n,f(x_n)))"""
    rvals=[i[0] for i in valpairs]
    fvals=[i[1] for i in valpairs]

    nlip=len(rvals)
    roots = [float(i) for i in range( -((nlip-1)/2) , nlip/2+1)]
    coeffs=np.zeros(nlip)
    for i in range(nlip):
        # Calculate base polynomials
        temp_roots=np.array(roots [:i] + roots [i+1:])
        factor=np.product(temp_roots-roots[i])
        basepol= np.poly1d (temp_roots, r=True) / factor 
        coeffs+=fvals[i]*basepol.coeffs
    lp=LocalPoly(coeffs,rvals[nlip/2],rvals[1]-rvals[0])
    lp.__dict__['fvals']=fvals
    return lp

class Function1D:
    """Numerical representation of a 1D function from the (x,f(x)) value
    pairs, using ncell cells with npoint sampling points in each, using
    fem_init as basis function generator"""
    def __init__(self,valpairs,npoint,fem_init=gen_LIP):
        self.np_tot=len(valpairs)
        self.fem=fem_init
        if self.np_tot%(npoint-1)!=1:
            raise ValueError("{0:d} points cannot be fitted into cells\
 containing {1:d} points".format(self.np_tot,npoint))
        self.npoint=npoint
        self.ncell=(self.np_tot-1)/(npoint-1)
        self.cells=[]
        self.starts=[]
        for icell in range(self.ncell):
            self.starts.append(valpairs[icell*(npoint-1)][0])
            self.cells.append(\
                    self.fem(valpairs[icell*(npoint-1):(icell+1)*(npoint-1)+1]))
        self.starts.append(valpairs[-1][0])
        return

    def __call__(self,x):
        """Evaluate the function at x"""
        if isinstance(x,np.ndarray):
            icell=[self.get_icell(x_i) for x_i in x]
            res=np.zeros(len(icell))
            for i in range(len(x)):
                if icell[i]>=0:
                    res[i]=self.cells[icell[i]](x[i])
            return res
        else:
            i=self.get_icell(x)
            if i>=0:
                return self.cells[i](x)
            else:
                return 0.0

    def get_icell(self,x):
        """Return the cell index to which x belongs"""
        if x > self.starts[-1] or x < self.starts[0]:
            return -1
        elif x==self.starts[-1]:
            return self.ncell-1
        else:
            maxi=self.ncell
            mini=0
            j=0
            while maxi-mini>1:
                j+=1
                midi=(mini+maxi)/2
                if x < self.starts[midi]:
                    maxi=midi
                else:
                    mini=midi
            return mini

    def deriv(self,x,m=1):
        w_cell=self.cells[self.get_icell(x)]
        return w_cell.deriv(m)(x)

    def integrate(self,a,b):
        try:
            if a<self.starts[0]:  a= self.starts[0]
            if b>self.starts[-1]: b= self.starts[-1]

            cella=self.get_icell(a)
            cellb=self.get_icell(b)
            s=0.0
            while cella<cellb:
                s+=self.F[cella](self.starts[cella+1])-\
                        self.F[self.get_icell(a)](a)
                cella+=1
                a=self.starts[cella]
            s+=self.F[cellb](b)-self.F[cella](a)
            return s
        except AttributeError:
            self.__dict__['F']=[]
            for cell in self.cells:
                self.F.append(cell.integ())
            return self.integrate(a,b)

    def prune(self,eps=1.e-8):
        i=0
        while i < self.ncell-1:
            # Generate child
            x=np.linspace(self.starts[i],self.starts[i+2],self.npoint)
            child=self.fem([[ix,Function1D.__call__(self,ix)] for ix in x])

            a=-(self.npoint/2)
            b=self.npoint-a
            sm=0.

            p1=self.cells[i].translate(0.,1.)
            q=((self.starts[i+2]-self.starts[i])/
                    (self.starts[i+1]-self.starts[i]))
            ch1=child.translate((self.npoint/2)*(q-1.),q)
            diff1=((ch1-p1)**2).integ()
            sm+=diff1(b)-diff1(a)

            p2=self.cells[i+1].translate(0.,1.)
            q=((self.starts[i+2]-self.starts[i])/
                    (self.starts[i+2]-self.starts[i+1]))
            ch2=child.translate((self.npoint/2)*(1.-q),q)
            diff2=((ch2-p2)**2).integ()
            sm+=diff2(b)-diff2(a)
            if(sm<=eps):
                self.cells.pop(i+1)
                self.starts.pop(i+1)
                self.ncell-=1
                self.cells[i]=child
            else:
                i+=1

if __name__=="__main__":
    ncell=4
    npoint=7
    f=lambda x: np.sin(5./np.pi*x)
    x=np.linspace(0.,10.,ncell*(npoint-1)+1)
    valpairs=[]
    for i in x:
        valpairs.append([i,f(i)])

    f_num=Function1D(valpairs,npoint)
    a=f_num(1.)
    integral=f_num.integrate(1.,3.)
