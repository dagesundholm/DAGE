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

import numpy as np, scipy.special, sys, os, re

class LIP:
    def __init__(self,nlip=7):
        self.nlip = nlip
        self.start = -((nlip-1)/2)
        self.end   = nlip/2
        self.roots = range(self.start,self.end+1)
        self.poly = []
        for i in range(nlip):
            # Calculate polynomials
            temp_roots=np.array(self.roots [:i] + self.roots [i+1:],dtype=float)
            factor=np.product(temp_roots-self.roots[i])
            self.poly.append( np.poly1d (temp_roots, r=True) / factor )
        self.gen_integrals()

    def gen_integrals(self,k=0):
        tmppol = [ np.polyint( poly ) for poly in self.poly ]
        tmp_ints = np.array ( [ [ pol(pos) for pol in tmppol ] for pos in self.roots ] )
        self.int_out = np.array ( [ row -  tmp_ints[0] for row in tmp_ints[1:]  ] )
        self.int_in  = np.array ( [ tmp_ints[-1] - row for row in tmp_ints[:-1] ] )
        return

class Cell:
    def __init__(self,lip,start,end,func):
        self.lip=lip
        self.r,self.step=np.linspace(start,end,self.lip.nlip,retstep=True)
        self.start=start
        self.length=end-start
        self.step=self.length/(self.lip.nlip-1.)
        self.gen_f=func
        self.f=self.gen_f(self.r)
        self.int_out=self.integrate_outwards()
        self.int_in =self.integrate_inwards()

    def integrate_outwards(self,k=0):
        """
        Return a nlip-1 dimensional array, where a[i] is the integral of the
        function integrated from the 1st point up to the i+1-th point.
        """
        dens_w = np.transpose( np.matrix ( self.f * self.r**(k+2) ) )
        tmpres = np.transpose ( np.matrix(self.lip.int_out ) * dens_w ) * self.step
        return tmpres

    def integrate_inwards(self,k=0):
        dens_w = np.transpose( np.matrix ( self.f * self.r**(-k+1) ) )
        if self.f[0]==0.0:
            dens_w[0]=0.0
        tmpres = np.transpose ( np.matrix(self.lip.int_in ) * dens_w ) * self.step
        return tmpres

    def split(self):
        self.children= [ Cell( self.lip, self.start+i*self.length/2,\
                self.start+(i+1)*self.length/2, self.gen_f) for i in range(2) ]
        return abs(self.int_in [0,0] -self.children[0].int_in[0,0] -self.children[1].int_in[0,0]),\
               abs(self.int_out[0,-1]-self.children[0].int_out[0,-1]-self.children[1].int_out[0,-1])

class Bubble:
    def __init__(self,origin,id,z=0.0,rmax=20.0,prec=1.e-10,nlip=7,ncell0=4,k=0,label=None):
        try:
            if (nlip%2!=1): raise ValueError
        except ValueError:
            print 'The number of LIP per cell must be an odd number ({0} given)'.format(nlip)
            exit(1)
        self.lip=LIP(nlip)
        self.origin=origin
        self.z=z
        self.initialize_origin()
        try:
            self.rmax=self.data_r[-1]
        except AttributeError:
            self.rmax=rmax
        # Calculate the initial lims of the cells (0, d, 2d,...,rmax)
#        initlims = [ i * self.rmax / ncell0 for i in range(ncell0+1) ]
#       # The new scheme...
        ncell0=int(self.z**0.25*ncell0)
        initlims = np.array([i * self.rmax/ncell0 for i in range(ncell0+1)])
        c=8*self.z**-1.5
        initlims[1:] = c*(1./((self.rmax+c)/initlims[1:]-1.))
#        print initlims
#        import matplotlib.pyplot as plt
#        plt.plot(range(ncell0+1),initlims)
#        plt.show()
#        quit()
        # Initialize the cells using the given limits
        self.cells = [ Cell(self.lip,initlims[i],initlims[i+1],self.get_rho) for i in range(ncell0) ]
        self.ncell0=ncell0
        self.ncell=ncell0
        self.npoint=self.ncell*(self.lip.nlip-1)+1
        self.k=k
        self.prec=prec
        self.id=id[0:8]
        if label!=None:
            self.label=label
        i=0
#        while ( i < self.ncell ):
#            current=self.cells[i]
#            outerr,inerr=current.split()
#            if(max(outerr,inerr)>self.prec):
#                self.cells=self.cells[:i]+current.children+self.cells[i+1:]
#                del current
#                self.ncell+=1
#                self.npoint+=self.lip.nlip-1
#            else:
#                i+=1
        self.num_el=4.0*np.pi*sum((np.trapz\
                (i.f*i.r**2,i.r) for i in self.cells))

    def initialize_origin(self):
        """
        self.origin can be several things:
        -----------------------------------------------------------------------------
        self.origin           | Type          |                 Description
        -----------------------------------------------------------------------------
          [r_values,f_values]   Two    iterables    (r, rho(r)) pairs

            g<a>                <str>           (a/pi)^(3/2)*exp(-a*r²)
                                                   <a> is a real number
                                                   Gaussian with an exponent a and a total
                                                   charge of 1.0
            s<z>                <str>           (z³/pi)*exp(-2.0*z*r)
                                                   <z> is a real number
                                                   Square of a 1s H-like orbital
                                                   (containing one electron) for a nuclear
                                                   charge Z
            filename            <str>           filename containing radial charge
                                                   for the moment it won't work...
        """
        if isinstance(self.origin,str):
            # Check if we want Slater or Gaussian
            match=re.match(r"([gsb])([\d\.]*)",self.origin)
            # Create a Gaussian or Slater charge distribution
            if match:
                self.origin_type=match.group(1)
                self.origin_expo=float(match.group(2))
                if (self.origin_type=="g"):
                    self.get_rho=self.__rho_gaussian
                    self.genpot_an=self.__v_gaussian
                    self.label="rho(r) = ({0:5.3f}/pi)^(3/2)*exp(-{0:5.3f}*r²)".format(self.origin_expo)
                    self.z=self.origin_expo
                elif (self.origin_type=="s"):
                    self.get_rho=self.__rho_slater
                    self.genpot_an=self.__v_slater
                    self.label="rho(r) = ({0:5.3f}³/pi)*exp(-2*{0:5.3f}*r)".format(self.origin_expo)
                    self.z=self.origin_expo
                elif (self.origin_type=="b"):
                    self.get_rho=self.__rho_bubble
                    self.genpot_an=self.__v_bubble
                    self.label="rho(r) = ({0:5.3f}³/pi)*exp(-2*{0:5.3f}*r) plus more stuff".format(self.origin_expo)
                    self.z=self.origin_expo
            # Otherwise read input data
            else:
                # Check for existence of the file
                try:
                    if(not(os.access(self.origin,os.F_OK))):
                        raise IOError("The file "+self.origin+" doesn't exist!")
                except IOError as detail:
                    print detail
                    quit()
                # Initialize and read MOs (mos.py is not working...)
#            self.get_rho=mos.Density(origin)
#            self.label="MOs from "+os.path.realpath(origin)
                if '.dat' in self.origin:
                # Read from numerical input
                    """
                    .dat format
                    Line 1  :   Comment
                    Lines 2-:   r(bohr) density

                    The grid must be equidistant!
                    """
                    try:
                        self.file=open(self.origin,'r')
                    except IOError as detail:
                        print detail
                        quit()
                    self.data_r=[]
                    self.data_d=[]
                    for line in self.file:
                        if(re.match("\s*#",line)):
                            continue
                        elif line.strip()=="":
                            break
                        else:
                            input=line.split() 
                            self.data_r.append( float (input[0]))
                            self.data_d.append( float (input[1]))
                    self.data_np=len(self.data_r)
                    # Initialize interpolation
                    self.nlip=7 # Our favourite 7 is hard-coded
                    self.interpol=LIP(self.nlip)
                    self.step=self.data_r[1]
                    self.get_rho=self.__rho_interpolate_from_data
                    self.label="Electronic density from "+os.path.realpath(self.origin)
        else:
            try:
                self.data_r=self.origin[0]
                self.data_d=self.origin[1]
                self.data_np=len(self.data_r)
                if len(self.data_d)!=self.data_np:
                    raise ValueError("r and rho array length mismatch")
                # Initialize interpolation
                self.nlip=7 # Our favourite 7 is hard-coded
                self.interpol=LIP(self.nlip)
                self.step=self.data_r[1]
                self.get_rho=self.__rho_interpolate_from_data
                self.label=""
            except TypeError as err:
                print err
                quit()
            except ValueError as err:
                print err
                quit()

    def __rho_interpolate_from_data(self,r_in):
        """ Interpolates rho(r) from the set of given (r, rho(r)) pairs,
        for each r given (r can be one single value or an array-like object).
        The interpolation is done by finding the data point which is closest
        to r, and the interpolation polynomials are centered around that point.
        """
        # OK, so the point where we center the polynomials: it cannot be smaller than
        # nlip/2 or np-nlip/2+1
        ans=[]
        for r in r_in:
            pos = min ( max ( int( round( r/self.step ) ) , self.nlip/2), self.data_np-self.nlip/2-1)
            # Transform into the LIP coordinate
            x=(r-self.data_r [pos])/self.step
            # Evaluate all the polynomials
            evals = np.array ([ poly(x) for poly in self.interpol.poly ])
            # Get slice with the necessary coeffs
            coeffs = self.data_d[pos-self.nlip/2:pos+(self.nlip+1)/2]
            ans.append( np.sum (evals*coeffs) )
        return ans

    def __rho_slater(self,r):
        """ Returns rho(r)=|Psi(r)|²=Z³/pi*exp(-2*Z*r)"""
        return self.origin_expo**3/np.pi*np.exp(-2.0*self.origin_expo*r)

    def __v_slater(self,r):
        """ Returns V(r)=(1-exp(-2*Z*r)*(1+Z*r))/r """
        if(r>0.):
            zr=self.origin_expo*r
            return (1.-np.exp(-2.*zr)*(1.+zr))/r
        else:
            return self.origin_expo

    def __rho_bubble(self,r):
        """ Returns rho(r)=|Psi(r)|²=Z³/pi*exp(-2*Z*r)+
            (Z-2)*2*a^(5/2)/(3*pi^(3/2))*r²*exp(-a*r²)"""
        a=2*self.origin_expo**2/(2*3)**2
        return 2*self.origin_expo**3/np.pi*np.exp(-2.0*self.origin_expo*r)+\
                (self.origin_expo-2)*a**2.5\
                /1.5/np.pi**1.5*r**2*np.exp(-a*r**2)

    def __v_bubble(self,r):
        """ Returns V(r)=(1-exp(-2*Z*r)*(1+Z*r))/r """
        a=2*self.origin_expo**2/(2*3)**2
        if(r>0.):
            zr=self.origin_expo*r
            return 2*(1.-np.exp(-2.*zr)*(1.+zr))/r+\
                    (self.origin_expo-2)*(scipy.special.erf(np.sqrt(a)*r)/r-\
                    np.sqrt(a)/1.5/np.pi**0.5*\
                    np.exp(-a*r**2))

        else:
            return 2*self.origin_expo+(self.origin_expo-2.)*\
                    4./3.*np.sqrt(a/np.pi)

    def __rho_gaussian(self,r):
        """ Returns rho(r)=|Psi(r)|²=(alpha/pi)**1.5*exp(-alpha*r²)"""
        return np.exp(-self.origin_expo*r*r)*(self.origin_expo/np.pi)**1.5

    def __v_gaussian(self,r):
        """ Returns V(r)=erf(sqrt(alpha)*r)/r """
        if(r>0.):
            return scipy.special.erf(np.sqrt(self.origin_expo)*r)/r
        else:
            return 2.0*np.sqrt(self.origin_expo/np.pi)

    def __str__(self):
        fmtstring="\n{0:21.15e}\t{1:21.15e}"
        out=''
        out+="#  "+self.id+"\n"
        out+="#  "+self.label+"\n"
        out+="#  {0:6d}{1:6d}{2:6d}{3:6d}\n".format(\
                self.npoint,self.lip.nlip,self.ncell,self.k)
        out+="#  {0:8.3f}{1:14.8f}{2:10.2e}{3:8.3f}{4:6d}".format(\
                self.z,self.num_el,self.prec,self.rmax,self.ncell0)
        for cell in self.cells:
            for i in range(self.lip.nlip-1):
                out+=fmtstring.format(cell.r[i],cell.f[i])
        out+=fmtstring.format(cell.r[i+1],cell.f[i+1])
        return out

    def genpot(self):
        """
        Wrap up the integrals to calculate the final potential.
        """
        # Make the final r_grid from all the cells
        self.gen_rgrid()

        # Outward integrals
        outpot=np.zeros(self.ncell*(self.lip.nlip-1)+1)
        outpot[0]=0.0
        for i in range(len(self.cells)):
            idx=i*(self.lip.nlip-1)
            outpot[idx+1:idx+self.lip.nlip]=outpot[idx]+\
                    np.array(self.cells[i].int_out)
        # This already includes the 1/r final multiplication
        outpot[1:]=outpot[1:]*self.r[1:]**(-self.k-1)
        outpot[0]=0.0

        # Inward integrals
        inpot=np.zeros(self.ncell*(self.lip.nlip-1)+1)
        for i in range(len(self.cells)-1,-1,-1):
            idx=i*(self.lip.nlip-1)
            inpot[idx:idx+self.lip.nlip-1]= inpot[idx+self.lip.nlip-1]+\
                    np.array(self.cells[i].int_in)
        # This already includes the 1/r final multiplication
        if (self.k>0): inpot=inpot*self.r**(self.k)

        self.pot=(inpot+outpot)*4.0*np.pi
        return

    def gen_rgrid(self):
        rgrid=np.zeros(self.ncell*(self.lip.nlip-1)+1)
        for i,cell in enumerate(self.cells):
            idx=i*(self.lip.nlip-1)
            for j in range(self.lip.nlip-1):
                rgrid[idx+j]=cell.r[j]
        rgrid[-1]=cell.r[j+1]
        self.r=rgrid
        return

if(__name__=="__main__"):
    import matplotlib.pyplot as plt
    import argparse
    parser=argparse.ArgumentParser(usage="%(prog)s <origin> [args]")
    parser.add_argument('origin',\
			action="store",\
            help="""Origin data for the bubble. There are three possibilities:
     - <filename>  File <filename> containing (r,rho(r)) pairs
     - g<a>        Gaussian with an exponent <a> and a total charge of 1.0
     - s<z>        Square of a 1s H-like orbital (containing one electron) 
                   for a nuclear charge <z>""")
    parser.add_argument('-z',\
            dest='z',\
            type=float,\
            default=0.0,\
            help="Nuclear charge (if a data file was given as origin)")
    parser.add_argument('-i','--id',\
            dest='id',\
            default='XX-XXXXX',\
            help="Eight-character string to identify the bubble entry in the library.\
            The first one or two characters should be the symbol of the element")
    parser.add_argument('-p','--precision',\
            dest='prec',\
            type=float,\
            default=1.e-8,\
            help="Required precision in the potential. Default: 1.e-8")
    parser.add_argument('-n','--ncell',\
            dest='ncell0',\
            type=int,\
            default=4,\
            help="Initial number of cells. Default: 4")
    parser.add_argument('-r','--rmax',\
            dest='rmax',\
            type=float,\
            default=10.0,\
            help="Maximum r (in a.u.). Default: 10.0. Ignored ir <ORIGIN> is a file, the last\
            r value is taken")
    parser.add_argument('-g','--genpot',
            action='store_true',\
            dest='genpot',\
            default=False,\
            help="Calculate the potential for the resulting bubble")
    parser.add_argument('-e','--plot-error',
            action='store_true',\
            dest='plot',\
            default=False,\
            help="Plot the error in the potential on screen (requires -g; only for gaussian and slater densities)")
    args=parser.parse_args()

    origin=args.origin
    bubbletest=Bubble(origin,args.id,z=args.z,prec=args.prec,ncell0=args.ncell0,rmax=args.rmax)
    if args.genpot:
        bubbletest.genpot()
        if args.plot:
            plt.plot(bubbletest.r,np.array( [ bubbletest.genpot_an(x) \
                    for x in bubbletest.r ] ) - bubbletest.pot )
            plt.show()
        else:
            for i in range(bubbletest.npoint):
                print "{0:20.10e}{1:20.10e}{2:20.10e}".format(bubbletest.r[i],\
                        bubbletest.pot[i],bubbletest.genpot_an(bubbletest.r[i])-bubbletest.pot[i])
    else:
        print bubbletest
