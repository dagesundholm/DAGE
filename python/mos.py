#!/usr/bin/python
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

import numpy, re, scipy.special
import bubbleblow

def norm_const(alpha,l):
    """Returns the normalization constant for a Cartesian GTO of given alpha and i,j,k"""
    # returns (2*x-1)!!
#    fact = lambda x: reduce(lambda y,z: y*z,range(1,max(2,2*x),2))
    fact = lambda x: 1
    return 2.0**(sum(l)+0.75)* alpha**(0.5*sum(l)+0.75) / numpy.pi**0.75 /numpy.sqrt( fact(l[0]) * fact(l[1]) * fact(l[2]))

sb_rad={1:0.35,6:0.70}
def a_factor(z1,z2):
    chi=sb_rad[z1]/sb_rad[z2]
    u=(chi-1.0)/(chi+1.0)
    return u/(u**2-1.0)

class Gaussian:
    def __init__(self,coeffs,expos,l):
        self.expos=numpy.array(expos)
        self.coeffs=numpy.array(coeffs) * numpy.array([norm_const(alpha,l) for alpha in self.expos])
        self.l=numpy.array(l) # l_x, l_y and l_z

    def __call__(self,pos,ext=False):
#        r=numpy.array(pos) # Pos is an array or tuple with the position
        r_sq=numpy.dot(pos,pos)
        if ext:
            factor=1.-numpy.exp(-4.*r_sq)
        else:
            factor=1.
        #          C_i * x**l_x * y**l_y * z**l_z * exp(-alpha*r**2)
        return (pos**self.l).prod() * sum(self.coeffs*numpy.exp(-self.expos*r_sq)) *factor

    def __repr__(self):
        return str(self.expos)+"   "+str(self.coeffs)+"   "+str(self.l)+"\n"

class Atom:
    def __init__(self,name,z,coords):
        self.name=name
        self.z=z
        self.coords=numpy.array(coords)
        self.basis=[]
        self.mo=[]
        return

    def orbital(self,i,pos):
#        return sum(mos[i,:]*( self.basis[i](pos-self.coords) for i in len(self.mos)))
        return self.basis[i](numpy.array(pos)-self.coords)

    def __str__(self):
        return self.name+"   "+repr(self.z)+"  "+repr(self.coords)

class WF:
    def __init__(self,filename='molden.input'):
        """ Initializes a  WF from a molden format file """
        moldenfile=open(filename)
        line=""
        #Skip first lines
        while not re.match("\[Atoms\]",line):
            line=moldenfile.readline()
        #Read atom types and coordinates
        self.atoms=[]
        while True:
            line=moldenfile.readline()
            if re.match("\[GTO\]",line): break
            vals=line.split()
            self.atoms.append(Atom(vals[0],int(vals[2]), [float(i) for i in vals[3:]]))
#        self.basis_table={}
        self.basis_table=[]
        #Read basis sets
        while True:
            line=moldenfile.readline()
            if re.match("\[MO\]",line): break
            # Read which atom basis sets are we reading
            curr_at=int(line.split()[0]) - 1
            # Read basis for this atom
            while True:
                # Read the type and the number of primitives
                line=moldenfile.readline()
                if re.match("\s*$",line): break

                vals=line.split()
                type,nprim=vals[0],int(vals[1])
                expos=[]
                coeffs=[]
                # Read primitives
                for i in range(nprim):
                    vals=moldenfile.readline().split()
                    expos.append(float(vals[0]))
                    coeffs.append(float(vals[1]))
                # Generate each primitive corresponding to each set i,j,k set of exponents
                for i in expos_from_type(type):
#                    self.basis_table[len(self.basis_table)]=curr_at
                    self.basis_table.append(curr_at)
                    self.atoms[curr_at].basis.append(Gaussian(coeffs,expos,i))
#    print atoms[curr_at].basis
        self.basis_size=sum([len(i.basis) for i in self.atoms])

#Read MO coefficients
        self.occs=[]
#        self.mos=[]
        i=-1
        while True:
            line=moldenfile.readline()
            match=re.match(" Occup=(.*)",line)
            if match:
                i+=1
                tmp=float(match.group(1))
                if tmp==0.0: break
                self.occs.append(tmp)
                for j in self.atoms:
                    j.mo.append([])
                for j in range(self.basis_size):
                    line=moldenfile.readline()
                    vals=line.split()
                    self.atoms[self.basis_table[j]].mo[i].append(float(vals[1]))
            else:
                continue

    def plot_mo(self,i,pos):
        """ Return phi_i(r) """
        wf=0.
        for atom in self.atoms:
            bas=[]
            x=numpy.array(pos)-atom.coords
            for basis in atom.basis:
                bas.append(basis(x))
            wf+=sum(numpy.array(atom.mo[i]) *numpy.array(bas))
        return wf

    def plot_mo_sq_bub(self,mo_id,atom_id,pos):
        """ Return stuff """
        bub=numpy.zeros(len(pos))
        rest=0.
        for i,atom in enumerate(wf.atoms):
            for j,basis in enumerate(atom.basis):
                if i==atom_id:
                    if sum(basis.l)==0:
                        for k,l in enumerate(pos):
                            x=numpy.array(l)
                            bub[k]+=basis(x)*atom.mo[mo_id][j]
                else:
                    x=wf.atoms[atom_id].coords-atom.coords
                    rest+=basis(x)*atom.mo[mo_id][j]
        return bub*(bub+2.*rest)

    def dens(self,pos):
        """ Return |Psi(r)|² """
        wf=numpy.zeros(len(self.occs))
        for i,atom in enumerate(self.atoms):
            x=numpy.array(pos)-atom.coords
            bas=[]
            for basis in atom.basis:
                bas.append(basis(x))
            wf+=[sum(numpy.array(j)*bas) for j in atom.mo]
        wf*=wf
        return sum( numpy.array(self.occs) * wf)

    def dens_bub2(self,atom_id,pos):
        """ Return |Psi(r)|² """
        wf=numpy.zeros(len(self.occs))
        for i,atom in enumerate(self.atoms):
            x=numpy.array(pos)-atom.coords
            bas=[]
            for basis in atom.basis:
                if (atom==self.atoms[atom_id] and sum(basis.l)>0):
                    bas.append(0.0)
                else:
                    bas.append(basis(x))
            wf+=[sum(numpy.array(j)*bas) for j in atom.mo]
        wf*=wf
        return sum( numpy.array(self.occs) * wf)

    def dens_bub3(self,atom_id,pos):
        """ Return stuff """
        return sum(numpy.array([self.plot_mo_sq_bub(j,atom_id,pos)*k for j,k in enumerate (self.occs)]))

    def dens_bub(self,atom_id,pos):
        """ Return |Psi(r)|² using only the basis functions centered
        at atom atom_id"""
        wf=numpy.zeros(len(self.occs))
        atom=self.atoms[atom_id]
        x=numpy.array(pos)-atom.coords
        bas=[]
        for basis in atom.basis:
            bas.append(basis(x))
        wf+=[sum(numpy.array(j)*bas) for j in atom.mo]
        wf*=wf
        return sum( numpy.array(self.occs) * wf)

    def s_f(self,atom_id,pos):
        a=sb_rad[wf.atoms[atom_id].z]/0.52917726
        return 0.5*scipy.special.erfc(4.0*(numpy.linalg.norm(wf.atoms[atom_id].coords-pos)-a))
        f=lambda x: 0.5*x*(3.-x**2)
#        f=lambda x: scipy.special.erf(4.0*x/(1.0-x**2))
#        f=lambda x: scipy.special.erf(10.0*x)
        s=[]
        for atom1 in self.atoms:
            refpos=atom1.coords
            s.append(1.0)
            for atom2 in self.atoms:
#                if atom2!=self.atoms[atom_id]:
                if atom2!=atom1:
                    r1=numpy.linalg.norm(pos-refpos)
                    r2=numpy.linalg.norm(pos-atom2.coords)
                    r12=numpy.linalg.norm(refpos-atom2.coords)
                    mu=(r1-r2)/r12
                    mu=mu+a_factor(atom1.z,atom2.z)*(1.-mu**2)

                    for j in range(5):
                        mu=f(mu)
#                    mu=f(mu)
                    s[-1]*=0.5*(1.-mu)
        return s[atom_id]/sum(s)
#        factor=0.5*scipy.special.erfc(numpy.linalg.norm(pos-self.atoms[atom_id].coords)-3)
#        return s[atom_id]/sum(s) * factor

    def bubble(self,atom_id,r_array):
        """ Return a density bubble by multiplying rho with Becke's fuzzy
        function centered on that atom, and averaging radially  over each
        spherical shell"""
        bub=numpy.zeros(len(r_array))
        pts=sampling_points(1)
        np=len(pts)
        for r in enumerate(r_array):
            for point in r*pts:
                pos=point+wf.atoms[atom_id].coords
                bub[j]+=wf.dens(pos)*wf.s_f(atom_id,pos)/np
        return bub

def sampling_points(i):
    """Return an array of points for integration over the spherical surface
    for a sphere of r=1"""
        if(i==1):
            s=1.0
            pts=[[ s,0.0,0.0],[-s,0.0,0.0],[0.0, s,0.0],[0.0,-s,0.0],[0.0,0.0, s],[0.0,0.0,-s]]
        else if (i==2):
            s=1.0/numpy.sqrt(3.0)
            pts=   [[0,s,s],[0,s,-s],[0,-s,s],[0,-s,-s],\
                    [s,0,s],[s,0,-s],[-s,0,s],[-s,0,-s],\
                    [s,s,0],[s,-s,0],[-s,s,0],[-s,-s,0]]
        else if (i==3):
            s=1.0/numpy.sqrt(3.0)
            pts=[[s,s,s],[s,s,-s],[s,-s,s],[s,-s,-s],[-s,s,s],[-s,s,-s],[-s,-s,s],[-s,-s,-s]]
    return pts

def expos_from_type(type):
    if type=="s":
        return [[0,0,0]]
    elif type=="p":
        return [[1,0,0],[0,1,0],[0,0,1]]
    elif type=="d":
        return [[2,0,0],[0,2,0],[0,0,2],\
                [1,1,0],[1,0,1],[0,1,1]]
    elif type=="f":
        return [[3,0,0],[0,3,0],[0,0,3],\
                [1,2,0],[2,1,0],[2,0,1],\
                [1,0,2],[0,1,2],[0,2,1],\
                [1,1,1]]

if __name__=="__main__":
    wf=WF()

    n=1000
    maxr=8.
    d=maxr/n

#    x=[-maxr+2*maxr/(n-1)*i for i in range (n)]

#    for j in [5,6]:
#    for j in range(len(wf.occs)):
#        f=open('orb_'+str(j)+'.dat','w')
#        for i in x:
#            f.write("{0:14.10f}   {1:14.10f}\n".format(i,wf.plot_mo(j,[0.0,0.0,i])))
#        f.close()

##################################################
# Print density along z axis
#    x=[-maxr+2*maxr/(n-1)*i for i in range (n)]
#    for i in x:
#        print i,wf.dens([0.0,0.0,i])
##################################################

##################################################
# Generate bublib
#    x=[maxr/n*i for i in range (n+1)]
#    f=open('bublib.dat','w')
#    g=open('coord.xyz','w')
#    g.write('{0:6d}\n'.format(len(wf.atoms)))
#    g.write('\n')
#    for i,atom in enumerate(wf.atoms):
#        print "Generating radial density "+str(i)
#        bubble=wf.bubble(i,x)
#        print "Generating bubble "+str(i)
#        bubble_trimmed=bubbleblow.Bubble([x,bubble],id=atom.name+str(i),z=atom.z)
#        print "Writing bubble "+str(i)
#        g.write('{0:10s}{1:14.10f}{2:14.10f}{3:14.10f}\n'.format\
#                (atom.name+str(i),atom.coords[0]*0.52917726,\
#                atom.coords[1]*0.52917726,atom.coords[2]*0.52917726))
#        f.write(str(bubble_trimmed)+"\n")
#    f.close()
##################################################
### Substract bubbles from dens_z
    x=[-maxr+2*maxr/(n-1)*i for i in range (n)]
    for i in x:
        pos=numpy.array([0.0,0.0,i])
        d=wf.dens(pos)
        b=[wf.bubble(j,[numpy.linalg.norm(pos-wf.atoms[j].coords)])[0] for j in range(len(wf.atoms))]
        print i,d,b[0],b[1],b[2],b[3],d-sum(b)
##################################################

#    for j in [0,2]:
#        f=open(str(j)+"_bubble_fuzz.dat","w")
#        dens=[]
#        nel=0.
#        outcore=False
#        factor=wf.dens(wf.atoms[j].coords)/wf.dens_bub3(j,wf.atoms[j].coords)
#        for i in x:
#            avg=(\
#                    wf.dens_bub3(j,numpy.array([ i,0.0,0.0])+wf.atoms[j].coords)+\
#                    wf.dens_bub3(j,numpy.array([-i,0.0,0.0])+wf.atoms[j].coords)+\
#                    wf.dens_bub3(j,numpy.array([0.0, i,0.0])+wf.atoms[j].coords)+\
#                    wf.dens_bub3(j,numpy.array([0.0,-i,0.0])+wf.atoms[j].coords)+\
#                    wf.dens_bub3(j,numpy.array([0.0,0.0, i])+wf.atoms[j].coords)+\
#                    wf.dens_bub3(j,numpy.array([0.0,0.0,-i])+wf.atoms[j].coords))/6\
#                    *factor
#            avg=(\
#                    wf.dens(numpy.array([ i,0.0,0.0])+wf.atoms[j].coords)+\
#                    wf.dens(numpy.array([-i,0.0,0.0])+wf.atoms[j].coords)+\
#                    wf.dens(numpy.array([0.0, i,0.0])+wf.atoms[j].coords)+\
#                    wf.dens(numpy.array([0.0,-i,0.0])+wf.atoms[j].coords)+\
#                    wf.dens(numpy.array([0.0,0.0, i])+wf.atoms[j].coords)+\
#                    wf.dens(numpy.array([0.0,0.0,-i])+wf.atoms[j].coords))/6
#            avg=0.0
#            for k in [[ i,0.0,0.0],[-i,0.0,0.0],[0.0, i,0.0],[0.0,-i,0.0],[0.0,0.0, i],[0.0,0.0,-i]]:
#                r=k+wf.atoms[j].coords
#                avg+=wf.dens(r)*wf.s_f(j,r)/6.

#            if not outcore:
#                avg=min(\
#                        wf.dens(numpy.array([ i,0.0,0.0])+wf.atoms[j].coords),\
#                        wf.dens(numpy.array([-i,0.0,0.0])+wf.atoms[j].coords),\
#                        wf.dens(numpy.array([0.0, i,0.0])+wf.atoms[j].coords),\
#                        wf.dens(numpy.array([0.0,-i,0.0])+wf.atoms[j].coords),\
#                        wf.dens(numpy.array([0.0,0.0, i])+wf.atoms[j].coords),\
#                        wf.dens(numpy.array([0.0,0.0,-i])+wf.atoms[j].coords))
#                dens.append(avg)

#                if(len(dens)>1):
#                    nel+=4*numpy.pi*(dens[-1]*i**2+dens[-2]*(i-d)**2)*d*0.5
#                    z=numpy.log(dens[-2]/dens[-1])/d
#                    c=dens[-1]*numpy.exp(z*i)
#                    nel_tot=nel+4*numpy.pi*c*numpy.exp(-z*i)*(2+z*i*(2+z*i))/z**3
#                    if i>0.2 and nel_tot>min(wf.atoms[j].z,2):
#                        print nel_tot
#                        outcore=True
#            else:
#                avg=c*numpy.exp(-z*i)

#            f.write("{0:14.10f}   {1:14.10f}\n".format(i,avg))
#            f.write("{0:14.10f}   {1:14.10f}   {2:14.10f}   {3:14.10f}   {4:14.10f}   {5:14.10f}   {6:14.10f}\n".format(i,\
#                    wf.dens(numpy.array([ i,0.0,0.0])+wf.atoms[j].coords),\
#                    wf.dens(numpy.array([-i,0.0,0.0])+wf.atoms[j].coords),\
#                    wf.dens(numpy.array([0.0, i,0.0])+wf.atoms[j].coords),\
#                    wf.dens(numpy.array([0.0,-i,0.0])+wf.atoms[j].coords),\
#                    wf.dens(numpy.array([0.0,0.0, i])+wf.atoms[j].coords),\
#                    wf.dens(numpy.array([0.0,0.0,-i])+wf.atoms[j].coords)))
#        quit()
#        g=open(str(j)+"_d.dat","w")
#        for j in range(len(dens)-1):
#            g.write("{0:14.10f}   {1:14.10f}\n".format(x[j],dens[j+1]/dens[j]))
