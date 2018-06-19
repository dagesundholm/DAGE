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

import numpy as np, scipy, fem1d
import copy, sys, subprocess, os.path
import matplotlib.pyplot as plt
import struct

A2au=1.889725949
ptable={"H" :1 ,"He":2 ,
        "Li":3 ,"Be":4 ,"B" :5 ,"C" :6 ,"N" :7 ,"O" :8 ,"F" :9 ,"Ne":10,
        "Na":11,"Mg":12,"Al":13,"Si":14,"P" :15,"S" :16,"Cl":17,"Ar":18,
        "Zn":30}
ptable_inv=dict(((ptable[i],i) for i in ptable))

orb_dens= [ [(lambda expo,r: expo**3/8/np.pi*np.exp(-expo*r)) ],
            [(lambda expo,r: expo**3/32/np.pi*(2.-expo*r)**2*np.exp(-expo*r)),
             (lambda expo,r: expo**3/96/np.pi*(expo*r)**2*np.exp(-expo*r))  ] ]

def sl_const(z,n):
    return z**(3+n)/(4.*np.pi*scipy.factorial(n+2))
#    """Generate np points for spherical integration.
#    
#    Returns np points distributed over a sphere and the corresponding
#    weights for numerical integration. The format is 
#    [(theta_0,phi_0,w_0),(theta_1,phi_1,w_1),...(theta_np-1,phi_np-1,w_np-1)]
#    """
sph_int_points={}
sph_int_points[1]= [(0.0,0.0,1.)]
sph_int_points[2]= [(np.pi/2,0.0,0.5),(np.pi/2,np.pi,0.5)]
sph_int_points[6]= [(0.,0.,1./6)]+\
                   [(np.pi/2,np.pi/2*i,1./6) for i in range(4)]+\
                   [(np.pi,0.,1./6)]
sph_int_points[12]=[(np.pi/4,  np.pi/2*i,      1./12) for i in range(4)]+\
                   [(np.pi/2,  np.pi/4*(2*i+1),1./12) for i in range(4)]+\
                   [(3*np.pi/4,np.pi/2*i,      1./12) for i in range(4)]

def Y(l,m,phi,theta):
    if (l==0):
        return 1.
    elif (l==1):
        if(m==-1):  #p_y
            return np.sin(phi)*np.sin(theta)
        if(m==0):  #p_z
            return np.cos(phi)
        if(m==1): #p_x
            return np.sin(phi)*np.cos(theta)

class Bubble(fem1d.Function1D):
    def __init__(self,valpairs,npoint,l,m,atom):
        self.l,self.m,self.atom=l,m,atom
        self.rmax=valpairs[-1,0]
        valpairs[:,1]*= (2*l+1)
        fem1d.Function1D.__init__(self,valpairs,npoint)
        return

    def __call__(self,pos):
        x,y,z=copy.copy(pos)
        atpos=self.atom.pos
        x-=atpos[0]
        y-=atpos[1]
        z-=atpos[2]
        r=np.sqrt(x*x+y*y+z*z)
        return fem1d.Function1D.__call__(self,r) * self.Y(x,y,z, r)

    def eval_rad(self,r):
        return fem1d.Function1D.__call__(self,r)

    def Y(self,x,y,z,r):
        try:
            iter(x)
        except TypeError:
            return self.__Yp( (x,y,z) ,r)
        else:
            res=copy.copy(r)
            for i,r_v in enumerate(r):
                res[i]=self.__Yp( (x[i],y[i],z[i]) , r_v)
            return res

    def __Yp(self,crd,r):
        """
        This should only take scalars as arguments, i.e.
        Bubble.__Yp([2.,3.,6.],7.)
        """
        if (self.l==0 or r < 1.e-18):
        # Workaround to copy r (array or real...)
            return 1.
        elif (self.l==1):
            if(self.m==-1): #p_y
                return crd[1]/r
            if(self.m==0):  #p_z
                return crd[2]/r
            if(self.m==1):  #p_x
                return crd[0]/r

class Atom:
    def __init__(self,pos,z):
        self.pos=np.array(pos)
        self.z=z
        # Generate parameters for analytical approximate density (Atom.adens)
        self.expo=[]
        self.const=[]
#### Guessed parameters...
#        if z==1:
#            self.rfunc=lambda x:0.446806*np.exp(-2.57306*abs(x))
#        elif z==6:
#            a=2*sl_const(12,0)
#            b=2*sl_const(6,1)
#            c=2*sl_const(3,2)
#            self.rfunc=lambda x:a*np.exp(-12.*abs(x))+
#                                b*np.exp(-6.*abs(x))*abs(x)+
#                                c*np.exp(-3.*abs(x))*x**2

### analytical constants
#        rem_el=self.z
#        n=0
#        while rem_el>0:
### calculate parameters shell by shell
#            n+=1
### calculate how many electrons are in this shell
###        n_el=min(2*n**2,rem_el); rem_el-=n_el
#            self.expo.append([])
#            self.const.append([])
#            for l in range(0,n):
### calculate how many electrons are in this subshell
#                n_el=min(2*(2*l+1),rem_el); rem_el-=n_el
###   # it seems that dividing z by n+l does the trick for c...
#                expo=float(2*self.z)/(n+3*l)
#                const=float(n_el) #* expo**(3+2*n)/4/np.pi/scipy.factorial(2*n)
#                self.expo[-1].append(expo)
#                self.const[-1].append(const)
#        self.nmax=n

        return

    def adens(self,pos):
        """Generate approximate atomic electronic density at point pos.
        
        It is calculated as a sum of layers:
        rho(r)=SUM_{k=0}^{k_max} {c_k r^{2k} e^{-2 z/(k+1) r}
        Each layer contains a maximum of 2*(k+1)^2 (2,8,18...) electrons

        pos can be an iterable containing 3 numbers or 3 np.array
        """
        x,y,z=pos
        r_sq=(x-self.pos[0])**2+(y-self.pos[1])**2+(z-self.pos[2])**2
        r=np.sqrt(r_sq)

        return np.exp(-2*r)
#        return self.rfunc(r)

#        d= self.const[0][0] * orb_dens[0][0] (self.expo[0][0], r)
#        for n in range(1,self.nmax+1):
#            for l in range(0,n):
#                d+= self.const[n-1][l] * orb_dens[n-1][l] (self.expo[n-1][l], r)
#        return d

    def eval_bubbles(self,pos):
        return self.bubbles[0](pos)
#        return sum([j(pos) for j in self.bubbles])
#
    def init_rgrid(self,ncell0,npoint,rmax):
        self.ncell=int(self.z**0.25*ncell0)
        r0 = np.array([i * rmax/self.ncell for i in range(self.ncell+1)])
        c=8*self.z**-1.5
        r0[1:] = c*(1./((rmax+c)/r0[1:]-1.))
#        for u in r0:
#            print u
#        quit()
        nptot=self.ncell*(npoint-1)+1
        self.r=np.zeros(nptot)
        for i in range(self.ncell):
            self.r[i*(npoint-1):(i+1)*(npoint-1)+1]=\
                np.linspace(r0[i],r0[i+1],npoint)
        return

class AtomEnsemble:
    def __init__(self,infile,rmax,ncell0,npoints):
        self.rmax,self.ncell0,self.npoints=rmax,ncell0,npoints
        self.basename,ext=infile.split(".")

#    Check that file exists
        if not os.path.isfile(infile):
            print "File "+infile+" not found"
            quit()

        if ext=="t21":
        #    Process t21 with dmpkf
            outfile=open(self.basename+'.dmp','w')
            proc=subprocess.Popen(['dmpkf',infile],
                    stdout=outfile,
                    stderr=open("/dev/null","w"))
            proc.wait()
            outfile.close()
        #    Generate atoms
            self.atoms=self.atoms_from_t21(infile)
        #    Process dmp with dgrid
            self.init_dgridProc(self.basename+'.dmp').wait()
#            subprocess.Popen(["/home/sergio/bin/dgrid-4.5",basename+'.dmp'])
#                    .wait()
        # Find most recent file
            self.infile=max([ i for i in os.listdir('.') if ".adf" in i],
                    key=lambda x:os.stat(x).st_mtime)
        else:
            fp=open(xyzfile)
            self.atoms=[]
            self.infile=infile

            for curline,line in enumerate(fp):
                if curline<2: continue
                indata=line.split()
                self.atoms.append(Atom([float(i)*A2au for i in indata[1:]],ptable[indata[0].capitalize()]))
        for atom in self.atoms:
            atom.init_rgrid(self.ncell0,self.npoints,self.rmax)
        return

    def atoms_from_t21(self,filename):

        proc=subprocess.Popen(
                ['adfreport','-r','Geometry%nr of atoms',filename],
                stdout=subprocess.PIPE)
        proc.wait()
        self.numatoms=int(proc.communicate()[0])

        proc=subprocess.Popen(
                ['adfreport','-r','Geometry%atomtype',filename],
                stdout=subprocess.PIPE)
        proc.wait()
        types=proc.communicate()[0].replace('}','').replace('{','').split()

        proc=subprocess.Popen(['adfreport','-r',
                'Geometry%fragment and atomtype index',filename]
                ,stdout=subprocess.PIPE)
        proc.wait()
        type_ids=[types[int(j)-1] for j in 
                proc.communicate()[0].split()[self.numatoms:]]

        proc=subprocess.Popen(
                ['adfreport','-r','Geometry%xyz',filename],
                stdout=subprocess.PIPE)
        proc.wait()
        coords=proc.communicate()[0].split()
        atoms=[]
        for i in range(self.numatoms):
            atoms.append(Atom([float(j) for j in coords[3*i:3*(i+1)]],
                    ptable[type_ids[i].capitalize()]))
        return atoms

    def mask_factor(self,iatom,pos):
        s=0.

####    BUBBLES
        for jatom,atom in enumerate(self.atoms):
            if jatom==iatom: continue
            s+=atom.adens(pos)

        rho=self.atoms[iatom].adens(pos)
        return np.where(s+rho!=0.,rho/(s+rho),0.)
####    BECKE
#        p=lambda x: 0.5*x*(3. - x**2)
#        npos=np.size(pos,1)
#        ans=np.zeros(npos)
#        for ipos in range(npos):
#            dists=[np.sqrt(np.sum((pos[:,ipos]-at.pos)**2))for at in self.atoms]
#            s=np.zeros([self.numatoms,self.numatoms])
#            for j in range(self.numatoms):
#                for k in range(j+1,self.numatoms):
#                    djk=np.sqrt(np.sum((self.atoms[j].pos-self.atoms[k].pos)**2))
#                    mu=(dists[j]-dists[k])/djk
#                    s[j,k]=0.5 * ( 1.- p(p(p( mu ))) )
#                    s[k,j]=0.5 * ( 1.- p(p(p(-mu ))) )
#            P=np.ones(self.numatoms)
#            for j in range(self.numatoms):
#                for k in range(self.numatoms):
#                    if j==k: continue
#                    P[j]*=s[j][k]
#            if(np.sum(P)!=0.):
#                ans[ipos]=P[iatom]/np.sum(P)
#            else:
#                ans[ipos]=0.
#        return  ans

    def get_rho_line(self,p1,p2,npoints):
        vect=p2-p1
        header="""
::
 basis="""+self.infile+"""
 compute=rho
 result=outfile
 vectors
 """
        footer=""" 0.0 0.0 0.0 0
 0.0 0.0 0.0 0
 format=grace
 END\n"""
        infile=open("infile","w")
        infile.write(header+
                '{0:10.8f} {1:10.8f} {2:10.8f}\n'.
                format(p1[0],p1[1],p1[2])+
                '{0:10.8f} {1:10.8f} {2:10.8f} {3:8d}\n'.
                format(vect[0],vect[1],vect[2],npoints-1)+
                footer)
        infile.close()
        proc=self.init_dgridProc("infile")
        proc.wait()
        f=np.loadtxt("outfile.rho_r",usecols=[1])
        subprocess.call(["rm","outfile.rho_r"])
        return f

    def get_rho(self,point):
        header="""
::
 basis="""+self.infile+"""
 compute=rho
 result=outfile
 point="""
        footer="""END\n"""
        infile=open("infile","w")
        infile.write(header+
                '{0:10.8f} {1:10.8f} {2:10.8f}\n'.format(point[0],point[1],point[2])+
                footer)
        infile.close()
        dgrid_proc=self.init_dgridProc("infile")
        dgrid_proc.wait()
        grep_proc=subprocess.Popen(["grep","rho"],stdin=dgrid_proc.stdout,stdout=subprocess.PIPE)
        rho_val=float(grep_proc.communicate()[0].split()[-1])
        return rho_val

    def gen_rho_cube(self,radius,step):
        nlip=7
        glims=[]
        gdims=[]
        gspan=[]
        for dim in range(3):
            crd=[at.pos[dim] for at in self.atoms]
            glims.append([min(crd)-radius,max(crd)+radius])
            d=glims[dim][1]-glims[dim][0]
            gdims.append(int(np.ceil(d/(step*2*(nlip-1)))*2))
            diff=gdims[dim]*(nlip-1)*step-d
            glims[dim][0]-=diff*0.5
            glims[dim][1]+=diff*0.5
            gspan.append(glims[dim][1]-glims[dim][0])
#            print glims[dim][0],glims[dim][0]+gdims[dim]*(nlip-1)*step
#        quit()
        # Cubify: RETARDED BUG IN GENPOT!!!!!
        mn=min([i[0] for i in glims])
        mx=max([i[1] for i in glims])
        gdims=[max(gdims)]*3
        for i in glims:
            i[0]=mn
            i[1]=mx
        gspan=[i[1]-i[0] for i in glims]

        header="""
::
 basis="""+self.infile+"""
 compute=rho
 result=outfile
 format=cube
  vectors
  """
        footer="""END\n"""
        infile=open("infile","w")
        infile.write(header)
        infile.write('{0:14.10f} {1:14.10f} {2:14.10f}\n'.format(
                glims[0][0],glims[1][0],glims[2][0]))
        infile.write('{0:14.10f} {1:14.10f} {2:14.10f} {3:5d}\n'.format(
                gspan[0],   0.,   0.,   gdims[0]*(nlip-1)))
        infile.write('{0:14.10f} {1:14.10f} {2:14.10f} {3:5d}\n'.format(
                0.,   gspan[1],   0.,   gdims[1]*(nlip-1)))
        infile.write('{0:14.10f} {1:14.10f} {2:14.10f} {3:5d}\n'.format(
                0.,   0.,   gspan[2],   gdims[2]*(nlip-1)))
        infile.write(footer)
        infile.close()
        dgrid_proc=self.init_dgridProc("infile")
        dgrid_proc.wait()
        cubefile=max([ i for i in os.listdir('.') if "outfile" in i],
                key=lambda x:os.stat(x).st_mtime)
        new=self.basename+"_"+"{0:G}".format(1000*step)+".cub"
        os.rename(cubefile,new)
        return new

    def calc_bubbles(self,lmax,numiter=2):
        self.lmax=lmax
        if numiter==0: return
#        print "Calculating bubbles... "+str(numiter)+" iterations to go"
#        np_tot=self.ncell*(self.npoints-1)+1
        sph_points=sph_int_points[12]
        lm=[]
        for l in range(lmax+1):
            lm+=[(l,m) for m in range(-l,l+1)] 
        bubbles=[]
        for iatom,atom in enumerate(self.atoms):
            r=atom.r
            np_tot=len(r)
            pos=atom.pos
            f=np.zeros([(lmax+1)**2,np_tot])

            for phi,theta,w in sph_points:
#                theta+=np.pi/4
                
                x=np.sin(phi)*np.cos(theta)*r+pos[0]
                y=np.sin(phi)*np.sin(theta)*r+pos[1]
                z=np.cos(phi)              *r+pos[2]

                Y_factor=np.array([[Y(l,m,phi,theta)] for l,m in lm ])

#                endpoint=pos+r[-1]*np.array([np.sin(phi)*np.cos(theta),
#                                   np.sin(phi)*np.sin(theta),
#                                   np.cos(phi)])
#                temp_f=self.mask_factor(iatom,np.row_stack((x,y,z)) )*\
#                       self.get_rho_line(pos,endpoint,np_tot)
#
#                endpoint=pos+r[-1]*np.array([np.sin(phi)*np.cos(theta),
#                                   np.sin(phi)*np.sin(theta),
#                                   np.cos(phi)])
#                np2=(np_tot-1)*10+1
#                flin=fem1d.Function1D(np.column_stack((
#                    np.linspace(0.,r[-1],np2),
#                    self.get_rho_line(pos,endpoint,np2))),npoints)
#                temp_f=self.mask_factor(iatom,np.row_stack((x,y,z)) )*\
#                        flin(r)

                mask=self.mask_factor(iatom,np.row_stack((x,y,z)) )
                d=np.array([ self.get_rho( [x[i],y[i],z[i]] )
                        for i in range(np_tot) ])
                temp_f= mask * d
#
                f+=w * Y_factor * temp_f

            bubbles.append([ Bubble( np.column_stack((r,f[j,:])),
                    self.npoints,l,m,atom) for j,(l,m) in enumerate(lm)])
        self.bubbles=bubbles
        for iatom,atom in enumerate(self.atoms):
            atom.bubbles=self.bubbles[iatom]
            atom.adens=atom.eval_bubbles
#        self.print_bubs(str(numiter))
        self.calc_bubbles(lmax,numiter=numiter-1)
        return

    def print_bubs(self,app=""):
        """Output the generated bubbles"""
        for i,atom in enumerate(self.atoms):
            outfile=open(ptable_inv[atom.z]+str(i)+"_"+app,"w")
            outfile.write("#")
            for crd in atom.pos:
                outfile.write("{0:23.14e}".format(crd))
            outfile.write("\n")
            bb=atom.bubbles[0]
            for icell in range(bb.ncell):
                for ipoint in range(bb.npoint):
                    r=bb.cells[icell].middle+(ipoint-bb.npoint/2)*bb.cells[icell].step
                    outfile.write("{0:23.14e}".format(r))
                    for bub in atom.bubbles:
                        outfile.write("{0:23.14e}".format(
                                bub.cells[icell].fvals[ipoint]))
                    outfile.write("\n")
            outfile.close()
        return

    def print_bublib(self,filename="bublib.dat"):
        """Output the generated bubbles in the bublib format"""
        outfile=open(filename,"w")
        outfile.write(  struct.pack("ii",len(self.atoms),self.lmax)  )
        for atom in self.atoms:
            outfile.write(  struct.pack("d",atom.z)  )
            for crd in atom.pos:
                outfile.write(  struct.pack("d",crd)  )
            bb=atom.bubbles[0]
            # Output steps
            outfile.write(  struct.pack("ii",bb.ncell,bb.npoint)  )
            for cell in bb.cells:
                outfile.write(  struct.pack("d",cell.step)   )
                      
            # Output radial functions
            for bub in atom.bubbles:
                for cell in bub.cells:
                    for ipoint in range(bb.npoint-1):
                        outfile.write(  struct.pack("d",cell.fvals[ipoint])  )
                outfile.write(  struct.pack("d",bub.cells[-1].fvals[-1])  )
        return

    def print_bublib_old(self,filename="bublib.dat"):
        """Output the generated bubbles in the bublib format"""
        outfile=open(filename,"w")
        for i,atom in enumerate(self.atoms):
            outfile.write("#  {0:s}-{1:d}\n".format(ptable_inv[atom.z],i))
            outfile.write("#  Bubble from ADF\n")
            bb=atom.bubbles[0]
            outfile.write("#{0:8d} {1:8d} {2:8d} {3:8d}\n".
                format(bb.np_tot,bb.npoints,bb.ncell,1))
            outfile.write("#  {0:10.5f} ...and other stuff\n".format(atom.z))
            for j in range(bb.np_tot):
                r_j=j*bb.rmax/(bb.np_tot-1)
                outfile.write("{0:20.10e}".format(r_j))
            for bub in atom.bubbles:
                outfile.write("{0:20.10e}".format(bub.eval_rad(r_j)))
            outfile.write("\n")
        return

    def init_dgridProc(self,infile):
        proc=subprocess.Popen(["dgrid-4.5",infile],
                stdout=subprocess.PIPE)
        proc.wait()
        subprocess.call(["rm",infile])
        return proc

    def write_diff(self,id1,id2,ncell=400,npoint=7,pad=8.,npi=10):
        atom1=self.atoms[id1]
        atom2=self.atoms[id2]

        npint=       ncell*(npoint-1)+1
        nptot=npi*ncell*(npoint-1)+1

        dv=atom2.pos-atom1.pos
        dr=np.sqrt(sum(dv*dv))
        start=atom1.pos-pad*dv/dr
        end=  atom2.pos+pad*dv/dr
        umax=0.5*dr+pad
        disp=dv/dr*(dr+2*pad)/(npint-1)

        pts=np.array( [ start + i * disp for i in range(npint) ] )
        u=np.linspace(-umax,umax,npint)
        d=atoms.get_rho_line(start,end,npint)
        diff=np.zeros(npint)
        nbub=sum( [len(at.bubbles) for at in atoms.atoms] )
        bubs=[]
        for at in atoms.atoms:
            for bub in at.bubbles:
                bubs.append(np.array( [ bub(pt) for pt in pts ] ))
#        print len(d),len(bubs[0]),len(np.sum(bubs,axis=0))
        diff=d-np.sum(bubs,axis=0)

        d=   fem1d.Function1D(np.column_stack((u,d)),npoints)
        diff=fem1d.Function1D(np.column_stack((u,diff)),npoints)
        bubs=[fem1d.Function1D(np.column_stack((u,b)),npoints) for b in bubs]

        # Subtract
        fout=open("{0:s}_subtr_{1:d}_{2:d}.dat".format(self.basename,id1,id2),"w")
        subtr=[]

        disp=dv/dr*(dr+2*pad)/(nptot-1)
        du=np.sqrt(sum(disp*disp))
        fout.write( "# {0:4d} {1:18.12f} {2:18.12f} {3:18.12f}\n".format(
            id1,-0.5*dr,d(-0.5*dr) ,diff(-0.5*dr) ) )
        fout.write( "# {0:4d} {1:18.12f} {2:18.12f} {3:18.12f}\n".format(
            id2, 0.5*dr,d( 0.5*dr) ,diff( 0.5*dr) ) )
            
        for i in range(nptot):
            u_val=u[0]+i*du
            pt=start+i*disp
            fout.write( "{0:18.12e} {1:18.12e} {2:18.12e}".format( u_val,d(u_val),diff(u_val) ) )
            for b in bubs:
                fout.write( " {0:18.12e}".format( b(u_val) ) )
            fout.write("\n")
        fout.close()
        return

    def write_masked_dens(self,id1,id2,ncell=400,npoint=7,pad=8.,npi=1):
        atom1=self.atoms[id1]
        atom2=self.atoms[id2]

        npint=       ncell*(npoint-1)+1
        nptot=npi*ncell*(npoint-1)+1

        dv=atom2.pos-atom1.pos
        dr=np.sqrt(sum(dv*dv))
        start=atom1.pos-pad*dv/dr
        end=  atom2.pos+pad*dv/dr
        umax=0.5*dr+pad
        disp=dv/dr*(dr+2*pad)/(npint-1)

        pts=np.array( [ start + i * disp for i in range(npint) ] )
        u=np.linspace(-umax,umax,npint)
        d=atoms.get_rho_line(start,end,npint)
        dm=[]
        for i in range(self.numatoms):
            dm.append(d * self.mask_factor(i,np.transpose(pts)))

        d=   fem1d.Function1D(np.column_stack((u,d)),npoints)
        dm=[fem1d.Function1D(np.column_stack((u,j)),npoints) for j in dm]

        # Subtract
        fout=open("{0:s}_mask_{1:d}_{2:d}.dat".format(self.basename,id1,id2),"w")
        subtr=[]

        disp=dv/dr*(dr+2*pad)/(nptot-1)
        du=np.sqrt(sum(disp*disp))
            
        for i in range(nptot):
            u_val=u[0]+i*du
            pt=start+i*disp
            fout.write( "{0:18.12e} {1:18.12e}".format( u_val,d(u_val)) )
            for j in dm:
                fout.write( " {0:18.12e}".format( j(u_val) ) )
            fout.write("\n")
        fout.close()
        return

    def write_xyz(self,filename=None):
        if filename==None:
            filename=self.basename+".xyz"
        outfile=open(filename,"w")
        outfile.write("{0:d}\n\n".format(len(self.atoms)))
        for atom in self.atoms:
            outfile.write("{0:2s} {1:12.6f}{2:12.6f}{3:12.6f}\n".format(
                ptable_inv[atom.z],
                atom.pos[0]/A2au,atom.pos[1]/A2au,atom.pos[2]/A2au))
        outfile.close()

def init_genpotProc(cubefile,bublib,onlysetup):
    infile=open("genpot.inp","w")
    infile.write("""nuclear_pot=on
bubbles=on

density {
     origin="""+cubefile+"""
     bubbles="""+bublib+"""
}

potential {
     file=pot.cub
}
""")
    infile.close()
    if not onlysetup:
        proc=subprocess.Popen(["genpot"],stdout=open("genpot.out","w"),
                stderr=open("/dev/null","w"))
    else:

        proc=subprocess.Popen(["echo"])
    return proc

def t21_to_xyz(t21filename):
    proc=subprocess.Popen(
            ['adfreport','-r','Geometry%nr of atoms',t21filename],
            stdout=subprocess.PIPE)
    proc.wait()
    numatoms=int(proc.communicate()[0])

    proc=subprocess.Popen(
            ['adfreport','-r','Geometry%atomtype',t21filename],
            stdout=subprocess.PIPE)
    proc.wait()
    types=proc.communicate()[0].replace('}','').replace('{','').split()

    proc=subprocess.Popen(['adfreport','-r',
            'Geometry%fragment and atomtype index',t21filename]
            ,stdout=subprocess.PIPE)
    proc.wait()
    type_ids=[types[int(j)-1] for j in 
            proc.communicate()[0].split()[numatoms:]]

    proc=subprocess.Popen(
            ['adfreport','-r','Geometry%xyz',t21filename],
            stdout=subprocess.PIPE)
    proc.wait()
    coords=proc.communicate()[0].split()
    string="{:d}\n".format(numatoms)
    string+=t21filename+"\n"
    for i in range(numatoms):
        string+="  {0:2s}".format(type_ids[i].capitalize())
        for j in coords[3*i:3*(i+1)]:
            string+="{0:20.10f}".format(float(j)/A2au)
        string+="\n"
    return string

if __name__=="__main__":

    np.seterr(all="ignore")

    import argparse,shutil

    parser=argparse.ArgumentParser(usage="%(prog)s <origin> <step> <radius> [options]")
    parser.add_argument('origin',
            action="store")
    parser.add_argument('--no-gen-bubbles',
            dest='genbub',
            action='store_false',
            default=True)
    parser.add_argument('--plot',
            dest='plot',
            action='store',
            nargs='+',
            default=[],
            help="Plot subtracted density between given atom pairs")
    parser.add_argument('--xyz',
            dest='printxyz',
            action='store_true',
            default=False)
    parser.add_argument('--no-print-lib',
            dest='printlib',
            action='store_false',
            default=True)
    parser.add_argument('--print-bubbles',
            dest='printbubs',
            action='store_true',
            default=False)
    parser.add_argument('--genpot',
            dest='genpot',
            action='store_true',
            default=False,
            help="Plot subtracted density")
    parser.add_argument('--only-setup',
            dest='onlysetup',
            action='store_true',
            default=False,
            help="Don't run genpot, just generate input")
    parser.add_argument('--step','-s',
            nargs='+',
            type=float,
            help="Grid step in au")
    parser.add_argument('--radius','-r',
            type=float,
            default=8.0,
            help="Minimum distance to any atom to the box sides in au")
    parser.add_argument('--numiter','-i',
            nargs='?',
            dest='numiter',
            default=2,
            type=int,
            help="Number of iterations in bubbles generation")
    parser.add_argument('--ncell',
            nargs='?',
            dest='ncell',
            default=100,
            type=int,
            help="Number of cells")
    args=parser.parse_args()

# PARAMETERS
    rmax=20.
    ncell=args.ncell
    npoints=7

    basename=args.origin.split('.')[0]
# READ T21 FILE
    atoms=AtomEnsemble(args.origin,rmax,ncell,npoints)
# GENERATE BUBBLES
    if args.printxyz:
        atoms.write_xyz()
        quit()
    if args.genbub:
        atoms.calc_bubbles(lmax=1,numiter=args.numiter)

# PRINT BUBLIB
        if args.printlib:
            atoms.print_bublib()
# PRINT BUBBLES
        if args.printbubs:
            atoms.print_bubs(".bub")
#            atoms.print_bubs("pre")
#            atoms.atoms[0].bubbles[0].prune()
#            atoms.print_bubs("pos")

# PRINT SUBTRACTED DENSITY
# Plot
        for pair in args.plot:
            i,j=[int(k) for k in pair.split(',')]
            atoms.write_diff(i,j)
            atoms.write_masked_dens(i,j)
#            atoms.write_diff(0,1,ncell=20)
#            plt.plot(z, subtr)
#            plt.show()

        # Subtract on plane
        #npoi=40
        #for z_val in [-8.+16./npoi*i for i in range(npoi+1)]:
        #    for y_val in [-8.+16./npoi*i for i in range(npoi+1)]:
        #        s=[]
        #        d_v=atoms.get_rho([0.,y_val,z_val])
        #        for g in f:
        #            for h in g:
        #                s.append(h((0.,y_val,z_val)))
        #        print "{0:18.12e} {1:18.12e} {2:18.12e} {3:18.12e}".format(
        #                z_val,y_val,d_v,sum(s))
        #    print
    if args.genpot:
        try:
            os.mkdir("genpot")
        except:
            pass

        results=[]
        for step in args.step:
            cube=atoms.gen_rho_cube(radius=args.radius,step=step)

            newdir=os.path.join("genpot","{0:G}".format(1000*step))
            try:
                os.mkdir(newdir)
            except:
                pass
            shutil.copy(cube,newdir)
            shutil.copy("bublib.dat",newdir)
            
            olddir=os.getcwd()
            os.chdir(newdir)
            init_genpotProc(cube,"bublib.dat",args.onlysetup).wait()
            results.append(( step,os.path.join(newdir,"genpot.out") ))
            os.chdir(olddir)
        if args.onlysetup: quit()
        # grep ADF result
        ADFval=float(subprocess.Popen(["grep","Coulomb.*=",basename+".out"],
               stdout=subprocess.PIPE).communicate()[0].split()[-1])
        print "# Step (bohr)      Value (Hartree)    Error (mHartree)     Rel.err.(ppm)"
        for step,file in results:
            val=float(subprocess.Popen(["grep","Total   ",file],
                stdout=subprocess.PIPE).communicate()[0].split()[-1])
            print "{0:12.6f}{1:24.8f}{2:18.6f}{3:18.6f}".\
                format(step,val,(val-ADFval)*1.e3,(val-ADFval)/ADFval*1.e6)
        print " ADF value  {0:24.8f}".format(ADFval)

                       
