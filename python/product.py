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
import matplotlib.pyplot as plt
from . import fem1d,bubblify_adf

f=open("c2h2_aa.xyz")
f.readline()
f.readline()
atoms=[]
for j in range(4):
    atoms.append(bubblify_adf.Atom([1.889725949*float(i) for i in\
        f.readline().split()[1:]],6.))
class Funcz:
    def __init__(self,name):
        self.name=name
        fs=[]
        fp=[]
        for j in range(4):
            fn=name+"_bub_"+str(j+1)+".dat"

            bub=np.loadtxt(fn              )
            fs.append(bubblify_adf.Bubble(bub,7,0,0,atoms[j]))

            bub=np.loadtxt(fn,usecols=[0,4])
            bub[:,1]/=3.0
            fp.append(bubblify_adf.Bubble(bub,7,1,1,atoms[j]))

        self.f_d=fem1d.Function1D( np.loadtxt(name+"_ns_z_p.dat"), 7 )
        self.bub_s=fs
        self.bub_p=fp
    
    def __call__(self,z):
        s=0.
        s+=self.f_d(z)
        for j in range(4): 
            r=self.bub_s[j]([0.,0.,z]) + self.bub_p[j] ([0.,0.,z])
            s+=r
#            for i in ( self.bub_s[j], self.bub_p[j] ):
#                r=i([0.,0.,z])
#                print '{0:20.14e}'.format(r),
#                s+=r
#            print
        return s

#d_t=fem1d.Function1D( np.loadtxt("dens_z_p.dat"), 7 )
d_d=fem1d.Function1D( np.loadtxt("dens_ns_z_p.dat"), 7 )
v_d=fem1d.Function1D( np.loadtxt("pot_ns_z_p.dat"), 7 )
u_d=fem1d.Function1D( np.loadtxt("rhov_ns_z_p.dat"), 7 )
dens=Funcz("dens")
pot=Funcz("pot")

def der(f,x):
    dd=1e-10
    return (f(x+dd)-f(x))/dd

def der2(f,x):
    dd=1e-6
    return (f(x)+f(x+2*dd)-2*f(x+dd))/dd**2

dens_0=[ dens(atoms[i].pos[2]) - dens.bub_s[i].eval_rad(0.) for i in range(4)]
pot_0=[ pot(atoms[i].pos[2]) - pot.bub_s[i].eval_rad(0.) for i in range(4)]

dens_1=[ der(dens,atoms[i].pos[2]) - der(dens.bub_s[i].eval_rad,0.)- der(dens.bub_p[i].eval_rad,0.) for i in range(4)]
pot_1=[ der(pot,atoms[i].pos[2]) - der(pot.bub_s[i].eval_rad,0.)- der(pot.bub_p[i].eval_rad,0.) for i in range(4)]

dens_2=[ der2(dens,atoms[i].pos[2]) - der2(dens.bub_s[i].eval_rad,0.)- der2(dens.bub_p[i].eval_rad,0.) for i in range(4)]
pot_2=[ der2(pot,atoms[i].pos[2]) - der2(pot.bub_s[i].eval_rad,0.)- der2(pot.bub_p[i].eval_rad,0.) for i in range(4)]

def func_s(z):
    s=0.
    for i,bs in enumerate(dens.bub_s):
        s+=          bs([0.,0.,z])*( pot.bub_s[i]([0.,0.,z]) + pot_0[i]) + \
           pot.bub_s[i]([0.,0.,z]) * dens_0[i]
    return s

ds=dens.bub_s[0].eval_rad
ps=pot.bub_s[0].eval_rad
#for r in np.linspace(0,10,2401):
#    print r,ds(r)*(ps(r) + pot_0[0]) + ps(r) * dens_0[0]
#quit()

#def func_p(z):
#    s=0.
#    for i,bs in enumerate(dens.bub_p):
#        s+=          bs([0.,0.,z])*( pot.bub_s[i]([0.,0.,z]) + pot_0[i]) + \
#           pot.bub_p[i]([0.,0.,z]) * dens_0[i]
#    for i,bs in enumerate(dens.bub_s):
#        s+=bs([0.,0.,z])*( pot.bub_p[i]([0.,0.,z]) + pot_1[i]* (z-atoms[i].pos[2])) + \
#           pot.bub_s[i]([0.,0.,z]) * dens_1[i] * (z-atoms[i].pos[2])
#    return s

#def func_d(z):
#    s=0.
#    for i,bs in enumerate(dens.bub_s):
#        s+=bs([0.,0.,z])*pot_2[i]*0.5*(z-atoms[i].pos[2])**2
#    for i,bp in enumerate(dens.bub_p):
#        s+=bp([0.,0.,z])*( pot.bub_p[i]([0.,0.,z]) + pot_1[i]* (z-atoms[i].pos[2])) + \
#           pot.bub_p[i]([0.,0.,z]) * dens_1[i] * (z-atoms[i].pos[2])
#    return s

def func_p(z):
    s=0.
    for i,bs in enumerate(dens.bub_s):
        s+=bs([0.,0.,z])*( pot.bub_p[i]([0.,0.,z]) + pot_1[i]* (z-atoms[i].pos[2]))
    for i,bp in enumerate(dens.bub_p):
        s+=bp([0.,0.,z])*( pot.bub_s[i]([0.,0.,z]) + pot_0[i])
    return s

def func_d(z):
    s=0.
    for i,bs in enumerate(dens.bub_s):
        s+=bs([0.,0.,z])*pot_2[i]*0.5*(z-atoms[i].pos[2])**2
    for i,bp in enumerate(dens.bub_p):
        s+=bp([0.,0.,z])*( pot.bub_p[i]([0.,0.,z]) + pot_1[i]* (z-atoms[i].pos[2]))
    return s

for at in atoms:
    plt.axvline(at.pos[2],color='grey',linestyle='-')

zin= np.loadtxt("pot_ns_z_p.dat",usecols=[0] )

npoints=1000
z=np.linspace(zin[0],zin[-1],npoints)

outfile=open("rhov_ns_z_py.dat","w")
for i in z:
    outfile.write("{0:20.12f} {1:20.12f}\n".format(i,pot(i)*dens(i)-func_s(i)-func_p(i)))
outfile.close()

#plt.plot(z, [dens.bub_s[0]([0.,0.,i]) for i in z],label="dens")
#plt.plot(z, [dens(i)-dens.bub_s[0]([0.,0.,i]) for i in z],label="dens")
#plt.plot(z, [pot(i)*dens(i) for i in z],label="dens*pot")
#plt.plot(z, [pot(i)*dens(i)-func_s(i) for i in z],label="dens*pot minus s")
#plt.plot(z, [func_p(i) for i in z],label="dens*pot p")
#plt.plot(z, [pot(i)*dens(i)-func_s(i)-func_p(i) for i in z],label="dens*pot minus p")
plt.plot(z, [pot(i)*dens(i)-func_s(i)-func_p(i) -func_d(i) for i in z],label="dens*pot minus d")
plt.plot(z, [u_d(i) for i in z],label="rhov_d")
plt.plot(z, [  sum([ff([0.,0.,i]) for ff in pot.bub_s+pot.bub_p])*d_d(i) for i in z],label="pot_bubs * rho_diff")
plt.plot(z, [  sum([(pot.bub_s[j]([0.,0.,i])+pot.bub_p[j]([0.,0.,i]))*dens_0[j] for j in range(4)]) for i in z],label="pot_bubs * rho_diff")
plt.plot(z, [d_d(i) for i in z],label="rhov_d")

plt.legend()
#plt.ylim(-0.1,0.6)
plt.show()
