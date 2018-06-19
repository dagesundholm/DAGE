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

import numpy, sys, os, re

run_ridft="/v/users/sergio/TURBOMOLE/bin/x86_64-unknown-linux-gnu/ridft -prop"
AU2A = 0.52917726

class Atom:
    def __init__(self,input):
        self.elem=input[3]
        self.crd=[]
        for i in input[0:3]:
            self.crd.append(float(i))

controlfile=open("control")

coordfile=open("coord")
atoms=[]
if (re.match("\$coord",coordfile.readline())):
    for line in coordfile:
        if(not(re.match("\$",line))):
            atoms.append(Atom(line.split()))
        else:
            break

newcontrolfile=open("control_new","w")
done=False
for line in controlfile:
# Substitute pointval section, or add it if it was not found
    if( re.match("\$pointval",line) or  (re.match("\$end",line) and not(done))  ):
        newcontrolfile.write("$pointval dens geo=point fmt=bub\n")
        for i in atoms:
            newcontrolfile.write("{0:20.10f}{1:20.10f}{2:20.10f}\n".format(\
                    i.crd[0],i.crd[1],i.crd[2]))
        for line2 in controlfile:
            if(re.match("\$",line2)):
                newcontrolfile.write(line2)
                break
        if re.match("\$end",line):
            newcontrolfile.write("$end\n")
        done=True
    else:
        newcontrolfile.write(line)

controlfile.close()
newcontrolfile.close()

os.system("mv control control.backup")
os.system("mv control_new control")

os.system(run_ridft)
cuspsfile=open("td.bub")
counter=0
for line in cuspsfile:
    if not(re.match("#",line)):
        data=line.split()
        atoms[counter].cusp=float(data[-1])
        counter=counter+1
        if(counter==len(atoms)): break
cuspsfile.close()

os.system("mv control control_new")
os.system("mv control.backup control")

output=open("nuclei","w")
output.write("%d \n" % len(atoms))
output.write("Nuclei ready for genpot\n")
for i in atoms:
    # Swap x and z coordinates, because turbomole plots them wrong!
    output.write("{0:s}{1:20.10f}{2:20.10f}{3:20.10f}{4:20.10f}\n".format(\
            i.elem,i.crd[2]*AU2A,i.crd[1]*AU2A,i.crd[0]*AU2A,i.cusp))
output.close()
