#!/usr/bin/env python3
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
 *--------------------------------------------------------------------------"""
# chewtm.py
# Convert turbomole RHF wavefunction (coordinates, basis set, mo coefficients)
# into easy-to-parse format. Run in the same directory of the turbomole
# calculation, where the coord, control and mos file are located, with no
# arguments. Output to stdout.

# The calculation must be performed with:
#  - a spherical GTO basis (5d/7f)
#  - c1 symmetry group (ie no symmetry)
import re
import numpy as np
import sys

# Stuff needed to reorder the shells.

######## SPHERICAL HARMONICS MAPPINGS #########
# TURBOMOLE                             SANE
#  s                                   S(0, 0)
#  px                                  S(1, 1) = x
#  py                                  S(1,-1) = y
#  pz                                  S(1, 0) = z
#  d0 = (-xx-yy+2zz)/sqrt(12)          S(2, 0) = (2zz-xx-yy)/2
#  d1a = xz                            S(2, 1) = √3 xz
#  d1b = yz                            S(2,-1) = √3 yz
#  d2a = xy                            S(2,-2) = √3 xy
#  d2b = (xx-yy)/2                     S(2, 2) = √3/2 (xx-yy)

c=3.**(-0.5)
trans_list=[ [ (0, 1.0) ],
             [ (1, 1.0),  (2, 1.0),  (0, 1.0) ],
             [ (3, 1.0),  (2, 1.0),    (0, 1.0),   (1, 1.0), (4, 1.0) ] ]

reorder_mos=lambda l,inp:[ coeff*inp[pos] for pos, coeff in trans_list[l] ]

elements=[ "h",                                   "he",
           "li", "be",   "b", "c", "n", "o", "f", "ne" ]
ptable={ element:i+1 for i, element in enumerate(elements) }

dfact = lambda n: np.product( range(1,n+1,2) )

overlap   = lambda l, expo1, expo2: (
    2.**(l+1.5) * (expo1*expo2)**(0.5*l+0.75) * (expo1+expo2)**-(l+1.5) )

# Contracted shell normalization factor
def the_factor(l, expos, coeffs):
    values=[ coeffs[i] * coeffs[j] *
              overlap(l, expos[i], expos[j] )
              for j in range(len(expos))
              for i in range(len(expos)) ]
    return np.sqrt(1./sum( values ))

# Read basis set
def parse_tm_basis(basisfilename):
    with open(basisfilename) as basisfile:
        raw_text=re.sub("\s*#.*","",re.search(r"\$basis\s*\n(.*)\n\$end",basisfile.read(),flags=re.DOTALL).group(1))
    atom_blocks=re.findall("\*\s*\n([A-Za-z]*)\s*.*\n\*([^*]*)", raw_text)
    atom_dict={}
    ldict={"s":0, "p":1, "d":2, "f":3, "g":4, "h":5, "i":6}
    basis=[]
    for iatom,atom in enumerate(atom_blocks):
        basis.append([])
        atom_dict[atom[0]]=iatom
        chunks=re.split("[0-9]*.*([spdfghi])",atom[1].strip())
        for i in range(1,len(chunks),2):
            l=ldict[chunks[i]]
            while l >= len(basis[-1]):
                basis[-1].append([])
            basis[-1][l].append( [[float(k) for k in j.split()]
                for j in chunks[i+1].strip().split('\n')] )
            shell=basis[-1][l][-1]
            factor=the_factor( l, [ i[0] for i in shell ],
                                  [ i[1] for i in shell ] )
            for primitive in shell:
                primitive[1]*=factor
#            for primitive in :
    return atom_dict,basis

# Convert basis set to string
def machinize_basis(basis):
    output=""
    sumcum = lambda list: [ 1+ sum(list[:i]) for i in range(len(list)+1) ]
    # Starting point of each atom type + total number of "l's"
    iatt = sumcum([len(atomtype) for atomtype in basis ])
    il   = sumcum([len(l    )    for atomtype in basis for l in atomtype ])
    iprim= sumcum([len(shell)    for atomtype in basis for l in atomtype for
        shell in l])
    expos =[ prim[0] for atomtype in basis for l in atomtype for shell in l for prim in shell]
    coeffs=[ prim[1] for atomtype in basis for l in atomtype for shell in l for prim in shell]
    output+= "{:5d}".format(len(iatt)-1)+'\n'
    output+="".join(["{:5d}".format(i) for i in iatt])+'\n'
    output+="".join(["{:5d}".format(i) for i in il])+'\n'
    output+="".join(["{:5d}".format(i) for i in iprim])+'\n'
    output+="".join(["{:24.16e}".format(i) for i in coeffs])+'\n'
    output+="".join(["{:24.16e}".format(i) for i in expos])+'\n'
    return output

# Convert coords to string
def machinize_coord(coords, atom_dict):
    output=str(len(coords))+"\n"
    output+="\n".join([ "{:6d}  {:6d}".format(atom_dict[c[1]]+1,ptable[c[1]])+"   "+"".join([ "{:24.16f}".format(i) for i in c[0]]) for c in
        coords])+'\n'
    return output

# Convert mo coefficients to string
def machinize_mos(coords, atom_dict, basis, mos):
    output=""
    for mo in mo_blocks:
        k=0
        mo_out=[]
        for pos, atom in coords:
            for l, shellblock in enumerate(basis[atom_dict[atom]]):
                for shell in shellblock:
                    mo_out+=reorder_mos(l, mo[k:k+2*l+1])
                    k+=2*l+1
        for c in mo_out:
            output+="{0:22.14e}".format(c)
        output+="\n"
    return output

# Read coords
def parse_tm_coord(coordfilename):
    lines=[]
    with open(coordfilename) as coordfile:
        lines=coordfile.readlines()
    start=next( i for i in range(len(lines)) if re.match("\$coord",lines[i]))+1
    end  =next( i for i in range(start,len(lines)) if re.match("\$",lines[i]))
    coords=[]
    for line in lines[start:end]:
        caca=line.split()
        coords.append([ [float(i) for i in caca[:3]], caca[3]])
    return coords

# Get the number of occupied shells from control file.
def get_nocc():
    with open("control","r") as controlfile:
        for line in controlfile:
            if "$closed shells" in line:
                return int(controlfile.readline().split()[1].split("-")[-1])

# Read mo coefficients
def get_mo_blocks():
    def split20(string):
        if len(string)>=20:
            return [float(string[:20].replace("D","e"))] + split20(string[20:])
        else:
            return []

    with open("mos","r") as mosin:
        while(not re.search("eigenvalue=", mosin.readline())):
            pass
        mo_blocks=[""]
        for line in mosin:
            if re.search("\$end",line):
                mo_blocks[-1]= split20(mo_blocks[-1])
                break
            else:
                if re.search("eigenvalue=", line):
                    mo_blocks[-1]= split20(mo_blocks[-1])
                    mo_blocks.append("")
                else:
                    mo_blocks[-1]+=line[:line.index("\n")]
    return mo_blocks

if __name__=="__main__":
    #with open("bubbles_input","w") as output:
    with sys.stdout as output:
        atom_dict,basis=parse_tm_basis("basis")

        output.write("GTO\n")

        output.write(machinize_basis(basis))

        coords=parse_tm_coord("coord")
        output.write(machinize_coord(coords,atom_dict))

        nocc=get_nocc()
        mo_blocks=get_mo_blocks()
        ntot=len(mo_blocks)
        output.write("{:6d}{:6d}\n".format(ntot,nocc))
        output.write(machinize_mos(coords, atom_dict, basis, mo_blocks))
