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
#!@PYTHON_EXECUTABLE@
# -*- coding: latin-1 -*-
#
# Written by Jonas Juselius <jonas.juselius@chem.uit.no> 
# University of Tromsø, 2006
#
# vim:syntax=python

import os
import re
import argparse
import getkw

def main():
    usage="usage: %(prog)s [options] [config] infile"
    cmdln=argparse.ArgumentParser(usage=usage)
    cmdln.add_argument('infile', action='store',nargs="?",
            help='input file', default='genpot.inp')
    cmdln.add_argument('-t','--title', action='store', dest='title',
            help='title of job')
    cmdln.add_argument('-d','--debug', action='store', dest='debug',
            help='debug level')
    cmdln.add_argument('-M','--mpi', action='store_true', dest='mpirun',
            default=False, help='mpi parallel run')
    cmdln.add_argument('-o','--output', action='store', dest='outfile',
            help='base name for output file(s)')
    cmdln.add_argument('-l','--lip', action='store', dest='lip',
            help='lip order (default 7)')
    cmdln.add_argument('-D','--debug-file', action='store', dest='debugf',
            help='debug file name')
    cmdln.add_argument('-T','--test', action='store', dest='test',
            help='use test grid of dimension NxNxN')
    cmdln.add_argument('-m','--mode', action='store', dest='mode',
            help='calulation mode (incore|iomode)')
#    cmdln.add_argument('-v','--verbose', action='store_true', dest='verbose',
#            help='be verbose')
#    cmdln.add_argument('-q','--quiet', action='store_false', dest='verbose',
#            help='be quiet')
    args=cmdln.parse_args()

    start=getkw.Section('start')
    start.set_status(True)
    start.add_kw('title', 'STR', ('',))
    start.add_kw('debug_file', 'STR', ('DEBUG.out',))
    start.add_kw('debug', 'INT', (0,))
    start.add_kw('verbosity', 'INT', (1,))
    start.add_kw('outfile', 'STR', ('kusse.out',))
    start.add_kw('mpirun', 'BOOL', (False,))
    start.add_kw('mode', 'STR', ('incore',))
    start.add_kw('calc', 'STR', None)
    start.add_kw('selfint', 'BOOL', (True,))
    start.add_kw('selfint_an', 'BOOL', (True,))
    start.add_kw('errpot', 'BOOL', (False,))
    start.add_kw('bubbles', 'BOOL', (False,))
    start.add_kw('gaussbub', 'BOOL', (False,))
    start.add_kw('nuclear_pot', 'BOOL', (True,))
    start.add_kw('lip_points', 'INT', (7,))
    start.add_kw('spectrum', 'BOOL', (False,))
# Marbles
#     start.add_kw('marbles', 'BOOL', None)

#     grid=getkw.Section('grid')
#     grid.add_kw('lip_points', 'INT', (7,))
#     grid.add_kw('dim', 'INT_ARRAY', 3)
#     grid.add_kw('xrange', 'DBL_ARRAY', 2)
#     grid.add_kw('yrange', 'DBL_ARRAY', 2)
#     grid.add_kw('zrange', 'DBL_ARRAY', 2)

    density=getkw.Section('density')
    density.add_kw('origin','STR', None)
    density.add_kw('pbc','STR', None)
    density.add_kw('ncells', 'INT_ARRAY', 3)
    density.add_kw('xrange', 'DBL_ARRAY', 2)
    density.add_kw('yrange', 'DBL_ARRAY', 2)
    density.add_kw('zrange', 'DBL_ARRAY', 2)
    density.add_kw('gaussian','STR', None)
    density.add_kw('nucfile','STR', None)
    density.add_kw('writetestdens','BOOL', (True,))
    density.add_kw('slater','STR', None)
    density.add_kw('bubbles', 'STR', None)
    density.add_kw('save','STR',None)

    integral=getkw.Section('integral')
    integral.add_kw('nlin', 'INT', (12,))
    integral.add_kw('nlog', 'INT', (8,))
    integral.add_kw('tlog', 'DBL', (2.0,))
    integral.add_kw('tend', 'DBL', (500.0,))
    integral.add_kw('trange', 'DBL_ARRAY', (1.e-4,1.e5,))

    potential=getkw.Section('potential')
    potential.add_kw('file','STR', None)
    potential.add_kw('gopenmol','STR', None)
    potential.add_kw('save','STR', None)
    potential.add_kw('gopenmol_step', 'INT', (2,))
    potential.add_kw('cubeplot','STR', None)
    potential.add_kw('cubeplot_step', 'INT', (2,))
    potential.add_kw('new_grid', 'BOOL', (False,))
    potential.add_kw('analytical', 'BOOL', (True,))
    potential.add_kw('ncells', 'INT_ARRAY', 3)
    potential.add_kw('xrange', 'DBL_ARRAY', 2)
    potential.add_kw('yrange', 'DBL_ARRAY', 2)
    potential.add_kw('zrange', 'DBL_ARRAY', 2)
    potential.add_kw('pbc','STR', None)
    potential.add_kw('nslices', 'INT', (0,))

#     dens=getkw.Section('dens')
#     dens.set_arg('BOOL', (False,))
#     dens.add_kw('density','STR', None)

#     start.add_sect(grid)
    start.add_sect(density)
    start.add_sect(integral)
    start.add_sect(potential)
#     start.add_sect(dens)
    
    if args.infile is not None:
        input=getkw.GetkwParser(start)
        inkw=input.parseFile(args.infile)
        inkw.equalize(start)
    else:
        inkw=start

    if args.title:
        inkw.setkw('title', args.title)
    if args.debug:
        inkw.setkw('debug', args.debug)
    if args.mpirun:
        inkw.setkw('mpirun', args.mpirun)
    if args.outfile:
        inkw.setkw('outfile', args.outfile)
    if args.lip:
        inkw.setkw('lip', args.lip)
    if args.debugf:
        inkw.setkw('debug_file', args.debugf)
    if args.test:
        tmp=inkw.getsect('grid')
        tmp.setkw('dim', (args.test, args.test, args.test))
    if args.mode:
        inkw.setkw('mode', args.mode)
#     if density is not None:
#         inkw.setkw('density', density)

    inkw.xvalidate(start)
#     inkw.validate()

#     args.infile='GENPOT.' + str(os.getpid())
    args.infile='GENPOT'
    fd=open(args.infile,'w')
    print >> fd, inkw
    fd.close()
#     os.system('genpot.x < ' + args.infile)
    os.system('@CMAKE_INSTALL_PREFIX@/bin/@GENPOT_EXECUTABLE@')
#     os.unlink(args.infile)


if __name__ == '__main__':
    main()

